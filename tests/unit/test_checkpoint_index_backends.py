# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The atomic checkpoint index each storage backend keeps for the adapter (issue #301).

Redis runs against a dict-backed fake of the client (strings, hashes and
pipelines); PostgreSQL and MySQL against the statement-recording stubs
their own test modules use; OpenSearch against a mocked client; Oracle
against a fake connection over one dict of rows. The live behaviour — two
processes writing one thread and losing nothing — is
``tests/integration/test_checkpoint_index_concurrency.py``.
"""

from __future__ import annotations

import json
import sys
import types
import warnings
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest

from locus.memory.backends import CheckpointIndexStore
from locus.memory.backends.adapters import StorageBackendAdapter
from locus.memory.backends.checkpoint_index import index_key, keeps_index_natively, newest_first
from locus.memory.backends.mysql import MySQLBackend, _like_literal
from locus.memory.backends.opensearch import OpenSearchBackend
from locus.memory.backends.oracle import OracleBackend, _read_json_column
from locus.memory.backends.postgresql import PostgreSQLBackend
from locus.memory.backends.redis import RedisBackend
from tests.unit.test_memory_backends_mysql import (
    _stub_mysql_connector,
)
from tests.unit.test_memory_backends_mysql import (
    _StubConn as _MySQLConn,
)
from tests.unit.test_memory_backends_mysql import (
    _StubCursor as _MySQLCursor,
)
from tests.unit.test_memory_backends_postgresql import _stub_asyncpg
from tests.unit.test_memory_backends_postgresql import _StubConn as _PgConn


def _entry(checkpoint_id: str, timestamp: str = "2026-01-01T00:00:00+00:00") -> dict[str, Any]:
    return {
        "checkpoint_id": checkpoint_id,
        "timestamp": timestamp,
        "metadata": {},
        "iteration": 0,
        "size": 10,
    }


# ---------------------------------------------------------------------------
# The shared module
# ---------------------------------------------------------------------------


class TestCheckpointIndexModule:
    def test_index_key_is_the_adapter_suffix(self) -> None:
        assert index_key("t") == "t:_checkpoints"

    def test_newest_first_orders_by_timestamp_and_is_stable(self) -> None:
        entries = [
            _entry("old", "2026-01-01T00:00:00"),
            _entry("new", "2026-01-03T00:00:00"),
            _entry("mid-a", "2026-01-02T00:00:00"),
            _entry("mid-b", "2026-01-02T00:00:00"),
            {"checkpoint_id": "no-time"},
        ]
        assert [e["checkpoint_id"] for e in newest_first(entries)] == [
            "new",
            "mid-a",
            "mid-b",
            "old",
            "no-time",
        ]

    @pytest.mark.parametrize(
        "backend",
        [
            RedisBackend(url="redis://x"),
            PostgreSQLBackend(),
            MySQLBackend(),
            OpenSearchBackend(hosts=["http://x:9200"]),
            OracleBackend(dsn="x"),
        ],
        ids=["redis", "postgresql", "mysql", "opensearch", "oracle"],
    )
    def test_every_shipped_backend_keeps_the_index_itself(self, backend: Any) -> None:
        assert isinstance(backend, CheckpointIndexStore)
        assert keeps_index_natively(backend), "all four index methods, each a coroutine function"


# ---------------------------------------------------------------------------
# Redis: a hash per thread, the legacy string migrated
# ---------------------------------------------------------------------------


class _FakePipeline:
    """``redis.asyncio``'s pipeline on the fake.

    Commands run at once after ``WATCH`` and queue after ``MULTI``, as in
    redis-py; ``EXEC`` is discarded with a ``WatchError`` when a watched key
    was written since the ``WATCH``.
    """

    def __init__(self, client: _FakeRedis) -> None:
        self._client = client
        self._ops: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        self._watched: dict[str, int] = {}
        self._explicit = False

    async def __aenter__(self) -> _FakePipeline:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._ops.clear()
        self._watched.clear()

    async def watch(self, *keys: str) -> bool:
        self._client.calls.append("watch")
        for key in keys:
            self._watched[key] = self._client.versions.get(key, 0)
        return True

    def multi(self) -> None:
        self._explicit = True

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._client, name)
        if self._watched and not self._explicit:
            return method

        def queue(*args: Any, **kwargs: Any) -> _FakePipeline:
            self._ops.append((method, args, kwargs))
            return self

        return queue

    async def execute(self) -> list[Any]:
        from redis.exceptions import WatchError

        ops, self._ops = self._ops, []
        changed = any(self._client.versions.get(k, 0) != v for k, v in self._watched.items())
        self._watched.clear()
        self._explicit = False
        if changed:
            raise WatchError("Watched variable changed.")
        return [await method(*args, **kwargs) for method, args, kwargs in ops]


class _FakeRedis:
    """The slice of ``redis.asyncio.Redis`` the backend uses, on dicts.

    ``calls`` records every command; ``versions`` counts the writes to each
    key, which is what a ``WATCH`` compares at ``EXEC``; ``after_get`` runs
    once, right after the next ``GET`` read its value, to model another
    process acting between a read and the ``EXEC`` that depends on it.
    """

    def __init__(self) -> None:
        self.strings: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.ttls: dict[str, int] = {}
        self.calls: list[str] = []
        self.versions: dict[str, int] = {}
        self.after_get: Callable[[], Awaitable[None]] | None = None

    def _touch(self, key: str) -> None:
        self.versions[key] = self.versions.get(key, 0) + 1

    async def get(self, key: str) -> str | None:
        self.calls.append("get")
        value = self.strings.get(key)
        if self.after_get is not None:
            hook, self.after_get = self.after_get, None
            await hook()
        return value

    async def set(self, key: str, value: str) -> None:
        self.calls.append("set")
        self.strings[key] = value
        self._touch(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.calls.append("setex")
        self.strings[key] = value
        self.ttls[key] = ttl
        self._touch(key)

    async def exists(self, key: str) -> int:
        self.calls.append("exists")
        return int(key in self.strings or key in self.hashes)

    async def delete(self, *keys: str) -> int:
        self.calls.append("delete")
        removed = 0
        for key in keys:
            if self.strings.pop(key, None) is not None:
                removed += 1
                self._touch(key)
            if self.hashes.pop(key, None) is not None:
                removed += 1
                self._touch(key)
        return removed

    async def hset(self, key: str, field: str, value: str) -> int:
        self.calls.append("hset")
        fields = self.hashes.setdefault(key, {})
        created = field not in fields
        fields[field] = value
        self._touch(key)
        return int(created)

    async def hsetnx(self, key: str, field: str, value: str) -> int:
        self.calls.append("hsetnx")
        fields = self.hashes.setdefault(key, {})
        if field in fields:
            return 0
        fields[field] = value
        self._touch(key)
        return 1

    async def hdel(self, key: str, *fields: str) -> int:
        self.calls.append("hdel")
        stored = self.hashes.get(key, {})
        removed = sum(1 for field in fields if stored.pop(field, None) is not None)
        if not stored:
            self.hashes.pop(key, None)
        if removed:
            self._touch(key)
        return removed

    async def hgetall(self, key: str) -> dict[str, str]:
        self.calls.append("hgetall")
        return dict(self.hashes.get(key, {}))

    async def expire(self, key: str, ttl: int) -> int:
        self.calls.append("expire")
        self.ttls[key] = ttl
        return 1

    def pipeline(self, transaction: bool = True) -> _FakePipeline:
        return _FakePipeline(self)


def _redis(**kwargs: Any) -> tuple[RedisBackend, _FakeRedis]:
    backend = RedisBackend(url="redis://x", prefix="p:", **kwargs)
    client = _FakeRedis()
    backend._client = client  # type: ignore[assignment]
    return backend, client


_HASH = "p:t:_checkpoints:hash"
_LEGACY = "p:t:_checkpoints"


class TestRedisIndex:
    async def test_add_is_a_hash_field_and_list_reads_the_fields(self) -> None:
        backend, client = _redis()

        await backend.index_add("t", _entry("cp-0"))
        await backend.index_add("t", _entry("cp-1"))

        assert set(client.hashes[_HASH]) == {"cp-0", "cp-1"}
        assert json.loads(client.hashes[_HASH]["cp-1"]) == _entry("cp-1")
        assert sorted(e["checkpoint_id"] for e in await backend.index_list("t")) == ["cp-0", "cp-1"]
        assert _LEGACY not in client.strings, "nothing is written at the legacy key"

    async def test_add_replaces_the_entry_with_the_same_id(self) -> None:
        backend, client = _redis()
        await backend.index_add("t", _entry("cp-0", "2026-01-01T00:00:00"))

        await backend.index_add("t", _entry("cp-0", "2026-01-02T00:00:00"))

        [entry] = await backend.index_list("t")
        assert entry["timestamp"] == "2026-01-02T00:00:00"
        assert len(client.hashes[_HASH]) == 1

    async def test_remove_and_clear(self) -> None:
        backend, client = _redis()
        await backend.index_add("t", _entry("cp-0"))
        await backend.index_add("t", _entry("cp-1"))

        assert await backend.index_remove("t", "cp-0") is True
        assert await backend.index_remove("t", "cp-0") is False
        assert [e["checkpoint_id"] for e in await backend.index_list("t")] == ["cp-1"]
        assert await backend.index_clear("t") is True
        assert await backend.index_clear("t") is False
        assert await backend.index_list("t") == []
        assert _HASH not in client.hashes

    async def test_the_ttl_applies_to_the_hash(self) -> None:
        backend, client = _redis(ttl_seconds=60)

        await backend.index_add("t", _entry("cp-0"))

        assert client.ttls[_HASH] == 60

    async def test_without_a_ttl_the_hash_never_expires(self) -> None:
        backend, client = _redis()

        await backend.index_add("t", _entry("cp-0"))

        assert _HASH not in client.ttls

    async def test_a_legacy_string_is_read_until_the_next_write_migrates_it(self) -> None:
        backend, client = _redis()
        legacy = {"checkpoints": [_entry("old-1"), _entry("old-0"), {"timestamp": "no id"}]}
        client.strings[_LEGACY] = json.dumps(legacy)

        listed = await backend.index_list("t")
        assert sorted(e["checkpoint_id"] for e in listed) == ["old-0", "old-1"]
        assert _LEGACY in client.strings, "a read alone leaves the string in place"

        await backend.index_add("t", _entry("new"))

        assert _LEGACY not in client.strings
        assert set(client.hashes[_HASH]) == {"old-0", "old-1", "new"}
        assert sorted(e["checkpoint_id"] for e in await backend.index_list("t")) == [
            "new",
            "old-0",
            "old-1",
        ]

    async def test_a_removal_migrates_the_legacy_string_too(self) -> None:
        backend, client = _redis()
        client.strings[_LEGACY] = json.dumps({"checkpoints": [_entry("old-0"), _entry("old-1")]})

        assert await backend.index_remove("t", "old-0") is True

        assert _LEGACY not in client.strings
        assert [e["checkpoint_id"] for e in await backend.index_list("t")] == ["old-1"]

    async def test_a_removal_that_migrates_puts_the_ttl_on_the_hash(self) -> None:
        backend, client = _redis(ttl_seconds=60)
        client.strings[_LEGACY] = json.dumps({"checkpoints": [_entry("old-0"), _entry("old-1")]})

        assert await backend.index_remove("t", "old-0") is True

        assert client.ttls[_HASH] == 60, "the index expires with its checkpoints"

    async def test_without_a_legacy_string_a_write_costs_one_exists(self) -> None:
        backend, client = _redis()

        await backend.index_add("t", _entry("cp-0"))

        assert client.calls == ["exists", "hset"]

    async def test_the_migration_runs_under_a_watch_on_the_string(self) -> None:
        backend, client = _redis()
        client.strings[_LEGACY] = json.dumps({"checkpoints": [_entry("old-0")]})

        await backend.index_add("t", _entry("new"))

        assert client.calls[:3] == ["exists", "watch", "get"]
        assert client.calls[3:] == ["hsetnx", "delete", "hset"]

    async def test_migrating_a_legacy_string_warns_that_reading_them_ends(self) -> None:
        from locus.core.warnings import LocusDeprecationWarning

        backend, client = _redis()
        client.strings[_LEGACY] = json.dumps({"checkpoints": [_entry("old-0")]})

        with pytest.warns(LocusDeprecationWarning, match="7.0.0"):
            await backend.index_add("t", _entry("new"))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            await backend.index_add("t", _entry("newer"))  # no string left: silent

    async def test_a_migration_that_loses_the_race_writes_nothing(self) -> None:
        """Process one reads the string; process two migrates it and removes ``old-0``
        before process one's ``EXEC``, which Redis then discards: ``old-0`` stays gone."""
        backend, client = _redis()
        other = RedisBackend(url="redis://x", prefix="p:")
        other._client = client  # type: ignore[assignment]
        client.strings[_LEGACY] = json.dumps({"checkpoints": [_entry("old-0"), _entry("old-1")]})

        async def process_two() -> None:
            assert await other.index_remove("t", "old-0") is True

        client.after_get = process_two

        await backend.index_add("t", _entry("new"))

        assert set(client.hashes[_HASH]) == {"old-1", "new"}, "old-0 is not put back"
        assert _LEGACY not in client.strings

    async def test_migration_never_replaces_a_hash_entry_with_its_older_copy(self) -> None:
        backend, client = _redis()
        await backend.index_add("t", _entry("cp-0", "2026-01-02T00:00:00"))
        client.strings[_LEGACY] = json.dumps(
            {"checkpoints": [_entry("cp-0", "2026-01-01T00:00:00")]}
        )

        await backend.index_add("t", _entry("cp-1"))

        by_id = {e["checkpoint_id"]: e for e in await backend.index_list("t")}
        assert by_id["cp-0"]["timestamp"] == "2026-01-02T00:00:00"

    async def test_the_hash_wins_over_the_legacy_string_on_a_read(self) -> None:
        backend, client = _redis()
        await backend.index_add("t", _entry("cp-0", "2026-01-02T00:00:00"))
        client.strings[_LEGACY] = json.dumps(
            {"checkpoints": [_entry("cp-0", "2026-01-01T00:00:00")]}
        )

        [entry] = await backend.index_list("t")

        assert entry["timestamp"] == "2026-01-02T00:00:00"

    async def test_the_adapter_lists_newest_first_and_delete_clears_the_hash(self) -> None:
        from locus.core.state import AgentState

        backend, client = _redis()
        adapter = StorageBackendAdapter(backend)
        state = AgentState(agent_id="a")

        for step in range(3):
            await adapter.save(state, "t", f"cp-{step}")

        assert await adapter.list_checkpoints("t") == ["cp-2", "cp-1", "cp-0"]
        assert [i.checkpoint_id for i in await adapter.list("t")] == ["cp-2", "cp-1", "cp-0"]
        assert _LEGACY not in client.strings
        assert await adapter.delete("t") is True
        assert _HASH not in client.hashes
        assert await adapter.list_checkpoints("t") == []


# ---------------------------------------------------------------------------
# Oracle: the index row rewritten under its row lock
# ---------------------------------------------------------------------------


class _OracleLob:
    """A temporary CLOB as the backend writes it; read back, an async locator."""

    def __init__(self, text: str | None = None) -> None:
        self.text = text

    async def write(self, text: str) -> None:
        self.text = text

    def read(self) -> Any:
        async def _read() -> str | None:
            return self.text

        return _read()


class _SyncLob:
    def __init__(self, text: str) -> None:
        self.text = text

    def read(self) -> str:
        return self.text


class _OracleCursor:
    def __init__(self, conn: _OracleConn) -> None:
        self._conn = conn
        self._row: Any = None
        self.rowcount = 0

    async def __aenter__(self) -> _OracleCursor:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def execute(self, sql: str, binds: dict[str, Any] | None = None) -> None:
        self._row, self.rowcount = await self._conn.run(" ".join(sql.split()), binds or {})

    async def fetchone(self) -> Any:
        return self._row


class _OracleConn:
    """A fake connection over one dict of rows: ``rows[thread_id]`` is the data document.

    Statements and binds are recorded. ``failing_inserts`` makes that many
    ``INSERT`` statements raise ``insert_error`` after putting ``other_row``
    in place, which is what losing the first-write race to another process
    looks like: the winner's row is there when the loser retries.
    """

    def __init__(
        self,
        rows: dict[str, dict[str, Any]] | None = None,
        *,
        failing_inserts: int = 0,
        insert_error: str = "ORA-00001: unique constraint (LOCUS.SYS_C007) violated",
        other_row: dict[str, Any] | None = None,
    ) -> None:
        self.rows = rows if rows is not None else {}
        self.statements: list[str] = []
        self.binds: list[dict[str, Any]] = []
        self.commits = 0
        self.rollbacks = 0
        self.failing_inserts = failing_inserts
        self.insert_error = insert_error
        self.other_row = other_row

    def cursor(self) -> _OracleCursor:
        return _OracleCursor(self)

    async def createlob(self, _type: Any) -> _OracleLob:
        return _OracleLob()

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def run(self, sql: str, binds: dict[str, Any]) -> tuple[Any, int]:
        self.statements.append(sql)
        self.binds.append(binds)
        head = sql.upper()
        key = binds.get("thread_id")
        if "USER_TABLES" in head:
            return (1,), 0
        if head.startswith("SELECT DATA FROM"):
            # The server's JSON type: oracledb returns the document as a dict.
            return ((self.rows[key],), 0) if key in self.rows else (None, 0)
        if head.startswith("SELECT 1 FROM"):
            return ((1,), 0) if key in self.rows else (None, 0)
        if head.startswith("INSERT"):
            if self.failing_inserts:
                self.failing_inserts -= 1
                if self.other_row is not None:
                    self.rows[key] = self.other_row
                raise RuntimeError(self.insert_error)
            self.rows[key] = json.loads(binds["data"].text)
            return None, 1
        if head.startswith(("UPDATE", "MERGE")):
            self.rows[key] = json.loads(binds["data"].text)
            return None, 1
        if head.startswith("DELETE"):
            return None, int(self.rows.pop(key, None) is not None)
        raise AssertionError(f"unexpected SQL: {sql}")


class _OraclePool:
    def __init__(self, conn: _OracleConn) -> None:
        self._conn = conn

    def acquire(self) -> Any:
        conn = self._conn

        class _Acquired:
            async def __aenter__(self) -> _OracleConn:
                return conn

            async def __aexit__(self, *exc: object) -> None:
                return None

        return _Acquired()

    async def close(self, *, force: bool = False) -> None:
        return None


def _oracle(monkeypatch: pytest.MonkeyPatch, conn: _OracleConn) -> OracleBackend:
    fake = types.ModuleType("oracledb")
    fake.create_pool_async = lambda **kwargs: _OraclePool(conn)  # type: ignore[attr-defined]
    fake.DB_TYPE_CLOB = object()  # type: ignore[attr-defined]
    fake.AsyncConnectionPool = _OraclePool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oracledb", fake)
    return OracleBackend(dsn="x", user="u", password="p")  # noqa: S106


def _oracle_row(*entries: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {_ORACLE_KEY: {"checkpoints": list(entries)}}


_ORACLE_KEY = "t:_checkpoints"


class TestOracleJsonColumn:
    @pytest.mark.parametrize(
        "raw",
        [
            {"checkpoints": []},
            '{"checkpoints": []}',
            b'{"checkpoints": []}',
            _SyncLob('{"checkpoints": []}'),
            _OracleLob('{"checkpoints": []}'),
        ],
        ids=["dict", "str", "bytes", "sync-lob", "async-lob"],
    )
    async def test_every_shape_the_driver_returns_reads_as_the_document(self, raw: Any) -> None:
        assert await _read_json_column(raw) == {"checkpoints": []}

    async def test_nothing_reads_as_none(self) -> None:
        assert await _read_json_column(None) is None
        assert await _read_json_column("") is None


class TestOracleIndex:
    async def test_add_inserts_the_row_under_a_row_lock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _OracleConn()
        backend = _oracle(monkeypatch, conn)

        await backend.index_add("t", _entry("cp-0"))

        assert conn.rows == _oracle_row(_entry("cp-0"))
        select, insert = conn.statements[1:]  # [0] probes user_tables
        assert select.upper().startswith("SELECT DATA FROM")
        assert select.upper().endswith("FOR UPDATE")
        assert insert.upper().startswith("INSERT INTO")
        assert conn.binds[-1]["thread_id"] == _ORACLE_KEY
        assert (conn.commits, conn.rollbacks) == (1, 0)

    async def test_add_rewrites_the_row_newest_first_replacing_the_same_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _OracleConn(
            _oracle_row(
                _entry("cp-0", "2026-01-01T00:00:00"), _entry("cp-1", "2026-01-02T00:00:00")
            )
        )
        backend = _oracle(monkeypatch, conn)

        await backend.index_add("t", _entry("cp-0", "2026-01-03T00:00:00"))

        entries = conn.rows[_ORACLE_KEY]["checkpoints"]
        assert [e["checkpoint_id"] for e in entries] == ["cp-0", "cp-1"]
        assert entries[0]["timestamp"] == "2026-01-03T00:00:00"
        assert conn.statements[-1].upper().startswith("UPDATE")
        assert conn.commits == 1

    async def test_remove_reports_whether_the_entry_was_there_and_writes_only_then(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _OracleConn(_oracle_row(_entry("cp-0"), _entry("cp-1")))
        backend = _oracle(monkeypatch, conn)

        assert await backend.index_remove("t", "cp-0") is True
        assert conn.rows == _oracle_row(_entry("cp-1"))
        assert conn.commits == 1

        assert await backend.index_remove("t", "cp-9") is False
        assert conn.commits == 1, "nothing to write: the row lock goes with a rollback"
        assert conn.rollbacks == 1

    async def test_remove_without_a_row_never_creates_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _OracleConn()
        backend = _oracle(monkeypatch, conn)

        assert await backend.index_remove("t", "cp-0") is False

        assert conn.rows == {}
        assert not any(s.upper().startswith("INSERT") for s in conn.statements)
        assert conn.rollbacks == 1

    async def test_losing_the_first_insert_retries_against_the_winners_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _OracleConn(failing_inserts=1, other_row={"checkpoints": [_entry("theirs")]})
        backend = _oracle(monkeypatch, conn)

        await backend.index_add("t", _entry("mine"))

        ids = sorted(e["checkpoint_id"] for e in conn.rows[_ORACLE_KEY]["checkpoints"])
        assert ids == ["mine", "theirs"]
        assert [s.split()[0].upper() for s in conn.statements[1:]] == [
            "SELECT",
            "INSERT",
            "SELECT",
            "UPDATE",
        ]
        assert (conn.commits, conn.rollbacks) == (1, 1)

    async def test_an_insert_that_keeps_losing_gives_up_after_the_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _OracleConn(failing_inserts=10)
        backend = _oracle(monkeypatch, conn)

        with pytest.raises(RuntimeError, match="ORA-00001"):
            await backend.index_add("t", _entry("mine"))

        assert sum(s.upper().startswith("INSERT") for s in conn.statements) == 3

    async def test_any_other_insert_error_propagates_at_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _OracleConn(failing_inserts=10, insert_error="ORA-12899: value too large")
        backend = _oracle(monkeypatch, conn)

        with pytest.raises(RuntimeError, match="ORA-12899"):
            await backend.index_add("t", _entry("mine"))

        assert sum(s.upper().startswith("INSERT") for s in conn.statements) == 1
        assert conn.rollbacks == 1

    async def test_list_and_clear_go_through_the_index_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _OracleConn(_oracle_row(_entry("cp-0")))
        backend = _oracle(monkeypatch, conn)

        assert await backend.index_list("t") == [_entry("cp-0")]
        assert await backend.index_clear("t") is True
        assert await backend.index_clear("t") is False
        assert await backend.index_list("t") == []

    async def test_the_adapter_uses_the_native_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from locus.core.state import AgentState

        conn = _OracleConn()
        backend = _oracle(monkeypatch, conn)
        adapter = StorageBackendAdapter(backend)
        assert adapter._native_index is backend

        state = AgentState(agent_id="a")
        for step in range(3):
            await adapter.save(state, "t", f"cp-{step}")

        assert await adapter.list_checkpoints("t") == ["cp-2", "cp-1", "cp-0"]
        assert await adapter.delete("t", "cp-1") is True
        assert await adapter.list_checkpoints("t") == ["cp-2", "cp-0"]
        assert await adapter.delete("t") is True
        assert conn.rows == {}


# ---------------------------------------------------------------------------
# PostgreSQL: one statement per change on the index row
# ---------------------------------------------------------------------------


def _pg(monkeypatch: pytest.MonkeyPatch, conn: _PgConn) -> PostgreSQLBackend:
    _stub_asyncpg(monkeypatch, conn)
    return PostgreSQLBackend()


class TestPostgreSQLIndex:
    async def test_add_is_one_upsert_that_rewrites_the_array(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _PgConn(fetchval="public.checkpoints")
        backend = _pg(monkeypatch, conn)

        await backend.index_add("t", _entry("cp-1"))

        sql, args = conn.execute_calls[-1]
        assert "ON CONFLICT (thread_id) DO UPDATE" in sql
        assert "jsonb_agg(e ORDER BY e->>'timestamp' DESC)" in sql
        assert "IS DISTINCT FROM $4::text" in sql
        assert args[0] == "t:_checkpoints"
        assert json.loads(args[1]) == _entry("cp-1")
        assert args[3] == "cp-1"
        assert len(conn.execute_calls) == 1, "no read before the write"

    async def test_remove_is_one_update_and_reports_whether_the_entry_was_there(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _PgConn(fetchval="public.checkpoints", execute_results=["UPDATE 1", "UPDATE 0"])
        backend = _pg(monkeypatch, conn)

        assert await backend.index_remove("t", "cp-1") is True
        assert await backend.index_remove("t", "cp-9") is False

        sql, args = conn.execute_calls[0]
        assert sql.lstrip().startswith("UPDATE")
        assert "@> jsonb_build_array" in sql
        assert args[:2] == ("t:_checkpoints", "cp-1")

    async def test_list_and_clear_go_through_the_index_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        row = {"data": json.dumps({"checkpoints": [_entry("cp-0")]})}
        conn = _PgConn(fetchval="public.checkpoints", fetchrow=row, execute_results=["DELETE 1"])
        backend = _pg(monkeypatch, conn)

        assert await backend.index_list("t") == [_entry("cp-0")]
        assert await backend.index_clear("t") is True
        assert conn.execute_calls[-1][1] == ("t:_checkpoints",)

    async def test_list_of_an_unknown_thread_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _pg(monkeypatch, _PgConn(fetchval="public.checkpoints", fetchrow=None))

        assert await backend.index_list("t") == []


# ---------------------------------------------------------------------------
# MySQL: one statement per change on the index row
# ---------------------------------------------------------------------------


def _mysql(monkeypatch: pytest.MonkeyPatch, cursor: _MySQLCursor) -> MySQLBackend:
    _stub_mysql_connector(monkeypatch, _MySQLConn(cursor))
    return MySQLBackend()


class TestMySQLIndex:
    def test_like_literal_escapes_the_search_wildcards(self) -> None:
        assert _like_literal("cp_1%") == "cp\\_1\\%"
        assert _like_literal("a\\b") == "a\\\\b"
        assert _like_literal("plain-id") == "plain-id"

    async def test_add_is_one_upsert_that_rewrites_the_array(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cur = _MySQLCursor()
        backend = _mysql(monkeypatch, cur)

        await backend.index_add("t", _entry("cp_1"))

        sql, args = cur.execute_calls[-1]
        assert "ON DUPLICATE KEY UPDATE" in sql
        assert "JSON_ARRAY_INSERT" in sql
        assert "JSON_REMOVE" in sql
        assert "'$[0]'" in sql, "the new entry goes first"
        assert args[0] == "t:_checkpoints"
        assert json.loads(args[1]) == _entry("cp_1")
        assert args[4] == "cp\\_1", "the id is matched literally, not as a LIKE pattern"
        assert args[5] == "\\"
        assert json.loads(args[-1]) == _entry("cp_1")
        assert sum("INSERT INTO" in call[0] for call in cur.execute_calls) == 1

    async def test_remove_is_one_update_and_reports_whether_the_entry_was_there(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cur = _MySQLCursor(rowcount=1)
        backend = _mysql(monkeypatch, cur)

        assert await backend.index_remove("t", "cp-1") is True
        sql, args = cur.execute_calls[-1]
        assert sql.lstrip().startswith("UPDATE")
        assert "IS NOT NULL" in sql
        assert args[0] == "cp-1"
        assert args[3] == "t:_checkpoints"

        cur.rowcount = 0
        assert await backend.index_remove("t", "cp-9") is False

    async def test_list_and_clear_go_through_the_index_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cur = _MySQLCursor(fetchone=(json.dumps({"checkpoints": [_entry("cp-0")]}),), rowcount=1)
        backend = _mysql(monkeypatch, cur)

        assert await backend.index_list("t") == [_entry("cp-0")]
        assert await backend.index_clear("t") is True
        assert cur.execute_calls[-1][1] == ("t:_checkpoints",)

    async def test_list_of_an_unknown_thread_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _mysql(monkeypatch, _MySQLCursor(fetchone=None))

        assert await backend.index_list("t") == []


# ---------------------------------------------------------------------------
# OpenSearch: one scripted update per change on the index document
# ---------------------------------------------------------------------------


def _opensearch(**updates: Any) -> tuple[OpenSearchBackend, AsyncMock]:
    backend = OpenSearchBackend(hosts=["http://x:9200"])
    client = AsyncMock()
    client.update = AsyncMock(return_value={"result": "updated"})
    client.delete = AsyncMock()
    client.get = AsyncMock(return_value={"_source": {"data": {"checkpoints": [_entry("cp-0")]}}})
    for name, value in updates.items():
        setattr(client, name, value)
    backend._client = client  # type: ignore[assignment]
    backend._initialized = True
    return backend, client


class TestOpenSearchIndex:
    async def test_add_is_one_scripted_upsert_retried_on_conflict(self) -> None:
        backend, client = _opensearch()

        await backend.index_add("t", _entry("cp-1"))

        client.update.assert_awaited_once()
        kwargs = client.update.await_args.kwargs
        assert kwargs["id"] == "t:_checkpoints"
        assert kwargs["refresh"] is True
        assert kwargs["retry_on_conflict"] > 0
        body = kwargs["body"]
        assert body["scripted_upsert"] is True
        assert body["script"]["params"]["entry"] == _entry("cp-1")
        assert "removeIf" in body["script"]["source"]
        assert body["upsert"]["data"] == {"checkpoints": []}
        assert body["upsert"]["thread_id"] == "t:_checkpoints"

    async def test_remove_reports_updated_noop_and_a_missing_document(self) -> None:
        backend, client = _opensearch()

        assert await backend.index_remove("t", "cp-1") is True
        kwargs = client.update.await_args.kwargs
        assert kwargs["id"] == "t:_checkpoints"
        assert kwargs["body"]["script"]["params"]["checkpoint_id"] == "cp-1"
        assert "upsert" not in kwargs["body"], "a removal never creates the document"

        client.update.return_value = {"result": "noop"}
        assert await backend.index_remove("t", "cp-9") is False

        exceptions = pytest.importorskip("opensearchpy.exceptions")
        client.update.side_effect = exceptions.NotFoundError(404, "document_missing_exception", {})
        assert await backend.index_remove("t", "cp-9") is False

    async def test_remove_raises_anything_but_a_missing_document(self) -> None:
        """A conflict, a refusal or a script error must not read as "nothing to remove"."""
        exceptions = pytest.importorskip("opensearchpy.exceptions")
        backend, client = _opensearch()

        for error in (
            exceptions.ConflictError(409, "version_conflict_engine_exception", {}),
            exceptions.AuthorizationException(403, "security_exception", {}),
            RuntimeError("connection reset"),
        ):
            client.update.side_effect = error
            with pytest.raises(type(error)):
                await backend.index_remove("t", "cp-1")

    async def test_list_and_clear_go_through_the_index_document(self) -> None:
        backend, client = _opensearch()

        assert await backend.index_list("t") == [_entry("cp-0")]
        assert client.get.await_args.kwargs["id"] == "t:_checkpoints"
        assert await backend.index_clear("t") is True
        assert client.delete.await_args.kwargs["id"] == "t:_checkpoints"

    async def test_list_of_an_unknown_thread_is_empty(self) -> None:
        backend, _client = _opensearch(get=AsyncMock(side_effect=RuntimeError("404")))

        assert await backend.index_list("t") == []
