# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The atomic checkpoint index each storage backend keeps for the adapter (issue #301).

Redis runs against a dict-backed fake of the client (strings, hashes and
pipelines); PostgreSQL and MySQL against the statement-recording stubs
their own test modules use; OpenSearch against a mocked client. The live
behaviour — two processes writing one thread and losing nothing — is
``tests/integration/test_checkpoint_index_concurrency.py``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from locus.memory.backends import CheckpointIndexStore
from locus.memory.backends.adapters import StorageBackendAdapter
from locus.memory.backends.checkpoint_index import index_key, newest_first
from locus.memory.backends.mysql import MySQLBackend, _like_literal
from locus.memory.backends.opensearch import OpenSearchBackend
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
        ],
        ids=["redis", "postgresql", "mysql", "opensearch"],
    )
    def test_every_shipped_backend_keeps_the_index_itself(self, backend: Any) -> None:
        assert isinstance(backend, CheckpointIndexStore)


# ---------------------------------------------------------------------------
# Redis: a hash per thread, the legacy string migrated
# ---------------------------------------------------------------------------


class _FakePipeline:
    def __init__(self, client: _FakeRedis) -> None:
        self._client = client
        self._ops: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._client, name)

        def queue(*args: Any, **kwargs: Any) -> _FakePipeline:
            self._ops.append((method, args, kwargs))
            return self

        return queue

    async def execute(self) -> list[Any]:
        return [await method(*args, **kwargs) for method, args, kwargs in self._ops]


class _FakeRedis:
    """The slice of ``redis.asyncio.Redis`` the backend uses, on dicts."""

    def __init__(self) -> None:
        self.strings: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        return self.strings.get(key)

    async def set(self, key: str, value: str) -> None:
        self.strings[key] = value

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.strings[key] = value
        self.ttls[key] = ttl

    async def exists(self, key: str) -> int:
        return int(key in self.strings or key in self.hashes)

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if self.strings.pop(key, None) is not None:
                removed += 1
            if self.hashes.pop(key, None) is not None:
                removed += 1
        return removed

    async def hset(self, key: str, field: str, value: str) -> int:
        fields = self.hashes.setdefault(key, {})
        created = field not in fields
        fields[field] = value
        return int(created)

    async def hsetnx(self, key: str, field: str, value: str) -> int:
        fields = self.hashes.setdefault(key, {})
        if field in fields:
            return 0
        fields[field] = value
        return 1

    async def hdel(self, key: str, *fields: str) -> int:
        stored = self.hashes.get(key, {})
        removed = sum(1 for field in fields if stored.pop(field, None) is not None)
        if not stored:
            self.hashes.pop(key, None)
        return removed

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    async def expire(self, key: str, ttl: int) -> int:
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

        client.update.side_effect = RuntimeError("404 document missing")
        assert await backend.index_remove("t", "cp-9") is False

    async def test_list_and_clear_go_through_the_index_document(self) -> None:
        backend, client = _opensearch()

        assert await backend.index_list("t") == [_entry("cp-0")]
        assert client.get.await_args.kwargs["id"] == "t:_checkpoints"
        assert await backend.index_clear("t") is True
        assert client.delete.await_args.kwargs["id"] == "t:_checkpoints"

    async def test_list_of_an_unknown_thread_is_empty(self) -> None:
        backend, _client = _opensearch(get=AsyncMock(side_effect=RuntimeError("404")))

        assert await backend.index_list("t") == []
