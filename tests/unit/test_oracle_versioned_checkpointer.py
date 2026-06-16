# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.memory.backends.oracle_versioned`` (OracleCheckpointSaver).

Stubs ``oracledb`` so the suite never needs a real database. The cursor
stub records every ``execute`` call and returns canned ``fetchone`` /
``fetchall`` rows — enough to pin the SQL shape, the CLOB-binding hints,
and the idempotent put_writes pattern.

Coverage matrix:

- Constructor: identifier validation rejects bad table_name / schema_name.
- DDL generation: both tables + the thread index.
- ``_ensure_tables``: creates only when ``user_tables`` says they're
  missing, and skipped entirely on ``auto_create_table=False``.
- ``put``: emits INSERT with the CLOB hints for ``checkpoint_data`` and
  ``metadata``.
- ``get``: latest-row branch uses ORDER BY created_at DESC + FETCH
  FIRST 1 ROWS ONLY; specific-id branch binds ``checkpoint_id``.
- ``list_checkpoints``: ``before`` adds the right subquery; without
  ``before``, simple newest-first ordering.
- ``put_writes``: DELETE-then-INSERT in one cursor pass, monotonic
  ``idx``, CLOB hint on ``value``.
- ``get_writes``: ``task_id`` filter optional.
- ``delete_thread``: hits both tables.
"""

from __future__ import annotations

import json
import sys
import types
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from locus.memory.backends.oracle_versioned import OracleCheckpointSaver


# ---------------------------------------------------------------------------
# Oracledb stub — same shape as tests/unit/test_oracle_store.py
# ---------------------------------------------------------------------------


class _StubCursor:
    """Records every execute() call and returns canned rows.

    ``fetchone_queue`` / ``fetchall_queue`` let a single cursor return
    different rows on successive calls — handy for tests that walk
    multiple table-existence checks before the DML call.
    """

    def __init__(
        self,
        *,
        fetchone: Any | None = None,
        fetchall: list[Any] | None = None,
        fetchone_queue: list[Any] | None = None,
        fetchall_queue: list[list[Any]] | None = None,
    ) -> None:
        self.fetchone_value = fetchone
        self.fetchall_value = fetchall or []
        self.fetchone_queue = list(fetchone_queue) if fetchone_queue else None
        self.fetchall_queue = list(fetchall_queue) if fetchall_queue else None
        self.execute_calls: list[tuple[str, dict[str, Any]]] = []
        self.input_sizes_calls: list[dict[str, Any]] = []
        self.rowcount: int = 0

    async def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        self.execute_calls.append((sql, params or {}))
        if sql.lstrip().upper().startswith("DELETE"):
            self.rowcount = 1

    async def fetchone(self) -> Any:
        if self.fetchone_queue is not None and self.fetchone_queue:
            return self.fetchone_queue.pop(0)
        return self.fetchone_value

    async def fetchall(self) -> list[Any]:
        if self.fetchall_queue is not None and self.fetchall_queue:
            return self.fetchall_queue.pop(0)
        return self.fetchall_value

    def setinputsizes(self, **kwargs: Any) -> None:
        # Snapshot each call separately so we can assert per-statement
        # hints (e.g. checkpoint INSERT vs writes INSERT).
        self.input_sizes_calls.append(dict(kwargs))

    async def __aenter__(self) -> _StubCursor:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _StubConn:
    def __init__(self, cursor: _StubCursor) -> None:
        self._cursor = cursor
        self.commit_count = 0

    def cursor(self) -> _StubCursor:
        return self._cursor

    async def commit(self) -> None:
        self.commit_count += 1

    async def __aenter__(self) -> _StubConn:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _StubPool:
    def __init__(self, conn: _StubConn) -> None:
        self._conn = conn
        self.closed = False

    @asynccontextmanager
    async def acquire(self) -> Any:
        yield self._conn

    async def close(self) -> None:
        self.closed = True


def _install_oracledb_stub(
    monkeypatch: pytest.MonkeyPatch,
    cursor: _StubCursor,
) -> _StubPool:
    """Install a fake ``oracledb`` module returning our stub pool."""
    pool = _StubPool(_StubConn(cursor))

    def fake_create_pool_async(*args: Any, **kwargs: Any) -> _StubPool:
        return pool

    def fake_makedsn(host: str, port: int, *, service_name: str) -> str:
        return f"(HOST={host})(PORT={port})(SERVICE={service_name})"

    fake = types.ModuleType("oracledb")
    fake.create_pool_async = fake_create_pool_async  # type: ignore[attr-defined]
    fake.makedsn = fake_makedsn  # type: ignore[attr-defined]
    fake.DB_TYPE_CLOB = "CLOB-SENTINEL"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oracledb", fake)
    return pool


def _make_saver(**kwargs: Any) -> OracleCheckpointSaver:
    base: dict[str, Any] = {"dsn": "x", "user": "u", "password": "p"}  # noqa: S106
    base.update(kwargs)
    return OracleCheckpointSaver(**base)


# ---------------------------------------------------------------------------
# Constructor / identifier validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_defaults_repr(self) -> None:
        saver = _make_saver()
        # repr exposes the user-supplied prefix, not the suffixed
        # checkpoints table the OracleConfig sees internally.
        assert "table_name='locus'" in repr(saver)
        assert "dsn='x'" in repr(saver)

    def test_repr_with_host(self) -> None:
        saver = OracleCheckpointSaver(
            host="adb.example.com",
            port=1522,
            service_name="svc.high",
            user="u",
            password="p",  # noqa: S106
        )
        r = repr(saver)
        assert "host='adb.example.com'" in r
        assert "service='svc.high'" in r

    def test_bad_table_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid table_name"):
            OracleCheckpointSaver(dsn="x", user="u", password="p", table_name="bad table")  # noqa: S106

    def test_sql_injection_table_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid table_name"):
            OracleCheckpointSaver(
                dsn="x",
                user="u",
                password="p",  # noqa: S106
                table_name="locus; DROP TABLE users--",
            )

    def test_bad_schema_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid schema_name"):
            OracleCheckpointSaver(
                dsn="x",
                user="u",
                password="p",  # noqa: S106
                schema_name="1bad",
            )

    def test_custom_prefix_used_in_table_names(self) -> None:
        saver = _make_saver(table_name="myapp")
        assert saver._checkpoints_table == "myapp_checkpoints"
        assert saver._writes_table == "myapp_writes"

    def test_schema_qualified_table_names(self) -> None:
        saver = _make_saver(schema_name="LOCUS_APP", table_name="cp")
        assert saver._checkpoints_table == "LOCUS_APP.cp_checkpoints"
        assert saver._writes_table == "LOCUS_APP.cp_writes"


# ---------------------------------------------------------------------------
# DDL generation
# ---------------------------------------------------------------------------


class TestDDL:
    def test_checkpoints_ddl_shape(self) -> None:
        ddl = _make_saver()._checkpoints_ddl()
        assert "CREATE TABLE locus_checkpoints" in ddl
        assert "thread_id            VARCHAR2(255) NOT NULL" in ddl
        assert "checkpoint_ns        VARCHAR2(255) DEFAULT 'default' NOT NULL" in ddl
        assert "checkpoint_data      CLOB CHECK (checkpoint_data IS JSON)" in ddl
        assert "metadata             CLOB DEFAULT '{}' CHECK (metadata IS JSON)" in ddl
        assert "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)" in ddl
        assert "CONSTRAINT pk_locus_checkpoints" in ddl

    def test_checkpoints_index_ddl_shape(self) -> None:
        ddl = _make_saver()._checkpoints_index_ddl()
        assert "CREATE INDEX idx_locus_checkpoints_thread" in ddl
        assert "ON locus_checkpoints (thread_id, checkpoint_ns, created_at DESC)" in ddl

    def test_writes_ddl_shape(self) -> None:
        ddl = _make_saver()._writes_ddl()
        assert "CREATE TABLE locus_writes" in ddl
        assert "task_id       VARCHAR2(255) NOT NULL" in ddl
        assert "idx           NUMBER(10)    NOT NULL" in ddl
        assert "value         CLOB" in ddl
        assert "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)" in ddl

    def test_ddl_uses_schema_when_set(self) -> None:
        saver = _make_saver(schema_name="APP", table_name="cp")
        cp_ddl = saver._checkpoints_ddl()
        wr_ddl = saver._writes_ddl()
        assert "CREATE TABLE APP.cp_checkpoints" in cp_ddl
        assert "CREATE TABLE APP.cp_writes" in wr_ddl
        # The PK constraint name uses the unqualified prefix.
        assert "CONSTRAINT pk_cp_checkpoints" in cp_ddl
        assert "CONSTRAINT pk_cp_writes" in wr_ddl


# ---------------------------------------------------------------------------
# _ensure_tables — idempotent DDL
# ---------------------------------------------------------------------------


class TestEnsureTables:
    @pytest.mark.asyncio
    async def test_creates_both_tables_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Two existence checks both return (0,) — neither table exists.
        cursor = _StubCursor(fetchone_queue=[(0,), (0,)])
        _install_oracledb_stub(monkeypatch, cursor)

        saver = _make_saver()
        await saver._ensure_tables()

        sqls = [c[0] for c in cursor.execute_calls]
        # 2 existence checks + 3 DDLs (cp table, cp index, writes table)
        assert len(sqls) == 5
        assert "user_tables" in sqls[0]
        assert "CREATE TABLE locus_checkpoints" in sqls[1]
        assert "CREATE INDEX idx_locus_checkpoints_thread" in sqls[2]
        assert "user_tables" in sqls[3]
        assert "CREATE TABLE locus_writes" in sqls[4]

    @pytest.mark.asyncio
    async def test_skips_ddl_when_both_tables_exist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Both checks return (1,) — tables exist; only the two SELECTs fire.
        cursor = _StubCursor(fetchone_queue=[(1,), (1,)])
        _install_oracledb_stub(monkeypatch, cursor)

        saver = _make_saver()
        await saver._ensure_tables()

        sqls = [c[0] for c in cursor.execute_calls]
        assert len(sqls) == 2
        assert all("user_tables" in s for s in sqls)

    @pytest.mark.asyncio
    async def test_creates_writes_only_when_checkpoints_already_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # checkpoints exists (1), writes does not (0).
        cursor = _StubCursor(fetchone_queue=[(1,), (0,)])
        _install_oracledb_stub(monkeypatch, cursor)

        saver = _make_saver()
        await saver._ensure_tables()

        sqls = [c[0] for c in cursor.execute_calls]
        # 2 existence checks + 1 CREATE TABLE for writes only.
        assert len(sqls) == 3
        assert "CREATE TABLE locus_checkpoints" not in " ".join(sqls)
        assert "CREATE TABLE locus_writes" in sqls[2]

    @pytest.mark.asyncio
    async def test_auto_create_false_skips_everything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cursor = _StubCursor()
        _install_oracledb_stub(monkeypatch, cursor)

        saver = _make_saver(auto_create_table=False)
        await saver._ensure_tables()

        # No SQL at all — DBA-managed mode trusts the schema.
        assert cursor.execute_calls == []
        # And the flag is sticky.
        assert saver._initialized is True

    @pytest.mark.asyncio
    async def test_ensure_is_idempotent_across_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchone_queue=[(1,), (1,)])
        _install_oracledb_stub(monkeypatch, cursor)

        saver = _make_saver()
        await saver._ensure_tables()
        await saver._ensure_tables()  # second call short-circuits.

        # Only the first call's two existence checks ran.
        assert len(cursor.execute_calls) == 2


# ---------------------------------------------------------------------------
# put — CLOB binding + INSERT shape
# ---------------------------------------------------------------------------


class TestPut:
    @pytest.mark.asyncio
    async def test_put_emits_insert_with_clob_hints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchone_queue=[(1,), (1,)])  # tables exist
        _install_oracledb_stub(monkeypatch, cursor)

        saver = _make_saver()
        await saver.put(
            thread_id="t1",
            checkpoint_id="c1",
            checkpoint_data={"step": 0, "values": {"x": 1}},
            metadata={"src": "test"},
        )

        # Last execute is the INSERT.
        sql, params = cursor.execute_calls[-1]
        assert sql.lstrip().startswith("INSERT INTO locus_checkpoints")
        assert params["thread_id"] == "t1"
        assert params["checkpoint_id"] == "c1"
        assert params["checkpoint_ns"] == "default"
        assert params["parent_checkpoint_id"] is None
        # CLOB hints were set on the cursor before the INSERT.
        last_hints = cursor.input_sizes_calls[-1]
        assert last_hints["checkpoint_data"] == "CLOB-SENTINEL"
        assert last_hints["metadata"] == "CLOB-SENTINEL"
        # JSON encoded.
        assert json.loads(params["checkpoint_data"]) == {"step": 0, "values": {"x": 1}}
        assert json.loads(params["metadata"]) == {"src": "test"}

    @pytest.mark.asyncio
    async def test_put_with_parent_and_ns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchone_queue=[(1,), (1,)])
        _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        await saver.put(
            thread_id="t1",
            checkpoint_id="c2",
            checkpoint_data={"step": 1},
            checkpoint_ns="branch-a",
            parent_checkpoint_id="c1",
        )
        _, params = cursor.execute_calls[-1]
        assert params["checkpoint_ns"] == "branch-a"
        assert params["parent_checkpoint_id"] == "c1"
        # No metadata supplied → defaulted to '{}'.
        assert params["metadata"] == "{}"

    @pytest.mark.asyncio
    async def test_put_commits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchone_queue=[(1,), (1,)])
        pool = _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        await saver.put(
            thread_id="t1",
            checkpoint_id="c1",
            checkpoint_data={},
        )
        assert pool._conn.commit_count >= 1


# ---------------------------------------------------------------------------
# get — latest vs specific id
# ---------------------------------------------------------------------------


class TestGet:
    @pytest.mark.asyncio
    async def test_get_latest_orders_desc_with_fetch_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ts = datetime(2026, 5, 18, tzinfo=UTC)
        cursor = _StubCursor(
            fetchone_queue=[
                (1,),  # checkpoints exists
                (1,),  # writes exists
                ("c9", "c8", '{"step": 9}', "{}", ts),  # SELECT row
            ]
        )
        _install_oracledb_stub(monkeypatch, cursor)

        saver = _make_saver()
        result = await saver.get(thread_id="t1")

        sql, params = cursor.execute_calls[-1]
        assert "ORDER BY created_at DESC" in sql
        assert "FETCH FIRST 1 ROWS ONLY" in sql
        assert "checkpoint_id" not in params  # bind only when supplied
        assert params == {"thread_id": "t1", "checkpoint_ns": "default"}
        assert result == {
            "checkpoint_id": "c9",
            "parent_checkpoint_id": "c8",
            "checkpoint": {"step": 9},
            "metadata": {},
            "created_at": ts.isoformat(),
        }

    @pytest.mark.asyncio
    async def test_get_by_id_binds_checkpoint_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ts = datetime(2026, 5, 18, tzinfo=UTC)
        cursor = _StubCursor(
            fetchone_queue=[
                (1,),
                (1,),
                ("c5", None, '{"step": 5}', '{"src": "x"}', ts),
            ]
        )
        _install_oracledb_stub(monkeypatch, cursor)

        saver = _make_saver()
        result = await saver.get(thread_id="t1", checkpoint_id="c5", checkpoint_ns="nsX")

        sql, params = cursor.execute_calls[-1]
        assert "checkpoint_id = :checkpoint_id" in sql
        assert "ORDER BY" not in sql
        assert params == {"thread_id": "t1", "checkpoint_ns": "nsX", "checkpoint_id": "c5"}
        assert result is not None
        assert result["checkpoint_id"] == "c5"
        assert result["metadata"] == {"src": "x"}

    @pytest.mark.asyncio
    async def test_get_returns_none_when_no_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchone_queue=[(1,), (1,), None])
        _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        result = await saver.get(thread_id="missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_decodes_clob_object_via_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate a LOB object — oracledb returns these for big CLOBs.
        class _LOB:
            def __init__(self, s: str) -> None:
                self._s = s

            def read(self) -> str:
                return self._s

        ts = datetime(2026, 5, 18, tzinfo=UTC)
        cursor = _StubCursor(
            fetchone_queue=[
                (1,),
                (1,),
                ("c1", None, _LOB('{"big": "payload"}'), _LOB("{}"), ts),
            ]
        )
        _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        result = await saver.get(thread_id="t1")
        assert result is not None
        assert result["checkpoint"] == {"big": "payload"}
        assert result["metadata"] == {}

    @pytest.mark.asyncio
    async def test_get_accepts_dict_clob_native_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Oracle 23ai+ may return JSON columns as Python dicts already.
        ts = datetime(2026, 5, 18, tzinfo=UTC)
        cursor = _StubCursor(
            fetchone_queue=[
                (1,),
                (1,),
                ("c1", None, {"already": "dict"}, {"m": 1}, ts),
            ]
        )
        _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        result = await saver.get(thread_id="t1")
        assert result is not None
        assert result["checkpoint"] == {"already": "dict"}
        assert result["metadata"] == {"m": 1}


# ---------------------------------------------------------------------------
# list_checkpoints — before clause + ordering
# ---------------------------------------------------------------------------


class TestListCheckpoints:
    @pytest.mark.asyncio
    async def test_list_without_before(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ts = datetime(2026, 5, 18, tzinfo=UTC)
        cursor = _StubCursor(
            fetchone_queue=[(1,), (1,)],
            fetchall_queue=[
                [
                    ("c3", "c2", '{"step": 3}', "{}", ts),
                    ("c2", "c1", '{"step": 2}', "{}", ts),
                ]
            ],
        )
        _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        rows = await saver.list_checkpoints(thread_id="t1", limit=5)
        sql, params = cursor.execute_calls[-1]
        assert "ORDER BY created_at DESC" in sql
        assert "FETCH FIRST :lim ROWS ONLY" in sql
        assert "before" not in sql
        assert params == {"thread_id": "t1", "checkpoint_ns": "default", "lim": 5}
        assert len(rows) == 2
        assert rows[0]["checkpoint_id"] == "c3"
        assert rows[1]["checkpoint_id"] == "c2"

    @pytest.mark.asyncio
    async def test_list_with_before_adds_subquery(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchone_queue=[(1,), (1,)], fetchall_queue=[[]])
        _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        await saver.list_checkpoints(thread_id="t1", before="c5", limit=10)
        sql, params = cursor.execute_calls[-1]
        # The before-clause is a correlated subquery against the same
        # checkpoints table — the test pins both the WHERE shape and
        # the bind dict.
        assert "AND created_at <" in sql
        assert "SELECT created_at FROM locus_checkpoints" in sql
        assert "AND checkpoint_id = :before" in sql
        assert params == {
            "thread_id": "t1",
            "checkpoint_ns": "default",
            "before": "c5",
            "lim": 10,
        }

    @pytest.mark.asyncio
    async def test_list_empty_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchone_queue=[(1,), (1,)], fetchall_queue=[[]])
        _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        rows = await saver.list_checkpoints(thread_id="t1")
        assert rows == []


# ---------------------------------------------------------------------------
# put_writes — delete-then-insert idempotence
# ---------------------------------------------------------------------------


class TestPutWrites:
    @pytest.mark.asyncio
    async def test_put_writes_deletes_then_inserts_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cursor = _StubCursor(fetchone_queue=[(1,), (1,)])
        _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        await saver.put_writes(
            thread_id="t1",
            checkpoint_id="c1",
            task_id="node-a",
            writes=[("x", 1), ("y", 2), ("z", 3)],
        )

        # First two executes are the table-exists checks (tables exist
        # so no DDL). Then: DELETE, then INSERT 0, INSERT 1, INSERT 2.
        post_ddl = cursor.execute_calls[2:]
        assert len(post_ddl) == 4
        assert post_ddl[0][0].lstrip().startswith("DELETE FROM locus_writes")
        assert post_ddl[0][1] == {
            "thread_id": "t1",
            "checkpoint_ns": "default",
            "checkpoint_id": "c1",
            "task_id": "node-a",
        }
        # INSERTs with monotonic idx 0..2.
        for i, (sql, params) in enumerate(post_ddl[1:]):
            assert sql.lstrip().startswith("INSERT INTO locus_writes")
            assert params["idx"] == i
            assert params["task_id"] == "node-a"
            assert params["channel"] == ["x", "y", "z"][i]
            assert json.loads(params["value"]) == [1, 2, 3][i]

    @pytest.mark.asyncio
    async def test_put_writes_pins_value_clob_each_insert(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cursor = _StubCursor(fetchone_queue=[(1,), (1,)])
        _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        await saver.put_writes(
            thread_id="t1",
            checkpoint_id="c1",
            task_id="node-a",
            writes=[("x", 1), ("y", 2)],
        )
        # setinputsizes called once per INSERT (2 inserts).
        value_hints = [h for h in cursor.input_sizes_calls if "value" in h]
        assert len(value_hints) == 2
        assert all(h["value"] == "CLOB-SENTINEL" for h in value_hints)

    @pytest.mark.asyncio
    async def test_put_writes_empty_list_only_deletes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cursor = _StubCursor(fetchone_queue=[(1,), (1,)])
        _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        await saver.put_writes(
            thread_id="t1",
            checkpoint_id="c1",
            task_id="node-a",
            writes=[],
        )
        # Only the DELETE fires after the existence checks.
        post_ddl = cursor.execute_calls[2:]
        assert len(post_ddl) == 1
        assert post_ddl[0][0].lstrip().startswith("DELETE")

    @pytest.mark.asyncio
    async def test_put_writes_respects_ns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchone_queue=[(1,), (1,)])
        _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        await saver.put_writes(
            thread_id="t1",
            checkpoint_id="c1",
            task_id="node-a",
            writes=[("x", 1)],
            checkpoint_ns="branch",
        )
        for _, params in cursor.execute_calls[2:]:
            assert params["checkpoint_ns"] == "branch"


# ---------------------------------------------------------------------------
# get_writes — task_id filter optional
# ---------------------------------------------------------------------------


class TestGetWrites:
    @pytest.mark.asyncio
    async def test_get_writes_all_tasks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(
            fetchone_queue=[(1,), (1,)],
            fetchall_queue=[
                [
                    ("node-a", 0, "x", '"v1"'),
                    ("node-a", 1, "y", '"v2"'),
                    ("node-b", 0, "z", "true"),
                ]
            ],
        )
        _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        rows = await saver.get_writes(thread_id="t1", checkpoint_id="c1")

        sql, params = cursor.execute_calls[-1]
        assert "ORDER BY task_id, idx" in sql
        assert "task_id = :task_id" not in sql
        assert params == {
            "thread_id": "t1",
            "checkpoint_ns": "default",
            "checkpoint_id": "c1",
        }
        assert rows == [
            {"task_id": "node-a", "idx": 0, "channel": "x", "value": "v1"},
            {"task_id": "node-a", "idx": 1, "channel": "y", "value": "v2"},
            {"task_id": "node-b", "idx": 0, "channel": "z", "value": True},
        ]

    @pytest.mark.asyncio
    async def test_get_writes_filtered_by_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(
            fetchone_queue=[(1,), (1,)],
            fetchall_queue=[
                [
                    ("node-a", 0, "x", '"v1"'),
                ]
            ],
        )
        _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        rows = await saver.get_writes(thread_id="t1", checkpoint_id="c1", task_id="node-a")

        sql, params = cursor.execute_calls[-1]
        assert "AND task_id = :task_id" in sql
        assert "ORDER BY idx" in sql
        assert params == {
            "thread_id": "t1",
            "checkpoint_ns": "default",
            "checkpoint_id": "c1",
            "task_id": "node-a",
        }
        assert rows == [{"task_id": "node-a", "idx": 0, "channel": "x", "value": "v1"}]


# ---------------------------------------------------------------------------
# delete_thread — both tables
# ---------------------------------------------------------------------------


class TestDeleteThread:
    @pytest.mark.asyncio
    async def test_delete_thread_hits_both_tables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchone_queue=[(1,), (1,)])
        pool = _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        await saver.delete_thread("t1")

        post_ddl = cursor.execute_calls[2:]
        assert len(post_ddl) == 2
        # Writes table is deleted first (referential safety even though
        # there's no FK), then the parent checkpoints table.
        assert post_ddl[0][0].lstrip().startswith("DELETE FROM locus_writes")
        assert post_ddl[1][0].lstrip().startswith("DELETE FROM locus_checkpoints")
        assert post_ddl[0][1] == {"thread_id": "t1"}
        assert post_ddl[1][1] == {"thread_id": "t1"}
        assert pool._conn.commit_count >= 1


# ---------------------------------------------------------------------------
# close — pool cleanup
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_closes_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _StubCursor(fetchone_queue=[(1,), (1,)])
        pool = _install_oracledb_stub(monkeypatch, cursor)
        saver = _make_saver()
        # Force pool creation.
        await saver._get_pool()
        assert saver._pool is pool
        await saver.close()
        assert pool.closed is True
        assert saver._pool is None

    @pytest.mark.asyncio
    async def test_close_without_pool_is_noop(self) -> None:
        saver = _make_saver()
        # No pool was ever created; close() must not raise.
        await saver.close()
        assert saver._pool is None
