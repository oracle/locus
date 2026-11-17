# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The ``expires_at`` column of the PostgreSQL and MySQL storage backends, on stubbed drivers.

The adapter hands ``save(..., expires_at=)`` to a backend whose ``save``
takes it; these backends keep it in a column, filter their reads on it and
sweep it with ``expire()``. An existing table is only probed on start: the
column is added by the first save that carries an expiry (a checkpointer
without a ``ttl`` never alters a table it did not create), and a role that
may not alter the table goes without the column, told once, with the
adapter's own record of the time still applying. The drivers are stubbed
as in ``test_memory_backends_postgresql.py`` and
``test_memory_backends_mysql.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from locus.memory.backends.mysql import MySQLBackend
from locus.memory.backends.postgresql import PostgreSQLBackend
from tests.unit import test_memory_backends_mysql as mysql_stubs
from tests.unit import test_memory_backends_postgresql as pg_stubs


EXPIRES = datetime(2030, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------


class _ProbingConn(pg_stubs._StubConn):
    """``fetchval`` answers the table probe and the column probe separately."""

    def __init__(self, *, table_exists: bool, column_exists: bool, alter_fails: bool = False):
        super().__init__()
        self.table_exists = table_exists
        self.column_exists = column_exists
        self.alter_fails = alter_fails

    async def fetchval(self, sql: str, *args: Any) -> Any:
        if "to_regclass" in sql:
            return "public.checkpoints" if self.table_exists else None
        if "information_schema.columns" in sql:
            return 1 if self.column_exists else None
        return None

    async def execute(self, sql: str, *args: Any) -> str:
        if self.alter_fails and "ALTER TABLE" in sql:
            raise RuntimeError("permission denied")
        return await super().execute(sql, *args)


class TestPostgresExpiresColumn:
    async def test_a_new_table_has_the_column_and_save_binds_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _ProbingConn(table_exists=False, column_exists=False)
        pg_stubs._stub_asyncpg(monkeypatch, conn)
        backend = PostgreSQLBackend()
        await backend.save("t:cp", {"x": 1}, expires_at=EXPIRES)
        ddl = [sql for sql, _ in conn.execute_calls if "CREATE TABLE" in sql]
        assert ddl
        assert "expires_at TIMESTAMPTZ" in ddl[0]
        assert not any("ALTER TABLE" in sql for sql, _ in conn.execute_calls)
        sql, args = conn.execute_calls[-1]
        assert "expires_at = EXCLUDED.expires_at" in sql
        assert args[5] == EXPIRES
        assert backend._expires_column is True

    async def test_without_a_ttl_an_old_table_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _ProbingConn(table_exists=True, column_exists=False)
        pg_stubs._stub_asyncpg(monkeypatch, conn)
        backend = PostgreSQLBackend()
        await backend.save("t:cp", {"x": 1})
        await backend.load("t:cp")
        await backend.exists("t:cp")
        assert not any("ALTER TABLE" in sql for sql, _ in conn.execute_calls), "no DDL asked for"
        assert backend._expires_column is False
        assert backend._live == "", "reads do not name a column the table lacks"
        sql, args = conn.execute_calls[0]
        assert "expires_at" not in sql
        assert len(args) == 6
        assert await backend.expire() == 0

    async def test_the_first_save_with_an_expiry_adds_the_column_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _ProbingConn(table_exists=True, column_exists=False)
        pg_stubs._stub_asyncpg(monkeypatch, conn)
        backend = PostgreSQLBackend()
        await backend.save("t:cp", {"x": 0})
        assert not any("ALTER TABLE" in sql for sql, _ in conn.execute_calls)
        await backend.save("t:cp", {"x": 1}, expires_at=EXPIRES)
        await backend.save("t:cp", {"x": 2}, expires_at=EXPIRES)
        alters = [sql for sql, _ in conn.execute_calls if "ALTER TABLE" in sql]
        assert len(alters) == 1
        assert "ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ" in alters[0]
        assert backend._expires_column is True
        assert conn.execute_calls[-1][1][5] == EXPIRES
        assert "expires_at IS NULL OR expires_at > now()" in backend._live

    async def test_an_existing_column_needs_no_ddl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = _ProbingConn(table_exists=True, column_exists=True)
        pg_stubs._stub_asyncpg(monkeypatch, conn)
        backend = PostgreSQLBackend()
        await backend.load("t:cp")
        assert not any("ALTER TABLE" in sql for sql, _ in conn.execute_calls)
        assert backend._expires_column is True

    async def test_a_role_that_cannot_alter_goes_without_the_column(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        conn = _ProbingConn(table_exists=True, column_exists=False, alter_fails=True)
        pg_stubs._stub_asyncpg(monkeypatch, conn)
        backend = PostgreSQLBackend()
        with caplog.at_level("WARNING"):
            await backend.save("t:cp", {"x": 1}, expires_at=EXPIRES)
            await backend.save("t:cp", {"x": 2}, expires_at=EXPIRES)
        assert backend._expires_column is False
        [record] = [r for r in caplog.records if "could not add the expires_at column" in r.message]
        assert record.exc_info is None, "one line, once"
        assert "permission denied" in record.getMessage()
        assert backend._expires_column_tried is True, "tried once, not on every save"
        sql, args = conn.execute_calls[-1]
        assert "expires_at" not in sql
        assert len(args) == 6
        assert await backend.expire() == 0
        assert backend._live == ""

    async def test_reads_filter_and_expire_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = _ProbingConn(table_exists=True, column_exists=True)
        conn.execute_results = ["DELETE 3"]
        pg_stubs._stub_asyncpg(monkeypatch, conn)
        backend = PostgreSQLBackend()
        await backend.load("t:cp")
        await backend.exists("t:cp")
        assert await backend.expire() == 3
        assert "expires_at IS NULL OR expires_at > now()" in backend._live
        sql, _ = conn.execute_calls[-1]
        assert "DELETE FROM" in sql
        assert "expires_at <= now()" in sql

    async def test_expire_tolerates_an_odd_status_line(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _ProbingConn(table_exists=True, column_exists=True)
        conn.execute_results = ["weird"]
        pg_stubs._stub_asyncpg(monkeypatch, conn)
        assert await PostgreSQLBackend().expire() == 0


# ---------------------------------------------------------------------------
# MySQL
# ---------------------------------------------------------------------------


class _ProbingCursor(mysql_stubs._StubCursor):
    """``fetchone`` answers the column probe with a count; everything else is canned."""

    def __init__(self, *, column_count: int | None, alter_fails: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.column_count = column_count
        self.alter_fails = alter_fails
        self._last_sql = ""

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        if self.alter_fails and "ALTER TABLE" in sql:
            raise RuntimeError("permission denied")
        self._last_sql = sql
        await super().execute(sql, params)

    async def fetchone(self) -> Any:
        if "information_schema.columns" in self._last_sql:
            return None if self.column_count is None else (self.column_count,)
        return await super().fetchone()


def _mysql(monkeypatch: pytest.MonkeyPatch, cursor: _ProbingCursor) -> tuple[MySQLBackend, Any]:
    conn = mysql_stubs._StubConn(cursor)
    mysql_stubs._stub_mysql_connector(monkeypatch, conn)
    return MySQLBackend(), conn


class TestMySQLExpiresColumn:
    async def test_the_table_ddl_has_the_column_and_save_binds_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cursor = _ProbingCursor(column_count=1)
        backend, _ = _mysql(monkeypatch, cursor)
        await backend.save("t:cp", {"x": 1}, metadata={"a": 1}, expires_at=EXPIRES)
        ddl = [sql for sql, _ in cursor.execute_calls if "CREATE TABLE" in sql]
        assert ddl
        assert "expires_at TIMESTAMP(6) NULL" in ddl[0]
        assert not any("ALTER TABLE" in sql for sql, _ in cursor.execute_calls)
        sql, params = cursor.execute_calls[-1]
        assert "expires_at = VALUES(expires_at)" in sql
        assert params[5] == EXPIRES.replace(tzinfo=None), "bound naive, in UTC"
        assert params[-1] == '{"a": 1}', "metadata stays the last parameter"

    async def test_without_a_ttl_a_missing_column_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cursor = _ProbingCursor(column_count=0)
        backend, _ = _mysql(monkeypatch, cursor)
        await backend.save("t:cp", {"x": 1})
        await backend.load("t:cp")
        assert not any("ALTER TABLE" in sql for sql, _ in cursor.execute_calls)
        assert backend._expires_column is False
        assert backend._live == ""
        inserts = [(sql, p) for sql, p in cursor.execute_calls if "INSERT INTO" in sql]
        assert "expires_at" not in inserts[0][0]
        assert len(inserts[0][1]) == 6

    async def test_the_first_save_with_an_expiry_adds_the_column_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cursor = _ProbingCursor(column_count=0)
        backend, _ = _mysql(monkeypatch, cursor)
        await backend.save("t:cp", {"x": 0})
        await backend.save("t:cp", {"x": 1}, expires_at=EXPIRES)
        await backend.save("t:cp", {"x": 2}, expires_at=EXPIRES)
        alters = [sql for sql, _ in cursor.execute_calls if "ALTER TABLE" in sql]
        assert len(alters) == 1
        assert "ADD COLUMN expires_at TIMESTAMP(6) NULL" in alters[0]
        assert backend._expires_column is True
        assert cursor.execute_calls[-1][1][5] == EXPIRES.replace(tzinfo=None)

    async def test_a_driver_that_gives_no_answer_assumes_the_column(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cursor = _ProbingCursor(column_count=None)
        backend, _ = _mysql(monkeypatch, cursor)
        await backend.load("t:cp")
        assert backend._expires_column is True
        assert "UTC_TIMESTAMP(6)" in backend._live

    async def test_a_role_that_cannot_alter_goes_without_the_column(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        cursor = _ProbingCursor(column_count=0, alter_fails=True)
        backend, _ = _mysql(monkeypatch, cursor)
        with caplog.at_level("WARNING"):
            await backend.save("t:cp", {"x": 1}, expires_at=EXPIRES)
            await backend.save("t:cp", {"x": 2}, expires_at=EXPIRES)
        assert backend._expires_column is False
        [record] = [r for r in caplog.records if "could not add the expires_at column" in r.message]
        assert record.exc_info is None
        assert backend._expires_column_tried is True
        sql, params = cursor.execute_calls[-1]
        assert "expires_at" not in sql
        assert len(params) == 6
        assert backend._live == ""
        assert await backend.expire() == 0

    async def test_reads_filter_and_expire_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cursor = _ProbingCursor(column_count=1, rowcount=2)
        backend, conn = _mysql(monkeypatch, cursor)
        await backend.load("t:cp")
        assert "expires_at IS NULL OR expires_at > UTC_TIMESTAMP(6)" in cursor.execute_calls[-1][0]
        await backend.exists("t:cp")
        assert await backend.expire() == 2
        assert "expires_at <= UTC_TIMESTAMP(6)" in cursor.execute_calls[-1][0]
        assert conn.commits >= 2
