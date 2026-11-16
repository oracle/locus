# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The database-backed approval stores on live servers, and decided from another process.

The conformance suite is the unit suite's (``tests/unit/approval_store_conformance.py``);
here it runs against PostgreSQL (``POSTGRES_HOST`` / ``_PORT`` / ``_DB`` / ``_USER`` /
``_PASSWORD``), Redis (``REDIS_URL``) and Oracle (``ORACLE_DSN`` / ``ORACLE_USER`` /
``ORACLE_PASSWORD``, optionally ``ORACLE_WALLET`` / ``ORACLE_WALLET_PASSWORD``), behind
the markers the rest of the integration suite uses. Each class also has a hold decided
by ``locus approvals`` in a subprocess, which is what an operator does.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from locus.control import OracleApprovals, PostgresApprovals, RedisApprovals
from tests.unit.approval_store_conformance import ApprovalStoreConformance


pytestmark = pytest.mark.integration


def _pg_dsn() -> str:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.environ["POSTGRES_DB"]
    user = os.environ["POSTGRES_USER"]
    password = os.getenv("POSTGRES_PASSWORD", "")
    auth = f"{user}:{password}" if password else user
    return f"postgresql://{auth}@{host}:{port}/{db}"


def _decide_from_another_process(spec: str, approval_id: str) -> str:
    """``locus approvals approve`` in a subprocess, as an operator would run it."""
    completed = subprocess.run(  # noqa: S603 - our own interpreter, fixed arguments
        [
            sys.executable,
            "-m",
            "locus.cli",
            "approvals",
            "--store",
            spec,
            "approve",
            approval_id,
            "--by",
            "operator@example.com",
            "--reason",
            "checked the ticket",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


class _Live:
    """A hold decided from another process is seen by the agent's store."""

    def test_an_operator_decides_from_the_command_line(
        self, make_store: Callable[[Any], Any], operator_spec: Callable[[Any], str]
    ) -> None:
        store = make_store(None)
        approval_id = store.submit("svc", "issue_refund", {"order_id": "o1"}, reason="payment")

        out = _decide_from_another_process(operator_spec(store), approval_id)

        assert f"{approval_id}: approved by operator@example.com" in out
        record = store.get(approval_id)
        assert record is not None
        assert (record.status, record.decided_by) == ("approved", "operator@example.com")
        assert record.approvals[-1]["reason"] == "checked the ticket"
        assert store.consume(approval_id).status == "consumed"


@pytest.mark.requires_postgres
class TestPostgresApprovalsLive(ApprovalStoreConformance, _Live):
    @pytest.fixture
    def table(self) -> Iterator[str]:
        import asyncio

        import asyncpg

        name = f"locus_approvals_it_{uuid.uuid4().hex[:8]}"
        yield name

        async def drop() -> None:
            conn = await asyncpg.connect(_pg_dsn())
            try:
                await conn.execute(f"DROP TABLE IF EXISTS public.{name}")
            finally:
                await conn.close()

        asyncio.run(drop())

    @pytest.fixture
    def make_store(self, table: str) -> Iterator[Callable[[Any], PostgresApprovals]]:
        opened: list[PostgresApprovals] = []

        def make(authority: Any) -> PostgresApprovals:
            store = PostgresApprovals(dsn=_pg_dsn(), table_name=table, authority=authority)
            opened.append(store)
            return store

        yield make
        for store in opened:
            store.close()

    @pytest.fixture
    def reopen(self, make_store: Callable[[Any], PostgresApprovals]) -> Callable[[Any], Any]:
        return lambda store: make_store(store.authority)

    @pytest.fixture
    def operator_spec(self) -> Callable[[Any], str]:
        return lambda store: f"{_pg_dsn()}?table=public.{store.table_name}"


@pytest.mark.requires_redis
class TestRedisApprovalsLive(ApprovalStoreConformance, _Live):
    @pytest.fixture
    def prefix(self) -> Iterator[str]:
        import redis

        name = f"locus:approvals:it:{uuid.uuid4().hex[:8]}:"
        yield name
        client = redis.Redis.from_url(os.environ["REDIS_URL"])
        keys = client.keys(f"{name}*")
        if keys:
            client.delete(*keys)
        client.close()

    @pytest.fixture
    def make_store(self, prefix: str) -> Iterator[Callable[[Any], RedisApprovals]]:
        opened: list[RedisApprovals] = []

        def make(authority: Any) -> RedisApprovals:
            store = RedisApprovals(os.environ["REDIS_URL"], prefix=prefix, authority=authority)
            opened.append(store)
            return store

        yield make
        for store in opened:
            store.close()

    @pytest.fixture
    def reopen(self, make_store: Callable[[Any], RedisApprovals]) -> Callable[[Any], Any]:
        return lambda store: make_store(store.authority)

    @pytest.fixture
    def operator_spec(self) -> Callable[[Any], str]:
        return lambda store: f"{os.environ['REDIS_URL']}?prefix={store.prefix}"


def _oracle_available() -> bool:
    return bool(
        os.getenv("ORACLE_DSN") and os.getenv("ORACLE_USER") and os.getenv("ORACLE_PASSWORD")
    )


@pytest.mark.skipif(
    not _oracle_available(),
    reason="Oracle not configured (set ORACLE_DSN, ORACLE_USER, ORACLE_PASSWORD; ORACLE_WALLET)",
)
class TestOracleApprovalsLive(ApprovalStoreConformance, _Live):
    @pytest.fixture
    def table(self) -> Iterator[str]:
        import asyncio

        name = f"locus_appr_it_{uuid.uuid4().hex[:8]}"
        yield name

        async def drop() -> None:
            import oracledb

            store = OracleApprovals(**_oracle_connection(), table_name=name)
            pool = await store._get_pool()  # noqa: SLF001 - teardown through the same pool code
            async with pool.acquire() as conn, conn.cursor() as cursor:
                try:
                    await cursor.execute(f"DROP TABLE {name}")
                except oracledb.DatabaseError as error:
                    # A test that never reached the database made no table.
                    if "ORA-00942" not in str(error):
                        raise
                await conn.commit()
            await pool.close()

        asyncio.run(drop())

    @pytest.fixture
    def make_store(self, table: str) -> Iterator[Callable[[Any], OracleApprovals]]:
        opened: list[OracleApprovals] = []

        def make(authority: Any) -> OracleApprovals:
            store = OracleApprovals(**_oracle_connection(), table_name=table, authority=authority)
            opened.append(store)
            return store

        yield make
        for store in opened:
            store.close()

    @pytest.fixture
    def reopen(self, make_store: Callable[[Any], OracleApprovals]) -> Callable[[Any], Any]:
        return lambda store: make_store(store.authority)

    @pytest.fixture
    def operator_spec(self) -> Callable[[Any], str]:
        # user / password / dsn / wallet come from the ORACLE_* environment.
        return lambda store: f"oracle://?table={store.config.table_name}"


def _oracle_connection() -> dict[str, Any]:
    connection: dict[str, Any] = {
        "dsn": os.environ["ORACLE_DSN"],
        "user": os.environ["ORACLE_USER"],
        "password": os.environ["ORACLE_PASSWORD"],
    }
    if os.getenv("ORACLE_WALLET"):
        connection["wallet_location"] = os.environ["ORACLE_WALLET"]
        connection["wallet_password"] = os.getenv("ORACLE_WALLET_PASSWORD")
    return connection
