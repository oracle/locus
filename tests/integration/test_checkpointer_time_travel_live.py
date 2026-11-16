# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Time travel on the checkpointers that need a service: the shared suite, live.

The cases are ``tests/unit/checkpointer_time_travel_conformance.py``'s, run
over the generic PostgreSQL adapter and ``PgCheckpointer`` (``POSTGRES_USER``
and ``POSTGRES_DB``; ``POSTGRES_HOST`` / ``POSTGRES_PORT`` optional), Redis
(``REDIS_URL``), Oracle through ``oracle_checkpointer()`` and the versioned
``OracleCheckpointSaver`` (``ORACLE_DSN`` / ``ORACLE_USER`` /
``ORACLE_PASSWORD`` and a wallet in ``ORACLE_WALLET``), and an OCI Object
Storage bucket (``OCI_TT_BUCKET`` / ``OCI_TT_NAMESPACE``, with
``OCI_PROFILE`` and ``OCI_REGION`` for the client). Every checkpointer a
test builds is emptied of the threads the cases write when the test ends;
the tables are named ``locus_tt_*`` and dropped when the module ends.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import pytest

from locus.core.state import AgentState
from tests.unit.checkpointer_time_travel_conformance import (
    BoundedHistoryConformance,
    CheckpointerTimeTravelConformance,
)


pytestmark = pytest.mark.integration

#: The threads the conformance cases write, and the ledger sidecars they leave.
_THREADS = ("t", "t-alt", "t~ledger", "t-alt~ledger")


def _postgres_dsn() -> str | None:
    user, db = os.getenv("POSTGRES_USER"), os.getenv("POSTGRES_DB")
    if not user or not db:
        return None
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = os.getenv("POSTGRES_PORT", "5432")
    password = os.getenv("POSTGRES_PASSWORD", "")
    auth = f"{user}:{password}" if password else user
    return f"postgresql://{auth}@{host}:{port}/{db}"


def _oracle_available() -> bool:
    return bool(
        os.getenv("ORACLE_DSN") and os.getenv("ORACLE_PASSWORD") and os.getenv("ORACLE_WALLET")
    )


def _oracle_kwargs() -> dict[str, Any]:
    return {
        "dsn": os.environ["ORACLE_DSN"],
        "user": os.environ.get("ORACLE_USER", "ADMIN"),
        "password": os.environ["ORACLE_PASSWORD"],
        "wallet_location": os.environ["ORACLE_WALLET"],
        "wallet_password": os.environ.get("ORACLE_WALLET_PASSWORD") or None,
    }


async def _empty(cp: Any) -> None:
    """Delete the threads a case wrote (named ones and unnamed forks), then close."""
    for thread in _THREADS:
        await cp.delete(thread)
    if cp.capabilities.list_threads:
        for thread in await cp.list_threads(limit=1000, pattern="t-fork-*"):
            await cp.delete(thread)
    await cp.close()


class _Tracked:
    """Builds checkpointers through ``_build`` and empties every one at the end of the test."""

    def _build(self, keep: int | None) -> Any:  # pragma: no cover - each class provides one
        raise NotImplementedError

    @pytest.fixture
    def make_checkpointer(self, _built: list[Any]) -> Callable[..., Any]:
        def make(keep: int | None = None) -> Any:
            cp = self._build(keep)
            _built.append(cp)
            return cp

        return make

    @pytest.fixture
    def make_bounded(self, make_checkpointer: Callable[..., Any]) -> Callable[..., Any]:
        return make_checkpointer

    @pytest.fixture
    async def _built(self) -> AsyncIterator[list[Any]]:
        built: list[Any] = []
        yield built
        for cp in built:
            await _empty(cp)


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

_PG_TABLES = ("locus_tt_adapter", "locus_tt_native")


@pytest.fixture(scope="module")
def _drop_postgres_tables() -> Iterator[None]:
    yield
    dsn = _postgres_dsn()
    if dsn is None:
        return
    import asyncio

    import asyncpg

    async def drop() -> None:
        conn = await asyncpg.connect(dsn)
        try:
            for table in _PG_TABLES:
                await conn.execute(f"DROP TABLE IF EXISTS {table}")
        finally:
            await conn.close()

    asyncio.run(drop())


@pytest.mark.requires_postgres
@pytest.mark.skipif(_postgres_dsn() is None, reason="POSTGRES_USER / POSTGRES_DB not set")
@pytest.mark.usefixtures("_drop_postgres_tables")
class TestPostgresAdapterTimeTravel(
    _Tracked, CheckpointerTimeTravelConformance, BoundedHistoryConformance
):
    def _build(self, keep: int | None) -> Any:
        from locus.memory.backends import postgresql_checkpointer

        return postgresql_checkpointer(
            dsn=_postgres_dsn(), table_name="locus_tt_adapter", max_checkpoints_per_thread=keep
        )


@pytest.mark.requires_postgres
@pytest.mark.skipif(_postgres_dsn() is None, reason="POSTGRES_USER / POSTGRES_DB not set")
@pytest.mark.usefixtures("_drop_postgres_tables")
class TestPgCheckpointerTimeTravel(
    _Tracked, CheckpointerTimeTravelConformance, BoundedHistoryConformance
):
    def _build(self, keep: int | None) -> Any:
        from locus.memory.backends import PgCheckpointer

        dsn = _postgres_dsn()
        assert dsn is not None
        return PgCheckpointer(dsn, table="locus_tt_native", keep_checkpoints=keep)


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------


@pytest.mark.requires_redis
@pytest.mark.skipif(not os.getenv("REDIS_URL"), reason="REDIS_URL not set")
class TestRedisTimeTravel(_Tracked, CheckpointerTimeTravelConformance, BoundedHistoryConformance):
    def _build(self, keep: int | None) -> Any:
        from locus.memory.backends import redis_checkpointer

        return redis_checkpointer(
            url=os.environ["REDIS_URL"],
            prefix=f"locus:tt:{uuid.uuid4().hex[:8]}:",
            max_checkpoints_per_thread=keep,
        )


# ---------------------------------------------------------------------------
# Oracle AI Database
# ---------------------------------------------------------------------------

_ORACLE_TABLES = ("LOCUS_TT_ADAPTER", "LOCUS_TT_V_CHECKPOINTS", "LOCUS_TT_V_WRITES")


@pytest.fixture(scope="module")
def _drop_oracle_tables() -> Iterator[None]:
    yield
    if not _oracle_available():
        return
    import oracledb

    kwargs = _oracle_kwargs()
    conn = oracledb.connect(
        user=kwargs["user"],
        password=kwargs["password"],
        dsn=kwargs["dsn"],
        config_dir=kwargs["wallet_location"],
        wallet_location=kwargs["wallet_location"],
        wallet_password=kwargs["wallet_password"],
    )
    try:
        with conn.cursor() as cursor:
            for table in _ORACLE_TABLES:
                try:
                    cursor.execute(f"DROP TABLE {table} PURGE")
                except oracledb.DatabaseError as exc:  # ORA-00942: never created
                    if "ORA-00942" not in str(exc):
                        raise
        conn.commit()
    finally:
        conn.close()


@pytest.mark.skipif(not _oracle_available(), reason="ORACLE_* credentials not set")
@pytest.mark.usefixtures("_drop_oracle_tables")
class TestOracleAdapterTimeTravel(
    _Tracked, CheckpointerTimeTravelConformance, BoundedHistoryConformance
):
    def _build(self, keep: int | None) -> Any:
        from locus.memory.backends import oracle_checkpointer

        return oracle_checkpointer(
            table_name="LOCUS_TT_ADAPTER", max_checkpoints_per_thread=keep, **_oracle_kwargs()
        )


@pytest.mark.skipif(not _oracle_available(), reason="ORACLE_* credentials not set")
@pytest.mark.usefixtures("_drop_oracle_tables")
class TestOracleVersionedSaverTimeTravel:
    """The reference saver: one row per checkpoint, the parent link kept on a fork."""

    @pytest.fixture
    async def saver(self) -> AsyncIterator[tuple[Any, str]]:
        from locus.memory.backends import OracleCheckpointSaver

        store = OracleCheckpointSaver(table_name="LOCUS_TT_V", **_oracle_kwargs())
        thread = f"t-{uuid.uuid4().hex[:8]}"
        try:
            yield store, thread
        finally:
            await store.delete_thread(thread)
            await store.delete_thread(f"{thread}-alt")
            await store.close()

    async def test_list_and_fork_keep_the_parent_link(self, saver: tuple[Any, str]) -> None:
        store, thread = saver
        previous: str | None = None
        for step in range(3):
            await store.put(
                thread_id=thread,
                checkpoint_id=f"cp-{step}",
                checkpoint_data=AgentState().with_iteration(step).to_checkpoint(),
                parent_checkpoint_id=previous,
                metadata={"note": step},
            )
            previous = f"cp-{step}"

        infos = await store.list(thread)
        assert [i.checkpoint_id for i in infos] == ["cp-2", "cp-1", "cp-0"]
        assert [i.parent_checkpoint_id for i in infos] == ["cp-1", "cp-0", None]
        assert [i.iteration for i in infos] == [2, 1, 0]
        assert all(i.size and i.created_at for i in infos)

        forked = await store.fork(thread, "cp-1", new_thread_id=f"{thread}-alt")
        [first] = await store.list(forked)
        assert first.parent_checkpoint_id == "cp-1"
        assert first.forked_from == thread
        assert first.source == "fork"
        assert first.iteration == 1
        assert len(await store.list(thread)) == 3
        with pytest.raises(LookupError):
            await store.fork(thread, "no-such")
        with pytest.raises(ValueError, match="different"):
            await store.fork(thread, new_thread_id=thread)


# ---------------------------------------------------------------------------
# OCI Object Storage
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.getenv("OCI_TT_BUCKET") and os.getenv("OCI_TT_NAMESPACE")),
    reason="OCI_TT_BUCKET / OCI_TT_NAMESPACE not set",
)
class TestOCIBucketTimeTravel(
    _Tracked, CheckpointerTimeTravelConformance, BoundedHistoryConformance
):
    # Object Storage keeps modification times to the millisecond.
    settle_seconds = 0.05

    def _build(self, keep: int | None) -> Any:
        from locus.memory.backends import OCIBucketBackend

        return OCIBucketBackend(
            bucket_name=os.environ["OCI_TT_BUCKET"],
            namespace=os.environ["OCI_TT_NAMESPACE"],
            prefix=f"locus-tt/{uuid.uuid4().hex[:8]}/",
            profile_name=os.getenv("OCI_PROFILE", "DEFAULT"),
            auth_type=os.getenv("OCI_AUTH_TYPE", "api_key"),
            region=os.getenv("OCI_REGION"),
            max_checkpoints_per_thread=keep,
        )
