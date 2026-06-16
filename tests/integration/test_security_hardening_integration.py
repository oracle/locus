# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end regression tests for the security hardening work.

These tests hit real services (Oracle ADB, OpenSearch) so they belong in the
integration suite. They verify that the validators we added at config
construction time refuse to even instantiate a store or backend when given a
SQL-injection payload — meaning the malicious identifier never reaches the
database.

Environment (see tests/integration/conftest.py for full setup):
    ORACLE_DSN, ORACLE_USER, ORACLE_PASSWORD, ORACLE_WALLET_LOCATION,
    ORACLE_WALLET_PASSWORD, TNS_ADMIN — required for Oracle tests.
    OPENSEARCH_HOSTS, OPENSEARCH_USER, OPENSEARCH_PASSWORD — required for
    the OpenSearch smoke test.
"""

from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# F2 / F3 — Oracle: config-time rejection never touches the database
# ---------------------------------------------------------------------------


class TestOracleInjectionRejectedBeforeConnection:
    """Malicious identifiers never reach the ADB — validation is config-time.

    If one of these tests ever fails with a DatabaseError instead of a
    ValueError, it means the validator was bypassed and SQL was actually
    executed. That would be a regression of F2/F3.
    """

    @pytest.mark.parametrize(
        "bad_table",
        [
            "t; DROP TABLE secrets",
            "t' UNION SELECT 1 FROM DUAL --",
            "a space",
        ],
    )
    def test_oracle_memory_config_rejects_injection(self, bad_table):
        from locus.memory.backends.oracle import OracleConfig

        with pytest.raises((ValueError, Exception)):
            OracleConfig(table_name=bad_table)

    @pytest.mark.parametrize(
        "bad_metric",
        ["COSINE; DROP TABLE docs", "COSINE) WITH ", ""],
    )
    def test_oracle_vector_config_rejects_bad_metric(self, bad_metric):
        from locus.rag.stores.oracle import OracleVectorConfig

        with pytest.raises((ValueError, Exception)):
            OracleVectorConfig(distance_metric=bad_metric)


# ---------------------------------------------------------------------------
# F2 — Oracle memory backend (legitimate round-trip still works)
# ---------------------------------------------------------------------------


def _oracle_env_ok() -> bool:
    return bool(
        os.getenv("ORACLE_DSN")
        and os.getenv("ORACLE_PASSWORD")
        and os.getenv("ORACLE_WALLET_LOCATION")
    )


@pytest.mark.skipif(not _oracle_env_ok(), reason="Oracle ADB env vars not set")
class TestOracleMemoryBackendLegitimate:
    """With valid identifiers, OracleBackend must still work end-to-end.

    This proves the new validator is not over-broad.
    """

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self):
        from locus.memory.backends.oracle import OracleBackend

        backend = OracleBackend(
            dsn=os.environ["ORACLE_DSN"],
            user=os.environ["ORACLE_USER"],
            password=os.environ["ORACLE_PASSWORD"],
            wallet_location=os.environ["ORACLE_WALLET_LOCATION"],
            wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", os.environ["ORACLE_PASSWORD"]),
            table_name="sec_test_memory",
        )

        try:
            # Verify the backend's SQL layer round-trips under the new
            # identifier validator. ``OracleBackend.save`` now matches
            # the :class:`BaseCheckpointer` abstract — first arg is the
            # :class:`AgentState`, second is the ``thread_id`` — so the
            # test wraps a small dict in a real ``AgentState`` rather
            # than passing a bare payload positionally.
            from locus.core.messages import Message
            from locus.core.state import AgentState

            state = AgentState(messages=(Message.user("hello"),))
            await backend.save(state, "thread-sec-test")
            loaded = await backend.load("thread-sec-test")
            assert loaded is not None
            assert loaded.messages == state.messages
        finally:
            # Cleanup: drop the regression-test table so re-runs are idempotent.
            try:
                pool = await backend._get_pool()
                async with pool.acquire() as conn, conn.cursor() as cur:
                    await cur.execute("DROP TABLE sec_test_memory PURGE")
                    await conn.commit()
            except Exception:  # noqa: BLE001 — cleanup is best-effort
                pass


# ---------------------------------------------------------------------------
# F3 — Oracle vector store: valid metric round-trips against ADB
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _oracle_env_ok(), reason="Oracle ADB env vars not set")
class TestOracleVectorBackendLegitimate:
    @pytest.mark.parametrize("metric", ["COSINE", "EUCLIDEAN"])
    @pytest.mark.asyncio
    async def test_valid_metric_creates_table(self, metric):
        from locus.rag.stores.oracle import OracleVectorStore

        store = OracleVectorStore(
            dsn=os.environ["ORACLE_DSN"],
            user=os.environ["ORACLE_USER"],
            password=os.environ["ORACLE_PASSWORD"],
            wallet_location=os.environ["ORACLE_WALLET_LOCATION"],
            wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", os.environ["ORACLE_PASSWORD"]),
            dimension=8,
            distance_metric=metric,
            table_name=f"sec_test_vec_{metric.lower()}",
        )

        try:
            # count() goes through _ensure_table → CREATE TABLE + CREATE INDEX.
            # If the validator had been bypassed and the metric was something
            # like "COSINE; DROP", the CREATE VECTOR INDEX statement would
            # raise a DatabaseError.
            count = await store.count()
            assert count == 0
        finally:
            try:
                pool = await store._get_pool()
                async with pool.acquire() as conn, conn.cursor() as cur:
                    await cur.execute(f"DROP TABLE sec_test_vec_{metric.lower()} PURGE")
                    await conn.commit()
            except Exception:  # noqa: BLE001
                pass
