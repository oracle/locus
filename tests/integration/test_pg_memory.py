# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for :class:`PgMemory` against a real Postgres + pgvector.

These run only when a Postgres is configured (``POSTGRES_HOST`` / ``_PORT`` /
``_USER`` / ``_DB``); otherwise they skip. They validate the enterprise memory
store end-to-end on real infrastructure — CRUD, HRR-over-pgvector semantic
recall, and, most importantly, **tenant isolation enforced by Row-Level
Security using a real non-superuser role** (a superuser bypasses RLS, so the
isolation guarantee can only be proven with an unprivileged app role).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest

from tulip.memory.store_backends.postgresql import PgMemory


try:
    from tests.integration.conftest import skip_without_openai, skip_without_postgres
except ImportError:  # pragma: no cover - conftest is always importable under pytest
    skip_without_postgres = pytest.mark.skipif(True, reason="conftest unavailable")
    skip_without_openai = pytest.mark.skipif(True, reason="conftest unavailable")

pytestmark = [pytest.mark.integration, skip_without_postgres]

_TABLE = "tulip_memories_it"
_RLS_ROLE = "tulip_mem_rls_test"
_RLS_PW = "rls_test_pw"  # noqa: S105 - ephemeral local test role, dropped on teardown


def _admin_dsn() -> str:
    host = os.environ["POSTGRES_HOST"]
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.environ["POSTGRES_USER"]
    pw = os.getenv("POSTGRES_PASSWORD", "")
    db = os.environ["POSTGRES_DB"]
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def _role_dsn() -> str:
    host = os.environ["POSTGRES_HOST"]
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.environ["POSTGRES_DB"]
    return f"postgresql://{_RLS_ROLE}:{_RLS_PW}@{host}:{port}/{db}"


@pytest.fixture
async def store() -> AsyncIterator[PgMemory]:
    """A PgMemory on a dedicated test table, dropped clean before and after."""
    import asyncpg

    admin = await asyncpg.connect(_admin_dsn())
    await admin.execute(f"DROP TABLE IF EXISTS {_TABLE} CASCADE")
    await admin.close()

    s = PgMemory(_admin_dsn(), table=_TABLE, dim=256)
    try:
        yield s
    finally:
        await s.close()
        admin = await asyncpg.connect(_admin_dsn())
        await admin.execute(f"DROP TABLE IF EXISTS {_TABLE} CASCADE")
        await admin.close()


# ── CRUD on real Postgres ─────────────────────────────────────────────────────
async def test_put_get_delete_list_and_upsert(store: PgMemory) -> None:
    ns = ("acme", "user", "fede")
    await store.put(ns, "fy", {"content": "fiscal year starts in April"}, {"type": "user"})
    await store.put(ns, "style", "prefers concise answers")

    assert await store.get(ns, "fy") == {"content": "fiscal year starts in April"}
    assert await store.get(ns, "missing") is None
    assert set(await store.list_keys(ns)) == {"fy", "style"}

    # upsert bumps the version, keeps one row
    await store.put(ns, "fy", {"content": "fiscal year starts in July"})
    hit = await store.get(ns, "fy")
    assert hit == {"content": "fiscal year starts in July"}
    items = await store.search(ns, None)
    assert len([i for i in items if i.key == "fy"]) == 1
    assert next(i for i in items if i.key == "fy").version == 2

    assert await store.delete(ns, "fy") is True
    assert await store.delete(ns, "fy") is False
    assert await store.get(ns, "fy") is None


async def test_associative_recall_over_pgvector_hrr(store: PgMemory) -> None:
    """The HRR fallback (no embedder) does lexical/associative cosine recall in
    Postgres: a query sharing tokens with a fact recalls it. HRR is NOT semantic
    — paraphrase matching is covered by the real-embedder test below."""
    ns = ("acme", "user", "fede")
    await store.put(ns, "fy", {"content": "the fiscal year starts in April"})
    await store.put(ns, "pet", {"content": "has a golden retriever named Argus"})
    await store.put(ns, "car", {"content": "drives an electric hatchback"})

    top = await store.search(ns, "which month does the fiscal year start", limit=1)
    assert top[0].key == "fy"  # matched on shared tokens, via pgvector cosine ANN


@skip_without_openai
async def test_true_semantic_recall_with_real_embedder() -> None:
    """With a real embedder, recall matches *meaning* — paraphrases with zero
    shared tokens land on the right fact (what HRR alone cannot do)."""
    import asyncpg

    from tulip.rag.embeddings.openai import OpenAIEmbeddings

    table = "tulip_memories_sem_it"
    admin = await asyncpg.connect(_admin_dsn())
    await admin.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    await admin.close()

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    store = PgMemory(_admin_dsn(), table=table, embedder=embedder)
    try:
        assert store.capabilities.semantic_search is True
        ns = ("acme", "user", "fede")
        await store.put(ns, "fy", {"content": "the company fiscal year starts in April"})
        await store.put(ns, "stack", {"content": "the platform runs on AWS EKS with Aurora"})
        await store.put(ns, "coffee", {"content": "prefers oat milk flat whites"})

        # "accounting period" shares no token with "fiscal year" — semantics only.
        top = await store.search(ns, "when does the accounting period begin", limit=1)
        assert top[0].key == "fy"
        assert (await store.search(ns, "what do they like to drink", limit=1))[0].key == "coffee"
    finally:
        await store.close()
        admin = await asyncpg.connect(_admin_dsn())
        await admin.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await admin.close()


async def test_search_without_query_returns_most_recent(store: PgMemory) -> None:
    ns = ("acme", "t")
    await store.put(ns, "a", "first")
    await store.put(ns, "b", "second")
    recent = await store.search(ns, None, limit=1)
    assert recent[0].key == "b"


async def test_memory_persists_and_recalls_across_store_instances(store: PgMemory) -> None:
    """Learn in "run 1", recall in "run 2": a fresh store (new pool/connection)
    over the same table recalls what a prior instance persisted — the durable
    cross-run memory guarantee, on real Postgres."""
    ns = ("acme", "user", "fede")
    await store.put(ns, "fy", {"content": "the fiscal year starts in April"})

    run2 = PgMemory(_admin_dsn(), table=_TABLE, dim=256)
    try:
        top = await run2.search(ns, "when does the fiscal year start", limit=1)
        assert top[0].value == {"content": "the fiscal year starts in April"}
    finally:
        await run2.close()


# ── tenant isolation — the query layer ────────────────────────────────────────
async def test_query_layer_tenant_isolation(store: PgMemory) -> None:
    """PgMemory's own API never returns another tenant's rows, even same key/ns."""
    await store.put(("acme", "u", "x"), "secret", {"content": "acme quarterly plan"})
    await store.put(("globex", "u", "x"), "secret", {"content": "globex quarterly plan"})

    assert await store.get(("acme", "u", "x"), "secret") == {"content": "acme quarterly plan"}
    assert await store.get(("globex", "u", "x"), "secret") == {"content": "globex quarterly plan"}

    acme_hits = await store.search(("acme", "u", "x"), "quarterly plan", limit=10)
    assert [i.value["content"] for i in acme_hits] == ["acme quarterly plan"]
    assert await store.list_keys(("globex", "u", "x")) == ["secret"]


# ── tenant isolation — Row-Level Security with a real non-superuser role ───────
async def test_rls_blocks_cross_tenant_access_for_app_role(store: PgMemory) -> None:
    """The hard guarantee: even a raw table scan by an app role is RLS-confined.

    A superuser bypasses RLS, so this creates an unprivileged role, grants it
    only DML, and proves that with ``tulip.tenant`` pinned to one tenant it can
    neither *read* nor *write* another tenant's rows — no ``WHERE`` needed.
    """
    import asyncpg

    # Seed two tenants via the (superuser) store, and create the app role.
    await store.put(("acme", "u", "x"), "k", {"content": "acme only"})
    await store.put(("globex", "u", "x"), "k", {"content": "globex only"})

    admin = await asyncpg.connect(_admin_dsn())
    await admin.execute(
        f"DO $$ BEGIN "
        f"IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='{_RLS_ROLE}') THEN "
        f"CREATE ROLE {_RLS_ROLE} LOGIN PASSWORD '{_RLS_PW}' NOSUPERUSER NOBYPASSRLS; "
        f"END IF; END $$"
    )
    await admin.execute(f"GRANT USAGE ON SCHEMA public TO {_RLS_ROLE}")
    await admin.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {_TABLE} TO {_RLS_ROLE}")
    await admin.close()

    role_conn = await asyncpg.connect(_role_dsn())
    try:
        async with role_conn.transaction():
            await role_conn.execute("SELECT set_config('tulip.tenant', 'acme', true)")
            # A raw scan with NO tenant predicate — RLS must confine it to acme.
            rows = await role_conn.fetch(f"SELECT tenant, key FROM {_TABLE}")
            assert rows, "app role should see its own tenant's rows"
            assert {r["tenant"] for r in rows} == {"acme"}

        async with role_conn.transaction():
            await role_conn.execute("SELECT set_config('tulip.tenant', 'globex', true)")
            rows = await role_conn.fetch(f"SELECT tenant, key FROM {_TABLE}")
            assert {r["tenant"] for r in rows} == {"globex"}

        # WITH CHECK: pinned to acme, the role cannot write a globex row.
        # SET LOCAL + INSERT run in one implicit transaction (a single execute).
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await role_conn.execute(
                "SET LOCAL tulip.tenant = 'acme'; "
                f"INSERT INTO {_TABLE} (tenant, ns, key, value) "
                "VALUES ('globex', 'globex', 'evil', '{}'::jsonb)"
            )
    finally:
        await role_conn.close()
        admin = await asyncpg.connect(_admin_dsn())
        await admin.execute(f"REVOKE ALL ON {_TABLE} FROM {_RLS_ROLE}")
        await admin.execute(f"REVOKE ALL ON SCHEMA public FROM {_RLS_ROLE}")
        await admin.execute(f"DROP ROLE IF EXISTS {_RLS_ROLE}")
        await admin.close()
