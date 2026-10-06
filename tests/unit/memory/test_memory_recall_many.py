# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Recall in fewer round trips, across several namespaces, without crossing tenants.

* ``PgMemory`` encodes a text once per tenant (an LRU keyed by ``(tenant, text)``),
  ranks several namespaces in one transaction and one ``SELECT``
  (``search_many``), refuses a call that spans tenants, and says loudly when it
  has no numpy and recall is a substring match.
* ``BaseStore.search_many`` loops ``search`` for every other store.
* ``LLMMemoryManager.retrieve`` makes two store calls (ranking + recency) for
  every memory type of every namespace, and a resolver's ``MemoryScope`` adds
  read-only namespaces of the same tenant to a run's recall. A recall namespace
  of another tenant is dropped and never read.

No database here: ``PgMemory`` talks to a stub pool that records statements.
The live behaviour, RLS included, is in ``tests/integration/test_pg_memory.py``.
"""

from __future__ import annotations

import json
import logging
import warnings
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from tulip.core.messages import Message
from tulip.core.state import AgentState
from tulip.memory import MemoryScope
from tulip.memory.manager import LLMMemoryManager, Memory, MemoryType
from tulip.memory.store import BaseStore, InMemoryStore, StoreItem
from tulip.memory.store_backends import postgresql as pgmod
from tulip.memory.store_backends.postgresql import PgMemory


_DSN = "postgresql://u:p@127.0.0.1:5432/db"  # never connected to


# ---------------------------------------------------------------------------
# A stub asyncpg pool that records what PgMemory sends
# ---------------------------------------------------------------------------


class _Conn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.statements: list[tuple[str, tuple[Any, ...]]] = []
        self.transactions = 0

    @asynccontextmanager
    async def transaction(self) -> Any:
        self.transactions += 1
        yield

    async def execute(self, sql: str, *args: Any) -> str:
        self.statements.append((sql, args))
        return "SELECT 1"

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.statements.append((sql, args))
        return self.rows


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn
        self.acquired = 0

    @asynccontextmanager
    async def acquire(self) -> Any:
        self.acquired += 1
        yield self.conn


def _row(idx: int, key: str, content: str) -> dict[str, Any]:
    now = datetime(2026, 10, 6, tzinfo=UTC)
    return {
        "idx": idx,
        "key": key,
        "value": json.dumps({"type": "user", "key": key, "content": content, "metadata": {}}),
        "metadata": "{}",
        "created_at": now,
        "updated_at": now,
        "version": 1,
        "_rank": 0.1,
    }


def _stubbed(rows: list[dict[str, Any]] | None = None, **kwargs: Any) -> tuple[PgMemory, _Conn]:
    store = PgMemory(_DSN, **kwargs)
    conn = _Conn(rows or [])
    store._pool = _Pool(conn)  # type: ignore[assignment]
    return store, conn


def _count_encodings(store: PgMemory, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    async def embed(content: str) -> str | None:
        seen.append(content)
        return "[" + ",".join(["0.5"] * store._vdim) + "]"

    monkeypatch.setattr(store, "_embed", embed)
    return seen


# ---------------------------------------------------------------------------
# PgMemory: the encoding cache
# ---------------------------------------------------------------------------


async def test_a_text_is_encoded_once_per_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    store, _ = _stubbed()
    seen = _count_encodings(store, monkeypatch)

    for _ in range(4):
        await store.search(("acme", "user"), "build me a castle")
    assert seen == ["build me a castle"]

    # Another tenant never gets acme's entry: it encodes its own.
    await store.search(("globex", "user"), "build me a castle")
    assert seen == ["build me a castle", "build me a castle"]
    assert set(store._encodings) == {
        ("acme", "build me a castle"),
        ("globex", "build me a castle"),
    }


async def test_the_encoding_cache_is_bounded_and_can_be_turned_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = _stubbed(encoding_cache=2)
    seen = _count_encodings(store, monkeypatch)
    for text in ("a", "b", "c", "a"):
        await store.search(("t", "user"), text)
    assert len(store._encodings) == 2
    assert seen == ["a", "b", "c", "a"]  # "a" was evicted by "c"

    off, _ = _stubbed(encoding_cache=0)
    seen_off = _count_encodings(off, monkeypatch)
    await off.search(("t", "user"), "same")
    await off.search(("t", "user"), "same")
    assert seen_off == ["same", "same"]
    assert not off._encodings

    with pytest.raises(ValueError, match="encoding_cache"):
        PgMemory(_DSN, encoding_cache=-1)


# ---------------------------------------------------------------------------
# PgMemory: search_many
# ---------------------------------------------------------------------------


async def test_search_many_is_one_transaction_and_one_select(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_row(1, "k1", "likes castles"), _row(1, "k2", "likes towers"), _row(3, "k3", "fam")]
    store, conn = _stubbed(rows)
    seen = _count_encodings(store, monkeypatch)
    namespaces = [("acme", "kid", "user"), ("acme", "kid", "feedback"), ("acme", "family", "user")]

    found = await store.search_many(namespaces, "castle", limit=4)

    assert store._pool.acquired == 1  # type: ignore[union-attr]
    assert conn.transactions == 1
    # The tenant pin, then ONE select for all three namespaces.
    assert len(conn.statements) == 2
    assert "set_config" in conn.statements[0][0]
    assert conn.statements[0][1] == ("acme",)
    sql, args = conn.statements[1]
    assert "unnest($2::text[]) WITH ORDINALITY" in sql
    assert "CROSS JOIN LATERAL" in sql
    assert "tenant=$1" in sql
    assert args[0] == "acme"
    assert args[1] == ["\x1f".join(ns) for ns in namespaces]
    assert args[-1] == 4
    assert seen == ["castle"]  # the query, encoded once for every namespace

    assert [[i.key for i in items] for items in found] == [["k1", "k2"], [], ["k3"]]
    assert found[2][0].namespace == namespaces[2]


async def test_search_many_without_a_query_ranks_by_recency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, conn = _stubbed([])
    seen = _count_encodings(store, monkeypatch)
    assert await store.search_many([("t", "a"), ("t", "b")], None, limit=3) == [[], []]
    sql = conn.statements[-1][0]
    assert "ORDER BY updated_at DESC" in sql
    assert "<=>" not in sql
    assert seen == []
    assert await store.search_many([], "x") == []


async def test_search_many_refuses_to_span_tenants() -> None:
    store, conn = _stubbed([])
    with pytest.raises(ValueError, match="spans tenants"):
        await store.search_many([("acme", "user"), ("globex", "user")], "anything")
    assert conn.statements == []  # nothing was sent


# ---------------------------------------------------------------------------
# PgMemory without numpy: loud, not silent
# ---------------------------------------------------------------------------


async def test_without_numpy_it_warns_at_construction_and_on_the_first_substring_search(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(pgmod, "_numpy", lambda: None)
    with pytest.warns(RuntimeWarning, match="substring match"):
        store = PgMemory(_DSN)
    conn = _Conn([])
    store._pool = _Pool(conn)  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING, logger=pgmod.__name__):
        await store.search(("t", "user"), "castle")
        await store.search_many([("t", "user")], "castle")
    warned = [r for r in caplog.records if "substring match" in r.getMessage()]
    assert len(warned) == 1  # once per store, not once per search
    assert "ILIKE" in conn.statements[-1][0]


def test_with_an_embedder_and_no_numpy_nothing_is_said(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Embedder:
        dimension = 8

    monkeypatch.setattr(pgmod, "_numpy", lambda: None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PgMemory(_DSN, embedder=_Embedder())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# BaseStore.search_many: the default every store gets
# ---------------------------------------------------------------------------


async def test_the_default_search_many_is_search_per_namespace() -> None:
    store = InMemoryStore()
    await store.put(("t", "a"), "k1", {"content": "castle"})
    await store.put(("t", "b"), "k2", {"content": "tower"})
    found = await store.search_many([("t", "a"), ("t", "b"), ("t", "c")], None, limit=5)
    assert [[i.key for i in items] for items in found] == [["k1"], ["k2"], []]


# ---------------------------------------------------------------------------
# LLMMemoryManager: two store calls per recall, and MemoryScope
# ---------------------------------------------------------------------------


class _CountingStore(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.many: list[tuple[list[tuple[str, ...]], str | None]] = []
        self.single = 0

    async def search_many(  # type: ignore[override]
        self, namespaces: Any, query: str | None = None, limit: int = 10
    ) -> list[list[StoreItem]]:
        self.many.append((list(namespaces), query))
        return await super().search_many(namespaces, query, limit)

    async def search(self, *args: Any, **kwargs: Any) -> list[StoreItem]:
        self.single += 1
        return await super().search(*args, **kwargs)


def _mem(key: str, content: str, kind: MemoryType = MemoryType.USER) -> Memory:
    return Memory(type=kind, key=key, content=content)


async def test_a_recall_is_two_store_calls_whatever_the_number_of_types() -> None:
    store = _CountingStore()
    manager = LLMMemoryManager(store=store, namespace_prefix=("acme", "kid"))
    await manager.save(
        [_mem("castle", "likes castles"), _mem("tnt", "no tnt", MemoryType.FEEDBACK)]
    )
    store.many.clear()

    found = await manager.retrieve(query="castles")

    assert len(store.many) == 2
    ranked, recent = store.many
    assert ranked[1] == "castles"
    assert recent[1] is None
    assert ranked[0] == [("acme", "kid", t.value) for t in MemoryType]
    assert {m.key for m in found} == {"castle", "tnt"}
    assert found[0].key == "castle"  # the match ranks ahead of the top-up


async def test_a_store_without_search_many_still_recalls() -> None:
    class _Plain:
        """Duck-typed: search but no search_many."""

        def __init__(self) -> None:
            self.inner = InMemoryStore()

        async def search(self, ns: Any, query: Any = None, limit: int = 10) -> Any:
            return await self.inner.search(ns, query=query, limit=limit)

        async def put(self, *a: Any, **k: Any) -> None:
            await self.inner.put(*a, **k)

        async def list_keys(self, *a: Any, **k: Any) -> Any:
            return await self.inner.list_keys(*a, **k)

        async def get(self, *a: Any, **k: Any) -> Any:
            return await self.inner.get(*a, **k)

        async def delete(self, *a: Any, **k: Any) -> Any:
            return await self.inner.delete(*a, **k)

    manager = LLMMemoryManager(store=_Plain(), namespace_prefix=("t",))  # type: ignore[arg-type]
    await manager.save([_mem("k", "likes castles")])
    assert [m.key for m in await manager.retrieve(query="castles")] == ["k"]


async def test_retrieve_over_explicit_namespaces_of_one_tenant() -> None:
    store = InMemoryStore()
    manager = LLMMemoryManager(store=store)
    with manager.scoped(("acme", "kid")):
        await manager.save([_mem("mine", "likes castles")])
    with manager.scoped(("acme", "family")):
        await manager.save([_mem("ours", "the family builds castles")])

    found = await manager.retrieve_relevant(
        "castles", namespaces=[("acme", "kid"), ("acme", "family")]
    )
    assert {m.key for m in found} == {"mine", "ours"}

    with pytest.raises(ValueError, match="more than one tenant"):
        await manager.retrieve(namespaces=[("acme", "kid"), ("globex", "kid")])


def _state(text: str, **metadata: Any) -> AgentState:
    return AgentState(messages=(Message.user(text),), metadata=metadata)


def _block(state: AgentState) -> str:
    return "\n".join(m.content or "" for m in state.messages if m.role.value == "system")


async def test_a_memory_scope_recalls_own_and_shared_in_one_block_and_writes_only_own() -> None:
    store = _CountingStore()

    async def extract(messages: list[Message]) -> list[Memory]:
        return [_mem("new", "built a tower today")]

    manager = LLMMemoryManager(
        store=store,
        extract_fn=extract,
        namespace_resolver=lambda run: MemoryScope(
            namespace=("acme", "kid", run.metadata["kid"]),
            recall=(("acme", "family"),),
        ),
    )
    with manager.scoped(("acme", "kid", "sofia")):
        await manager.save([_mem("mine", "Sofia likes castles")])
    with manager.scoped(("acme", "family")):
        await manager.save([_mem("ours", "the family names castles after cats")])
    store.many.clear()

    state = await manager.on_session_start(_state("castles!", kid="sofia"))
    block = _block(state)
    assert "Sofia likes castles" in block
    assert "the family names castles after cats" in block
    assert sum(1 for m in state.messages if m.role.value == "system") == 1  # one block
    assert len(store.many) == 2  # own + family, every type: two calls

    await manager.on_session_end(_state("I built a tower", kid="sofia"))
    own = await store.list_keys(("acme", "kid", "sofia", "user"))
    family = await store.list_keys(("acme", "family", "user"))
    assert "new" in own
    assert "new" not in family  # the recall namespace is read-only


async def test_a_recall_namespace_of_another_tenant_is_never_read(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Tenant isolation: acme's resolver naming globex's namespace recalls nothing of it."""
    store = _CountingStore()
    manager = LLMMemoryManager(
        store=store,
        namespace_resolver=lambda run: MemoryScope(
            namespace=("acme", "kid"),
            recall=(("globex", "secrets"),),
        ),
    )
    with manager.scoped(("globex", "secrets")):
        await manager.save([_mem("plan", "globex launch plan castles")])
    with manager.scoped(("acme", "kid")):
        await manager.save([_mem("mine", "acme kid likes castles")])
    store.many.clear()

    with caplog.at_level(logging.WARNING):
        state = await manager.on_session_start(_state("castles"))

    block = _block(state)
    assert "acme kid likes castles" in block
    assert "globex" not in block
    read = {ns for namespaces, _ in store.many for ns in namespaces}
    assert all(ns[0] == "acme" for ns in read)
    assert any("dropped" in r.getMessage() for r in caplog.records)


async def test_a_bare_prefix_from_the_resolver_still_works() -> None:
    store = InMemoryStore()
    manager = LLMMemoryManager(store=store, namespace_resolver=lambda run: ("acme", "kid"))
    with manager.scoped(("acme", "kid")):
        await manager.save([_mem("mine", "likes castles")])
    assert "likes castles" in _block(await manager.on_session_start(_state("castles")))
    assert manager.active_recall == (("tulip_memory",),)  # outside a scope


# ---------------------------------------------------------------------------
# _prune: one read, not one per key
# ---------------------------------------------------------------------------


async def test_prune_reads_the_type_once() -> None:
    class _Store(_CountingStore):
        gets = 0

        async def get(self, *args: Any, **kwargs: Any) -> Any:
            type(self).gets += 1
            return await super().get(*args, **kwargs)

    store = _Store()
    manager = LLMMemoryManager(store=store, namespace_prefix=("t",), max_memories=2)
    for i in range(4):
        await manager.save([_mem(f"k{i}", f"fact {i}")])
    assert _Store.gets == 0
    assert sorted(await store.list_keys(("t", "user"))) == ["k2", "k3"]


async def test_prune_falls_back_when_search_fails() -> None:
    class _NoSearch(InMemoryStore):
        async def search(self, *args: Any, **kwargs: Any) -> list[StoreItem]:
            raise RuntimeError("no search here")

    store = _NoSearch()
    manager = LLMMemoryManager(store=store, namespace_prefix=("t",), max_memories=1)
    for i in range(3):
        await manager.save([_mem(f"k{i}", f"fact {i}")])
    assert await store.list_keys(("t", "user")) == ["k2"]


def test_base_store_has_search_many() -> None:
    assert BaseStore.search_many is InMemoryStore.search_many
