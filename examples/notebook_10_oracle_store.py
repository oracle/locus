#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 10: a Locus agent with cross-thread memory on Oracle 26ai.

``oracle_checkpointer`` (notebook 07) persists *per-thread* agent
state — same ``thread_id`` resumes a conversation. :class:`OracleStore`
is the other half of long-term memory: a namespaced key/value store
for *cross-thread* facts. The Locus agent loop reaches into it through
:class:`~locus.memory.manager.LLMMemoryManager`, which extracts
memories at the end of a session and injects them into the system
prompt at the start of the next one.

The agent in this notebook is the protagonist. It learns a fact about
the user in thread A, the manager writes the fact to Oracle 26ai, and
a brand-new agent in thread B answers a question that depends on that
fact — even though thread B's checkpoint never saw thread A's
messages.

Key concepts:

- ``OracleStore`` implements the :class:`BaseStore` contract on top of
  Oracle 26ai. Pass ``dimension=N`` for vector-mode storage; leave it
  unset for plain JSON K/V.
- ``LLMMemoryManager(store=oracle_store, namespace_prefix=("users",
  user_id))`` is the agent-facing surface. Hand it to
  ``Agent(memory_manager=...)`` and the agent loop calls
  ``manager.extract(messages)`` at session end and
  ``manager.retrieve()`` at session start.
- ``namespace_prefix`` scopes the store per tenant or per user. The
  same OracleStore can host memories for thousands of users without
  cross-talk.
- For deterministic demos we pass an explicit ``extract_fn``; in
  production point it at an LLM-backed extractor.

Run it::

    export ORACLE_DSN=mydb_low                   # tnsnames alias
    export ORACLE_USER=locus_app                 # least-privileged app schema
    export ORACLE_PASSWORD='<app-password>'
    export ORACLE_WALLET=~/.oci/wallets/mydb
    export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if encrypted

    python examples/notebook_11_oracle_store.py

If those env vars aren't set the notebook prints the wiring snippet and
exits cleanly — no traceback, no half-initialised state.

Difficulty: Intermediate. Self-contained — no prior notebook required.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import uuid

from config import get_model, print_config

from locus.agent import Agent
from locus.core.messages import Message, Role
from locus.memory.manager import LLMMemoryManager, Memory, MemoryType
from locus.memory.store_backends import OracleStore


_REQUIRED_ENV = (
    "ORACLE_DSN",
    "ORACLE_USER",
    "ORACLE_PASSWORD",
    "ORACLE_WALLET",
)


TABLE_NAME = "locus_notebook_11_store"
USER_ID = "alice"


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


def _print_skip_banner(missing: list[str]) -> None:
    print("\n--- Notebook 11: OracleStore-backed cross-thread memory agent ---")
    print(
        "Required environment variables not set; skipping the live demo so "
        "this file still runs cleanly in CI.\n"
    )
    print("Missing:")
    for name in missing:
        print(f"  - {name}")
    print(
        "\nProvision an Autonomous Database, drop its wallet under "
        "$ORACLE_WALLET, then set the variables above and re-run."
    )
    print("\nMinimal wiring (what the live path below builds):")
    print(
        """
    from locus.agent import Agent
    from locus.memory.manager import LLMMemoryManager
    from locus.memory.store_backends import OracleStore

    store = OracleStore(dsn=..., user=..., password=..., wallet_location=...)
    manager = LLMMemoryManager(
        store=store,
        namespace_prefix=("users", user_id),
        extract_fn=my_extractor,
    )
    agent = Agent(model=model, memory_manager=manager, ...)

    # Thread A teaches a fact.
    agent.run_sync("I work on OCI GenAI and I prefer us-chicago-1.",
                   thread_id="thread-a")
    # Brand-new thread B recalls it.
    agent.run_sync("Remind me what region I prefer?", thread_id="thread-b")
        """.rstrip()
    )


def _build_store() -> OracleStore:
    return OracleStore(
        dsn=os.environ["ORACLE_DSN"],
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        wallet_location=os.path.expanduser(os.environ["ORACLE_WALLET"]),
        wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", ""),
        table_name=TABLE_NAME,
        # K/V mode is fine for this demo — no vector embedding required
        # for the manager's recall surface, which uses list_keys / get.
        dimension=None,
    )


async def _drop_table(store: OracleStore) -> None:
    """Drop the demo table so the notebook is re-runnable."""
    pool = await store._get_pool()  # noqa: SLF001 — cleanup helper, not public API
    async with pool.acquire() as conn, conn.cursor() as cursor:
        try:
            await cursor.execute(f"DROP TABLE {TABLE_NAME} PURGE")
            await conn.commit()
        except Exception as exc:  # pragma: no cover — cleanup is best-effort
            print(f"  (cleanup) DROP TABLE {TABLE_NAME} skipped: {exc}")


_FAVORITE_RX = re.compile(
    r"(?:i\s+(?:prefer|like|use|work\s+with|run\s+on)|my\s+(?:favourite|favorite)\s+\w+\s+is)\s+([^\.\n]+)",
    re.IGNORECASE,
)
_ROLE_RX = re.compile(r"i\s+(?:work\s+on|build|maintain|own)\s+([^\.\n]+)", re.IGNORECASE)


async def _deterministic_extractor(messages: list[Message]) -> list[Memory]:
    """Tiny regex extractor — deterministic for the demo.

    Production deployments should pass an LLM-backed extractor; the
    heuristic the LLMMemoryManager bundles works but is conservative.
    """
    memories: list[Memory] = []
    for msg in messages:
        if msg.role != Role.USER or not msg.content:
            continue
        text = msg.content.strip()

        m = _FAVORITE_RX.search(text)
        if m:
            preference = m.group(1).strip().rstrip(".")
            memories.append(
                Memory(
                    type=MemoryType.USER,
                    key=f"preference_{uuid.uuid4().hex[:8]}",
                    content=f"User stated a preference: {preference}.",
                    metadata={"source": "regex", "raw": text[:200]},
                )
            )

        m = _ROLE_RX.search(text)
        if m:
            role_text = m.group(1).strip().rstrip(".")
            memories.append(
                Memory(
                    type=MemoryType.USER,
                    key=f"role_{uuid.uuid4().hex[:8]}",
                    content=f"User's role / focus: {role_text}.",
                    metadata={"source": "regex", "raw": text[:200]},
                )
            )
    return memories


def _build_manager(store: OracleStore) -> LLMMemoryManager:
    return LLMMemoryManager(
        store=store,
        namespace_prefix=("users", USER_ID),
        extract_fn=_deterministic_extractor,
    )


async def _drive_agent(agent: Agent, prompt: str, *, thread_id: str) -> None:
    """Async-friendly driver — we're already inside an event loop, so we
    can't call ``agent.run_sync`` (it spins up its own loop). ``agent.run``
    is the async generator that powers run_sync under the hood.
    """
    print(f"\n[{thread_id}] User : {prompt}")
    final = ""
    async for event in agent.run(prompt, thread_id=thread_id):
        if event.event_type == "terminate":
            final = getattr(event, "final_message", "") or ""
    print(f"[{thread_id}] Agent: {final.strip()}")


async def thread_a(manager: LLMMemoryManager) -> None:
    """Session 1 — the agent learns about Alice; the manager persists facts."""
    print("\n--- Thread A: agent learns about the user ---")
    agent = Agent(
        model=get_model(max_tokens=120),
        system_prompt=(
            "You are a helpful OCI infrastructure assistant. Be concise. "
            "Use any persistent facts you've been given about the user."
        ),
        memory_manager=manager,
    )
    await _drive_agent(
        agent,
        "Hi — I work on OCI GenAI infrastructure and I prefer us-chicago-1 "
        "for inference. Note that for next time.",
        thread_id="thread-a",
    )


async def show_persisted(store: OracleStore) -> None:
    """Inspect what the manager wrote to Oracle after thread A.

    Uses ``store.search`` instead of ``list_keys``/``get`` because the
    BaseStore's ``StoreItem`` wrapper makes the (namespace, key, value)
    triple available in one call — the same surface the LLM memory
    manager reaches for at session start.
    """
    print("\n--- What's in OracleStore now (admin view) ---")
    for memory_type in MemoryType:
        ns = ("users", USER_ID, memory_type.value)
        items = await store.search(ns, query=None, limit=10)
        if not items:
            continue
        print(f"  namespace = {ns}")
        for item in items:
            value = item.value
            content = value.get("content") if isinstance(value, dict) else value
            print(f"    {item.key}  →  {content}")


async def thread_b(manager: LLMMemoryManager) -> None:
    """Session 2 — fresh agent, fresh thread, no prior messages.

    The memory manager runs ``retrieve()`` against Oracle at session
    start and injects the recovered memories into the system prompt.
    The model never sees thread A's messages — only the extracted
    facts.
    """
    print("\n--- Thread B: brand-new agent recalls Alice's preference ---")
    agent = Agent(
        model=get_model(max_tokens=120),
        system_prompt=(
            "You are a helpful OCI infrastructure assistant. Be concise. "
            "Use any persistent facts you've been given about the user."
        ),
        memory_manager=manager,
    )
    await _drive_agent(
        agent,
        "Remind me which OCI region I prefer for inference, and what I work on.",
        thread_id="thread-b",
    )


async def main() -> None:
    missing = _missing_env()
    if missing:
        _print_skip_banner(missing)
        return

    print_config()
    print("Opening pool against Oracle 26ai…")
    store = _build_store()
    manager = _build_manager(store)
    try:
        await thread_a(manager)
        await show_persisted(store)
        await thread_b(manager)
        print(
            "\nThread B's agent never saw thread A's messages. It loaded "
            "Alice's preference from OracleStore at session start and "
            "answered from there. Same wallet, same pool kwargs as the "
            "RAG store (06), the checkpointer (07), and the versioned "
            "saver (12)."
        )
    finally:
        print("\n--- Cleanup: drop the demo table ---")
        await _drop_table(store)
        await store.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
