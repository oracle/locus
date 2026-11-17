# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 37: a Locus agent with cross-thread memory on Oracle 26ai.

``oracle_checkpointer`` (notebook 42) persists *per-thread* agent
state — same ``thread_id`` resumes a conversation. :class:`OracleStore`
is the other half of long-term memory: a namespaced key/value store
for *cross-thread* facts. The Locus agent loop reaches into it through
:class:`~locus.memory.manager.LLMMemoryManager`, which extracts
memories at the end of a session and, at the start of the next one,
inserts them as a system message right after the user's turn.

The agent in this notebook is the protagonist. It learns a fact about
the user in thread A, the manager writes the fact to Oracle 26ai, and
a brand-new agent in thread B answers a question that depends on that
fact — even though thread B never saw thread A's messages.

Key concepts:

- ``OracleStore`` implements the :class:`BaseStore` contract on top of
  Oracle 26ai. Pass ``dimension=N`` for vector-mode storage; leave it
  unset for plain JSON key/value rows.
- ``LLMMemoryManager(store=oracle_store, namespace_prefix=("users",
  user_id))`` is the agent-facing surface. Hand it to
  ``Agent(memory_manager=...)`` and the agent loop calls
  ``manager.on_session_end`` (extract and save) when a run finishes and
  ``manager.on_session_start`` (recall and inject) before the first
  model call.
- ``namespace_prefix`` scopes the store per tenant or per user. The
  same OracleStore can host memories for thousands of users without
  cross-talk.
- For deterministic demos we pass an explicit ``extract_fn``; in
  production point it at an LLM-backed extractor.

Run it:
    export ORACLE_DSN=mydb_low                   # tnsnames alias
    export ORACLE_USER=locus_app                 # least-privileged app schema
    export ORACLE_PASSWORD='<app-password>'
    export ORACLE_WALLET=~/.oci/wallets/mydb     # mTLS only; a TLS-only database has none
    export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if the wallet is encrypted
    # TLS-only (no wallet): a connect string in ORACLE_DSN, or an alias + ORACLE_CONFIG_DIR

    python examples/notebook_37_oracle_store.py

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
)


TABLE_NAME = "locus_notebook_37_store"
USER_ID = "alice"


def _missing_env() -> list[str]:
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    # A tnsnames alias in ORACLE_DSN needs a directory to resolve it: the
    # mTLS wallet, or ORACLE_CONFIG_DIR / TNS_ADMIN for a TLS-only database.
    # A full connect string needs neither.
    dsn = os.environ.get("ORACLE_DSN", "")
    if (
        "(" not in dsn
        and "/" not in dsn
        and not any(
            os.environ.get(name) for name in ("ORACLE_WALLET", "ORACLE_CONFIG_DIR", "TNS_ADMIN")
        )
    ):
        missing.append("ORACLE_WALLET (mTLS) or ORACLE_CONFIG_DIR (TLS-only alias)")
    return missing


def _oracle_connection() -> dict[str, str]:
    """dsn / user / password, plus the TNS directory or wallet the environment names."""
    kwargs = {
        "dsn": os.environ["ORACLE_DSN"],
        "user": os.environ["ORACLE_USER"],
        "password": os.environ["ORACLE_PASSWORD"],
    }
    if os.environ.get("ORACLE_CONFIG_DIR"):
        kwargs["config_dir"] = os.path.expanduser(os.environ["ORACLE_CONFIG_DIR"])
    if os.environ.get("ORACLE_WALLET"):  # mTLS only; a TLS-only database has no wallet
        kwargs["wallet_location"] = os.path.expanduser(os.environ["ORACLE_WALLET"])
        kwargs["wallet_password"] = os.environ.get("ORACLE_WALLET_PASSWORD", "")
    return kwargs


def _print_skip_banner(missing: list[str]) -> None:
    print("\n--- Notebook 37: OracleStore-backed cross-thread memory agent ---")
    print(
        "Required environment variables not set; skipping the live demo so "
        "this file still runs cleanly in CI.\n"
    )
    print("Missing:")
    for name in missing:
        print(f"  - {name}")
    print(
        "\nProvision an Autonomous Database. With mTLS, drop its wallet under "
        "$ORACLE_WALLET; without it (TLS-only), put the connect string in "
        "ORACLE_DSN or an alias in a tnsnames.ora under $ORACLE_CONFIG_DIR. "
        "Then set the variables above and re-run."
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
        **_oracle_connection(),
        table_name=TABLE_NAME,
        # Key/value mode is enough here: the manager recalls through
        # store.search() — a LIKE substring match against the new user turn,
        # then a recency pass with query=None — so no vector embedding is needed.
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
    r"(?:i\s+(?:prefer|like|use|work\s+with|run\s+on)|my\s+(?:favourite|favorite)\s+\w+\s+is)"
    r"\s+([^\.\n]+)",
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
    """One turn on ``thread_id`` — ``arun`` is the awaitable form of
    ``run_sync`` for code that is already inside an event loop."""
    print(f"\n[{thread_id}] User : {prompt}")
    result = await agent.arun(prompt, thread_id=thread_id)
    print(f"[{thread_id}] Agent: {result.message.strip()}")


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

    Uses ``store.search`` — the same call the LLM memory manager makes at
    session start (it only falls back to ``list_keys``/``get`` when a
    backend cannot search). The ``StoreItem`` wrapper hands back the
    (namespace, key, value) triple in one call.
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

    At session start the memory manager recalls the stored facts from
    Oracle and inserts them as a system message right after the prompt.
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
            "RAG store (notebook 38), the checkpointer (notebook 42), and "
            "the versioned saver (notebook 43)."
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
