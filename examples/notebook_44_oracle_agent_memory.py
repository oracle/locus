# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 44: a Locus agent with Oracle's official agent-memory client.

The Oracle-native memory path. Wires Locus's
:class:`~locus.memory.managers.OracleAgentMemoryManager` to
``oracleagentmemory.core.OracleAgentMemory`` (Oracle's official
``oracleagentmemory`` PyPI package). Compared with the BaseStore-flavoured
:class:`~locus.memory.manager.LLMMemoryManager` from notebook 37, this
manager gets you:

- LLM-backed memory extraction benchmarked on LongMemEval (figures in
  the upstream README)
- Prompt-ready context cards with summaries + recent messages
- Scoped retrieval (``user_id`` / ``agent_id`` / ``thread_id``)
- Temporal reasoning + automatic memory expiry

The trade-off: it's Oracle-only and the upstream is alpha — check the
``oracleagentmemory`` README on PyPI for its current status. For non-Oracle backends
(InMemoryStore / Redis / Postgres / OpenSearch) or for test
environments, stick with notebook 37.

Same two-session shape as notebook 37 so you can compare side-by-side:

- Session A (``thread_id=THREAD_A``, ``user_id="alice"``) — agent learns
  a fact about Alice; the manager's ``on_session_end`` hook ships the
  transcript to ``Thread.add_messages_async``, which triggers
  oracleagentmemory's LLM extractor.
- Session B (``thread_id=THREAD_B``, **same** ``user_id="alice"``) — a
  brand-new Agent; on session start the manager fetches a context card
  for Alice via ``Thread.get_context_card_async`` and injects it into
  the system prompt. The agent answers the follow-up.

Prerequisites::

    # Optional dependency:
    pip install 'locus-sdk[agentmemory]'

    # Or the bundle that covers every Oracle backend:
    pip install 'locus-sdk[checkpoints]'

Run it:
    export ORACLE_DSN=mydb_low                   # tnsnames alias
    export ORACLE_USER=locus_app                 # least-privileged app schema
    export ORACLE_PASSWORD='<app-password>'
    export ORACLE_WALLET=~/.oci/wallets/mydb
    export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if encrypted

    # oracleagentmemory drives its Embedder + Llm through litellm. The demo
    # defaults to OCI Generative AI (``oci/cohere.embed-v4.0`` for
    # embeddings, ``oci/openai.gpt-4.1`` for extraction) and Locus hands
    # litellm a signer built from your OCI profile — API key or
    # ``oci session authenticate`` token — plus the region and compartment:
    export OCI_PROFILE=DEFAULT                    # or OCI_SESSION_PROFILE=...
    export OCI_REGION=eu-frankfurt-1
    export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..xxx

    # Any other litellm model spec works too, with that provider's own
    # credentials, e.g.:
    #   export LOCUS_OAM_EMBEDDER=openai/text-embedding-3-small
    #   export LOCUS_OAM_LLM=openai/gpt-4o-mini  OPENAI_API_KEY=sk-...

    python examples/notebook_44_oracle_agent_memory.py

If ``ORACLE_DSN`` / ``ORACLE_PASSWORD`` aren't set the notebook prints
the wiring snippet and exits cleanly — no traceback, no
half-initialised state. Same goes for ``oracleagentmemory`` not being
installed: the manager raises a clear ImportError pointing at the
extras command.

Difficulty: Intermediate. Self-contained — no prior notebook required.
"""

from __future__ import annotations

import asyncio
import os
import sys

from config import get_model, print_config

from locus.agent import Agent, AgentConfig


_REQUIRED_ENV = (
    "ORACLE_DSN",
    "ORACLE_USER",
    "ORACLE_PASSWORD",
    "ORACLE_WALLET",
)


THREAD_A = "notebook_44_thread_a"
THREAD_B = "notebook_44_thread_b"
USER_ID = "alice"
EMBEDDER = os.environ.get("LOCUS_OAM_EMBEDDER", "oci/cohere.embed-v4.0")
LLM = os.environ.get("LOCUS_OAM_LLM", "oci/openai.gpt-4.1")
TABLE_PREFIX = "locus_notebook_44_oam_"


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


def _print_skip_banner(missing: list[str]) -> None:
    print("\n--- Notebook 44: Oracle agent memory ---")
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
    print("\nMinimal wiring (what the live path below builds):\n")
    print(
        """    import oracledb
    from locus.agent import Agent, AgentConfig
    from locus.memory.managers import OracleAgentMemoryManager

    # A pool, not a single connection: oracleagentmemory extracts
    # memories on a background thread and needs its own session.
    conn = oracledb.create_pool(
        min=1, max=4,
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        dsn=os.environ["ORACLE_DSN"],
        config_dir=os.environ["ORACLE_WALLET"],
        wallet_location=os.environ["ORACLE_WALLET"],
        wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", ""),
    )
    manager = OracleAgentMemoryManager(
        connection=conn,
        embedder="oci/cohere.embed-v4.0",      # litellm spec; auth from your OCI profile
        llm="oci/openai.gpt-4.1",
        table_name_prefix="locus_oam_",
    )
    agent = Agent(config=AgentConfig(model=model, memory_manager=manager))
    agent.run_sync("Hi, I'm Alice — I prefer us-chicago-1.",
                   metadata={"user_id": "alice", "thread_id": "t-a"})
        """.rstrip()
    )


def _build_connection():
    """Open a sync oracledb *pool* — oracleagentmemory wants sync, and it
    runs memory extraction on a background thread. A single
    ``oracledb.Connection`` shared between that thread and the agent's
    own queries is not thread-safe (it crashed the thin driver in
    testing); a pool hands each caller its own session."""
    import oracledb

    return oracledb.create_pool(
        min=1,
        max=4,
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        dsn=os.environ["ORACLE_DSN"],
        config_dir=os.path.expanduser(os.environ["ORACLE_WALLET"]),
        wallet_location=os.path.expanduser(os.environ["ORACLE_WALLET"]),
        wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", ""),
    )


def _build_manager(conn):
    from locus.memory.managers import OracleAgentMemoryManager

    return OracleAgentMemoryManager(
        connection=conn,
        embedder=EMBEDDER,
        llm=LLM,
        extract_memories=True,
        table_name_prefix=TABLE_PREFIX,
        # First-run provisioning of the oracleagentmemory schema. For
        # production swap to "managed" and create the schema as a DBA
        # before deploying the runtime user.
        schema_policy="create_if_necessary",
        # Force the LLM extractor to run on EVERY add_messages call so
        # the demo's session B sees Alice's preference immediately. In
        # production leave these as upstream defaults (-1) to batch
        # extraction over a windowing schedule.
        thread_defaults={
            "memory_extraction_frequency": 1,
            "memory_extraction_window": 0,
            "enable_context_summary": True,
        },
    )


async def _drive_agent(agent: Agent, prompt: str, *, thread_id: str, user_id: str) -> str:
    """Drive a single agent turn, returning the final assistant message.

    ``agent.run`` is the async generator under ``run_sync``. We use it
    directly so the on_session_start/on_session_end hooks (which call
    oracleagentmemory's async APIs) share our event loop instead of
    spinning up a second one inside ``run_sync``.
    """
    final = ""
    async for event in agent.run(
        prompt,
        thread_id=thread_id,
        metadata={"thread_id": thread_id, "user_id": user_id},
    ):
        if event.event_type == "terminate":
            final = getattr(event, "final_message", "") or ""
    return final


def _build_agent(manager) -> Agent:
    """Fresh Agent per session — the only thing the two sessions share is ``manager``."""
    return Agent(
        config=AgentConfig(
            model=get_model(max_tokens=120),
            system_prompt=(
                "You are a helpful OCI infrastructure assistant. Be concise. "
                "Use any persistent facts you've been given about the user."
            ),
            memory_manager=manager,
            max_iterations=2,
        )
    )


async def session_a(manager) -> None:
    """Alice tells the agent her region preference; the manager extracts it."""
    print(f"\n--- Session A: Alice teaches her preference (thread {THREAD_A}) ---")
    agent = _build_agent(manager)

    prompt = (
        "Hi — I'm Alice. I work on OCI GenAI infrastructure and I prefer "
        "us-chicago-1 for inference. Note that for next time."
    )
    print(f"[{THREAD_A}] User : {prompt}")
    reply = await _drive_agent(agent, prompt, thread_id=THREAD_A, user_id=USER_ID)
    print(f"[{THREAD_A}] Agent: {reply.strip()}")


async def session_b(manager) -> None:
    """Brand-new Agent on a different thread — same user_id.

    The manager's ``on_session_start`` pulls a context card for Alice
    and injects it. Thread B has zero prior messages, so the only way
    the agent knows about her region preference is via that card.
    """
    print(f"\n--- Session B: brand-new Agent recalls Alice's preference (thread {THREAD_B}) ---")
    agent = _build_agent(manager)

    prompt = "Remind me which OCI region I prefer for inference, and what I work on."
    print(f"[{THREAD_B}] User : {prompt}")
    reply = await _drive_agent(agent, prompt, thread_id=THREAD_B, user_id=USER_ID)
    print(f"[{THREAD_B}] Agent: {reply.strip()}")


async def show_what_was_extracted(conn) -> None:
    """Peek into the underlying oracleagentmemory client to show what
    memories the LLM extractor produced for Alice between sessions.

    Pure diagnostic — uses the client directly, not via the Locus
    adapter, so the comparison with notebook 37 stays apples-to-apples.
    """
    from oracleagentmemory.core import MemoryExtractionConfig, OracleAgentMemory
    from oracleagentmemory.core.embedders import Embedder
    from oracleagentmemory.core.llms import Llm

    from locus.memory.managers.oracle_agent_memory import oci_litellm_auth

    # Same wrappers the Locus manager builds: for ``oci/...`` specs the
    # signer, region and compartment come from your OCI profile.
    client = OracleAgentMemory(
        connection=conn,
        embedder=Embedder(model=EMBEDDER, **oci_litellm_auth(EMBEDDER)),
        llm=Llm(model=LLM, **oci_litellm_auth(LLM)),
        memory_extraction_config=MemoryExtractionConfig(extract_memories=False),  # read-only probe
        table_name_prefix=TABLE_PREFIX,
    )
    # Use the async variant — calling the sync `search` from inside an
    # `async def` trips oracleagentmemory's "async-in-sync-from-async" guard.
    try:
        hits = await client.search_async("region preference", user_id=USER_ID, max_results=5)
    finally:
        await client.close_async()
    print(
        f"\n--- Inspect: client.search_async(user_id={USER_ID!r}) returned {len(hits)} record(s) ---"
    )
    for i, hit in enumerate(hits, start=1):
        snippet = (getattr(hit, "content", None) or str(hit))[:160].replace("\n", " ")
        record_type = getattr(hit, "record_type", "?")
        print(f"  #{i}  [{record_type}]  {snippet}")


async def main() -> None:
    missing = _missing_env()
    if missing:
        _print_skip_banner(missing)
        return

    print_config()
    print("\nOpening sync oracledb pool for oracleagentmemory…")
    conn = _build_connection()
    manager = None
    try:
        manager = _build_manager(conn)
        await session_a(manager)
        await show_what_was_extracted(conn)
        await session_b(manager)
        print(
            "\nThread B's Agent never saw Thread A's messages. The context card "
            "from oracleagentmemory carried Alice's preference across. Same wallet, "
            "same Autonomous Database as notebook 37 — the upgrade is the manager."
        )
    finally:
        # Drain the library's background extraction and close its client
        # *before* the pool goes away — otherwise the pool is still busy.
        if manager is not None:
            await manager.aclose()
        conn.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
