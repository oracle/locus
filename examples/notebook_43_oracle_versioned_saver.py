# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 43: versioned agent history on Oracle Database 26ai.

Notebook 42's ``oracle_checkpointer`` keeps **one row per thread** —
``MERGE`` on save means the latest write wins and history is
destructive. That's the right shape for most agent threads. But when
you need a *history-preserving* saver — one row per
``(thread_id, checkpoint_ns, checkpoint_id)`` plus a sibling table for
intra-step pending writes — reach for
:class:`OracleCheckpointSaver`.

This notebook puts a Locus ``Agent`` on top of it. After every
turn the agent's :meth:`AgentState.to_checkpoint` snapshot is
written to the saver with a monotonic ``checkpoint_id`` and a parent
link, so the saver ends up with the agent's full audit trail. We then
demonstrate ``list_checkpoints`` (lineage), ``get(checkpoint_id=v1)``
(rewind read), and ``put_writes`` (intra-step durability) — every
operation driven by the agent's actual state, not synthetic blobs.

The same Autonomous Database wallet drives both this saver and the
single-row ``oracle_checkpointer`` side-by-side, so you can mix and
match per workload.

Key concepts:

- Two physical tables: ``<table>_checkpoints`` (one row per
  ``(thread_id, checkpoint_ns, checkpoint_id)`` with
  ``parent_checkpoint_id`` walking the lineage) and ``<table>_writes``
  (pending intra-step writes keyed by ``task_id`` + monotonic ``idx``).
- ``put(...)`` inserts; never upserts. Re-saving a different
  ``checkpoint_id`` for the same thread *appends* — that's the whole
  point of the versioned shape.
- ``get(thread_id=...)`` with no ``checkpoint_id`` returns the newest
  row; ``list_checkpoints(...)`` enumerates the lineage newest-first.
- ``put_writes(...)`` is idempotent: it replaces any prior writes for
  the same ``(thread, ns, checkpoint, task)`` tuple, so retries are
  safe.

Run it:
    export ORACLE_DSN=mydb_low                   # tnsnames alias
    export ORACLE_USER=locus_app                 # least-privileged app schema
    export ORACLE_PASSWORD='<app-password>'
    export ORACLE_WALLET=~/.oci/wallets/mydb
    export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if encrypted

    python examples/notebook_43_oracle_versioned_saver.py

If those env vars aren't set the notebook prints the wiring snippet and
exits cleanly — no traceback, no half-initialised state.

Difficulty: Intermediate. Self-contained — no prior notebook required.
"""

from __future__ import annotations

import asyncio
import os
import sys

from config import get_model, print_config

from locus.agent import Agent
from locus.core.state import AgentState
from locus.memory.backends import OracleCheckpointSaver, oracle_checkpointer


_REQUIRED_ENV = (
    "ORACLE_DSN",
    "ORACLE_USER",
    "ORACLE_PASSWORD",
    "ORACLE_WALLET",
)


TABLE_PREFIX = "locus_notebook_43"
THREAD_ID = "agent-incident-001"


_TURNS = (
    "Open the incident: checkout-api p99 is up 4× over the last 15 minutes.",
    "What service did we deploy at 13:58 and is anything correlated?",
    "Draft a one-sentence customer-facing status update.",
)


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


def _print_skip_banner(missing: list[str]) -> None:
    print("\n--- Notebook 43: OracleCheckpointSaver agent ---")
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
    from locus.memory.backends import OracleCheckpointSaver

    cp = oracle_checkpointer(dsn=..., table_name="locus_notebook_43_live")
    saver = OracleCheckpointSaver(dsn=..., user=..., wallet_location=...,
                                  table_name="locus_notebook_43")
    agent = Agent(model=model, system_prompt="...", checkpointer=cp,
                  checkpoint_every_n_iterations=1)

    parent = None
    for i, turn in enumerate(turns, start=1):
        result = agent.run_sync(turn, thread_id="t1")  # thread_id continues
        cid = f"v{i}"
        await saver.put(thread_id="t1", checkpoint_id=cid,
                        parent_checkpoint_id=parent,
                        checkpoint_data=result.state.to_checkpoint())
        parent = cid
        """.rstrip()
    )


def _build_saver() -> OracleCheckpointSaver:
    return OracleCheckpointSaver(
        dsn=os.environ["ORACLE_DSN"],
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        wallet_location=os.path.expanduser(os.environ["ORACLE_WALLET"]),
        wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", ""),
        table_name=TABLE_PREFIX,
    )


async def _drop_tables(saver: OracleCheckpointSaver) -> None:
    """Drop both demo tables so the notebook is re-runnable."""
    pool = await saver._get_pool()  # noqa: SLF001 — cleanup helper, not public API
    async with pool.acquire() as conn, conn.cursor() as cursor:
        for table in (f"{TABLE_PREFIX}_writes", f"{TABLE_PREFIX}_checkpoints"):
            try:
                await cursor.execute(f"DROP TABLE {table} PURGE")
            except Exception as exc:  # pragma: no cover — cleanup is best-effort
                print(f"  (cleanup) DROP TABLE {table} skipped: {exc}")
        await conn.commit()


async def run_agent_with_versioned_history(agent: Agent, saver: OracleCheckpointSaver) -> None:
    """Drive the agent through three turns, snapshotting each one.

    Each ``arun`` returns an :class:`AgentResult` whose ``.state``
    is the post-turn :class:`AgentState`. We serialise that with
    :meth:`AgentState.to_checkpoint` (the same call the in-process
    checkpointer uses) and write it to the saver under a fresh
    ``checkpoint_id`` linked to the previous one — building a lineage
    inside Oracle 26ai.
    """
    print("\n--- 1. Agent runs three turns; each snapshot lands in Oracle ---")

    parent: str | None = None
    for i, prompt in enumerate(_TURNS, start=1):
        # The agent's own checkpointer keeps the live thread going (turn N
        # sees turn N-1). ``arun`` is the awaitable form of ``run_sync`` for
        # code already inside an event loop; its result carries the
        # post-turn state we snapshot into the versioned saver.
        result = await agent.arun(prompt, thread_id=THREAD_ID)
        state = result.state
        final = result.message
        stop_reason = result.stop_reason

        cid = f"v{i}"
        snapshot = state.to_checkpoint()
        await saver.put(
            thread_id=THREAD_ID,
            checkpoint_id=cid,
            parent_checkpoint_id=parent,
            checkpoint_data=snapshot,
            metadata={
                "turn": i,
                "messages": len(state.messages),
                "stop_reason": str(stop_reason),
            },
        )
        print(f"\n  Turn {i} ({cid}, parent={parent!r}):")
        print(f"    User : {prompt}")
        print(f"    Agent: {final.strip()[:160]}")
        print(
            f"    saver.put -> checkpoint_id={cid!r}, "
            f"messages_in_snapshot={len(snapshot.get('messages', []))}"
        )
        parent = cid


async def list_lineage(saver: OracleCheckpointSaver) -> None:
    print("\n--- 2. list_checkpoints — the agent's full lineage ---")
    history = await saver.list_checkpoints(thread_id=THREAD_ID)
    print(f"  list_checkpoints({THREAD_ID!r}) -> {len(history)} rows (newest first):")
    for row in history:
        cid = row["checkpoint_id"]
        parent = row["parent_checkpoint_id"]
        meta = row.get("metadata") or {}
        print(
            f"    - id={cid!r}  parent={parent!r}  "
            f"turn={meta.get('turn')!r}  stop_reason={meta.get('stop_reason')!r}"
        )


async def rewind_to_v1(saver: OracleCheckpointSaver) -> None:
    """Pull a specific historical checkpoint back as an ``AgentState``."""
    print("\n--- 3. get(checkpoint_id='v1') — rewind read ---")
    row = await saver.get(thread_id=THREAD_ID, checkpoint_id="v1")
    if row is None:
        print("  no v1 row found (saver state inconsistent)")
        return
    rehydrated = AgentState.from_checkpoint(row["checkpoint"])
    print(f"  hydrated AgentState: {len(rehydrated.messages)} messages")
    for m in rehydrated.messages:
        role = getattr(m, "role", "?")
        text = (getattr(m, "content", "") or "").replace("\n", " ")[:80]
        print(f"    [{role}] {text}")
    print(
        "\n  This is enough to spin up an alternate-branch agent on a new "
        "thread by writing the snapshot back into its checkpointer. The "
        "lineage in 26ai is the source of truth — the live thread is "
        "just one realisation of it."
    )


async def write_pending(saver: OracleCheckpointSaver) -> None:
    """Demonstrate ``put_writes`` against the agent's latest checkpoint."""
    print("\n--- 4. put_writes — intra-step pending writes against v3 ---")
    await saver.put_writes(
        thread_id=THREAD_ID,
        checkpoint_id="v3",
        task_id="status-page-update",
        writes=[
            ("channel:status", "we are investigating"),
            ("channel:pagerduty", "warmed"),
        ],
    )
    writes = await saver.get_writes(thread_id=THREAD_ID, checkpoint_id="v3")
    print(f"  get_writes(thread={THREAD_ID!r}, checkpoint='v3') -> {len(writes)} rows:")
    for w in writes:
        print(
            f"    - task={w['task_id']!r} idx={w['idx']} "
            f"channel={w['channel']!r} value={w['value']!r}"
        )
    print(
        "\n  put_writes is idempotent: a retry for the same (thread, ns, "
        "checkpoint, task) replaces the prior set inside one transaction. "
        "Safe to call from a tool that may execute twice on retry."
    )


async def main() -> None:
    missing = _missing_env()
    if missing:
        _print_skip_banner(missing)
        return

    print_config()
    print("Opening pool against Oracle 26ai…")
    saver = _build_saver()

    # Live thread continuity comes from the single-row checkpointer
    # (notebook 42's primitive); the versioned saver adds an audit
    # trail on top. Both ride the same Autonomous Database wallet.
    live_cp = oracle_checkpointer(
        dsn=os.environ["ORACLE_DSN"],
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        wallet_location=os.path.expanduser(os.environ["ORACLE_WALLET"]),
        wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD", ""),
        table_name=f"{TABLE_PREFIX}_live",
    )

    agent = Agent(
        model=get_model(max_tokens=180),
        system_prompt=(
            "You are the on-call incident commander. Each turn, advance "
            "the incident: clarify what's known, identify likely causes, "
            "and draft the next public update. Stay concise — one short "
            "paragraph per turn."
        ),
        checkpointer=live_cp,
        checkpoint_every_n_iterations=1,
    )

    try:
        await run_agent_with_versioned_history(agent, saver)
        await list_lineage(saver)
        await rewind_to_v1(saver)
        await write_pending(saver)
        await saver.delete_thread(THREAD_ID)
        print(
            "\nThe agent ran live; Oracle 26ai kept the full lineage of "
            "its state. Use the single-row oracle_checkpointer (notebook "
            "42) for live durability and OracleCheckpointSaver for the "
            "auditable version history on top — same wallet, same pool."
        )
    finally:
        print("\n--- Cleanup: drop the demo tables ---")
        await _drop_tables(saver)
        # Also delete the live thread row from the single-row checkpointer.
        try:
            await live_cp.delete(THREAD_ID)
        except Exception as exc:  # pragma: no cover — best-effort
            print(f"  (cleanup) live_cp.delete skipped: {exc}")
        await saver.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
