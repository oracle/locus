#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 50: Durable support cases — checkpoint backends on object storage.

A real customer-support case runs for days and survives restarts,
shift handoffs, and redeploys. Locus persists that conversation state
in a durable store — the team's system of record for checkpointed case
state. The checkpointer contract is backend-agnostic; this notebook
first drives the time-travel part of it — listing a case's checkpoints,
forking the case at one of them, re-running a turn from one — on the
in-memory checkpointer, which needs no credentials, and then the storage
part against ``S3Backend`` — OCI Object Storage through its
S3-compatible API, or S3 / MinIO / Cloudflare R2, all via boto3 — a
durable store over a single bucket, so case state outlives any one
process. Object storage has no index, so it reports ``search=False``;
the notebook branches on the capability descriptor rather than assuming.
Notebook 05 covers the checkpointer contract itself. Portable SQL
deployments can use PostgreSQL or MySQL through the same adapter shape;
key/value deployments can use Redis.

- List a case's checkpoints and fork the case at an earlier one; re-run
  a turn from a past checkpoint; draw both branches (in memory, keyless).
- Save and load a support case's AgentState via S3Backend.
- Inspect the reported capabilities.
- Walk open cases with list_threads / list_checkpoints.
- Vacuum checkpoints past the conversation-retention window.
- Branch on the capability descriptor for search.

Run it:
    python examples/notebook_50_checkpoint_backends.py        # Part 0 only

    # OCI Object Storage (S3-compatible API) — the bucket lives in your
    # tenancy; the credentials are an OCI "Customer Secret Key" pair:
    export S3_ENDPOINT_URL=https://<namespace>.compat.objectstorage.<region>.oraclecloud.com
    export S3_BUCKET=locus-checkpoints
    export AWS_ACCESS_KEY_ID=<customer-secret-key-id>
    export AWS_SECRET_ACCESS_KEY=<customer-secret-key>
    python examples/notebook_50_checkpoint_backends.py

    # Any other S3-compatible endpoint works the same way, e.g. a local MinIO:
    export S3_ENDPOINT_URL=http://localhost:9000
    export AWS_ACCESS_KEY_ID=minioadmin
    export AWS_SECRET_ACCESS_KEY=minioadmin

Without the env vars the notebook runs Part 0, prints what's missing for
the rest and exits cleanly so CI stays green. The in-memory checkpointer
covered in notebook 05 is the developer default; S3 / object storage is
a production default.
"""

import asyncio
import os
import sys

from locus.agent import Agent
from locus.core.messages import Message, Role
from locus.core.state import AgentState
from locus.memory.backends import MemoryCheckpointer, S3Backend
from locus.multiagent.visualize import draw_mermaid_history
from locus.testing import ScriptedModel, text


_REQUIRED_ENV = (
    "S3_BUCKET",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


def _make_backend() -> S3Backend:
    return S3Backend(
        bucket=os.environ["S3_BUCKET"],
        endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
        prefix="locus_notebook_50/",
    )


def _user_texts(state: AgentState) -> list[str]:
    return [str(m.content) for m in state.messages if m.role == Role.USER]


async def time_travel_in_memory() -> None:
    """Part 0: the time-travel surface every checkpointer has, on the keyless one."""
    print("\n=== Part 0: List, fork and re-run a case (in memory, no credentials) ===\n")
    cp = MemoryCheckpointer()

    # Three days of a case, saved by hand so the checkpoint ids read well.
    # (An agent run mints its own ids; the same calls work on them.)
    day1 = AgentState(agent_id="support_agent").with_iteration(1)
    day1 = day1.with_message(Message.user("Day 1: customer can't log in after a password reset."))
    day1 = day1.with_message(Message.assistant("Sent a reset link; waiting for the customer."))
    day2 = day1.with_iteration(2).with_message(
        Message.user("Day 2: login works; asking for a refund.")
    )
    day2 = day2.with_message(Message.assistant("Refund policy checked: eligible."))
    day3 = day2.with_iteration(3).with_message(Message.user("Day 3: refund issued in full."))
    for checkpoint_id, state in (("day-1", day1), ("day-2", day2), ("day-3", day3)):
        await cp.save(state, "case_tkt_4021", checkpoint_id)

    # `list` describes the history newest first: the step each checkpoint
    # was written at and the checkpoint it descends from.
    print("case_tkt_4021 history (newest first):")
    for info in await cp.list("case_tkt_4021"):
        print(
            f"  {info.checkpoint_id:<6} iteration={info.iteration}  "
            f"parent={info.parent_checkpoint_id}"
        )

    # Fork at day 2: a what-if that never touches the original case.
    alt = await cp.fork("case_tkt_4021", "day-2", new_thread_id="case_tkt_4021-partial")
    forked_at = (await cp.list(alt))[0]
    alt_day3 = (await cp.load(alt)).with_iteration(3)
    alt_day3 = alt_day3.with_message(Message.user("Day 3: partial refund; escalated to billing."))
    await cp.save(alt_day3, alt, "alt-day-3")
    print(f"\nforked {forked_at.forked_from!r} at {forked_at.parent_checkpoint_id!r} -> {alt!r}")
    print("original still ends with:", _user_texts(await cp.load("case_tkt_4021"))[-1])
    print("fork ends with:          ", _user_texts(await cp.load(alt))[-1])

    # Both threads in one drawing: the fork is a branch of the checkpoint it
    # was taken at. The fork's first checkpoint has a generated id (a node id
    # and a shortened label); both are shown as `fork_root` so the printout
    # is stable.
    print("\nMermaid, both threads:")
    diagram = await draw_mermaid_history(cp, "case_tkt_4021", alt)
    generated = forked_at.checkpoint_id
    print(diagram.replace(generated, "fork_root").replace(generated[:12] + "…", "fork_root"))

    # The agent's own saves work the same way: re-run a turn from a past
    # checkpoint and the new turn descends from it, the old one stays.
    agent = Agent(
        model=ScriptedModel([text("Noted."), text("Refunded in full."), text("Refund held.")]),
        checkpointer=cp,
        reflexion=False,
        grounding=False,
    )
    agent.run_sync("Open ticket T-4821.", thread_id="case-4821")
    agent.run_sync("Refund the customer in full.", thread_id="case-4821")
    opened = (await cp.list("case-4821"))[-1]
    agent.run_sync("Hold the refund.", thread_id="case-4821", from_checkpoint=opened.checkpoint_id)
    history = await cp.list("case-4821")
    print("\ncase-4821 after re-running the second turn from the first checkpoint:")
    print(f"  checkpoints: {len(history)}")
    print(
        f"  newest descends from the first: {history[0].parent_checkpoint_id == opened.checkpoint_id}"
    )
    print(f"  the thread now reads: {_user_texts(await cp.load('case-4821'))}")


async def main() -> None:
    print("=" * 60)
    print("Notebook 50: Durable support-case state on S3-compatible storage")
    print("=" * 60)

    await time_travel_in_memory()

    missing = _missing_env()
    if missing:
        print(
            "\nRequired environment variables not set; skipping the live "
            "demo so this file still runs cleanly in CI.\n"
        )
        for name in missing:
            print(f"  - {name}")
        print(
            "\nCreate a bucket on OCI Object Storage (or any S3-compatible "
            "endpoint such as MinIO), then set the variables above and re-run."
        )
        return

    backend = _make_backend()

    # Part 1: round-trip a day-one support case through the
    # Checkpointer contract. Tomorrow's shift loads exactly this state.
    print("\n=== Part 1: Save / load via S3Backend ===\n")
    # S3Backend already implements BaseCheckpointer, so it is the checkpointer.
    # (RedisBackend and the SQL backends are storage only and need
    # StorageBackendAdapter — see notebook 65.)
    cp = backend

    state = AgentState(agent_id="support_agent")
    state = state.with_message(
        Message.user("Day 1: ticket T-7731 — customer can't log in after a password reset.")
    )
    state = state.with_message(
        Message.assistant("Logged. Sent reset link; awaiting customer reply, resuming tomorrow.")
    )

    checkpoint_id = await cp.save(state, "case_tkt_4021")
    print(f"Saved checkpoint id={checkpoint_id} into bucket={os.environ['S3_BUCKET']}")

    loaded = await cp.load("case_tkt_4021")
    print(f"Loaded case_tkt_4021 with {len(loaded.messages)} messages")

    # Day 2: the next shift resumes from exactly that state, appends, and
    # re-checkpoints. Multi-day continuity is just save → load → save.
    loaded = loaded.with_message(
        Message.user("Day 2: customer confirmed login works; now asking about a refund.")
    )
    await cp.save(loaded, "case_tkt_4021")
    print(f"Re-saved case_tkt_4021 after day-2 work ({len(loaded.messages)} messages)")

    # Part 2: the capability descriptor — drives feature detection at
    # runtime so generic code can ask whether search or vacuum exist.
    print("\n=== Part 2: Reported capabilities ===\n")
    caps = cp.capabilities
    print(f"  list_threads:              {caps.list_threads}")
    print(f"  persistent_checkpoint_ids: {caps.persistent_checkpoint_ids}")
    print(f"  search:                    {caps.search}")
    print(f"  metadata_query:            {caps.metadata_query}")
    print(f"  vacuum:                    {caps.vacuum}")

    # Part 3: enumerate open cases and their checkpoint history. `list`
    # works here exactly as it did in memory: one listing, one small
    # metadata object per checkpoint, never the state itself.
    print("\n=== Part 3: Enumerate open cases ===\n")
    # Save a second case so the listing has something to show.
    other = AgentState(agent_id="support_agent")
    other = other.with_message(Message.user("Day 1: new case — billing dispute on last invoice."))
    await cp.save(other, "case_tkt_4022")

    threads = await cp.list_threads()
    print(f"Cases on this backend: {threads}")
    for tid in threads:
        infos = await cp.list(tid)
        print(f"  {tid}: {len(infos)} checkpoint(s); newest at iteration {infos[0].iteration}")

    # Part 4: vacuum old rows. Conversation-retention policy in practice —
    # a periodic job prunes checkpoints older than the retention window.
    print("\n=== Part 4: Vacuum past the retention window ===\n")
    removed = await backend.vacuum(older_than_days=30)
    print(f"vacuum(older_than_days=30) removed {removed} stale object(s).")

    # Part 5: full-text search across every stored case — find which
    # support case mentioned a given topic. This is exactly what the
    # capability descriptor above is for: S3 reports search=False, so
    # generic code branches on it instead of calling and catching.
    print("\n=== Part 5: Search across support cases ===\n")
    if caps.search:
        hits = await backend.search("refund")
        print(f"search('refund') returned {len(hits)} case id(s): {hits[:5]}")
    else:
        print(
            "This backend reports search=False, so the call is skipped.\n"
            "Object storage has no index to query — use a backend that\n"
            "reports search=True (OpenSearch) when you need it."
        )

    print("\nDone — every checkpoint above survives restarts in object storage.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
