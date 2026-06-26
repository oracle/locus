# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 52: Durable support cases — checkpoint backends on object storage.

A real customer-support case runs for days and survives restarts,
shift handoffs, and redeploys. Tulip persists that conversation state
in a durable store — the team's system of record for checkpointed case
state. The checkpointer contract is backend-agnostic; this notebook
drives it against ``S3Backend`` — S3 / MinIO / Cloudflare R2 via boto3 —
a durable store with the full capability set (list_threads, search,
vacuum) over a single bucket, so case state outlives any one process.
Notebook 08 covers the checkpointer contract itself. Portable SQL
deployments can use PostgreSQL or MySQL through the same adapter shape;
key/value deployments can use Redis.

- Save and load a support case's AgentState via S3Backend.
- Inspect the reported capabilities.
- Walk open cases with list_threads / list_checkpoints.
- Vacuum checkpoints past the conversation-retention window.
- Full-text search across stored support cases.

Run it
    # Requires an S3-compatible endpoint (e.g. MinIO) + a bucket:
    export S3_ENDPOINT_URL=http://localhost:9000   # MinIO / R2 endpoint
    export S3_BUCKET=tulip-checkpoints
    export AWS_ACCESS_KEY_ID=minioadmin
    export AWS_SECRET_ACCESS_KEY=minioadmin
    python examples/notebook_52_checkpoint_backends.py

Without the env vars the notebook prints what's missing and exits cleanly
so CI stays green. The in-memory checkpointer covered in notebook 08 is
the developer default; S3 / object storage is a production default.
"""

import asyncio
import os
import sys

from tulip.core.messages import Message
from tulip.core.state import AgentState
from tulip.memory.backends import S3Backend


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
        prefix="tulip_notebook_52/",
    )


async def main() -> None:
    print("=" * 60)
    print("Notebook 52: Durable support-case state on S3-compatible storage")
    print("=" * 60)

    missing = _missing_env()
    if missing:
        print(
            "\nRequired environment variables not set; skipping the live "
            "demo so this file still runs cleanly in CI.\n"
        )
        for name in missing:
            print(f"  - {name}")
        print(
            "\nStand up an S3-compatible endpoint (MinIO / R2), create a "
            "bucket, then set the variables above and re-run."
        )
        return

    backend = _make_backend()

    # Part 1: round-trip a day-one support case through the
    # Checkpointer contract. Tomorrow's shift loads exactly this state.
    print("\n=== Part 1: Save / load via S3Backend ===\n")
    cp = backend.as_checkpointer()

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

    # Part 3: enumerate open cases and their checkpoint history.
    print("\n=== Part 3: Enumerate open cases ===\n")
    # Save a second case so the listing has something to show.
    other = AgentState(agent_id="support_agent")
    other = other.with_message(Message.user("Day 1: new case — billing dispute on last invoice."))
    await cp.save(other, "case_tkt_4022")

    threads = await cp.list_threads()
    print(f"Cases on this backend: {threads}")
    for tid in threads:
        cps = await cp.list_checkpoints(tid)
        print(f"  {tid}: {len(cps)} checkpoint(s)")

    # Part 4: vacuum old rows. Conversation-retention policy in practice —
    # a periodic job prunes checkpoints older than the retention window.
    print("\n=== Part 4: Vacuum past the retention window ===\n")
    removed = await backend.vacuum(older_than_days=30)
    print(f"vacuum(older_than_days=30) removed {removed} stale object(s).")

    # Part 5: full-text search across every stored case — find which
    # support case mentioned a given topic.
    print("\n=== Part 5: Search across support cases ===\n")
    hits = await backend.search("refund")
    print(f"search('refund') returned {len(hits)} case id(s): {hits[:5]}")

    print("\nDone — every checkpoint above survives restarts in object storage.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
