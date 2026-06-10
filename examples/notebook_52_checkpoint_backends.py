# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 52: Checkpoint backends on S3-compatible object storage.

The checkpointer contract is backend-agnostic. This notebook drives it
against ``S3Backend`` — S3 / MinIO / Cloudflare R2 via boto3 — a durable
store with the full capability set (list_threads, search, vacuum) over a
single bucket. Notebook 06 covers the checkpointer contract itself.
Portable SQL deployments can use PostgreSQL or MySQL through the same
adapter shape; key/value deployments can use Redis.

- Save and load AgentState via S3Backend.
- Inspect the reported capabilities.
- Walk thread history with list_threads / list_checkpoints.
- Vacuum old checkpoints with S3Backend.vacuum.
- Full-text search across stored conversations.

Run it
    # Requires an S3-compatible endpoint (e.g. MinIO) + a bucket:
    export S3_ENDPOINT_URL=http://localhost:9000   # MinIO / R2 endpoint
    export S3_BUCKET=tulip-checkpoints
    export AWS_ACCESS_KEY_ID=minioadmin
    export AWS_SECRET_ACCESS_KEY=minioadmin
    python examples/notebook_52_checkpoint_backends.py

Without the env vars the notebook prints what's missing and exits cleanly
so CI stays green. The in-memory checkpointer covered in notebook 06 is
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
    print("Notebook 52: Checkpoint backends on S3-compatible storage")
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

    # Part 1: round-trip an AgentState through the Checkpointer contract.
    print("\n=== Part 1: Save / load via S3Backend ===\n")
    cp = backend.as_checkpointer()

    state = AgentState(agent_id="demo_agent")
    state = state.with_message(Message.user("Hello from object storage!"))
    state = state.with_message(Message.assistant("Hi — your state lives in S3."))

    checkpoint_id = await cp.save(state, "thread_1")
    print(f"Saved checkpoint id={checkpoint_id} into bucket={os.environ['S3_BUCKET']}")

    loaded = await cp.load("thread_1")
    print(f"Loaded thread_1 with {len(loaded.messages)} messages")

    # Part 2: the capability descriptor — drives feature detection at
    # runtime so generic code can ask whether search or vacuum exist.
    print("\n=== Part 2: Reported capabilities ===\n")
    caps = cp.capabilities
    print(f"  list_threads:              {caps.list_threads}")
    print(f"  persistent_checkpoint_ids: {caps.persistent_checkpoint_ids}")
    print(f"  search:                    {caps.search}")
    print(f"  metadata_query:            {caps.metadata_query}")
    print(f"  vacuum:                    {caps.vacuum}")

    # Part 3: enumerate stored conversations and their checkpoint history.
    print("\n=== Part 3: Enumerate stored conversations ===\n")
    # Save a second thread so the listing has something to show.
    other = AgentState(agent_id="demo_agent")
    other = other.with_message(Message.user("Second thread"))
    await cp.save(other, "thread_2")

    threads = await cp.list_threads()
    print(f"Threads on this backend: {threads}")
    for tid in threads:
        cps = await cp.list_checkpoints(tid)
        print(f"  {tid}: {len(cps)} checkpoint(s)")

    # Part 4: vacuum old rows. Production deployments want a periodic
    # job that prunes checkpoints older than the retention window.
    print("\n=== Part 4: Vacuum old checkpoints ===\n")
    removed = await backend.vacuum(older_than_days=30)
    print(f"vacuum(older_than_days=30) removed {removed} stale object(s).")

    # Part 5: full-text search across every stored thread.
    print("\n=== Part 5: Search across checkpoints ===\n")
    hits = await backend.search("storage")
    print(f"search('storage') returned {len(hits)} thread id(s): {hits[:5]}")

    print("\nDone — every checkpoint above is durable in object storage.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
