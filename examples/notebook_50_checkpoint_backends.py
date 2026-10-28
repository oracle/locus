# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 50: Durable support cases — checkpoint backends on object storage.

A real customer-support case runs for days and survives restarts,
shift handoffs, and redeploys. Locus persists that conversation state
in a durable store — the team's system of record for checkpointed case
state. The checkpointer contract is backend-agnostic; this notebook
drives it against ``S3Backend`` — OCI Object Storage through its
S3-compatible API, or S3 / MinIO / Cloudflare R2, all via boto3 — a
durable store over a single bucket, so case state outlives any one
process. Object storage has no index, so it reports ``search=False``;
the notebook branches on the capability descriptor rather than assuming.
Notebook 05 covers the checkpointer contract itself. Portable SQL
deployments can use PostgreSQL or MySQL through the same adapter shape;
key/value deployments can use Redis.

- Save and load a support case's AgentState via S3Backend.
- Inspect the reported capabilities.
- Walk open cases with list_threads / list_checkpoints.
- Vacuum checkpoints past the conversation-retention window.
- Branch on the capability descriptor for search.

Run it:
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

Without the env vars the notebook prints what's missing and exits cleanly
so CI stays green. The in-memory checkpointer covered in notebook 05 is
the developer default; S3 / object storage is a production default.
"""

import asyncio
import os
import sys

from locus.core.messages import Message
from locus.core.state import AgentState
from locus.memory.backends import S3Backend


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


async def main() -> None:
    print("=" * 60)
    print("Notebook 50: Durable support-case state on S3-compatible storage")
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
