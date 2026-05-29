# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 49: Checkpoint backends on Oracle Autonomous Database 26ai.

The checkpointer contract is backend-agnostic, but the production
default on OCI is Oracle 26ai — native JSON columns, vector and
text indexes in one schema, and the full capability set (list_threads,
search, vacuum) over a single durable store. Notebook 06 covers the
checkpointer contract itself; this notebook drives it against a real
ADB. Portable SQL deployments can use PostgreSQL or MySQL through the
same adapter shape; MySQL uses the official Connector/Python asyncio
driver.

- Save and load AgentState via oracle_checkpointer.
- Inspect the reported capabilities.
- Walk thread history with list_threads / list_checkpoints.
- Vacuum old checkpoints with OracleBackend.vacuum.
- Full-text search across stored conversations.

Run it
    # Requires a running Autonomous Database with its wallet on disk:
    export ORACLE_DSN=mydb_low                   # tnsnames alias
    export ORACLE_USER=locus_app                 # least-privileged
    export ORACLE_PASSWORD='<app-password>'
    export ORACLE_WALLET=~/.oci/wallets/mydb
    export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if encrypted
    python examples/notebook_54_checkpoint_backends.py

Without the env vars the notebook prints what's missing and exits cleanly
so CI stays green. The in-memory checkpointer covered in notebook 06 is
the developer default; Oracle 26ai is the production default.
"""

import asyncio
import os
import sys

from locus.core.messages import Message
from locus.core.state import AgentState
from locus.memory.backends import oracle_checkpointer
from locus.memory.backends.oracle import OracleBackend


_REQUIRED_ENV = (
    "ORACLE_DSN",
    "ORACLE_USER",
    "ORACLE_PASSWORD",
    "ORACLE_WALLET",
)


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


def _common_kwargs() -> dict:
    return {
        "dsn": os.environ["ORACLE_DSN"],
        "user": os.environ["ORACLE_USER"],
        "password": os.environ["ORACLE_PASSWORD"],
        "wallet_location": os.path.expanduser(os.environ["ORACLE_WALLET"]),
        "wallet_password": os.environ.get("ORACLE_WALLET_PASSWORD", ""),
    }


async def main() -> None:
    print("=" * 60)
    print("Notebook 49: Checkpoint backends on Oracle 26ai")
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
            "\nProvision an Autonomous Database, drop its wallet on disk, "
            "then set the variables above and re-run."
        )
        return

    table = "locus_notebook_54"

    # Part 1: round-trip an AgentState through the Checkpointer contract.
    print("\n=== Part 1: Save / load via oracle_checkpointer ===\n")
    cp = oracle_checkpointer(table_name=table, **_common_kwargs())

    state = AgentState(agent_id="demo_agent")
    state = state.with_message(Message.user("Hello from Oracle 26ai!"))
    state = state.with_message(Message.assistant("Hi — your state lives in ADB."))

    checkpoint_id = await cp.save(state, "thread_1")
    print(f"Saved checkpoint id={checkpoint_id} into table={table}")

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
    backend = OracleBackend(table_name=table, **_common_kwargs())
    removed = await backend.vacuum(older_than_days=30)
    print(f"vacuum(older_than_days=30) removed {removed} stale row(s).")

    # Part 5: full-text search across every stored thread.
    print("\n=== Part 5: Search across checkpoints ===\n")
    hits = await backend.search("Oracle")
    print(f"search('Oracle') returned {len(hits)} thread id(s): {hits[:5]}")

    print("\nDone — every checkpoint above is durable in Oracle 26ai.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
