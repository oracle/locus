#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 42: a Locus agent that pulls rows from Autonomous Database.

The first link of any Oracle-native data flow is "get the source rows
out of the database". Locus ships
:class:`locus.rag.loaders.OracleADBLoader` for exactly this — an
async-pooled loader that turns an ADB ``SELECT`` into a stream of
:class:`locus.rag.stores.base.Document` objects. This notebook puts a
Locus ``Agent`` in the driver's seat: the loader is wrapped as a
``@tool``, the agent chooses which topic to pull, the ADB query
happens inside ``ToolStartEvent`` / ``ToolCompleteEvent``, and the
final answer is grounded in what came back.

Key concepts:

- The agent never sees ``OracleADBLoader`` directly. It sees
  ``fetch_articles_by_topic(topic)`` — a Locus ``@tool`` that wraps
  the loader and returns a deterministic list of rows.
- Each invocation is a fresh, parameter-bound ADB query —
  ``bind_params={"topic": ...}`` keeps user-controlled input out of
  the SQL text.
- ``await loader.load()`` is used here because the agent wants the
  whole result set in one turn. Switch to ``async for doc in
  loader.lazy_load()`` if you're streaming into a UI.
- The same connection envelope (DSN, user, wallet) feeds the loader,
  the vector store (notebook 06), the in-DB chunker (notebook 09),
  the in-DB embedder (notebook 10), and the checkpointer (07).

This notebook is self-contained: it creates a disposable demo table
``locus_notebook_08_articles``, populates a few rows, hands the loader
to an Agent as a tool, asks the agent two domain questions, and drops
the table on the way out.

Run it::

    export ORACLE_DSN=mydb_low                   # tnsnames alias in the wallet
    export ORACLE_USER=locus_app                 # least-privileged app schema
    export ORACLE_PASSWORD='<app-password>'
    export ORACLE_WALLET=~/.oci/wallets/mydb     # directory holding tnsnames.ora
    export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if the wallet is encrypted

    python examples/notebook_08_oracle_adb_loader.py

If ``ORACLE_DSN`` / ``ORACLE_PASSWORD`` / ``ORACLE_WALLET`` aren't set
the notebook prints the wiring snippet and exits cleanly — no
traceback, no half-initialised state.

Difficulty: Intermediate. Self-contained — no prior notebook required.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from config import get_model, print_config

from locus.agent import Agent
from locus.rag.loaders import OracleADBLoader
from locus.tools import tool


_REQUIRED_ENV = (
    "ORACLE_DSN",
    "ORACLE_USER",
    "ORACLE_PASSWORD",
    "ORACLE_WALLET",
)


_TABLE = "locus_notebook_08_articles"

_DEMO_ROWS = [
    (
        1,
        "oracle-26ai-vectors",
        "Oracle 26ai introduces a native VECTOR(N, FLOAT32) column type "
        "with HNSW and IVF organisations.",
        "AI Vector Search",
    ),
    (
        2,
        "oracle-26ai-similarity",
        "VECTOR_DISTANCE(col, :query, COSINE) returns the cosine distance "
        "between a stored vector and a query vector.",
        "AI Vector Search",
    ),
    (
        3,
        "adb-wallets",
        "Autonomous Database wallets bundle a tnsnames.ora alias per "
        "consumer group — _low, _medium, _high, _tp, _tpurgent.",
        "Autonomous Database",
    ),
    (
        4,
        "adb-consumer-groups",
        "_tp and _tpurgent are transaction-processing consumer groups; "
        "_low / _medium / _high target reporting workloads with different "
        "parallel-server limits.",
        "Autonomous Database",
    ),
]


_QUERIES = (
    "Summarise what we know about AI Vector Search from the content store.",
    "What does the content store say about Autonomous Database consumer groups?",
)


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


def _print_skip_banner(missing: list[str]) -> None:
    print("\n--- Notebook 08: OracleADBLoader agent ---")
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
    from locus.rag.loaders import OracleADBLoader
    from locus.tools import tool

    @tool
    async def fetch_articles_by_topic(topic: str) -> list[dict]:
        loader = OracleADBLoader(..., bind_params={"topic": topic}, ...)
        try:
            return [
                {"id": d.id, "body": d.content, "topic": d.metadata["topic"]}
                for d in await loader.load()
            ]
        finally:
            await loader.close()

    agent = Agent(model=model, tools=[fetch_articles_by_topic], ...)
    agent.run_sync("Summarise what we know about AI Vector Search.")
        """.rstrip()
    )


def _conn_kwargs() -> dict[str, str]:
    """Connection envelope shared by the helper DDL/DML and the loader."""
    return {
        "dsn": os.environ["ORACLE_DSN"],
        "user": os.environ["ORACLE_USER"],
        "password": os.environ["ORACLE_PASSWORD"],
        "wallet_location": os.path.expanduser(os.environ["ORACLE_WALLET"]),
        "wallet_password": os.environ.get("ORACLE_WALLET_PASSWORD", ""),
    }


async def _setup_demo_table() -> None:
    import oracledb

    kw = _conn_kwargs()
    params: dict[str, str] = {}
    if kw["wallet_location"]:
        params["config_dir"] = kw["wallet_location"]
        params["wallet_location"] = kw["wallet_location"]
        if kw["wallet_password"]:
            params["wallet_password"] = kw["wallet_password"]

    conn = await oracledb.connect_async(
        user=kw["user"],
        password=kw["password"],
        dsn=kw["dsn"],
        **params,
    )
    try:
        cursor = conn.cursor()
        try:
            await cursor.execute(f"DROP TABLE {_TABLE}")
        except Exception:  # noqa: BLE001 - table may not exist; fine.
            pass
        await cursor.execute(
            f"""
            CREATE TABLE {_TABLE} (
                id      NUMBER PRIMARY KEY,
                slug    VARCHAR2(64) NOT NULL,
                body    CLOB NOT NULL,
                topic   VARCHAR2(64) NOT NULL
            )
            """
        )
        await cursor.executemany(
            f"INSERT INTO {_TABLE} (id, slug, body, topic) VALUES (:1, :2, :3, :4)",
            _DEMO_ROWS,
        )
        await conn.commit()
    finally:
        await conn.close()


async def _drop_demo_table() -> None:
    import oracledb

    kw = _conn_kwargs()
    params: dict[str, str] = {}
    if kw["wallet_location"]:
        params["config_dir"] = kw["wallet_location"]
        params["wallet_location"] = kw["wallet_location"]
        if kw["wallet_password"]:
            params["wallet_password"] = kw["wallet_password"]

    conn = await oracledb.connect_async(
        user=kw["user"],
        password=kw["password"],
        dsn=kw["dsn"],
        **params,
    )
    try:
        cursor = conn.cursor()
        try:
            await cursor.execute(f"DROP TABLE {_TABLE}")
        except Exception:  # noqa: BLE001 - already gone; fine.
            pass
        await conn.commit()
    finally:
        await conn.close()


async def _drive_agent(agent: Agent, prompt: str) -> None:
    print(f"\n--- User: {prompt} ---")
    final = ""
    async for event in agent.run(prompt):
        et = event.event_type
        if et == "tool_start":
            args = getattr(event, "arguments", None) or {}
            topic = args.get("topic") if isinstance(args, dict) else None
            print(f"  [agent → ADB] tool_start: {event.tool_name}(topic={topic!r})")
        elif et == "tool_complete":
            raw = getattr(event, "result", "") or ""
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = []
            rows = payload if isinstance(payload, list) else []
            print(f"  [agent ← ADB] tool_complete: {len(rows)} row(s)")
            for r in rows:
                rid = r.get("id") if isinstance(r, dict) else None
                body = (r.get("body") if isinstance(r, dict) else "") or ""
                print(f"      id={rid!r}  {body[:80]}")
        elif et == "terminate":
            final = getattr(event, "final_message", "") or ""
    if final:
        print(f"\nAgent: {final.strip()}")


async def main() -> None:
    missing = _missing_env()
    if missing:
        _print_skip_banner(missing)
        return

    print_config()

    print(f"\nCreating disposable demo table {_TABLE}…")
    await _setup_demo_table()

    @tool
    async def fetch_articles_by_topic(topic: str) -> list[dict]:
        """Fetch every article tagged with the given topic from the
        content store.

        Args:
            topic: Exact topic label. Known values include 'AI Vector
                Search' and 'Autonomous Database'.

        Returns:
            A list of articles, each with id (slug), body (text), and
            topic. Empty list if no rows match.
        """
        # Fresh loader per call so the SELECT is parameter-bound — the
        # ``topic`` arg goes through ``bind_params``, not string
        # interpolation, even though the agent picked it.
        loader = OracleADBLoader(
            **_conn_kwargs(),
            sql=(f"SELECT id, slug, body, topic FROM {_TABLE} WHERE topic = :topic ORDER BY id"),
            bind_params={"topic": topic},
            content_column="body",
            id_column="slug",
            metadata_columns=["topic"],
        )
        try:
            docs = await loader.load()
            return [
                {
                    "id": d.id,
                    "body": d.content,
                    "topic": d.metadata.get("topic"),
                }
                for d in docs
            ]
        finally:
            await loader.close()

    agent = Agent(
        model=get_model(max_tokens=300),
        tools=[fetch_articles_by_topic],
        system_prompt=(
            "You are a docs analyst. The content store lives in Oracle "
            "Autonomous Database — use fetch_articles_by_topic with the "
            "exact topic label to retrieve rows before you answer. Cite "
            "the article slug for each claim. Do not invent rows."
        ),
        max_iterations=4,
    )

    try:
        for query in _QUERIES:
            await _drive_agent(agent, query)

        print(
            "\nThe agent drove every ADB round-trip itself — picking the "
            "topic, running the parameter-bound SELECT, grounding the "
            "answer in the returned rows. Locus + Oracle 26ai, no "
            "langchain glue."
        )
    finally:
        print(f"\nDropping demo table {_TABLE}…")
        await _drop_demo_table()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
