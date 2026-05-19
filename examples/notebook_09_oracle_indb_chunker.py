#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 09: a Locus agent driving an in-database chunker.

Oracle 23ai / 26ai ships a server-side chunking primitive,
``DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS``. It tokenises and segments text
without ever shipping the content back to Python — perfect for the
first stage of an ingest pipeline where the source CLOBs already live
in the database. Locus wraps it as
:class:`locus.rag.chunkers.OracleInDBChunker`, but the protagonist of
this notebook is the *Locus agent* that decides when and how to use it.

The agent gets two tools:

- ``chunk_paragraph(text)`` — wraps ``chunker.chunk_text``. Returns the
  chunk list for one Python string (e.g. a passage the user pasted).
- ``chunk_table_rows(limit)`` — wraps the streaming
  ``chunker.chunk_column`` over a demo table; returns per-row chunk
  counts. This is how an ingest agent inspects a corpus that already
  lives in ADB without pulling every CLOB back over the wire.

The agent picks which tool to call from the user's prompt, sees the
``ToolStartEvent`` / ``ToolCompleteEvent`` pair, and grounds its
summary in the chunk stats the database returned.

Prerequisite (one-time, out-of-band)::

    GRANT EXECUTE ON DBMS_VECTOR_CHAIN TO locus_app;

Run it::

    export ORACLE_DSN=mydb_low                   # tnsnames alias in the wallet
    export ORACLE_USER=locus_app                 # least-privileged app schema
    export ORACLE_PASSWORD='<app-password>'
    export ORACLE_WALLET=~/.oci/wallets/mydb     # directory holding tnsnames.ora
    export ORACLE_WALLET_PASSWORD='<wallet-pw>'  # if the wallet is encrypted

    python examples/notebook_09_oracle_indb_chunker.py

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
from collections import Counter

from config import get_model, print_config

from locus.agent import Agent
from locus.rag.chunkers import OracleInDBChunker
from locus.tools import tool


_REQUIRED_ENV = (
    "ORACLE_DSN",
    "ORACLE_USER",
    "ORACLE_PASSWORD",
    "ORACLE_WALLET",
)


_TABLE = "locus_notebook_09_docs"


_LONG_PARAGRAPH = (
    "Oracle Database 26ai introduces native vector search through the "
    "VECTOR column type and the VECTOR_DISTANCE SQL function. Vectors "
    "live in a real column with a configurable dimension and storage "
    "format such as FLOAT32 or INT8. The CREATE VECTOR INDEX statement "
    "builds an HNSW or IVF index asynchronously after the first INSERT, "
    "and the database keeps it incrementally updated as new rows arrive. "
    "DBMS_VECTOR_CHAIN exposes server-side primitives for chunking, "
    "embedding, and reranking so the whole pipeline can run inside the "
    "database when data residency rules require it."
)


_QUERIES = (
    f"I have this passage to ingest — chunk it for me and report stats:\n{_LONG_PARAGRAPH}",
    f"How many chunks would I get if I ingested every row in the {_TABLE} table right now?",
)


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


def _print_skip_banner(missing: list[str]) -> None:
    print("\n--- Notebook 09: OracleInDBChunker agent ---")
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
    print("\nPrerequisite (one-time, out-of-band):")
    print("    GRANT EXECUTE ON DBMS_VECTOR_CHAIN TO locus_app;")
    print("\nMinimal wiring (what the live path below builds):")
    print(
        """
    from locus.agent import Agent
    from locus.rag.chunkers import OracleInDBChunker
    from locus.tools import tool

    chunker = OracleInDBChunker(..., max_tokens=20)

    @tool
    async def chunk_paragraph(text: str) -> dict:
        chunks = await chunker.chunk_text(text)
        return {"count": len(chunks), "preview": chunks[:3]}

    agent = Agent(model=model, tools=[chunk_paragraph], ...)
    agent.run_sync("Chunk this passage for me: ...")
        """.rstrip()
    )


def _conn_kwargs() -> dict[str, str]:
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
        user=kw["user"], password=kw["password"], dsn=kw["dsn"], **params
    )
    try:
        cursor = conn.cursor()
        try:
            await cursor.execute(f"DROP TABLE {_TABLE}")
        except Exception:  # noqa: BLE001 - may not exist; fine.
            pass
        await cursor.execute(
            f"""
            CREATE TABLE {_TABLE} (
                id   NUMBER PRIMARY KEY,
                body CLOB NOT NULL
            )
            """
        )
        await cursor.execute(
            f"INSERT INTO {_TABLE} (id, body) VALUES (:1, :2)",
            (1, _LONG_PARAGRAPH),
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
        user=kw["user"], password=kw["password"], dsn=kw["dsn"], **params
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
    one_line = prompt.replace("\n", " ")
    head = one_line if len(one_line) < 120 else one_line[:117] + "..."
    print(f"\n--- User: {head} ---")
    final = ""
    async for event in agent.run(prompt):
        et = event.event_type
        if et == "tool_start":
            args = getattr(event, "arguments", None) or {}
            arg_preview = {
                k: (v if not isinstance(v, str) or len(v) < 40 else v[:37] + "...")
                for k, v in (args.items() if isinstance(args, dict) else [])
            }
            print(f"  [agent → 26ai] tool_start: {event.tool_name}({arg_preview})")
        elif et == "tool_complete":
            raw = getattr(event, "result", "") or ""
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                summary = {
                    k: (
                        v
                        if not isinstance(v, (list, dict))
                        else f"<{type(v).__name__} len={len(v)}>"
                    )
                    for k, v in payload.items()
                }
                print(f"  [agent ← 26ai] tool_complete: {summary}")
            else:
                snippet = raw if len(raw) < 200 else raw[:197] + "..."
                print(f"  [agent ← 26ai] tool_complete: {snippet!r}")
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

    chunker = OracleInDBChunker(
        **_conn_kwargs(),
        max_tokens=20,
        overlap=0,
        by="words",
    )
    await _setup_demo_table()

    @tool
    async def chunk_paragraph(text: str) -> dict:
        """Tokenise and segment a single passage inside the database.

        Wraps DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS. The text is bound to a
        PL/SQL local and chunked server-side at the chunker's configured
        max_tokens / overlap. Returns the chunk count, the total char
        length, and a preview of the first chunk.

        Args:
            text: The passage to chunk. Pass the user's full string.

        Returns:
            {"count": int, "total_chars": int, "first": str}
        """
        chunks = await chunker.chunk_text(text)
        first = (chunks[0]["text"] if chunks else "") or ""
        return {
            "count": len(chunks),
            "total_chars": len(text),
            "first": first[:160],
        }

    @tool
    async def chunk_table_rows(limit: int = 100) -> dict:
        """Stream chunks from every row in the demo CLOB table.

        Wraps OracleInDBChunker.chunk_column over the
        ``locus_notebook_09_docs`` table. The source CLOBs never leave
        the database — only the chunk records come back. Useful for
        an ingest agent that wants to know "how big is this corpus, in
        chunks" before committing to write to a vector store.

        Args:
            limit: Cap on rows to inspect (default 100). Set higher for
                bigger demos.

        Returns:
            {"rows": int, "chunks": int, "per_row": {source_id: count}}
        """
        per_row: Counter[str] = Counter()
        seen = 0
        async for chunk in chunker.chunk_column(
            table_name=_TABLE,
            text_column="body",
            id_column="id",
        ):
            seen += 1
            per_row[str(chunk["source_id"])] += 1
            if seen >= limit * 50:  # safety stop
                break
        return {
            "rows": len(per_row),
            "chunks": seen,
            "per_row": dict(per_row),
        }

    agent = Agent(
        model=get_model(max_tokens=400),
        tools=[chunk_paragraph, chunk_table_rows],
        system_prompt=(
            "You are an ingest planning agent. When the user hands you a "
            "passage, call chunk_paragraph to size it. When they ask about "
            "a table corpus, call chunk_table_rows. Always report the "
            "chunk count and the first chunk's preview so they know what "
            "the database produced. Do not invent numbers."
        ),
        max_iterations=4,
    )

    try:
        for query in _QUERIES:
            await _drive_agent(agent, query)

        print(
            "\nThe agent picked the right primitive for each request — "
            "single-string vs. column-streaming — and every chunk was "
            "produced by UTL_TO_CHUNKS inside Oracle 26ai. Pair with "
            "OracleInDBEmbeddings (notebook 10) for a fully in-database "
            "ingest pipeline."
        )
    finally:
        await chunker.close()
        await _drop_demo_table()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
