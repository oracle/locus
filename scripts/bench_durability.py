#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Overhead of each durability mode, with and without the tool-effect ledger.

Runs a scripted agent (one tool call per iteration, ten iterations, a 2 KB
tool result, then an answer) under ``durability="sync"``, ``"async"`` and
``"exit"``, with ``tool_ledger`` off and on, and prints the median wall time
per iteration. Model and tool cost nothing, so what is measured is the
checkpoint and ledger traffic alone. The numbers in
``docs/concepts/checkpointers.md`` come from this script.

Always measures ``MemoryCheckpointer``. With ``POSTGRES_USER`` and
``POSTGRES_DB`` set (``POSTGRES_HOST`` / ``POSTGRES_PORT`` / ``POSTGRES_PASSWORD``
optional) it also measures ``postgresql_checkpointer()`` — the adapter, whose
save is several statements — and ``PgCheckpointer`` — one upsert per save —
on that database, in throwaway tables it drops afterwards::

    POSTGRES_USER=$(whoami) POSTGRES_DB=postgres python scripts/bench_durability.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import statistics
import time
import uuid
from collections.abc import Callable
from typing import Any

from locus.agent import Agent
from locus.core.messages import Message, ToolCall
from locus.memory.backends.memory import MemoryCheckpointer
from locus.models.base import ModelResponse
from locus.tools.decorator import tool


ITERATIONS = 10
REPEATS = 11


@tool
def step(n: int) -> str:
    """One step of work."""
    return f"step {n} done: " + "x" * 2000


class _Model:
    def __init__(self, n: int) -> None:
        usage = {"prompt_tokens": 100, "completion_tokens": 20}
        self.script: list[ModelResponse] = [
            ModelResponse(
                message=Message.assistant(
                    content=None,
                    tool_calls=[ToolCall(id=f"c{i}", name="step", arguments={"n": i})],
                ),
                usage=usage,
            )
            for i in range(n)
        ]
        self.script.append(ModelResponse(message=Message.assistant("done"), usage=usage))

    async def complete(self, messages: Any, tools: Any = None, **_: Any) -> ModelResponse:
        return self.script.pop(0)


async def one_run(checkpointer: Any, mode: str, ledger: bool) -> float:
    agent = Agent(
        model=_Model(ITERATIONS),
        tools=[step],
        checkpointer=checkpointer,
        max_iterations=ITERATIONS + 2,
        reflexion=False,
        grounding=False,
        durability=mode,  # type: ignore[arg-type]
        tool_ledger=ledger,
        context_window=1_000_000,
    )
    thread = "bench-" + uuid.uuid4().hex[:8]
    started = time.perf_counter()
    async for _ in agent.run("go", thread_id=thread):
        pass
    return (time.perf_counter() - started) * 1000 / ITERATIONS


async def bench(name: str, make: Callable[[], Any]) -> None:
    print(f"\n{name}: ms per iteration (median of {REPEATS} runs, {ITERATIONS} iterations each)")
    print(f"{'mode':<8}{'ledger off':>12}{'ledger on':>12}")
    for mode in ("sync", "async", "exit"):
        row = []
        for ledger in (False, True):
            samples = [await one_run(make(), mode, ledger) for _ in range(REPEATS)]
            row.append(statistics.median(samples))
        print(f"{mode:<8}{row[0]:>12.2f}{row[1]:>12.2f}")


async def main() -> None:
    logging.disable(logging.WARNING)
    await bench("MemoryCheckpointer", MemoryCheckpointer)

    user = os.getenv("POSTGRES_USER")
    database = os.getenv("POSTGRES_DB")
    if not (user and database):
        print("\nPOSTGRES_USER / POSTGRES_DB not set; skipping PostgreSQL")
        return
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    password = os.getenv("POSTGRES_PASSWORD", "")

    from locus.memory.backends import PgCheckpointer, PostgreSQLBackend, StorageBackendAdapter

    backend = PostgreSQLBackend(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        table_name="bench_durability_" + uuid.uuid4().hex[:8],
    )
    adapter = StorageBackendAdapter(backend)
    try:
        await bench(f"postgresql_checkpointer ({host})", lambda: adapter)
    finally:
        for thread in await backend.list_threads(limit=10_000):
            await backend.delete(thread)
        await backend.close()

    auth = f"{user}:{password}@" if password else f"{user}@"
    table = "bench_durability_" + uuid.uuid4().hex[:8]
    pg = PgCheckpointer(f"postgresql://{auth}{host}:{port}/{database}", table=table)
    try:
        await bench(f"PgCheckpointer ({host})", lambda: pg)
    finally:
        import asyncpg

        await pg.close()
        conn = await asyncpg.connect(
            host=host, port=port, user=user, password=password, database=database
        )
        try:
            await conn.execute(f"DROP TABLE IF EXISTS public.{table}")
        finally:
            await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
