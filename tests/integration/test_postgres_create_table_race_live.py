# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Two adapters saving to a brand-new table at the same instant both succeed (needs PostgreSQL).

Gated on ``POSTGRES_HOST`` (plus the usual ``POSTGRES_PORT`` / ``POSTGRES_DB`` / ``POSTGRES_USER`` /
``POSTGRES_PASSWORD``), as the other live PostgreSQL tests are.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from locus.core.state import AgentState
from locus.memory.backends.adapters import postgresql_checkpointer


pytestmark = pytest.mark.skipif(not os.getenv("POSTGRES_HOST"), reason="POSTGRES_HOST not set")


def _kwargs(table: str) -> dict[str, object]:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "database": os.getenv("POSTGRES_DB", "locus_test"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
        "table_name": table,
    }


@pytest.mark.asyncio
async def test_concurrent_first_saves_on_a_fresh_table() -> None:
    table = "race_" + uuid.uuid4().hex[:10]
    a = postgresql_checkpointer(**_kwargs(table))
    b = postgresql_checkpointer(**_kwargs(table))
    thread = "t-" + uuid.uuid4().hex[:6]
    try:
        await asyncio.gather(
            a.save(AgentState(messages=[]), thread_id=thread),
            b.save(AgentState(messages=[]), thread_id=thread),
        )
        assert await a.load(thread) is not None
    finally:
        pool = await a._backend._get_pool()  # type: ignore[attr-defined]
        async with pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS public.{table}")
        await a.close()
        await b.close()
