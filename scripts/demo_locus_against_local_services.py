#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end smoke: drive Locus against local Postgres + pgvector + Redis.

Spins up nothing itself — assumes the three services are already up
(see ``scripts/start_local_test_services.sh`` or:

    docker run -d --rm --name locus-test-pg -p 5432:5432 \\
        -e POSTGRES_PASSWORD=locus -e POSTGRES_USER=locus \\
        -e POSTGRES_DB=locus_test pgvector/pgvector:pg16
    docker run -d --rm --name locus-test-redis -p 6379:6379 redis:7-alpine

Then exercises each backend via the public Locus API:

1. ``PostgreSQLBackend``  — save / load / delete a checkpoint
2. ``PgVectorStore``      — add docs, vector search, delete
3. ``RedisBackend``       — same save/load round-trip

This is the kind of smoke test a release engineer runs before tagging.
"""

from __future__ import annotations

import asyncio
import sys

from locus.memory.backends.postgresql import PostgreSQLBackend
from locus.memory.backends.redis import RedisBackend
from locus.rag.stores.base import Document
from locus.rag.stores.pgvector import PgVectorStore


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


async def smoke_postgres_checkpoint() -> bool:
    banner("PostgreSQLBackend (checkpoint)")
    backend = PostgreSQLBackend(
        host="localhost",
        port=5432,
        database="locus_test",
        user="locus",
        password="locus",  # noqa: S106 — local test container
        table_name="locus_demo_checkpoints",
    )
    payload = {"messages": [{"role": "user", "content": "hi"}], "iteration": 1}
    try:
        await backend.save("demo-thread", payload)
        loaded = await backend.load("demo-thread")
        assert loaded is not None
        assert loaded["iteration"] == 1
        deleted = await backend.delete("demo-thread")
        print(f"  save/load/delete round-trip OK (delete returned {deleted})")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {e}")
        return False
    finally:
        if hasattr(backend, "close"):
            await backend.close()


async def smoke_pgvector() -> bool:
    banner("PgVectorStore (RAG)")
    store = PgVectorStore(
        host="localhost",
        port=5432,
        database="locus_test",
        user="locus",
        password="locus",  # noqa: S106
        table_name="locus_demo_vectors",
        dimension=4,  # tiny for the demo
    )
    docs = [
        Document(
            id="d1",
            content="Locus is an Oracle agent framework.",
            metadata={"topic": "framework"},
            embedding=[0.10, 0.20, 0.30, 0.40],
        ),
        Document(
            id="d2",
            content="Postgres pgvector adds vector similarity search.",
            metadata={"topic": "database"},
            embedding=[0.05, 0.30, 0.10, 0.50],
        ),
        Document(
            id="d3",
            content="Redis stores checkpoints with TTL.",
            metadata={"topic": "cache"},
            embedding=[0.90, 0.05, 0.05, 0.05],
        ),
    ]
    try:
        ids = await store.add_batch(docs)
        print(f"  inserted {len(ids)} documents")
        # query nearest to docs[0]'s embedding
        hits = await store.search(query_embedding=docs[0].embedding, limit=2)
        assert hits, "search returned no hits"
        print(f"  top match for d1's embedding: {hits[0].document.id} (score={hits[0].score:.4f})")
        assert hits[0].document.id == "d1"
        cleared = await store.clear()
        print(f"  cleared table ({cleared} rows)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {e}")
        return False


async def smoke_redis() -> bool:
    banner("RedisBackend (checkpoint)")
    backend = RedisBackend(url="redis://localhost:6379", prefix="locus:demo:")
    payload = {"messages": [{"role": "user", "content": "ping"}], "iteration": 7}
    try:
        await backend.save("demo-thread", payload)
        loaded = await backend.load("demo-thread")
        assert loaded is not None
        assert loaded["iteration"] == 7
        await backend.delete("demo-thread")
        print("  save/load/delete round-trip OK")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {e}")
        return False
    finally:
        if hasattr(backend, "close"):
            await backend.close()


async def main() -> int:
    results = await asyncio.gather(
        smoke_postgres_checkpoint(),
        smoke_pgvector(),
        smoke_redis(),
    )
    print()
    if all(results):
        print("ALL SMOKES PASSED ✓")
        return 0
    print(f"FAILED: {sum(1 for r in results if not r)}/{len(results)} smokes")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
