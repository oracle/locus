# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Store conformance suite against served backends, behind environment gates.

The suite is the unit suite's (``tests/unit/rag/store_conformance.py``);
each class here only knows how to reach one backend. Gates:

- ``WEAVIATE_EMBEDDED=1`` — embedded Weaviate (downloads the binary once;
  ``WEAVIATE_EMBEDDED_PORT`` / ``WEAVIATE_EMBEDDED_GRPC_PORT`` default to
  8099 / 50099). Or ``WEAVIATE_URL=http://host:8080`` for a running instance
  (``WEAVIATE_GRPC_PORT`` default 50051, ``WEAVIATE_API_KEY`` optional).
- ``REDIS_VECTOR_URL`` — a Redis with the Query Engine (Redis Stack / Redis 8
  bundle), e.g. ``redis://localhost:6399``.
- ``ELASTICSEARCH_URL`` — an Elasticsearch node (``ELASTICSEARCH_API_KEY``
  optional).
- ``MONGODB_ATLAS_URI`` — an Atlas cluster; the suite creates the vector index
  on first use and waits for it (``MONGODB_ATLAS_DB`` default ``locus_test``).
- ``PINECONE_API_KEY`` — a Pinecone project; the suite creates / reuses the
  ``locus-conformance`` serverless index (``PINECONE_CLOUD`` /
  ``PINECONE_REGION`` default aws / us-east-1).
- ``PGVECTOR_DSN`` or ``POSTGRES_DSN`` — PostgreSQL with pgvector.
- ``OPENSEARCH_HOSTS`` — an OpenSearch cluster with the k-NN plugin (see
  ``conftest.get_opensearch_config`` for the auth / TLS variables).
- ``MILVUS_URI`` — a Milvus server or Zilliz Cloud (``MILVUS_TOKEN`` optional).
- ``HEATWAVE_DSN`` (or ``LOCUS_MYSQL_DSN``) — a MySQL HeatWave DB system or a
  community MySQL 9+ server, e.g. ``mysql://user:pw@host:3306/db``; the suite
  reports which search path the server took.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from tests.unit.rag.store_conformance import DIMENSION, VectorStoreConformance


pytestmark = pytest.mark.integration


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


# ---------------------------------------------------------------- Weaviate

_WEAVIATE_CLIENT: Any = None


def _weaviate_client() -> Any:
    """One embedded / remote client for the whole module (startup is slow)."""
    global _WEAVIATE_CLIENT  # noqa: PLW0603 — module-level cache for the embedded process
    if _WEAVIATE_CLIENT is not None:
        return _WEAVIATE_CLIENT
    import weaviate

    if _env("WEAVIATE_URL"):
        from urllib.parse import urlparse

        url = urlparse(os.environ["WEAVIATE_URL"])
        auth = None
        if _env("WEAVIATE_API_KEY"):
            auth = weaviate.classes.init.Auth.api_key(os.environ["WEAVIATE_API_KEY"])
        _WEAVIATE_CLIENT = weaviate.connect_to_local(
            host=url.hostname or "localhost",
            port=url.port or 8080,
            grpc_port=int(os.environ.get("WEAVIATE_GRPC_PORT", "50051")),
            auth_credentials=auth,
        )
    else:
        _WEAVIATE_CLIENT = weaviate.connect_to_embedded(
            port=int(os.environ.get("WEAVIATE_EMBEDDED_PORT", "8099")),
            grpc_port=int(os.environ.get("WEAVIATE_EMBEDDED_GRPC_PORT", "50099")),
            persistence_data_path=os.environ.get("WEAVIATE_EMBEDDED_PATH"),
        )
    return _WEAVIATE_CLIENT


@pytest.mark.skipif(
    not (_env("WEAVIATE_EMBEDDED") or _env("WEAVIATE_URL")),
    reason="set WEAVIATE_EMBEDDED=1 or WEAVIATE_URL",
)
class TestWeaviateConformance(VectorStoreConformance):
    supports_hybrid = True

    async def make_store(self):  # type: ignore[no-untyped-def]
        pytest.importorskip("weaviate")
        from locus.rag.stores.weaviate import WeaviateVectorStore

        client = await asyncio.to_thread(_weaviate_client)
        return WeaviateVectorStore(
            dimension=DIMENSION, collection_name="LocusConformance", _client=client
        )


# ------------------------------------------------------------------- Redis


@pytest.mark.skipif(not _env("REDIS_VECTOR_URL"), reason="set REDIS_VECTOR_URL")
class TestRedisConformance(VectorStoreConformance):
    async def make_store(self):  # type: ignore[no-untyped-def]
        pytest.importorskip("redis")
        from locus.rag.stores.redis import RedisVectorStore

        return RedisVectorStore(
            dimension=DIMENSION,
            index_name="locus_conformance",
            url=os.environ["REDIS_VECTOR_URL"],
            metadata_fields=["topic"],
        )


# ----------------------------------------------------------- Elasticsearch


@pytest.mark.skipif(not _env("ELASTICSEARCH_URL"), reason="set ELASTICSEARCH_URL")
class TestElasticsearchConformance(VectorStoreConformance):
    supports_hybrid = True

    async def make_store(self):  # type: ignore[no-untyped-def]
        pytest.importorskip("elasticsearch")
        from locus.rag.stores.elasticsearch import ElasticsearchVectorStore

        return ElasticsearchVectorStore(
            dimension=DIMENSION,
            index_name="locus_conformance",
            hosts=[os.environ["ELASTICSEARCH_URL"]],
            api_key=_env("ELASTICSEARCH_API_KEY"),
            verify_certs=os.environ.get("ELASTICSEARCH_VERIFY_CERTS", "true").lower() == "true",
        )


# ----------------------------------------------------------- MongoDB Atlas


@pytest.mark.skipif(not _env("MONGODB_ATLAS_URI"), reason="set MONGODB_ATLAS_URI")
class TestMongoDBAtlasConformance(VectorStoreConformance):
    # Atlas search indexes are eventually consistent: give writes a moment.
    async def make_store(self):  # type: ignore[no-untyped-def]
        pytest.importorskip("pymongo")
        from locus.rag.stores.mongodb import MongoDBAtlasVectorStore

        store = MongoDBAtlasVectorStore(
            dimension=DIMENSION,
            connection_string=os.environ["MONGODB_ATLAS_URI"],
            database=os.environ.get("MONGODB_ATLAS_DB", "locus_test"),
            collection="conformance",
            filter_fields=["topic", "year"],
        )
        await store.ensure_index(wait=True)
        return _SettlingStore(store, delay=2.0)


# ---------------------------------------------------------------- Pinecone


@pytest.mark.skipif(not _env("PINECONE_API_KEY"), reason="set PINECONE_API_KEY")
class TestPineconeConformance(VectorStoreConformance):
    async def make_store(self):  # type: ignore[no-untyped-def]
        pytest.importorskip("pinecone")
        from locus.rag.stores.pinecone import PineconeVectorStore

        store = PineconeVectorStore(
            dimension=DIMENSION,
            index_name="locus-conformance",
            namespace="conformance",
            cloud=os.environ.get("PINECONE_CLOUD", "aws"),
            region=os.environ.get("PINECONE_REGION", "us-east-1"),
        )
        return _SettlingStore(store, delay=10.0)


# ---------------------------------------------------------------- pgvector


@pytest.mark.skipif(not _env("PGVECTOR_DSN", "POSTGRES_DSN"), reason="set PGVECTOR_DSN")
class TestPgVectorConformance(VectorStoreConformance):
    # ``metadata->>'key' = $1`` compares text, so numeric filters need a string.
    supports_numeric_filter = False

    async def make_store(self):  # type: ignore[no-untyped-def]
        pytest.importorskip("asyncpg")
        from locus.rag.stores.pgvector import PgVectorStore

        return PgVectorStore(
            dsn=_env("PGVECTOR_DSN", "POSTGRES_DSN"),
            table_name="locus_conformance",
            dimension=DIMENSION,
        )


# -------------------------------------------------------------- OpenSearch


@pytest.mark.skipif(not _env("OPENSEARCH_HOSTS"), reason="set OPENSEARCH_HOSTS")
class TestOpenSearchConformance(VectorStoreConformance):
    supports_hybrid = True

    async def make_store(self):  # type: ignore[no-untyped-def]
        pytest.importorskip("opensearchpy")
        from locus.rag.stores.opensearch import OpenSearchVectorStore
        from tests.integration.rag.conftest import get_opensearch_config

        cfg = get_opensearch_config()
        return OpenSearchVectorStore(
            hosts=cfg["hosts"],
            http_auth=cfg["http_auth"],
            use_ssl=cfg["use_ssl"],
            verify_certs=cfg["verify_certs"],
            index_name="locus_conformance",
            dimension=DIMENSION,
        )


# ------------------------------------------------------------ Milvus server


@pytest.mark.skipif(not _env("MILVUS_URI"), reason="set MILVUS_URI")
class TestMilvusServerConformance(VectorStoreConformance):
    supports_hybrid = True

    async def make_store(self):  # type: ignore[no-untyped-def]
        pytest.importorskip("pymilvus")
        from locus.rag.stores.milvus import MilvusVectorStore

        return MilvusVectorStore(
            dimension=DIMENSION,
            collection_name="locus_conformance",
            uri=os.environ["MILVUS_URI"],
            token=_env("MILVUS_TOKEN"),
            enable_hybrid=True,
        )


# ---------------------------------------------------------------- HeatWave


@pytest.mark.skipif(not _env("HEATWAVE_DSN", "LOCUS_MYSQL_DSN"), reason="set HEATWAVE_DSN")
class TestHeatWaveConformance(VectorStoreConformance):
    supports_hybrid = True

    async def make_store(self):  # type: ignore[no-untyped-def]
        pytest.importorskip("mysql.connector")
        from locus.rag.stores.heatwave import HeatWaveVectorStore

        store = HeatWaveVectorStore(
            dsn=_env("HEATWAVE_DSN", "LOCUS_MYSQL_DSN"),
            dimension=DIMENSION,
            table_name="locus_conformance",
        )
        await store.ensure_ready()
        print(f"HeatWaveVectorStore server_side_distance={store.server_side_distance}")
        return store


# ------------------------------------------------------------------ helper


class _SettlingStore:
    """Wraps a store whose reads lag its writes; sleeps after every write.

    Atlas Vector Search and Pinecone index asynchronously, so a search
    right after an upsert can miss the row. The wrapper keeps the
    conformance tests unchanged and the production adapters honest.
    """

    def __init__(self, inner: Any, *, delay: float) -> None:
        self._inner = inner
        self._delay = delay

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def _settle(self) -> None:
        await asyncio.sleep(self._delay)

    async def add(self, document: Any) -> str:
        result: str = await self._inner.add(document)
        await self._settle()
        return result

    async def add_batch(self, documents: list[Any]) -> list[str]:
        result: list[str] = await self._inner.add_batch(documents)
        if documents:
            await self._settle()
        return result

    async def delete(self, doc_id: str) -> bool:
        result: bool = await self._inner.delete(doc_id)
        await self._settle()
        return result

    async def clear(self) -> int:
        result: int = await self._inner.clear()
        await self._settle()
        return result
