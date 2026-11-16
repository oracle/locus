# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Store conformance suite against the free, in-process backends.

In-memory, Qdrant (``:memory:``), Chroma (ephemeral) and Milvus Lite
(file-backed, embedded) all run here with no server and no network. The
served backends run the same suite in
``tests/integration/rag/test_store_conformance_live.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from locus.rag.stores.memory import InMemoryVectorStore
from tests.unit.rag.store_conformance import DIMENSION, VectorStoreConformance


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


class TestInMemoryConformance(VectorStoreConformance):
    async def make_store(self) -> InMemoryVectorStore:
        return InMemoryVectorStore(dimension=DIMENSION)


@pytest.mark.skipif(not _installed("qdrant_client"), reason="qdrant-client not installed")
class TestQdrantConformance(VectorStoreConformance):
    async def make_store(self):  # type: ignore[no-untyped-def]
        from locus.rag.stores.qdrant import QdrantVectorStore

        return QdrantVectorStore(location=":memory:", dimension=DIMENSION, collection_name="conf")


@pytest.mark.skipif(not _installed("chromadb"), reason="chromadb not installed")
class TestChromaConformance(VectorStoreConformance):
    async def make_store(self):  # type: ignore[no-untyped-def]
        from locus.rag.stores.chroma import ChromaVectorStore

        return ChromaVectorStore(dimension=DIMENSION, collection_name="conf")


@pytest.mark.skipif(
    not (_installed("pymilvus") and _installed("milvus_lite")),
    reason="pymilvus with Milvus Lite not installed",
)
class TestMilvusLiteConformance(VectorStoreConformance):
    @pytest.fixture(autouse=True)
    def _db_path(self, tmp_path: Path) -> None:
        self._uri = str(tmp_path / "conformance.db")

    async def make_store(self):  # type: ignore[no-untyped-def]
        from locus.rag.stores.milvus import MilvusVectorStore

        return MilvusVectorStore(uri=self._uri, dimension=DIMENSION, collection_name="conf")


@pytest.mark.skipif(
    not (_installed("pymilvus") and _installed("milvus_lite")),
    reason="pymilvus with Milvus Lite not installed",
)
class TestMilvusLiteHybridConformance(VectorStoreConformance):
    supports_hybrid = True

    @pytest.fixture(autouse=True)
    def _db_path(self, tmp_path: Path) -> None:
        self._uri = str(tmp_path / "conformance_hybrid.db")

    async def make_store(self):  # type: ignore[no-untyped-def]
        from locus.rag.stores.milvus import MilvusVectorStore

        return MilvusVectorStore(
            uri=self._uri, dimension=DIMENSION, collection_name="confh", enable_hybrid=True
        )
