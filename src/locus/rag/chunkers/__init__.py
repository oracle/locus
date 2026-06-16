# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Text chunkers — splitters that produce chunks for embedding.

Locus ships a small client-side chunker via :class:`ChunkConfig` on
``RAGRetriever``. This package adds **server-side** chunkers that run
their tokenisation and segmentation *inside* the database, sidestepping
the round-trip cost for large ingest jobs.

Currently:

* :class:`OracleInDBChunker` — wrapper around
  ``DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS`` (Oracle 23ai/26ai). Native locus,
  no langchain dep.
"""

from locus.rag.chunkers.oracle_indb import OracleInDBChunker
from locus.rag.chunkers.oracle_sync import OracleSyncInDBChunker


__all__ = ["OracleInDBChunker", "OracleSyncInDBChunker"]
