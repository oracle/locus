# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Document loaders for RAG ingestion.

Each loader yields :class:`locus.rag.stores.base.Document` instances from
some external source (SQL database, filesystem, HTTP, etc.) so the
downstream chunker / embedder / vector store pipeline can stay agnostic
of where the raw rows came from.
"""

from locus.rag.loaders.oracle import OracleADBLoader
from locus.rag.loaders.oracle_sync import OracleSyncADBLoader


__all__ = ["OracleADBLoader", "OracleSyncADBLoader"]
