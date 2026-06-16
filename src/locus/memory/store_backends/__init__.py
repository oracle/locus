# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Store backends for Locus long-term memory.

A :class:`~locus.memory.store.BaseStore` is the cross-thread persistent
key/value store used for long-term memory (see ``store.py``). Where the
``backends/`` package holds **checkpointer** drivers (per-thread agent
state), this package holds **store** drivers — namespace-keyed value
storage that survives across threads, optionally with vector search.

Available backends:

- :class:`OracleStore` — Oracle Database 26ai (native VECTOR + JSON).
  The locus-native equivalent of ``langgraph-oracledb.OracleStore`` /
  ``AsyncOracleStore``, but with **zero** langchain/langgraph imports.

Usage::

    from locus.memory.store_backends import OracleStore

    store = OracleStore(
        dsn="mydb_high",
        user="locus_app",
        password=os.environ["LOCUS_DB_PASSWORD"],
        wallet_location="~/.oci/wallets/mydb",
        dimension=1024,  # None for text-only mode
    )
    await store.put(("memory", "user-42"), "theme", {"value": "dark"})
    item = await store.get(("memory", "user-42"), "theme")
"""

from locus.memory.store_backends.oracle import OracleStore, OracleStoreConfig
from locus.memory.store_backends.oracle_sync import OracleSyncStore


__all__ = [
    "OracleStore",
    "OracleStoreConfig",
    "OracleSyncStore",
]
