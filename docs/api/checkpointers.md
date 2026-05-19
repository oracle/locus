# Checkpointers

State persistence between agent runs. **Oracle Autonomous Database** is
the recommended production backend: native JSON columns, the
`OracleCheckpointSaver` versioned saver matching the
langgraph-oracledb shape, and pool-based async access via `python-oracledb`.

For long-term memory (durable KV store, semantic recall), see
[Memory](memory.md). This page covers the **per-run state snapshot**
contract used by `AgentConfig.checkpointer`.

## Contract

::: locus.memory.checkpointer.BaseCheckpointer
::: locus.core.protocols.CheckpointerCapabilities

## Oracle Database 26ai

Three flavours, depending on how you want state stored:

- **`OracleBackend`** — async, JSON-column storage; the default
  full-checkpointer interface.
- **`OracleCheckpointSaver`** — versioned (LangGraph-shape) saver,
  with explicit checkpoint versions and parent IDs for time-travel.
- **`OracleSyncBackend` / `OracleSyncCheckpointSaver`** — sync
  companions for the langgraph-oracledb split, when you need to run
  inside a sync ORM session.

::: locus.memory.backends.OracleBackend
::: locus.memory.backends.oracle_versioned.OracleCheckpointSaver
::: locus.memory.backends.oracle_sync.OracleSyncBackend
::: locus.memory.backends.oracle_sync.OracleSyncCheckpointSaver

## OCI Object Storage

::: locus.memory.backends.OCIBucketBackend

## Other backends

When you can't run on Oracle / OCI, the same `BaseCheckpointer`
contract is implemented for Redis, PostgreSQL, OpenSearch, file
system, and an HTTP-API adapter.

::: locus.memory.backends.RedisBackend
::: locus.memory.backends.PostgreSQLBackend
::: locus.memory.backends.OpenSearchBackend
::: locus.memory.backends.FileCheckpointer
::: locus.memory.backends.HTTPCheckpointer
::: locus.memory.backends.MemoryCheckpointer

## Adapters

`StorageBackendAdapter` wraps any of the simple key-value backends
above into the full `BaseCheckpointer` interface. The convenience
factory functions below are one-line shortcuts.

::: locus.memory.backends.adapters.StorageBackendAdapter
::: locus.memory.backends.adapters.oracle_checkpointer
::: locus.memory.backends.adapters.oci_bucket_checkpointer
::: locus.memory.backends.adapters.opensearch_checkpointer
::: locus.memory.backends.adapters.postgresql_checkpointer
::: locus.memory.backends.adapters.redis_checkpointer
