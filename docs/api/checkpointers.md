# Checkpointers

State persistence between agent runs. Oracle Database 26ai and OCI Object
Storage are the Oracle-native production backends; S3-compatible object
storage (S3 / MinIO / R2 via boto3), Redis, PostgreSQL, MySQL, and
OpenSearch ship alongside.

For long-term memory (durable KV store, semantic recall), see
[Memory](memory.md). This page covers the **per-run state snapshot**
contract used by `AgentConfig.checkpointer`.

For the concepts, start with [Checkpointers](../concepts/checkpointers.md).

## Contract

::: locus.memory.checkpointer.BaseCheckpointer
::: locus.core.protocols.CheckpointerCapabilities

## Oracle Database 26ai

Three flavours, depending on how you want state stored (all need the
`oracledb` extra):

- **`OracleBackend`** — async, JSON-column key-value backend;
  `oracle_checkpointer()` wraps it in `StorageBackendAdapter` to give
  the full checkpointer interface.
- **`OracleCheckpointSaver`** — versioned saver with explicit checkpoint
  versions, parent IDs and pending writes for time-travel.
- **`OracleSyncBackend` / `OracleSyncCheckpointSaver`** — sync
  companions, when you need to run inside a sync ORM session.

::: locus.memory.backends.oracle.OracleBackend
::: locus.memory.backends.oracle_versioned.OracleCheckpointSaver
::: locus.memory.backends.oracle_sync.OracleSyncBackend
::: locus.memory.backends.oracle_sync.OracleSyncCheckpointSaver

## Object storage

OCI Object Storage (`oci` extra) and S3-compatible stores (`s3` extra).
Both implement `BaseCheckpointer` directly.

::: locus.memory.backends.oci_bucket.OCIBucketBackend
::: locus.memory.backends.s3.S3Backend

## Other backends

The file-system, HTTP-API, and in-memory backends implement the
`BaseCheckpointer` contract natively. Redis, PostgreSQL, MySQL, and
OpenSearch ship as simple key-value backends (Pydantic models) that the
`StorageBackendAdapter` (below) wraps into the same contract — use the
factory functions rather than passing them to `AgentConfig.checkpointer`
directly.

::: locus.memory.backends.RedisBackend
::: locus.memory.backends.PostgreSQLBackend
::: locus.memory.backends.MySQLBackend
::: locus.memory.backends.opensearch.OpenSearchBackend
::: locus.memory.backends.FileCheckpointer
::: locus.memory.backends.HTTPCheckpointer
::: locus.memory.backends.MemoryCheckpointer

## Adapters

`StorageBackendAdapter` wraps any of the simple key-value backends
above into the full `BaseCheckpointer` interface. For the common
backends, use the factory functions — `redis_checkpointer(...)`,
`postgresql_checkpointer(...)`, `mysql_checkpointer(...)`,
`opensearch_checkpointer(...)` — which build the adapter for you.
`oracle_checkpointer(...)` builds the adapter around `OracleBackend`.
`s3_checkpointer(...)` and `oci_bucket_checkpointer(...)` are the
exceptions: convenience aliases that return the native `S3Backend` /
`OCIBucketBackend` unchanged, since both already implement
`BaseCheckpointer`.

::: locus.memory.backends.adapters.StorageBackendAdapter
::: locus.memory.backends.adapters.oracle_checkpointer
::: locus.memory.backends.adapters.oci_bucket_checkpointer
::: locus.memory.backends.adapters.redis_checkpointer
::: locus.memory.backends.adapters.postgresql_checkpointer
::: locus.memory.backends.adapters.mysql_checkpointer
::: locus.memory.backends.adapters.opensearch_checkpointer
::: locus.memory.backends.adapters.s3_checkpointer
