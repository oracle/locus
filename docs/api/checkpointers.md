# Checkpointers

State persistence between agent runs. **S3-compatible object storage**
(S3 / MinIO / R2 via boto3) is a production backend, alongside Redis,
PostgreSQL, MySQL, and OpenSearch.

For long-term memory (durable KV store, semantic recall), see
[Memory](memory.md). This page covers the **per-run state snapshot**
contract used by `AgentConfig.checkpointer`.

## Contract

::: tulip.memory.checkpointer.BaseCheckpointer
::: tulip.core.protocols.CheckpointerCapabilities

## Object storage

::: tulip.memory.backends.s3.S3Backend

## Other backends

The same `BaseCheckpointer` contract is implemented for Redis,
PostgreSQL, MySQL, OpenSearch, file system, and an HTTP-API adapter.

::: tulip.memory.backends.RedisBackend
::: tulip.memory.backends.PostgreSQLBackend
::: tulip.memory.backends.MySQLBackend
::: tulip.memory.backends.opensearch.OpenSearchBackend
::: tulip.memory.backends.FileCheckpointer
::: tulip.memory.backends.HTTPCheckpointer
::: tulip.memory.backends.MemoryCheckpointer

## Adapters

`StorageBackendAdapter` wraps any of the simple key-value backends
above into the full `BaseCheckpointer` interface. Call `.as_checkpointer()`
on a backend for a one-line shortcut.

::: tulip.memory.backends.adapters.StorageBackendAdapter
