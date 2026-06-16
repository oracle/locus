# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Synchronous-API variants of the Oracle checkpoint backends.

These wrappers expose blocking equivalents of every public method on
:class:`locus.memory.backends.oracle.OracleBackend` (single-row) and
:class:`locus.memory.backends.oracle_versioned.OracleCheckpointSaver`
(versioned LangGraph-shape saver). Each wrapper holds an internal async
instance and drives it via :func:`locus._sync.run_sync`, so the wire
protocol, schema, and CLOB binding hardening are all shared with the
async path — only the surface call shape differs.

Same split-pattern ``langgraph-oracledb`` ships with its ``OracleSaver``
(sync) and ``AsyncOracleSaver`` (async). **Zero** langchain / langgraph
imports — locus owns the contract end-to-end.
"""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr

from locus._sync import run_sync
from locus.core.state import AgentState
from locus.memory.backends.oracle import OracleBackend
from locus.memory.backends.oracle_versioned import OracleCheckpointSaver


class OracleSyncBackend:
    """Sync companion to :class:`OracleBackend`.

    Constructor accepts the same arguments as
    :class:`OracleBackend`; every public async method is mirrored as a
    blocking method with the **same name** (no ``_sync`` suffix) so
    porting code between the two surfaces is a single import change.
    """

    def __init__(
        self,
        dsn: str | None = None,
        user: str = "admin",
        password: str | SecretStr = "",
        wallet_location: str | None = None,
        wallet_password: str | SecretStr | None = None,
        host: str | None = None,
        port: int = 1521,
        service_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._async = OracleBackend(
            dsn=dsn,
            user=user,
            password=password,
            wallet_location=wallet_location,
            wallet_password=wallet_password,
            host=host,
            port=port,
            service_name=service_name,
            **kwargs,
        )

    # -- Public surface ----------------------------------------------------

    def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return run_sync(
            self._async.save(state, thread_id, checkpoint_id=checkpoint_id, metadata=metadata)
        )

    def load(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> dict | None:
        return run_sync(self._async.load(thread_id, checkpoint_id=checkpoint_id))

    def delete(self, thread_id: str) -> bool:
        return run_sync(self._async.delete(thread_id))

    def exists(self, thread_id: str) -> bool:
        return run_sync(self._async.exists(thread_id))

    def list_threads(
        self,
        limit: int = 100,
        offset: int = 0,
        pattern: str = "%",
    ) -> list[str]:
        return run_sync(self._async.list_threads(limit=limit, offset=offset, pattern=pattern))

    def get_metadata(self, thread_id: str) -> dict[str, Any] | None:
        return run_sync(self._async.get_metadata(thread_id))

    def query_by_metadata(
        self,
        key: str,
        value: Any,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return run_sync(self._async.query_by_metadata(key, value, limit=limit))

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return run_sync(self._async.search(query, limit=limit))

    def vacuum(self, older_than_days: int = 30) -> int:
        return run_sync(self._async.vacuum(older_than_days=older_than_days))

    def count(self, pattern: str = "%") -> int:
        return run_sync(self._async.count(pattern=pattern))

    def close(self) -> None:
        run_sync(self._async.close())

    def __repr__(self) -> str:
        return f"OracleSyncBackend(wrapping={self._async!r})"


class OracleSyncCheckpointSaver:
    """Sync companion to :class:`OracleCheckpointSaver`.

    Versioned, history-preserving checkpoint saver with pending-writes
    durability — synchronous API. Method names match the async
    counterpart exactly (``put`` / ``get`` / ``list_checkpoints`` /
    ``put_writes`` / ``get_writes`` / ``delete_thread`` / ``close``).
    """

    def __init__(
        self,
        dsn: str | None = None,
        user: str = "admin",
        password: str | SecretStr = "",
        wallet_location: str | None = None,
        wallet_password: str | SecretStr | None = None,
        host: str | None = None,
        port: int = 1521,
        service_name: str | None = None,
        table_name: str = "locus",
        schema_name: str | None = None,
        min_pool_size: int = 1,
        max_pool_size: int = 5,
        auto_create_table: bool = True,
        **kwargs: Any,
    ) -> None:
        self._async = OracleCheckpointSaver(
            dsn=dsn,
            user=user,
            password=password,
            wallet_location=wallet_location,
            wallet_password=wallet_password,
            host=host,
            port=port,
            service_name=service_name,
            table_name=table_name,
            schema_name=schema_name,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            auto_create_table=auto_create_table,
            **kwargs,
        )

    # -- Public surface ----------------------------------------------------

    def put(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
        checkpoint_data: dict,
        checkpoint_ns: str = "",
        parent_checkpoint_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        return run_sync(
            self._async.put(
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                checkpoint_data=checkpoint_data,
                checkpoint_ns=checkpoint_ns,
                parent_checkpoint_id=parent_checkpoint_id,
                metadata=metadata,
            )
        )

    def get(
        self,
        *,
        thread_id: str,
        checkpoint_id: str | None = None,
        checkpoint_ns: str = "",
    ) -> dict | None:
        return run_sync(
            self._async.get(
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                checkpoint_ns=checkpoint_ns,
            )
        )

    def list_checkpoints(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str = "",
        limit: int = 10,
        before: str | None = None,
    ) -> list[dict]:
        return run_sync(
            self._async.list_checkpoints(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                limit=limit,
                before=before,
            )
        )

    def put_writes(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
        task_id: str,
        writes: list[tuple[str, Any]],
        checkpoint_ns: str = "",
    ) -> None:
        return run_sync(
            self._async.put_writes(
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                task_id=task_id,
                writes=writes,
                checkpoint_ns=checkpoint_ns,
            )
        )

    def get_writes(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
        checkpoint_ns: str = "",
        task_id: str | None = None,
    ) -> list[dict]:
        return run_sync(
            self._async.get_writes(
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                checkpoint_ns=checkpoint_ns,
                task_id=task_id,
            )
        )

    def delete_thread(self, thread_id: str) -> None:
        return run_sync(self._async.delete_thread(thread_id))

    def close(self) -> None:
        run_sync(self._async.close())

    def __repr__(self) -> str:
        return f"OracleSyncCheckpointSaver(wrapping={self._async!r})"
