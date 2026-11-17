# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The Oracle pool builders' host-triplet path and their small table helpers.

Companion to ``test_oracle_tls_only_config_dir.py``: every builder assembles
the connect string from ``host`` / ``port`` / ``service_name`` when no ``dsn``
is given, qualifies its table name with ``schema_name``, and skips table
creation once it has run (or when a DBA owns the table). Stubbed ``oracledb``,
no database.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.unit.test_oracle_tls_only_config_dir import (
    _CREDENTIALS,
    _builders,
    _ids,
    _install_capturing_oracledb,
)


@pytest.mark.parametrize("idx", range(10), ids=_ids())
def test_host_triplet_builds_the_connect_string(idx: int, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_capturing_oracledb(monkeypatch)
    _name, construct, build_pool = _builders()[idx]
    build_pool(construct(host="db.example", port=1522, service_name="svc_low", **_CREDENTIALS))
    assert len(calls) == 1
    assert calls[0]["dsn"] == "db.example:1522/svc_low"
    assert "config_dir" not in calls[0]
    assert "wallet_location" not in calls[0]


def test_backend_table_name_is_schema_qualified_when_a_schema_is_given() -> None:
    from locus.memory.backends.oracle import OracleBackend

    plain = OracleBackend(dsn="mydb_low", table_name="LOCUS_CKPT", **_CREDENTIALS)
    owned = OracleBackend(
        dsn="mydb_low", table_name="LOCUS_CKPT", schema_name="APP", **_CREDENTIALS
    )
    assert plain._full_table_name == "LOCUS_CKPT"
    assert owned._full_table_name == "APP.LOCUS_CKPT"


def test_ensure_table_is_a_no_op_once_initialised(monkeypatch: pytest.MonkeyPatch) -> None:
    """An initialised backend never touches the pool again for DDL."""
    calls = _install_capturing_oracledb(monkeypatch)
    from locus.memory.backends.oracle import OracleBackend
    from locus.memory.store_backends.oracle import OracleStore
    from locus.rag.stores.oracle import OracleVectorStore

    backend = OracleBackend(dsn="mydb_low", **_CREDENTIALS)
    backend._initialized = True
    asyncio.run(backend._ensure_table())

    vectors = OracleVectorStore(dsn="mydb_low", dimension=8, **_CREDENTIALS)
    vectors._initialized = True
    asyncio.run(vectors._ensure_table())

    # A DBA-provisioned store (auto_create_table=False) trusts the table exists.
    store = OracleStore(dsn="mydb_low", auto_create_table=False, **_CREDENTIALS)
    asyncio.run(store._ensure_table())
    assert store._initialized is True

    assert calls == [], "no pool was built: nothing to create"


def test_chunk_column_rejects_a_where_clause_with_statement_separators() -> None:
    from locus.rag.chunkers.oracle_indb import OracleInDBChunker

    chunker = OracleInDBChunker(dsn="mydb_low", **_CREDENTIALS)

    async def consume() -> list[Any]:
        rows = chunker.chunk_column(
            table_name="DOCS", text_column="BODY", where="1=1; DROP TABLE x"
        )
        return [row async for row in rows]

    with pytest.raises(ValueError, match="where clause"):
        asyncio.run(consume())


def test_approval_record_column_decodes_bytes() -> None:
    from locus.control.approvals_oracle import _document

    assert asyncio.run(_document(b'{"a": 1}')) == '{"a": 1}'
    assert asyncio.run(_document('{"a": 1}')) == '{"a": 1}'
    assert asyncio.run(_document({"a": 1})) == {"a": 1}
