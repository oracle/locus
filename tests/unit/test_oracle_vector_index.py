# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for OracleVectorStore vector-index DDL generation.

Covers the three ORGANIZATION variants Oracle 23ai/26ai accepts (HNSW,
IVF, NONE) plus bounds validation on every tuning knob. The DDL output
is asserted as plain strings so this suite runs without a database —
the live-fire smoke happens in ``tests/integration/test_oracle_rag.py``.

The grammar quirks worth pinning here:

* ``PARAMETERS (...)`` clause cannot coexist with the ``WITH DISTANCE``
  keyword; ORA-00922 if both appear. The DDL builder drops the
  ``WITH`` when params are present.
* IVF parameter is ``NEIGHBOR PARTITIONS`` (with a space) inside
  ``PARAMETERS (...)``, *not* ``NEIGHBOR_PARTITIONS``.
* HNSW lives under ``ORGANIZATION INMEMORY NEIGHBOR GRAPH``, IVF under
  ``ORGANIZATION NEIGHBOR PARTITIONS``.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import pytest

import locus.rag.stores.oracle as oracle_mod
from locus.core.errors import VectorStoreError
from locus.core.warnings import LocusWarning
from locus.rag.stores.oracle import OracleVectorStore, _needs_vector_memory


def _store(**kwargs) -> OracleVectorStore:
    """Convenience builder — no live connection is opened until
    ``_get_pool()`` is awaited, so unit tests can introspect the DDL
    string without env vars."""
    base = {
        "dsn": "x",
        "user": "u",
        "password": "p",
        "dimension": 1024,
        "distance_metric": "COSINE",
    }
    base.update(kwargs)
    return OracleVectorStore(**base)


class TestHNSWDDL:
    def test_defaults_emit_with_distance(self) -> None:
        ddl = _store(index_type="HNSW")._vector_index_ddl()
        assert "ORGANIZATION INMEMORY NEIGHBOR GRAPH" in ddl
        # No PARAMETERS → keep the WITH keyword.
        assert "WITH DISTANCE COSINE" in ddl
        assert "PARAMETERS" not in ddl

    def test_tuned_drops_with_keyword_before_distance(self) -> None:
        ddl = _store(
            index_type="HNSW",
            hnsw_neighbors=32,
            hnsw_ef_construction=256,
        )._vector_index_ddl()
        # PARAMETERS present → DISTANCE without WITH (ORA-00922 otherwise).
        assert " DISTANCE COSINE" in ddl
        assert "WITH DISTANCE" not in ddl
        assert "PARAMETERS (TYPE HNSW, NEIGHBORS 32, EFCONSTRUCTION 256)" in ddl

    def test_parallel_clause(self) -> None:
        ddl = _store(index_type="HNSW", index_parallel=4)._vector_index_ddl()
        assert ddl.endswith("PARALLEL 4")

    def test_target_accuracy_clause(self) -> None:
        ddl = _store(index_type="HNSW", index_accuracy=90)._vector_index_ddl()
        assert "WITH TARGET ACCURACY 90" in ddl


class TestIVFDDL:
    def test_defaults(self) -> None:
        ddl = _store(index_type="IVF")._vector_index_ddl()
        assert "ORGANIZATION NEIGHBOR PARTITIONS" in ddl
        assert "WITH DISTANCE COSINE" in ddl
        assert "PARAMETERS" not in ddl

    def test_tuned_uses_neighbor_partitions_with_space(self) -> None:
        ddl = _store(index_type="IVF", ivf_neighbor_partitions=128)._vector_index_ddl()
        # Critical: SPACE inside PARAMETERS, not underscore — Oracle 26ai
        # rejects NEIGHBOR_PARTITIONS as ORA-00922.
        assert "NEIGHBOR PARTITIONS 128" in ddl
        assert "NEIGHBOR_PARTITIONS" not in ddl
        assert "PARAMETERS (TYPE IVF, NEIGHBOR PARTITIONS 128)" in ddl
        assert "WITH DISTANCE" not in ddl
        assert " DISTANCE COSINE" in ddl


class TestNoneIndex:
    def test_returns_none(self) -> None:
        assert _store(index_type="NONE")._vector_index_ddl() is None


class TestValidation:
    def test_unknown_index_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="index_type"):
            _store(index_type="BOGUS")

    def test_bool_for_int_rejected(self) -> None:
        # bool is a subclass of int; Pydantic coerces True → 1 before the
        # validator sees it, so the rejection actually fires from the
        # bounds check ("must be at least 2"). Either way the rejection
        # is what we care about — the field is unreachable from the user
        # surface, so just confirm pydantic.ValidationError is raised.
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="hnsw_neighbors"):
            _store(index_type="HNSW", hnsw_neighbors=True)  # type: ignore[arg-type]

    def test_below_min_rejected(self) -> None:
        with pytest.raises(ValueError, match="hnsw_neighbors"):
            _store(index_type="HNSW", hnsw_neighbors=1)

    def test_above_max_rejected(self) -> None:
        with pytest.raises(ValueError, match="hnsw_neighbors"):
            _store(index_type="HNSW", hnsw_neighbors=4096)

    def test_ef_construction_bounds(self) -> None:
        with pytest.raises(ValueError, match="hnsw_ef_construction"):
            _store(index_type="HNSW", hnsw_ef_construction=70000)

    def test_ivf_partitions_bounds(self) -> None:
        with pytest.raises(ValueError, match="ivf_neighbor_partitions"):
            _store(index_type="IVF", ivf_neighbor_partitions=0)

    def test_accuracy_bounds(self) -> None:
        with pytest.raises(ValueError, match="index_accuracy"):
            _store(index_type="HNSW", index_accuracy=101)

    def test_parallel_min(self) -> None:
        with pytest.raises(ValueError, match="index_parallel"):
            _store(index_type="HNSW", index_parallel=0)


# ---------------------------------------------------------------------------
# ORA-51962: the HNSW index has no vector memory pool to live in
# ---------------------------------------------------------------------------

ORA_51962 = "ORA-51962: The vector memory area is out of space for the current container."


class _Cursor:
    """Records every statement; raises ``error`` on the vector index DDL."""

    def __init__(self, error: Exception | None) -> None:
        self.error = error
        self.executed: list[str] = []

    async def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        self.executed.append(" ".join(sql.split()))
        if self.error is not None and "CREATE VECTOR INDEX" in sql:
            raise self.error

    async def fetchone(self) -> tuple[int]:
        return (0,)  # the table does not exist yet

    async def __aenter__(self) -> _Cursor:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _Conn:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor
        self.commits = 0

    def cursor(self) -> _Cursor:
        return self._cursor

    async def commit(self) -> None:
        self.commits += 1


def _provisioning_store(
    monkeypatch: pytest.MonkeyPatch, error: Exception | None, **kwargs: Any
) -> tuple[OracleVectorStore, _Cursor, _Conn]:
    cursor = _Cursor(error)
    conn = _Conn(cursor)

    @asynccontextmanager
    async def fake_acquire(_store: object, _pool: object) -> AsyncIterator[_Conn]:
        yield conn

    monkeypatch.setattr(oracle_mod, "safe_acquire", fake_acquire)
    store = _store(**kwargs)
    monkeypatch.setattr(store, "_get_pool", AsyncMock(return_value=object()))
    return store, cursor, conn


def test_needs_vector_memory_is_ora_51962_only() -> None:
    assert _needs_vector_memory(RuntimeError(ORA_51962))
    assert not _needs_vector_memory(RuntimeError("ORA-00955: name is already used"))
    assert not _needs_vector_memory(RuntimeError("ORA-01031: insufficient privileges"))


class TestVectorMemoryFallback:
    async def test_hnsw_without_vector_memory_warns_and_keeps_the_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, cursor, conn = _provisioning_store(monkeypatch, RuntimeError(ORA_51962))

        with pytest.warns(LocusWarning, match=r'index_type="IVF"') as record:
            await store._ensure_table()

        assert store._initialized is True
        assert conn.commits == 1
        assert any(sql.startswith("CREATE TABLE locus_vectors") for sql in cursor.executed)
        assert any("ORGANIZATION INMEMORY NEIGHBOR GRAPH" in sql for sql in cursor.executed)
        message = str(record[0].message)
        assert "HNSW vector index on locus_vectors" in message
        assert "ORA-51962" in message
        assert message.endswith("Searching locus_vectors with an exact scan.")

    async def test_the_table_is_searched_without_an_index_afterwards(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, cursor, _conn = _provisioning_store(monkeypatch, RuntimeError(ORA_51962))
        with pytest.warns(LocusWarning):
            await store._ensure_table()

        # Already provisioned: a second pass runs no DDL and warns again never.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            await store._ensure_table()
        assert sum("CREATE" in sql for sql in cursor.executed) == 3  # table, index, created_at

    async def test_other_index_errors_propagate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store, _cursor, conn = _provisioning_store(
            monkeypatch, RuntimeError("ORA-01031: insufficient privileges")
        )
        with pytest.raises(RuntimeError, match="ORA-01031"):
            await store._ensure_table()
        assert store._initialized is False
        assert conn.commits == 0

    async def test_build_index_raises_vector_store_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, _cursor, conn = _provisioning_store(monkeypatch, RuntimeError(ORA_51962))

        with pytest.raises(VectorStoreError, match="ORA-51962") as excinfo:
            await store.build_index()

        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert 'index_type="IVF"' in str(excinfo.value)
        assert "HNSW vector index on locus_vectors" in str(excinfo.value)
        assert conn.commits == 0

    async def test_build_index_other_errors_propagate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, _cursor, _conn = _provisioning_store(
            monkeypatch, RuntimeError("ORA-01031: insufficient privileges")
        )
        with pytest.raises(RuntimeError, match="ORA-01031"):
            await store.build_index()
