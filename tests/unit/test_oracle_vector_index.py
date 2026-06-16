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

import pytest

from locus.rag.stores.oracle import OracleVectorStore


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
