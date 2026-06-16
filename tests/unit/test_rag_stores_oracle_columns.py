# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for OracleVectorStore column-name overrides.

Covers the read-only / foreign-schema attachment path so locus can attach
to tables written by other ingestion pipelines (e.g.
``langchain_oracledb.vectorstores.OracleVS``) without re-ingestion.

These tests exercise the pure-SQL-generation half of the contract — no
``oracledb`` server is needed.
"""

from __future__ import annotations

import pytest

from locus.rag.stores.oracle import OracleVectorConfig, OracleVectorStore


def test_default_config_uses_locus_native_columns() -> None:
    """Defaults match the locus-native schema written by `_ensure_table`."""
    config = OracleVectorConfig(dsn="x", user="u")
    assert config.id_column == "id"
    assert config.content_column == "content"
    assert config.embedding_column == "embedding"
    assert config.metadata_column == "metadata"
    assert config.created_at_column == "created_at"
    assert config.auto_create_table is True


def test_oraclevs_compat_config_overrides() -> None:
    """OracleVS-style table: 'text' column, no created_at, read-only."""
    config = OracleVectorConfig(
        dsn="x",
        user="u",
        table_name="VECTOR_DOCUMENTS",
        content_column="text",
        created_at_column=None,
        auto_create_table=False,
        dimension=1536,
    )
    assert config.content_column == "text"
    assert config.created_at_column is None
    assert config.auto_create_table is False
    assert config.dimension == 1536


@pytest.mark.parametrize(
    ("bad_col", "field"),
    [
        ("'; DROP TABLE users--", "content_column"),
        ("text--", "content_column"),
        ("text;", "id_column"),
        ("", "embedding_column"),
        ("1abc", "metadata_column"),
    ],
)
def test_column_names_are_validated_against_sql_injection(bad_col: str, field: str) -> None:
    """Identifier validation rejects anything that isn't a safe SQL ident."""
    with pytest.raises(ValueError):
        OracleVectorConfig(dsn="x", user="u", **{field: bad_col})


def test_created_at_column_validated_only_when_set() -> None:
    """``created_at_column=None`` is valid (disables the column entirely)."""
    cfg = OracleVectorConfig(dsn="x", user="u", created_at_column=None)
    assert cfg.created_at_column is None

    with pytest.raises(ValueError):
        OracleVectorConfig(dsn="x", user="u", created_at_column="bad'name")


def _make_store(**kwargs: object) -> OracleVectorStore:
    return OracleVectorStore(dsn="x", user="u", password="p", **kwargs)  # noqa: S106


def test_insert_sql_locus_native() -> None:
    """Default schema: id, content, embedding, metadata, created_at."""
    store = _make_store()
    sql = store._insert_sql()
    assert "INSERT INTO locus_vectors" in sql
    assert "id, content, embedding, metadata, created_at" in sql
    assert ":id, :content, TO_VECTOR(:embedding), :metadata, :created_at" in sql


def test_insert_sql_oraclevs_compat_drops_created_at() -> None:
    """OracleVS-style: 'text' column, no created_at — SQL must match."""
    store = _make_store(
        table_name="VECTOR_DOCUMENTS",
        content_column="text",
        created_at_column=None,
        auto_create_table=False,
    )
    sql = store._insert_sql()
    assert "INSERT INTO VECTOR_DOCUMENTS" in sql
    assert "id, text, embedding, metadata" in sql
    assert "created_at" not in sql
    # placeholders match
    assert ":id, :content, TO_VECTOR(:embedding), :metadata" in sql


def test_select_columns_sql_uses_overrides() -> None:
    """SELECT clause references the configured columns, aliased uniformly."""
    store = _make_store(
        content_column="text",
        metadata_column="meta_json",
        created_at_column=None,
    )
    sql = store._select_columns_sql()
    assert "text AS content_" in sql
    assert "meta_json AS metadata_" in sql
    # When created_at_column is None, fall back to NULL alias to keep the
    # tuple shape stable for the row parser.
    assert "NULL AS created_at_" in sql


def test_select_columns_sql_with_distance_expr() -> None:
    """``with_distance=`` appends a distance column for ORDER BY."""
    store = _make_store()
    distance = "VECTOR_DISTANCE(embedding, TO_VECTOR(:q), COSINE)"
    sql = store._select_columns_sql(with_distance=distance)
    assert sql.endswith(f"{distance} AS distance_")


def test_full_table_name_respects_schema() -> None:
    """Schema-qualified table when ``schema_name`` is set."""
    s_default = _make_store()
    s_schema = _make_store(schema_name="ADMIN", table_name="VEC_DOC")
    assert s_default._full_table_name == "locus_vectors"
    assert s_schema._full_table_name == "ADMIN.VEC_DOC"


def test_insert_params_omits_created_at_when_disabled() -> None:
    """``_insert_params`` only carries created_at when the column exists."""
    from locus.rag.stores.base import Document

    doc = Document(
        id="doc1",
        content="hello",
        embedding=[0.1, 0.2, 0.3],
        metadata={"src": "test"},
    )
    s_with = _make_store()
    s_without = _make_store(created_at_column=None)

    params_with = s_with._insert_params("doc1", doc)
    params_without = s_without._insert_params("doc1", doc)
    assert "created_at" in params_with
    assert "created_at" not in params_without
    # All other keys are present in both
    for k in ("id", "content", "embedding", "metadata"):
        assert k in params_with
        assert k in params_without
