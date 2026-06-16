# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for OracleVectorStore's Mongo-style metadata filter compiler.

Covers the operator surface and the SQL fragments the compiler emits.
Live behaviour against Oracle 26ai is verified in the
``test_oracle_rag.py`` integration suite — this file pins the SQL
shape so a future refactor doesn't silently change the operator
semantics.

The grammar mirrors langchain-oracle's OracleVS filter DSL so users
migrating between the two stacks don't have to rewrite their
metadata-filter dicts. Implemented natively — no langchain dep.
"""

from __future__ import annotations

import pytest

from locus.rag.stores.oracle import OracleVectorStore


def _store() -> OracleVectorStore:
    return OracleVectorStore(dsn="x", user="u", password="p", dimension=8, distance_metric="COSINE")  # noqa: S105, S106


class TestLeafOperators:
    def test_empty_filter_returns_empty_string(self) -> None:
        params: dict = {}
        sql = _store()._compile_metadata_filter(None, params)
        assert sql == ""
        assert params == {}

    def test_implicit_equality(self) -> None:
        params: dict = {}
        sql = _store()._compile_metadata_filter({"k": "v"}, params)
        assert "JSON_VALUE(metadata, '$.k')" in sql
        assert "= :" in sql
        assert "v" in params.values()

    def test_explicit_eq(self) -> None:
        params: dict = {}
        sql = _store()._compile_metadata_filter({"k": {"$eq": "v"}}, params)
        assert " = :" in sql
        assert "v" in params.values()

    def test_ne(self) -> None:
        params: dict = {}
        sql = _store()._compile_metadata_filter({"k": {"$ne": "v"}}, params)
        assert " != :" in sql

    def test_comparison_operators(self) -> None:
        store = _store()
        for op, sql_op in [("$gt", ">"), ("$gte", ">="), ("$lt", "<"), ("$lte", "<=")]:
            params: dict = {}
            sql = store._compile_metadata_filter({"year": {op: 2020}}, params)
            assert f" {sql_op} :" in sql, f"{op} should emit {sql_op}"
            assert "2020" in params.values()

    def test_multiple_ops_in_one_field_are_AND_ed(self) -> None:
        params: dict = {}
        sql = _store()._compile_metadata_filter({"year": {"$gte": 2020, "$lte": 2023}}, params)
        assert " AND " in sql


class TestSetOperators:
    def test_in_with_three_values(self) -> None:
        params: dict = {}
        sql = _store()._compile_metadata_filter({"tag": {"$in": ["a", "b", "c"]}}, params)
        assert " IN (" in sql
        # Three placeholders bind through.
        assert sum(1 for v in params.values() if v in {"a", "b", "c"}) == 3

    def test_nin(self) -> None:
        params: dict = {}
        sql = _store()._compile_metadata_filter({"tag": {"$nin": ["x"]}}, params)
        assert " NOT IN (" in sql

    def test_in_empty_list_returns_no_rows(self) -> None:
        params: dict = {}
        sql = _store()._compile_metadata_filter({"tag": {"$in": []}}, params)
        assert "1=0" in sql
        assert params == {}

    def test_nin_empty_list_returns_all_rows(self) -> None:
        params: dict = {}
        sql = _store()._compile_metadata_filter({"tag": {"$nin": []}}, params)
        assert "1=1" in sql


class TestLogicalOperators:
    def test_and_combines_predicates(self) -> None:
        params: dict = {}
        sql = _store()._compile_metadata_filter({"$and": [{"a": "1"}, {"b": "2"}]}, params)
        assert " AND " in sql

    def test_or_combines_predicates(self) -> None:
        params: dict = {}
        sql = _store()._compile_metadata_filter({"$or": [{"a": "1"}, {"b": "2"}]}, params)
        assert " OR " in sql

    def test_not_wraps_predicate(self) -> None:
        params: dict = {}
        sql = _store()._compile_metadata_filter({"$not": {"a": "1"}}, params)
        assert sql.startswith("(NOT ") or "NOT (" in sql

    def test_implicit_AND_across_top_level_keys(self) -> None:
        # Top-level dict with two keys = implicit AND.
        params: dict = {}
        sql = _store()._compile_metadata_filter({"a": "1", "b": "2"}, params)
        assert " AND " in sql

    def test_nested_and_or(self) -> None:
        params: dict = {}
        sql = _store()._compile_metadata_filter(
            {"$or": [{"a": "1"}, {"$and": [{"b": "2"}, {"c": "3"}]}]},
            params,
        )
        assert " OR " in sql
        assert " AND " in sql


class TestValidation:
    def test_unknown_top_level_op_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown top-level operator"):
            _store()._compile_metadata_filter({"$bogus": "x"}, {})

    def test_unknown_field_op_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown operator"):
            _store()._compile_metadata_filter({"k": {"$weird": "x"}}, {})

    def test_invalid_field_name_rejected(self) -> None:
        # Dots / spaces in field name would let user input escape the
        # JSON path expression — the compiler must reject them.
        with pytest.raises(ValueError, match="Invalid metadata field name"):
            _store()._compile_metadata_filter({"foo.bar": "x"}, {})

    def test_and_or_require_list(self) -> None:
        with pytest.raises(TypeError, match="\\$and expects a list"):
            _store()._compile_metadata_filter({"$and": {"a": "1"}}, {})

    def test_in_requires_list(self) -> None:
        with pytest.raises(ValueError, match="\\$in expects a list"):
            _store()._compile_metadata_filter({"k": {"$in": "x"}}, {})


class TestValueCoercion:
    def test_int_is_stringified(self) -> None:
        params: dict = {}
        _store()._compile_metadata_filter({"n": 42}, params)
        assert "42" in params.values()

    def test_bool_becomes_json_literal(self) -> None:
        params: dict = {}
        _store()._compile_metadata_filter({"flag": True}, params)
        assert "true" in params.values()
        params2: dict = {}
        _store()._compile_metadata_filter({"flag": False}, params2)
        assert "false" in params2.values()

    def test_float_is_stringified(self) -> None:
        params: dict = {}
        _store()._compile_metadata_filter({"n": 1.5}, params)
        assert "1.5" in params.values()
