# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the Oracle Text leg of ``OracleVectorStore.hybrid_search``.

No ``oracledb`` server: ``safe_acquire`` yields a fake connection whose
cursor records every statement, so the tests pin the SQL the store sends.
"""

from __future__ import annotations

from array import array
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import pytest

import locus.rag.stores.oracle as oracle_mod
from locus.rag.stores.oracle import OracleVectorStore, _oracle_text_query


QUERY_VECTOR = [0.1, 0.2, 0.3, 0.4]
TEXT_INDEX_JOIN = (
    "LEFT JOIN (SELECT id AS lex_id_, SCORE(1) AS lex_score_ FROM locus_vectors "
    "WHERE CONTAINS(content, :text_query, 1) > 0) lex_ ON lex_.lex_id_ = t.id"
)


class _Cursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, dict[str, Any] | None]] = []

    async def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        self.executed.append((sql, params))

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows

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


@pytest.fixture
def make_store(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., tuple[OracleVectorStore, _Cursor]]:
    def factory(rows: Iterable[tuple[Any, ...]] = ()) -> tuple[OracleVectorStore, _Cursor]:
        cursor = _Cursor(list(rows))
        conn = _Conn(cursor)

        @asynccontextmanager
        async def fake_acquire(_store: object, _pool: object) -> AsyncIterator[_Conn]:
            yield conn

        monkeypatch.setattr(oracle_mod, "safe_acquire", fake_acquire)
        store = OracleVectorStore(dsn="x", user="u", password="p", dimension=4)  # noqa: S106
        monkeypatch.setattr(store, "_ensure_table", AsyncMock())
        monkeypatch.setattr(store, "_get_pool", AsyncMock(return_value=object()))
        return store, cursor

    return factory


def _flat(sql: str) -> str:
    return " ".join(sql.split())


def test_text_query_escapes_each_term_and_accumulates() -> None:
    assert (
        _oracle_text_query("ORA-01555 snapshot too old")
        == "{ORA-01555} ACCUM {snapshot} ACCUM {too} ACCUM {old}"
    )
    assert _oracle_text_query("kubectl logs -p") == "{kubectl} ACCUM {logs} ACCUM {-p}"
    assert _oracle_text_query("a & b | c") == "{a} ACCUM {&} ACCUM {b} ACCUM {|} ACCUM {c}"


def test_text_query_drops_braces_and_has_no_empty_terms() -> None:
    assert _oracle_text_query("{a}b {} c") == "{ab} ACCUM {c}"
    assert _oracle_text_query("   ") is None
    assert _oracle_text_query("{}") is None


def test_text_query_caps_the_number_of_terms() -> None:
    query = _oracle_text_query(" ".join(f"t{i}" for i in range(50)))
    assert query is not None
    assert query.count("ACCUM") == 19


async def test_ensure_text_index_synchronises_on_commit(make_store: Callable[..., Any]) -> None:
    store, cursor = make_store()
    await store.ensure_text_index()
    assert [sql for sql, _ in cursor.executed] == [
        "CREATE INDEX idx_locus_vectors_txt ON locus_vectors (content) "
        "INDEXTYPE IS CTXSYS.CONTEXT PARAMETERS ('SYNC (ON COMMIT)')"
    ]


async def test_ensure_text_index_drops_the_old_index_first(make_store: Callable[..., Any]) -> None:
    store, cursor = make_store()
    await store.ensure_text_index(drop_existing=True)
    statements = [sql for sql, _ in cursor.executed]
    assert statements[0] == "DROP INDEX idx_locus_vectors_txt"
    assert "SYNC (ON COMMIT)" in statements[1]


async def test_text_leg_is_a_left_join_that_filters_nothing(make_store: Callable[..., Any]) -> None:
    store, cursor = make_store()
    await store.hybrid_search("decline code 05", QUERY_VECTOR, use_text_index=True)
    sql, params = cursor.executed[-1]
    flat = _flat(sql)
    assert TEXT_INDEX_JOIN in flat
    assert "NVL(lex_.lex_score_, 0) / 100.0" in flat
    assert "FROM locus_vectors t" in flat
    outer = flat.split(") lex_ ON", 1)[1]
    assert "WHERE" not in outer
    assert params is not None
    assert params["text_query"] == "{decline} ACCUM {code} ACCUM {05}"


async def test_metadata_filter_is_the_only_outer_where(make_store: Callable[..., Any]) -> None:
    store, cursor = make_store()
    await store.hybrid_search(
        "decline code 05",
        QUERY_VECTOR,
        use_text_index=True,
        metadata_filter={"area": "payments"},
    )
    sql, params = cursor.executed[-1]
    outer = _flat(sql).split(") lex_ ON", 1)[1]
    assert "WHERE (JSON_VALUE(metadata, '$.area')" in outer
    assert "CONTAINS" not in outer
    assert params is not None
    assert "payments" in params.values()


async def test_text_index_with_no_terms_is_vector_only(make_store: Callable[..., Any]) -> None:
    store, cursor = make_store()
    await store.hybrid_search("   ", QUERY_VECTOR, use_text_index=True)
    sql, params = cursor.executed[-1]
    assert "LEFT JOIN" not in sql
    assert "CONTAINS" not in sql
    assert "0.0 AS lexical_score_" in _flat(sql)
    assert params is not None
    assert "text_query" not in params


async def test_like_leg_filters_nothing_either(make_store: Callable[..., Any]) -> None:
    store, cursor = make_store()
    await store.hybrid_search("decline code", QUERY_VECTOR, use_text_index=False)
    sql, params = cursor.executed[-1]
    flat = _flat(sql)
    assert "WHERE" not in flat
    assert "LEFT JOIN" not in flat
    assert params is not None
    assert params["tok_0"] == "%decline%"
    assert params["tok_1"] == "%code%"


async def test_rows_the_text_index_misses_keep_their_vector_score(
    make_store: Callable[..., Any],
) -> None:
    vector = array("f", QUERY_VECTOR)
    row = ("d2", "ORA-01555 snapshot too old", vector, "{}", None, 0.3, None, 0.85)
    store, cursor = make_store(rows=[row])
    results = await store.hybrid_search(
        "card rejected", QUERY_VECTOR, use_text_index=True, alpha=1.0
    )
    assert [r.document.id for r in results] == ["d2"]
    assert results[0].score == pytest.approx(0.85)
    assert results[0]._lexical_score == 0.0
    assert results[0].document.embedding == pytest.approx(QUERY_VECTOR)
    _, params = cursor.executed[-1]
    assert params is not None
    assert params["alpha"] == 1.0


def test_select_fetches_the_vector_column_natively() -> None:
    store = OracleVectorStore(dsn="x", user="u", password="p", dimension=4)  # noqa: S106
    sql = store._select_columns_sql()
    assert "embedding AS embedding_" in sql
    assert "VECTOR_SERIALIZE" not in sql


async def test_read_embedding_accepts_arrays_text_lobs_and_null() -> None:
    read = OracleVectorStore._read_embedding

    class _Lob:
        async def read(self) -> str:
            return "[1.0, 2.0]\n"

    assert await read(array("f", [0.5, 0.25])) == [0.5, 0.25]
    assert await read("[0.5,0.25]") == [0.5, 0.25]
    assert await read(b"[0.5,0.25]") == [0.5, 0.25]
    assert await read(_Lob()) == [1.0, 2.0]
    assert await read(None) is None
    assert await read("") is None
