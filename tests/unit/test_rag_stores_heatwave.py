# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the MySQL HeatWave vector store.

A fake ``mysql.connector`` module is injected into ``sys.modules`` so the
store's lazy import resolves to an in-memory server and no driver is
needed. The fake understands exactly the statements the store emits and
can be switched between a HeatWave server (``DISTANCE()`` available) and
a community server (``ERROR 1305``), so both search paths run the shared
conformance suite here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sys
import types
from datetime import UTC, datetime
from typing import Any

import pytest

from locus.rag.stores.base import Document, supports_hybrid_search
from locus.rag.stores.heatwave import (
    HeatWaveVectorConfig,
    HeatWaveVectorStore,
    _parse_dsn,
    _SyncPool,
    build_filter_sql,
)
from tests.unit.rag.store_conformance import DIMENSION, VEC_A, VEC_B, VEC_C, VectorStoreConformance


# ------------------------------------------------------------- fake driver


class _FakeMySQLError(Exception):
    def __init__(self, errno: int, msg: str) -> None:
        super().__init__(f"{errno} (42000): {msg}")
        self.errno = errno
        self.msg = msg


class _FakeServer:
    """Rows plus the knobs that make it a HeatWave or a community server."""

    def __init__(self, *, has_distance: bool = True) -> None:
        self.has_distance = has_distance
        self.rows: dict[str, dict[str, Any]] = {}
        self.statements: list[tuple[str, tuple[Any, ...] | None]] = []
        self.connections: list[_FakeConnection] = []
        self.connect_kwargs: list[dict[str, Any]] = []
        self.fail_next_execute: Exception | None = None


def _vec(text: str) -> list[float]:
    return [float(x) for x in text.strip("[]").split(",")]


def _distance(metric: str, a: list[float], b: list[float]) -> float:
    """What ``DISTANCE()`` returns on HeatWave MySQL: for ``DOT`` the dot product itself."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    if metric == "DOT":
        return dot
    if metric == "EUCLIDEAN":
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / norm if norm else 1.0


class _FakeCursor:
    _FILTER = re.compile(r"JSON_EXTRACT\(`metadata`, %s\) = CAST\(%s AS JSON\)")

    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn
        self._server = conn.server
        self._rows: list[tuple[Any, ...]] = []
        self.description: Any = None
        self.rowcount = -1
        self.closed = False

    # -- helpers ------------------------------------------------------------

    def _row_tuple(self, row: dict[str, Any]) -> tuple[Any, ...]:
        vector = "[" + ",".join(f"{x:.5e}" for x in row["embedding"]) + "]"
        return (
            row["id"],
            row["content"],
            row["content_type"],
            vector,
            json.dumps(row["metadata"]),
            row["created_at"],
        )

    def _apply_filters(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        rows = list(self._server.rows.values())
        for _ in self._FILTER.findall(sql):
            path, value_json = params.pop(0), params.pop(0)
            key = path[3:-1].replace('\\"', '"').replace("\\\\", "\\")
            rows = [
                r
                for r in rows
                if key in r["metadata"] and json.dumps(r["metadata"][key]) == value_json
            ]
        return rows

    # -- DB-API surface -----------------------------------------------------

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        if self._conn.closed:
            raise _FakeMySQLError(2006, "MySQL server has gone away")
        self._server.statements.append((sql, params))
        if self._server.fail_next_execute is not None:
            exc, self._server.fail_next_execute = self._server.fail_next_execute, None
            raise exc
        flat = " ".join(sql.split())
        args = list(params or ())
        self.description = None
        self._rows = []
        self.rowcount = 0

        if flat.startswith(("CREATE TABLE", "ALTER TABLE")):
            return
        if flat.startswith("SELECT DISTANCE(STRING_TO_VECTOR('[1,0]')"):
            if not self._server.has_distance:
                raise _FakeMySQLError(1305, "FUNCTION locus.DISTANCE does not exist")
            self.description = [("d",)]
            self._rows = [(1.0,)]
            return
        if flat.startswith("SELECT COUNT(*)"):
            self.description = [("count",)]
            self._rows = [(len(self._server.rows),)]
            return
        if flat.startswith("DELETE FROM"):
            if "WHERE id = %s" in flat:
                self.rowcount = 1 if self._server.rows.pop(args[0], None) else 0
            else:
                self.rowcount = len(self._server.rows)
                self._server.rows.clear()
            return
        if flat.startswith("SELECT id, content, content_type, VECTOR_TO_STRING(embedding)"):
            self.description = [("id",)]
            if "WHERE id = %s" in flat:
                row = self._server.rows.get(args[0])
                self._rows = [self._row_tuple(row)] if row else []
                return
            if "AS distance_" in flat:
                if not self._server.has_distance:
                    raise _FakeMySQLError(1305, "FUNCTION locus.DISTANCE does not exist")
                metric = re.search(r"'(COSINE|DOT|EUCLIDEAN)'", flat).group(1)  # type: ignore[union-attr]
                negated = "-DISTANCE(" in flat
                query = _vec(args.pop(0))
                rows = self._apply_filters(flat, args)
                limit = args.pop(0)
                scored = sorted(
                    (
                        (
                            r,
                            -_distance(metric, query, r["embedding"])
                            if negated
                            else _distance(metric, query, r["embedding"]),
                        )
                        for r in rows
                    ),
                    key=lambda p: p[1],
                )
                self._rows = [(*self._row_tuple(r), d) for r, d in scored[:limit]]
                return
            if "AS lexical_" in flat:
                tokens = {t.lower() for t in args.pop(0).split()}
                args.pop(0)  # the WHERE MATCH(...) copy of the query text
                rows = self._apply_filters(flat, args)
                limit = args.pop(0)
                hits = []
                for r in rows:
                    words = set(re.findall(r"\w+", r["content"].lower()))
                    score = float(len(tokens & words))
                    if score > 0:
                        hits.append((r, score))
                hits.sort(key=lambda p: p[1], reverse=True)
                self._rows = [(*self._row_tuple(r), s) for r, s in hits[:limit]]
                return
            rows = self._apply_filters(flat, args)
            limit = args.pop(0)
            self._rows = [self._row_tuple(r) for r in rows[:limit]]
            return
        raise AssertionError(f"fake server does not understand: {flat[:80]}")

    def executemany(self, sql: str, seq: list[list[Any]]) -> None:
        if self._conn.closed:
            raise _FakeMySQLError(2006, "MySQL server has gone away")
        self._server.statements.append((sql, tuple(seq)))
        assert sql.startswith("INSERT INTO"), sql
        for doc_id, content, content_type, vector, metadata, created in seq:
            self._server.rows[doc_id] = {
                "id": doc_id,
                "content": content,
                "content_type": content_type,
                "embedding": _vec(vector),
                "metadata": json.loads(metadata),
                "created_at": created,
            }
        self.rowcount = len(seq)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)

    def close(self) -> None:
        self.closed = True


class _FakeConnection:
    def __init__(self, server: _FakeServer, **kwargs: Any) -> None:
        self.server = server
        self.kwargs = kwargs
        self.closed = False
        self.commits = 0
        self.rollbacks = 0
        self.healthy = True
        self.rollback_error: Exception | None = None
        server.connections.append(self)
        server.connect_kwargs.append(kwargs)

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        if self.rollback_error is not None:
            raise self.rollback_error
        self.rollbacks += 1

    def is_connected(self) -> bool:
        return self.healthy and not self.closed

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> _FakeServer:
    fake = _FakeServer()
    connector = types.ModuleType("mysql.connector")
    connector.connect = lambda **kwargs: _FakeConnection(fake, **kwargs)  # type: ignore[attr-defined]
    connector.Error = _FakeMySQLError  # type: ignore[attr-defined]
    package = types.ModuleType("mysql")
    package.connector = connector  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mysql", package)
    monkeypatch.setitem(sys.modules, "mysql.connector", connector)
    return fake


def _store(**kwargs: Any) -> HeatWaveVectorStore:
    kwargs.setdefault("dimension", DIMENSION)
    kwargs.setdefault("database", "kb")
    kwargs.setdefault("table_name", "t")
    return HeatWaveVectorStore(**kwargs)


def _doc(doc_id: str, vector: list[float], content: str = "", **metadata: Any) -> Document:
    return Document(
        id=doc_id, content=content or f"content {doc_id}", embedding=vector, metadata=metadata
    )


# ------------------------------------------------------------- conformance


class TestHeatWaveServerSideConformance(VectorStoreConformance):
    """The shared contract on a fake HeatWave server (``DISTANCE()`` on the server)."""

    supports_hybrid = True

    @pytest.fixture(autouse=True)
    def _fake(self, server: _FakeServer) -> None:
        self._server = server

    async def make_store(self) -> HeatWaveVectorStore:
        return _store()

    async def test_server_side_path_was_taken(self, store: HeatWaveVectorStore) -> None:
        await store.add(_doc("a", VEC_A))
        await store.search(VEC_A, limit=1)
        assert store.server_side_distance is True
        assert any("AS distance_" in sql for sql, _ in self._server.statements)


class TestHeatWaveClientSideConformance(VectorStoreConformance):
    """The same contract on a fake community server (no ``DISTANCE()``)."""

    supports_hybrid = True

    @pytest.fixture(autouse=True)
    def _fake(self, server: _FakeServer) -> None:
        server.has_distance = False
        self._server = server

    async def make_store(self) -> HeatWaveVectorStore:
        return _store()

    async def test_client_side_path_was_taken(self, store: HeatWaveVectorStore) -> None:
        await store.add(_doc("a", VEC_A))
        await store.search(VEC_A, limit=1)
        assert store.server_side_distance is False
        assert not any("AS distance_" in sql for sql, _ in self._server.statements)
        # The candidate scan is bounded.
        scans = [
            p
            for sql, p in self._server.statements
            if sql.endswith("LIMIT %s") and "WHERE" not in sql
        ]
        assert scans
        assert scans[-1][-1] == store.heatwave_config.candidate_limit


# ------------------------------------------------------------------ config


def test_config_rejects_bad_identifiers_metric_and_bounds() -> None:
    with pytest.raises(ValueError, match="table_name"):
        _store(table_name="bad name")
    with pytest.raises(ValueError, match="database"):
        _store(database="kb;DROP")
    with pytest.raises(ValueError, match="distance_metric"):
        _store(distance_metric="MANHATTAN")
    with pytest.raises(ValueError, match="dimension"):
        _store(dimension=0)
    with pytest.raises(ValueError, match="candidate_limit"):
        _store(candidate_limit=0)
    with pytest.raises(ValueError, match="pool_size"):
        _store(pool_size=0)
    with pytest.raises(ValueError, match="extra"):
        _store(namespace="oops")


def test_config_property_and_repr() -> None:
    store = _store(distance_metric="euclidean", server_side_distance=True)
    assert store.config.dimension == DIMENSION
    assert store.config.distance_metric == "l2"
    assert store.config.index_type == "hnsw"
    assert _store(distance_metric="DOT").config.distance_metric == "dot_product"
    assert _store().config.index_type == "flat"
    assert repr(_store(host="db.internal")) == (
        "HeatWaveVectorStore(host='db.internal', database='kb', table='t')"
    )
    assert repr(_store(dsn="mysql://u:p@h/kb")) == "HeatWaveVectorStore(dsn=..., table='t')"
    assert isinstance(HeatWaveVectorConfig(), HeatWaveVectorConfig)


def test_parse_dsn() -> None:
    cfg = _parse_dsn("mysql+connector://us%40er:p%23ss@db.internal:3307/kb?ssl_disabled=true")
    assert cfg == {
        "host": "db.internal",
        "port": 3307,
        "user": "us@er",
        "password": "p#ss",
        "database": "kb",
        "ssl_disabled": "true",
    }
    assert _parse_dsn("mysql://localhost/") == {"host": "localhost"}
    with pytest.raises(ValueError, match="scheme"):
        _parse_dsn("postgres://x")


async def test_connect_kwargs_from_fields_dsn_ssl_and_connect_args(server: _FakeServer) -> None:
    store = _store(
        host="db.internal",
        port=3307,
        user="locus",
        password="pw",  # noqa: S106 — fake credential for the fake driver
        ssl_ca="/certs/ca.pem",
        connect_args={"connection_timeout": 5, "collation": "utf8mb4_bin"},
    )
    await store.count()
    kwargs = server.connect_kwargs[-1]
    assert kwargs["host"] == "db.internal"
    assert kwargs["port"] == 3307
    assert kwargs["user"] == "locus"
    assert kwargs["password"] == "pw"  # noqa: S105 — fake credential
    assert kwargs["database"] == "kb"
    assert kwargs["ssl_ca"] == "/certs/ca.pem"
    assert kwargs["autocommit"] is False
    assert kwargs["charset"] == "utf8mb4"
    assert kwargs["collation"] == "utf8mb4_bin"  # connect_args win
    assert kwargs["connection_timeout"] == 5

    store = _store(dsn="mysql://u:p@h:3308/other")
    await store.count()
    kwargs = server.connect_kwargs[-1]
    assert (kwargs["host"], kwargs["port"], kwargs["user"], kwargs["database"]) == (
        "h",
        3308,
        "u",
        "other",
    )


def test_missing_driver_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mysql.connector", None)
    with pytest.raises(ImportError, match=r"locus-sdk\[heatwave\]"):
        asyncio.run(_store().count())


# ----------------------------------------------------------------- filters


def test_build_filter_sql_binds_paths_and_values() -> None:
    sql, params = build_filter_sql({"topic": "x", "year": 2026, "flag": True, 'we"ird': None})
    assert sql == " AND ".join(["JSON_EXTRACT(`metadata`, %s) = CAST(%s AS JSON)"] * 4)
    assert params == [
        '$."topic"',
        '"x"',
        '$."year"',
        "2026",
        '$."flag"',
        "true",
        '$."we\\"ird"',
        "null",
    ]
    assert build_filter_sql(None) == ("", [])
    assert build_filter_sql({}) == ("", [])
    with pytest.raises(TypeError, match="scalars"):
        build_filter_sql({"tags": ["a"]})
    with pytest.raises(TypeError, match="keys"):
        build_filter_sql({"": 1})
    with pytest.raises(ValueError, match="metadata_column"):
        build_filter_sql({"a": 1}, metadata_column="bad col")


# ------------------------------------------------------------- SQL shapes


async def test_table_ddl_and_probe(server: _FakeServer) -> None:
    store = _store()
    await store.ensure_ready()
    ddl = server.statements[0][0]
    assert ddl.startswith("CREATE TABLE IF NOT EXISTS `t`")
    assert f"embedding VECTOR({DIMENSION}) NOT NULL" in ddl
    assert "metadata JSON NOT NULL" in ddl
    assert "FULLTEXT KEY `ft_t_content` (content)" in ddl
    assert server.statements[1][0].startswith("SELECT DISTANCE(")
    assert store.server_side_distance is True
    # Ready state is cached: another call issues nothing.
    await store.ensure_ready()
    assert len(server.statements) == 2


async def test_no_fulltext_and_no_auto_create(server: _FakeServer) -> None:
    store = _store(fulltext=False, auto_create_table=False, server_side_distance=False)
    await store.count()
    assert not any(sql.startswith("CREATE TABLE") for sql, _ in server.statements)
    assert not any(sql.startswith("SELECT DISTANCE(") for sql, _ in server.statements)
    assert store.hybrid_search_available is False
    assert supports_hybrid_search(store) is False
    with pytest.raises(RuntimeError, match="FULLTEXT"):
        await store.hybrid_search("q", VEC_A)
    store = _store(fulltext=False)
    await store.ensure_ready()
    assert "FULLTEXT" not in server.statements[-2][0]


async def test_upsert_statement_shape_and_params(server: _FakeServer) -> None:
    store = _store()
    created = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    naive = created.replace(tzinfo=None)
    doc = Document(id="a", content="alpha", embedding=VEC_A, metadata={"k": 1}, created_at=created)
    assert await store.add(doc) == "a"
    sql, rows = next((s, p) for s, p in server.statements if s.startswith("INSERT INTO"))
    assert "VALUES (%s, %s, %s, STRING_TO_VECTOR(%s), %s, %s) AS new" in sql
    assert "ON DUPLICATE KEY UPDATE content = new.content" in sql
    assert rows == (["a", "alpha", "text", "[1.0,0.0,0.0,0.0]", '{"k": 1}', naive],)
    assert server.connections[0].commits == 2  # CREATE TABLE + the insert


async def test_add_rejects_missing_or_wrong_sized_embedding(server: _FakeServer) -> None:
    store = _store()
    with pytest.raises(ValueError, match="embedding"):
        await store.add(Document(id="x", content="no vector"))
    with pytest.raises(ValueError, match="dimensions"):
        await store.add(_doc("x", [1.0, 0.0]))
    assert not server.statements  # rejected before touching the server


async def test_search_sql_server_side(server: _FakeServer) -> None:
    store = _store()
    await store.add_batch([_doc("a", VEC_A, topic="x"), _doc("b", VEC_B, topic="y")])
    results = await store.search(VEC_A, limit=5, metadata_filter={"topic": "x"})
    assert [r.document.id for r in results] == ["a"]
    assert results[0].distance == pytest.approx(0.0)
    assert results[0].score == pytest.approx(1.0)
    sql, params = server.statements[-1]
    assert "DISTANCE(embedding, STRING_TO_VECTOR(%s), 'COSINE') AS distance_" in sql
    assert "WHERE JSON_EXTRACT(`metadata`, %s) = CAST(%s AS JSON)" in sql
    assert sql.endswith("ORDER BY distance_ ASC LIMIT %s")
    assert params == ("[1.0,0.0,0.0,0.0]", '$."topic"', '"x"', 5)


async def test_search_sql_client_side(server: _FakeServer) -> None:
    server.has_distance = False
    store = _store(candidate_limit=7)
    await store.add_batch([_doc("a", VEC_A), _doc("b", VEC_B), _doc("c", VEC_C)])
    results = await store.search(VEC_A, limit=2, threshold="0.5")
    assert [r.document.id for r in results] == ["a", "c"]
    assert results[0].distance == pytest.approx(0.0)
    assert results[1].score == pytest.approx(math.sqrt(0.5), abs=1e-4)
    sql, params = server.statements[-1]
    assert "DISTANCE" not in sql
    assert sql.endswith("FROM `t` LIMIT %s")
    assert params == (7,)
    # A non-numeric threshold string is ignored rather than raised.
    assert len(await store.search(VEC_A, limit=3, threshold="high")) == 3


async def test_detection_logs_once_and_honours_override(
    server: _FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    server.has_distance = False
    with caplog.at_level(logging.INFO, logger="locus.rag.stores.heatwave"):
        store = _store()
        await store.count()
        await store.count()
    assert sum("client-side" in r.message for r in caplog.records) == 1
    # Pinning the flag skips the probe and the DISTANCE() SQL is emitted even
    # though this server would reject it.
    pinned = _store(server_side_distance=True)
    assert pinned.server_side_distance is True
    with pytest.raises(_FakeMySQLError, match="DISTANCE"):
        await pinned.search(VEC_A)


async def test_probe_reraises_other_errors(server: _FakeServer) -> None:
    server.fail_next_execute = _FakeMySQLError(1045, "Access denied")
    store = _store(auto_create_table=False)
    with pytest.raises(_FakeMySQLError, match="Access denied"):
        await store.count()


@pytest.mark.parametrize(
    ("metric", "expected_distance", "expected_score"),
    [
        ("COSINE", 1.0 - math.sqrt(0.5), math.sqrt(0.5)),
        ("DOT", -math.sqrt(0.5), (1.0 + math.sqrt(0.5)) / 2.0),
        (
            "EUCLIDEAN",
            math.sqrt(2.0 - 2.0 * math.sqrt(0.5)),
            1.0 / (1.0 + math.sqrt(2.0 - 2.0 * math.sqrt(0.5))),
        ),
    ],
)
@pytest.mark.parametrize("has_distance", [True, False])
async def test_metrics_agree_on_both_paths(
    server: _FakeServer,
    metric: str,
    expected_distance: float,
    expected_score: float,
    has_distance: bool,
) -> None:
    server.has_distance = has_distance
    store = _store(distance_metric=metric)
    await store.add(_doc("c", VEC_C))
    [hit] = await store.search(VEC_A, limit=1)
    assert hit.distance == pytest.approx(expected_distance, abs=1e-4)
    assert hit.score == pytest.approx(expected_score, abs=1e-4)
    if has_distance:
        sql = server.statements[-1][0]
        assert f"'{metric}'" in sql
        # DISTANCE(..., 'DOT') is the dot product itself; the store negates it.
        assert ("-DISTANCE(" in sql) is (metric == "DOT")


def test_client_distance_edge_cases() -> None:
    store = _store()
    assert store._client_distance([0.0, 0.0, 0.0, 0.0], VEC_A) == 1.0
    with pytest.raises(ValueError, match="dimension mismatch"):
        store._client_distance([1.0], VEC_A)


async def test_hybrid_sql_and_blend(server: _FakeServer) -> None:
    store = _store()
    await store.add_batch(
        [
            _doc("a", VEC_A, "card refunds settle in five business days", topic="x"),
            _doc("b", VEC_B, "chargeback reason code for merchandise not received", topic="x"),
        ]
    )
    results = await store.hybrid_search(
        "chargeback merchandise", VEC_A, limit=2, alpha=0.5, metadata_filter={"topic": "x"}
    )
    assert {r.document.id for r in results} == {"a", "b"}
    assert results[0].score == pytest.approx(0.5)
    sql, params = server.statements[-1]
    assert "MATCH(content) AGAINST (%s IN NATURAL LANGUAGE MODE) AS lexical_" in sql
    assert "WHERE MATCH(content) AGAINST (%s IN NATURAL LANGUAGE MODE) AND JSON_EXTRACT(" in sql
    assert sql.endswith("ORDER BY lexical_ DESC LIMIT %s")
    assert params == ("chargeback merchandise", "chargeback merchandise", '$."topic"', '"x"', 20)
    # Blank query text is plain vector search.
    only_vector = await store.hybrid_search("   ", VEC_A, limit=1)
    assert [r.document.id for r in only_vector] == ["a"]
    assert "lexical_" not in server.statements[-1][0]
    with pytest.raises(ValueError, match="alpha"):
        await store.hybrid_search("q", VEC_A, alpha=-0.1)


async def test_get_delete_count_clear_and_close(server: _FakeServer) -> None:
    store = _store()
    assert await store.get("missing") is None
    await store.add(_doc("a", VEC_A, "alpha", topic="x"))
    got = await store.get("a")
    assert got is not None
    assert (got.id, got.content, got.metadata, got.content_type) == (
        "a",
        "alpha",
        {"topic": "x"},
        "text",
    )
    assert got.embedding == pytest.approx(VEC_A)
    assert got.created_at.tzinfo is UTC
    assert server.statements[-1] == (
        "SELECT id, content, content_type, VECTOR_TO_STRING(embedding), metadata, created_at FROM `t` WHERE id = %s",
        ("a",),
    )
    assert await store.count() == 1
    assert server.statements[-1] == ("SELECT COUNT(*) FROM `t`", None)
    assert await store.delete("a") is True
    assert server.statements[-1] == ("DELETE FROM `t` WHERE id = %s", ("a",))
    assert await store.delete("a") is False
    await store.add_batch([_doc("a", VEC_A), _doc("b", VEC_B)])
    assert await store.clear() == 2
    assert server.statements[-1] == ("DELETE FROM `t`", None)

    conn = server.connections[0]
    assert len(server.connections) == 1  # one pooled connection served everything
    await store.close()
    assert conn.closed is True
    # The store reconnects after close and re-checks the table.
    assert await store.count() == 0
    assert len(server.connections) == 2
    assert any(sql.startswith("CREATE TABLE") for sql, _ in server.statements[-3:])
    await store.close()
    await store.close()  # idempotent


def test_to_document_tolerates_driver_variants() -> None:
    row = ("a", b"bytes content", None, b"[1.0,0.0,0.0,0.0]", b'{"k": 1}', "not a datetime")
    doc = HeatWaveVectorStore._to_document(row)
    assert doc.content == "bytes content"
    assert doc.content_type == "text"
    assert doc.embedding == [1.0, 0.0, 0.0, 0.0]
    assert doc.metadata == {"k": 1}
    assert doc.created_at.tzinfo is UTC
    midnight = datetime(2026, 1, 1, tzinfo=UTC).replace(tzinfo=None)
    empty = HeatWaveVectorStore._to_document(("b", None, "text", "[]", None, midnight))
    assert empty.content == ""
    assert empty.embedding == []
    assert empty.metadata == {}
    assert empty.created_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert HeatWaveVectorStore._to_document(("c", "x", "text", None, "", None)).embedding is None


async def test_load_into_heatwave_statements(server: _FakeServer) -> None:
    store = _store()
    await store.load_into_heatwave()
    statements = [sql for sql, _ in server.statements]
    assert statements[-2:] == [
        "ALTER TABLE `t` SECONDARY_ENGINE = RAPID",
        "ALTER TABLE `t` SECONDARY_LOAD",
    ]


# -------------------------------------------------------------------- pool


def test_pool_reuses_discards_and_closes(server: _FakeServer) -> None:
    created: list[_FakeConnection] = []

    def connect() -> _FakeConnection:
        conn = _FakeConnection(server)
        created.append(conn)
        return conn

    pool = _SyncPool(connect, max_size=2)
    with pool.acquire() as c1:
        pass
    with pool.acquire() as c2:
        pass
    assert c1 is c2
    assert len(created) == 1
    assert c1.rollbacks == 2

    # An exception inside the block discards the connection.
    with pytest.raises(RuntimeError), pool.acquire() as c3:
        raise RuntimeError("boom")
    assert c3.closed is True
    # An unhealthy idle connection is replaced.
    with pool.acquire() as c4:
        pass
    c4.healthy = False
    with pool.acquire() as c5:
        pass
    assert c5 is not c4
    assert c4.closed is True
    # A connection that cannot roll back is not pooled.
    c5.rollback_error = _FakeMySQLError(2006, "gone away")
    with pool.acquire() as c6:
        pass
    assert c6 is c5
    assert c6.closed is True
    # Closing drains idle connections and refuses new work.
    with pool.acquire() as c7:
        pass
    pool.close()
    assert c7.closed is True
    with pytest.raises(RuntimeError, match="closed"), pool.acquire():
        pass


def test_pool_release_after_close_discards(server: _FakeServer) -> None:
    pool = _SyncPool(lambda: _FakeConnection(server), max_size=1)
    with pool.acquire() as conn:
        pool.close()
    assert conn.closed is True


def test_pool_discard_swallows_close_errors(server: _FakeServer) -> None:
    conn = _FakeConnection(server)

    def bad_close() -> None:
        raise OSError("socket")

    conn.close = bad_close  # type: ignore[method-assign]
    _SyncPool._discard(conn)  # no raise
