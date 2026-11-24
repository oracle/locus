# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Report stores and the trend across them.

Five stores, one interface: a directory, a SQLite file, an Oracle table, an OCI
bucket and an S3 bucket. These tests pin the round trip and the listing order
on each (the two local stores through one fixture, the others with doubles for
the clients), the spec strings that open them, and the arithmetic
of the trend: deltas against the previous point, a changed suite marked rather
than read as a regression, and the newest point's regression deciding the
verdict.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from locus.evaluation import (
    EvalCase,
    EvalReport,
    EvalResult,
    EvalRunner,
    FileReportStore,
    OCIBucketReportStore,
    OracleReportStore,
    ReportStore,
    S3ReportStore,
    SqliteReportStore,
    StoredReport,
    report_store,
    trend,
)
from locus.evaluation.stores import SPEC_ENV, trend_of
from locus.testing import ScriptedModel, text


def _report(
    *, passed: int, total: int, fingerprint: str | None = "abcdefabcdef", at: datetime | None = None
) -> EvalReport:
    results = [
        EvalResult(
            case_name=f"c{i}",
            passed=i < passed,
            score=1.0 if i < passed else 0.0,
            checks={"k": i < passed},
        )
        for i in range(total)
    ]
    report = EvalReport(
        results=results,
        total_cases=total,
        passed=passed,
        failed=total - passed,
        avg_score=passed / total if total else 0.0,
        fingerprint=fingerprint,
    )
    if at is not None:
        report.created_at = at
    return report


T0 = datetime(2026, 11, 16, 9, 0, tzinfo=UTC)


def _three_reports(store: ReportStore, suite: str = "warden") -> list[StoredReport]:
    return [
        store.put(_report(passed=3, total=3, at=T0), suite=suite),
        store.put(_report(passed=2, total=3, at=T0 + timedelta(hours=1)), suite=suite),
        store.put(
            _report(passed=3, total=3, fingerprint="123456789abc", at=T0 + timedelta(hours=2)),
            suite=suite,
        ),
    ]


# --------------------------------------------------------------------------
# The local stores: FileReportStore and SqliteReportStore, the same tests
# --------------------------------------------------------------------------


@pytest.fixture(params=["file", "sqlite"])
def local_store(request: pytest.FixtureRequest, tmp_path: Path) -> ReportStore:
    if request.param == "file":
        return FileReportStore(tmp_path)
    return SqliteReportStore(tmp_path / "evals.db")


def test_local_stores_name_their_spec(tmp_path: Path) -> None:
    assert FileReportStore(tmp_path).spec == f"file://{tmp_path}"
    store = SqliteReportStore(tmp_path / "evals.db")
    assert store.spec == f"sqlite://{tmp_path}/evals.db"
    assert (
        repr(store) == f"SqliteReportStore(path='{tmp_path}/evals.db', table='locus_eval_reports')"
    )


def test_local_store_round_trips_and_lists_newest_first(local_store: ReportStore) -> None:
    store = local_store
    stored = _three_reports(store)
    assert isinstance(store, ReportStore)
    assert stored[0].passed == 3
    assert stored[0].suite == "warden"
    loaded = store.get(stored[1].report_id)
    assert loaded is not None
    assert loaded.passed == 2
    assert loaded.fingerprint == "abcdefabcdef"
    assert store.get("missing") is None
    listed = store.list(suite="warden")
    assert [item.report_id for item in listed] == [s.report_id for s in reversed(stored)]
    assert listed[0].passed == 3
    assert listed[0].fingerprint == "123456789abc"
    assert listed[1].avg_score == pytest.approx(2 / 3)


def test_local_store_filters_by_suite_fingerprint_and_limit(local_store: ReportStore) -> None:
    store = local_store
    _three_reports(store)
    store.put(_report(passed=1, total=1, at=T0), suite="other suite/with slash")
    assert len(store.list()) == 4
    assert len(store.list(suite="warden")) == 3
    assert len(store.list(suite="warden", fingerprint="abcdefabcdef")) == 2
    assert len(store.list(fingerprint="123456789abc")) == 1
    assert len(store.list(suite="warden", limit=1)) == 1
    assert store.list(suite="warden", limit=0) == []
    assert len(store.list(suite="other suite/with slash")) == 1
    assert store.list(suite="nothing") == []


def test_a_suite_name_is_a_directory_in_the_file_store_and_itself_in_sqlite(
    tmp_path: Path,
) -> None:
    files = FileReportStore(tmp_path)
    files.put(_report(passed=1, total=1, at=T0), suite="other suite/with slash")
    assert files.list(suite="other suite/with slash")[0].suite == "other-suite-with-slash"
    assert (tmp_path / "other-suite-with-slash").is_dir()
    database = SqliteReportStore(tmp_path / "evals.db")
    database.put(_report(passed=1, total=1, at=T0), suite="other suite/with slash")
    assert database.list()[0].suite == "other suite/with slash"


def test_file_store_ignores_files_that_are_not_reports(tmp_path: Path) -> None:
    (tmp_path / "warden").mkdir()
    (tmp_path / "warden" / "notes.json").write_text("{}")
    store = FileReportStore(tmp_path)
    assert store.list(suite="warden") == []
    assert FileReportStore(tmp_path / "absent").list() == []


def test_a_report_the_runner_made_round_trips_with_its_results(local_store: ReportStore) -> None:
    from locus.agent import Agent

    report = EvalRunner(agent=Agent(model=ScriptedModel([text("yes")]))).run(
        [EvalCase(name="c", prompt="p", expected_output_contains=["yes"])]
    )
    store = local_store
    stored = store.put(report, suite="s")
    loaded = store.get(stored.report_id)
    assert loaded is not None
    assert loaded.results[0].checks == report.results[0].checks
    assert loaded.fingerprint == report.fingerprint


# --------------------------------------------------------------------------
# OCI and S3 with client doubles
# --------------------------------------------------------------------------


class _OCIClient:
    """Enough of ``ObjectStorageClient`` for the store: put, get, paginated list."""

    def __init__(self, page: int = 2) -> None:
        self.objects: dict[str, bytes] = {}
        self.page = page

    def put_object(
        self,
        *,
        namespace_name: str,
        bucket_name: str,
        object_name: str,
        put_object_body: bytes,
        content_type: str,
    ) -> None:
        assert content_type == "application/json"
        self.objects[object_name] = put_object_body

    def get_object(self, *, namespace_name: str, bucket_name: str, object_name: str) -> Any:
        if object_name not in self.objects:
            raise RuntimeError("ObjectNotFound: no such object")
        return type("R", (), {"data": type("D", (), {"content": self.objects[object_name]})()})()

    def list_objects(
        self,
        *,
        namespace_name: str,
        bucket_name: str,
        prefix: str,
        fields: str,
        limit: int,
        start: str | None = None,
    ) -> Any:
        names = sorted(n for n in self.objects if n.startswith(prefix))
        if start:
            names = [n for n in names if n >= start]
        page, rest = names[: self.page], names[self.page :]
        data = type(
            "L",
            (),
            {
                "objects": [type("O", (), {"name": n})() for n in page],
                "next_start_with": rest[0] if rest else None,
            },
        )()
        return type("R", (), {"data": data})()


def test_oci_bucket_store_round_trips_through_a_paginated_listing() -> None:
    client = _OCIClient(page=2)
    store = OCIBucketReportStore("evals", "ns", prefix="locus/eval/", _client=client)
    stored = _three_reports(store)
    assert store.spec == "oci://ns/evals/locus/eval/"
    assert stored[0].location.startswith("locus/eval/warden/20261116T090000000000Z-abcdefabcdef-")
    listed = store.list(suite="warden")
    assert [item.passed for item in listed] == [3, 2, 3]
    assert listed[0].created_at == T0 + timedelta(hours=2)
    loaded = store.get(stored[1].report_id)
    assert loaded is not None
    assert loaded.passed == 2
    assert store.get("0" * 32) is None
    assert len(client.objects) == 3


class _S3Client:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:  # noqa: N803
        self.objects[Key] = Body

    def get_object(self, *, Bucket: str, Key: str) -> Any:  # noqa: N803
        if Key not in self.objects:
            error = RuntimeError("NoSuchKey")
            error.response = {"Error": {"Code": "NoSuchKey"}}  # type: ignore[attr-defined]
            raise error
        body = self.objects[Key]
        return {"Body": type("B", (), {"read": lambda self: body})()}

    def list_objects_v2(self, **kwargs: Any) -> Any:
        bucket_prefix, token = kwargs["Prefix"], kwargs.get("ContinuationToken")
        names = sorted(n for n in self.objects if n.startswith(bucket_prefix))
        if token:
            names = [n for n in names if n > token]
        page, rest = names[:2], names[2:]
        return {
            "Contents": [{"Key": n} for n in page],
            "IsTruncated": bool(rest),
            "NextContinuationToken": page[-1] if rest else None,
        }


def test_s3_store_round_trips_through_a_paginated_listing() -> None:
    client = _S3Client()
    store = S3ReportStore("evals", prefix="evals/", _client=client)
    stored = _three_reports(store)
    assert store.spec == "s3://evals/evals/"
    listed = store.list(suite="warden", fingerprint="abcdefabcdef")
    assert [item.passed for item in listed] == [2, 3]
    loaded = store.get(stored[2].report_id)
    assert loaded is not None
    assert loaded.fingerprint == "123456789abc"
    assert store.get("0" * 32) is None


# --------------------------------------------------------------------------
# Oracle with a connection double
# --------------------------------------------------------------------------


class _OracleDB:
    """An in-memory stand-in for the one table the store uses."""

    def __init__(self, *, table_exists: bool = False) -> None:
        self.table_exists = table_exists
        self.rows: list[dict[str, Any]] = []
        self.statements: list[tuple[str, dict[str, Any] | None]] = []
        self.commits = 0


class _OracleCursor:
    def __init__(self, db: _OracleDB) -> None:
        self.db = db
        self._result: list[tuple[Any, ...]] = []

    def __enter__(self) -> _OracleCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, binds: dict[str, Any] | None = None) -> None:
        self.db.statements.append((sql, binds))
        head = sql.strip().upper()
        binds = binds or {}
        if head.startswith("SELECT COUNT(*) FROM ALL_TABLES"):
            self._result = [(1 if self.db.table_exists else 0,)]
        elif head.startswith("CREATE TABLE"):
            self.db.table_exists = True
        elif head.startswith("CREATE INDEX"):
            pass
        elif head.startswith("INSERT INTO"):
            self.db.rows.append(dict(binds))
        elif head.startswith("SELECT REPORT FROM"):
            self._result = [(r["report"],) for r in self.db.rows if r["report_id"] == binds["id"]]
        elif head.startswith("SELECT REPORT_ID, SUITE"):
            rows = [
                r
                for r in self.db.rows
                if ("suite" not in binds or r["suite"] == binds["suite"])
                and ("fingerprint" not in binds or r["fingerprint"] == binds["fingerprint"])
            ]
            rows.sort(key=lambda r: (r["created_at"], r["report_id"]), reverse=True)
            self._result = [
                (
                    r["report_id"],
                    r["suite"],
                    r["fingerprint"],
                    r["created_at"],
                    r["total_cases"],
                    r["passed"],
                    r["failed"],
                    r["not_graded"],
                    r["avg_score"],
                )
                for r in rows[: binds["limit"]]
            ]
        else:  # pragma: no cover - a statement the store does not issue
            raise AssertionError(sql)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._result[0] if self._result else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._result)


class _OracleConnection:
    def __init__(self, db: _OracleDB) -> None:
        self.db = db

    def cursor(self) -> _OracleCursor:
        return _OracleCursor(self.db)

    def commit(self) -> None:
        self.db.commits += 1


def _oracle_store(db: _OracleDB, **kwargs: Any) -> OracleReportStore:
    return OracleReportStore(
        dsn="evals_low",
        user="EVALS",
        password="pw",  # noqa: S106 — a double's password
        table_name="locus_eval_reports",
        _connect=lambda: _OracleConnection(db),
        **kwargs,
    )


def test_oracle_store_creates_the_table_once_and_round_trips() -> None:
    db = _OracleDB()
    store = _oracle_store(db)
    assert store.spec == "oracle://EVALS@/?dsn=evals_low&table=locus_eval_reports"
    assert "pw" not in store.spec
    stored = _three_reports(store)
    creates = [s for s, _ in db.statements if s.strip().upper().startswith("CREATE TABLE")]
    assert len(creates) == 1
    assert "created_at TIMESTAMP NOT NULL" in creates[0]
    assert db.commits >= 3
    inserted = db.rows[0]
    assert inserted["suite"] == "warden"
    assert inserted["fingerprint"] == "abcdefabcdef"
    assert inserted["created_at"] == datetime(2026, 11, 16, 9, 0)  # noqa: DTZ001 — UTC with no zone, as stored
    assert inserted["created_at"].tzinfo is None
    listed = store.list(suite="warden")
    assert [item.passed for item in listed] == [3, 2, 3]
    assert listed[0].created_at == T0 + timedelta(hours=2)
    assert listed[0].created_at.tzinfo is UTC
    assert listed[0].location == f"locus_eval_reports:{stored[2].report_id}"
    loaded = store.get(stored[1].report_id)
    assert loaded is not None
    assert loaded.passed == 2
    assert store.get("missing") is None


def test_oracle_store_filters_with_bind_variables_not_string_building() -> None:
    db = _OracleDB(table_exists=True)
    store = _oracle_store(db)
    _three_reports(store)
    store.list(suite="warden", fingerprint="abcdefabcdef", limit=5)
    sql, binds = db.statements[-1]
    assert "suite = :suite" in sql
    assert "fingerprint = :fingerprint" in sql
    assert binds == {"suite": "warden", "fingerprint": "abcdefabcdef", "limit": 5}
    assert "warden" not in sql


def test_oracle_store_reads_a_clob_report_back() -> None:
    db = _OracleDB(table_exists=True)
    store = _oracle_store(db)
    stored = store.put(_report(passed=1, total=1), suite="s")
    body = db.rows[0]["report"]
    db.rows[0]["report"] = type("LOB", (), {"read": lambda self: body})()
    loaded = store.get(stored.report_id)
    assert loaded is not None
    assert loaded.passed == 1


def test_oracle_store_refuses_to_run_without_the_table_when_not_allowed_to_create_it() -> None:
    store = _oracle_store(_OracleDB(), auto_create_table=False)
    with pytest.raises(RuntimeError, match="does not exist"):
        store.put(_report(passed=1, total=1), suite="s")


def test_oracle_store_needs_oracledb_for_a_real_pool() -> None:
    store = OracleReportStore(dsn="x", user="u", password="p")  # noqa: S106
    assert "oracle://u@/?dsn=x" in store.spec
    assert repr(store) == "OracleReportStore(table='locus_eval_reports')"
    assert "locus_eval_reports" in store.ddl()
    store.close()


# --------------------------------------------------------------------------
# Opening a store from a spec
# --------------------------------------------------------------------------


def test_report_store_opens_each_scheme(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(report_store(str(tmp_path)), FileReportStore)
    assert isinstance(report_store(f"file://{tmp_path}"), FileReportStore)
    sqlite = report_store(f"sqlite://{tmp_path}/evals.db")
    assert isinstance(sqlite, SqliteReportStore)
    assert sqlite.path == tmp_path / "evals.db"
    assert sqlite.table == "locus_eval_reports"
    monkeypatch.chdir(tmp_path)
    relative = report_store("sqlite://history/evals.db?table=nightly")
    assert isinstance(relative, SqliteReportStore)
    assert (relative.path, relative.table) == (Path("history/evals.db"), "nightly")
    relative.put(_report(passed=1, total=1, at=T0), suite="warden")
    assert (tmp_path / "history" / "evals.db").is_file()
    assert report_store(relative.spec).list() == [], "the spec names the file, not the table"
    oci = report_store(
        "oci://ns/bucket/some/prefix?profile=P&region=eu-frankfurt-1&auth=security_token"
    )
    assert isinstance(oci, OCIBucketReportStore)
    assert (oci.namespace, oci.bucket_name, oci.prefix) == ("ns", "bucket", "some/prefix/")
    assert oci._profile_name == "P"
    assert oci._region == "eu-frankfurt-1"
    assert oci._auth_type == "security_token"
    s3 = report_store("s3://bucket/prefix?endpoint_url=https://objects.example&region=r1")
    assert isinstance(s3, S3ReportStore)
    assert (s3.bucket, s3.prefix, s3._endpoint_url, s3._region_name) == (
        "bucket",
        "prefix/",
        "https://objects.example",
        "r1",
    )
    oracle = report_store("oracle://u:pw@/?dsn=db_low&wallet=/w&table=ops.reports")
    assert isinstance(oracle, OracleReportStore)
    assert oracle.config.dsn == "db_low"
    assert oracle.config.user == "u"
    assert oracle.config.wallet_location == "/w"
    assert (oracle.config.schema_name, oracle.config.table_name) == ("ops", "reports")
    hosted = report_store("oracle://u:pw@db.example:1522/svc")
    assert isinstance(hosted, OracleReportStore)
    assert (hosted.config.host, hosted.config.port, hosted.config.service_name) == (
        "db.example",
        1522,
        "svc",
    )


def test_report_store_reads_the_environment_for_a_bare_oracle_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "ORACLE_DSN",
        "ORACLE_USER",
        "ORACLE_PASSWORD",
        "ORACLE_WALLET",
        "ORACLE_WALLET_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="dsn="):
        report_store("oracle://")
    monkeypatch.setenv("ORACLE_DSN", "env_low")
    monkeypatch.setenv("ORACLE_USER", "ENV")
    monkeypatch.setenv("ORACLE_WALLET", "/env/wallet")
    monkeypatch.setenv("ORACLE_WALLET_PASSWORD", "wp")
    store = report_store("oracle://?table=LOCUS_EVAL_REPORTS")
    assert isinstance(store, OracleReportStore)
    assert store.config.dsn == "env_low"
    assert store.config.user == "ENV"
    assert store.config.wallet_location == "/env/wallet"
    assert store.config.wallet_password is not None
    assert store.config.table_name == "LOCUS_EVAL_REPORTS"


def test_report_store_rejects_what_it_cannot_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SPEC_ENV, raising=False)
    with pytest.raises(ValueError, match="no report store"):
        report_store(None)
    with pytest.raises(ValueError, match="unknown report store scheme"):
        report_store("ftp://x")
    with pytest.raises(ValueError, match="oci://<namespace>"):
        report_store("oci://ns")
    with pytest.raises(ValueError, match="s3://<bucket>"):
        report_store("s3:///prefix")
    monkeypatch.setenv(SPEC_ENV, "s3://from-env/p")
    assert isinstance(report_store(None), S3ReportStore)


# --------------------------------------------------------------------------
# The trend
# --------------------------------------------------------------------------


def test_trend_reads_the_last_reports_oldest_first_with_deltas(local_store: ReportStore) -> None:
    store = local_store
    _three_reports(store)
    result = trend(store, suite="warden", last=10)
    assert [p.passed for p in result.points] == [3, 2, 3]
    assert result.points[0].pass_delta is None
    assert result.points[1].pass_delta == -1
    assert result.points[1].avg_score_delta == pytest.approx(-1 / 3, abs=1e-4)
    assert result.points[1].regressed
    assert result.points[2].pass_delta == 1
    assert result.points[2].suite_changed
    assert not result.points[2].regressed
    assert not result.regressed
    text = result.summary()
    assert "3 report(s), oldest first" in text
    assert "(-1 passed, -0.33 avg)" in text
    assert "suite changed" in text
    markdown = result.to_markdown()
    assert markdown.count("| 2026-11-16") == 3
    assert "`123456789abc`" in markdown
    assert "suite changed" in markdown


def test_trend_narrowed_to_one_fingerprint_and_the_latest_regression(
    local_store: ReportStore,
) -> None:
    store = local_store
    _three_reports(store)
    result = trend(store, suite="warden", fingerprint="abcdefabcdef")
    assert [p.passed for p in result.points] == [3, 2]
    assert result.regressed
    assert "worse than the one before it" in result.summary()
    assert trend(store, suite="warden", last=1).points[0].pass_delta is None
    assert trend(store, suite="nothing").summary() == "Trend for nothing: no reports"


def test_trend_of_orders_rows_itself_and_reads_not_graded() -> None:
    rows = [
        StoredReport(
            report_id="b",
            suite="s",
            created_at=T0 + timedelta(hours=1),
            total_cases=4,
            passed=2,
            not_graded=2,
            avg_score=1.0,
        ),
        StoredReport(
            report_id="a", suite="s", created_at=T0, total_cases=4, passed=4, avg_score=1.0
        ),
    ]
    result = trend_of(rows, suite="s")
    assert [p.report_id for p in result.points] == ["a", "b"]
    assert result.points[1].pass_rate == 1.0
    assert result.points[1].pass_delta == -2
    assert result.latest is not None
    assert result.latest.regressed


def test_a_regression_is_not_read_across_a_suite_change() -> None:
    rows = [
        StoredReport(
            report_id="a",
            suite="s",
            fingerprint="aaaaaaaaaaaa",
            created_at=T0,
            total_cases=3,
            passed=3,
            avg_score=1.0,
        ),
        StoredReport(
            report_id="b",
            suite="s",
            fingerprint="bbbbbbbbbbbb",
            created_at=T0 + timedelta(hours=1),
            total_cases=5,
            passed=2,
            avg_score=0.4,
        ),
    ]
    result = trend_of(rows, suite="s")
    assert result.points[1].suite_changed
    assert result.points[1].pass_delta == -1
    assert not result.regressed


def test_stored_json_is_plain_and_readable(tmp_path: Path) -> None:
    store = FileReportStore(tmp_path)
    stored = store.put(_report(passed=1, total=1, at=T0), suite="warden")
    path = tmp_path / stored.location
    data = json.loads(path.read_text())
    assert data["passed"] == 1
    assert data["fingerprint"] == "abcdefabcdef"
    assert os.sep not in stored.location or os.sep == "/"


# --------------------------------------------------------------------------
# The clients the stores build themselves, and the error paths around them
# --------------------------------------------------------------------------


def test_name_helpers_reject_what_is_not_a_report_name() -> None:
    from locus.evaluation.stores import _parse_name, _utc_naive

    aware = datetime(2026, 11, 16, 9, 0, tzinfo=UTC)
    naive = aware.replace(tzinfo=None)
    assert _utc_naive(aware) == naive
    assert _utc_naive(naive) == naive
    assert _parse_name("elsewhere/x.json", prefix="locus/eval/") is None
    assert _parse_name("locus/eval/no-suite.json", prefix="locus/eval/") is None
    assert _parse_name("locus/eval/warden/notes.txt", prefix="locus/eval/") is None


def test_listing_skips_an_object_that_vanished_between_list_and_read(tmp_path: Path) -> None:
    store = FileReportStore(tmp_path)
    first, second, _third = _three_reports(store)
    (tmp_path / second.location).unlink()
    listed = store.list(suite="warden")
    assert [item.report_id for item in listed] == [_third.report_id, first.report_id]


def test_oci_bucket_store_builds_its_client_from_the_bucket_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from locus.memory.backends import oci_bucket

    built: list[Any] = []
    client = _OCIClient()

    def build(config: Any) -> Any:
        built.append(config)
        return client

    monkeypatch.setattr(oci_bucket, "build_object_storage_client", build)
    store = OCIBucketReportStore(
        "evals", "ns", profile_name="EVALS", auth_type="instance_principal", region="eu-x-1"
    )
    stored = store.put(_report(passed=1, total=1, at=T0), suite="warden")
    assert store.get(stored.report_id) is not None
    assert store.get("missing") is None
    (config,) = built
    assert (config.bucket_name, config.namespace, config.profile_name) == ("evals", "ns", "EVALS")
    assert (config.auth_type, config.region) == ("instance_principal", "eu-x-1")

    class _Broken(_OCIClient):
        def get_object(self, **kwargs: Any) -> Any:
            raise RuntimeError("ServiceUnavailable")

    with pytest.raises(RuntimeError, match="ServiceUnavailable"):
        OCIBucketReportStore("evals", "ns", _client=_Broken())._get_bytes("locus/eval/x.json")


def test_s3_store_builds_a_boto3_client_and_maps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    calls: list[tuple[str, dict[str, Any]]] = []
    client = _S3Client()

    def make_client(service: str, **kwargs: Any) -> Any:
        calls.append((service, kwargs))
        return client

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=make_client))
    store = S3ReportStore("evals", endpoint_url="http://object-store.local", region_name="r1")
    stored = store.put(_report(passed=1, total=1, at=T0), suite="warden")
    assert store.get(stored.report_id) is not None
    assert store.get("missing") is None
    assert calls == [
        (
            "s3",
            {
                "endpoint_url": "http://object-store.local",
                "region_name": "r1",
                "aws_access_key_id": None,
                "aws_secret_access_key": None,
            },
        )
    ]

    class _Broken(_S3Client):
        def get_object(self, **kwargs: Any) -> Any:
            raise RuntimeError("AccessDenied")

    with pytest.raises(RuntimeError, match="AccessDenied"):
        S3ReportStore("evals", _client=_Broken())._get_bytes("locus/eval/x.json")

    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(ImportError, match=r"locus-sdk\[s3\]"):
        S3ReportStore("evals").get("x")


class _DeadError(Exception):
    """The driver error a dropped socket raises."""


class _Pool:
    """A pool whose connections come from one in-memory table; the first can be dead."""

    def __init__(self, db: _OracleDB, *, dead_first: bool = False) -> None:
        self.db = db
        self.dead = dead_first
        self.closed: list[bool] = []

    def acquire(self) -> Any:
        pool = self

        class _Acquired:
            def __enter__(self) -> Any:
                if pool.dead:
                    pool.dead = False
                    raise _DeadError("DPY-4011: the database or network closed the connection")
                return _OracleConnection(pool.db)

            def __exit__(self, *exc: object) -> None:
                return None

        return _Acquired()

    def close(self, force: bool = False) -> None:
        self.closed.append(force)
        raise RuntimeError("already closed")  # the store treats teardown as best effort


def _fake_oracledb(monkeypatch: pytest.MonkeyPatch, pools: list[_Pool]) -> list[dict[str, Any]]:
    import sys
    import types

    created: list[dict[str, Any]] = []

    def create_pool(**kwargs: Any) -> _Pool:
        created.append(kwargs)
        return pools.pop(0)

    module = types.SimpleNamespace(
        makedsn=lambda host, port, service_name: f"{host}:{port}/{service_name}",
        create_pool=create_pool,
        DB_TYPE_CLOB="CLOB",
    )
    monkeypatch.setitem(sys.modules, "oracledb", module)
    return created


def test_oracle_store_pools_reconnects_once_and_closes_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import locus._oracle_pool_cache as pool_cache

    monkeypatch.setattr(pool_cache, "is_reconnectable", lambda exc: isinstance(exc, _DeadError))
    db = _OracleDB()
    pools = [_Pool(db, dead_first=True), _Pool(db)]
    created = _fake_oracledb(monkeypatch, pools)
    store = OracleReportStore(
        host="db.internal",
        port=1522,
        service_name="evals_low",
        user="EVALS",
        password="pw",  # noqa: S106 — a double's password
        wallet_location="/wallets/evals",
        wallet_password="wpw",  # noqa: S106 — a double's password
        schema_name="EVALS",
    )
    assert store._full_table_name == "EVALS.locus_eval_reports"
    stored = store.put(_report(passed=1, total=1, at=T0), suite="warden")
    # The first pool died on acquire; the store rebuilt it and ran the insert on the second.
    assert len(created) == 2
    assert created[0]["dsn"] == "db.internal:1522/evals_low"
    assert created[0]["config_dir"] == "/wallets/evals"
    assert created[0]["wallet_password"] == "wpw"  # noqa: S105 — a double's password
    assert store.get(stored.report_id) is not None
    assert pools == []
    store.close()
    assert store._pool is None

    def never(_exc: BaseException) -> bool:
        return False

    monkeypatch.setattr(pool_cache, "is_reconnectable", never)
    broken = OracleReportStore(dsn="x", user="u", password="p")  # noqa: S106
    _fake_oracledb(monkeypatch, [_Pool(_OracleDB(), dead_first=True)])
    with pytest.raises(_DeadError):
        broken.get("anything")


def test_oracle_store_writes_a_large_report_as_a_lob_and_reads_bytes_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    class _Lob:
        def __init__(self) -> None:
            self.text = ""

        def write(self, text: str) -> None:
            self.text += text

        def read(self) -> str:
            return self.text

    class _LobConnection(_OracleConnection):
        def createlob(self, kind: str) -> _Lob:
            assert kind == "CLOB"
            return _Lob()

    monkeypatch.setitem(sys.modules, "oracledb", types.SimpleNamespace(DB_TYPE_CLOB="CLOB"))
    db = _OracleDB()
    store = OracleReportStore(
        dsn="evals_low",
        user="EVALS",
        password="pw",  # noqa: S106 — a double's password
        _connect=lambda: _LobConnection(db),
    )
    big = _report(passed=300, total=400, at=T0)
    stored = store.put(big, suite="warden")
    assert isinstance(db.rows[-1]["report"], _Lob)  # over the inline limit: streamed as a LOB
    read = store.get(stored.report_id)
    assert read is not None
    assert read.total_cases == 400
    db.rows.append({**db.rows[-1], "report_id": "raw", "report": big.model_dump_json().encode()})
    raw = store.get("raw")
    assert raw is not None
    assert raw.passed == 300


def test_specs_carry_the_optional_connection_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    oracle = report_store("oracle://u:p@/?dsn=evals_low&table=reports&schema=EVALS")
    assert isinstance(oracle, OracleReportStore)
    assert oracle.config.schema_name == "EVALS"
    assert oracle.config.table_name == "reports"
    bucket = report_store(
        "oci://ns/evals/trend?profile=EVALS&auth=instance_principal&region=eu-x-1"
    )
    assert isinstance(bucket, OCIBucketReportStore)
    assert bucket.prefix == "trend/"
    assert (bucket._profile_name, bucket._auth_type, bucket._region) == (
        "EVALS",
        "instance_principal",
        "eu-x-1",
    )
    s3 = report_store("s3://evals/trend?endpoint_url=http://object-store.local&region=r1")
    assert isinstance(s3, S3ReportStore)
    assert (s3.prefix, s3._endpoint_url, s3._region_name) == (
        "trend/",
        "http://object-store.local",
        "r1",
    )


# --------------------------------------------------------------------------
# A caller's report id reads back from every object store
# --------------------------------------------------------------------------


@pytest.mark.parametrize("report_id", ["build-42", "build42", "nightly/2026-11-16 #1"])
def test_local_store_reads_back_a_report_stored_under_a_callers_id(
    local_store: ReportStore, report_id: str
) -> None:
    store = local_store
    store.put(_report(passed=2, total=3, at=T0), suite="warden")
    stored = store.put(
        _report(passed=3, total=3, at=T0 + timedelta(hours=1)), suite="warden", report_id=report_id
    )

    assert stored.report_id == report_id
    if isinstance(store, FileReportStore):
        assert stored.location.startswith("warden/"), "the id cannot split the suite from the file"
        assert stored.location.count("/") == 1
    else:
        assert stored.location.endswith(f"evals.db:{report_id}")
    loaded = store.get(report_id)
    assert loaded is not None
    assert loaded.passed == 3
    listed = store.list(suite="warden")
    assert listed[0].report_id == report_id
    assert listed[0].passed == 3
    history = trend(store, suite="warden")
    assert len(history.points) == 2
    assert not history.regressed


def test_bucket_stores_read_back_a_report_stored_under_a_callers_id() -> None:
    stores: list[ReportStore] = [
        OCIBucketReportStore("evals", "ns", prefix="locus/eval/", _client=_OCIClient(page=1)),
        S3ReportStore("evals", prefix="evals/", _client=_S3Client()),
    ]
    for store in stores:
        stored = store.put(_report(passed=1, total=1, at=T0), suite="warden", report_id="build-42")
        assert stored.location.endswith("-abcdefabcdef-build-42.json")
        loaded = store.get("build-42")
        assert loaded is not None, store.spec
        assert loaded.passed == 1
        assert [item.report_id for item in store.list(suite="warden")] == ["build-42"]
        assert store.get("build-43") is None
        assert len(trend(store, suite="warden").points) == 1


def test_a_report_stored_under_a_generated_id_still_parses(tmp_path: Path) -> None:
    legacy = "0123456789abcdef0123456789abcdef"
    (tmp_path / "warden").mkdir()
    (tmp_path / "warden" / f"20261116T090000000000Z-abcdefabcdef-{legacy}.json").write_text(
        _report(passed=1, total=1, at=T0).model_dump_json()
    )
    store = FileReportStore(tmp_path)

    listed = store.list(suite="warden")

    assert [item.report_id for item in listed] == [legacy]
    assert listed[0].created_at == T0
    assert listed[0].fingerprint == "abcdefabcdef"
    assert store.get(legacy) is not None
    stored = store.put(_report(passed=1, total=1, at=T0 + timedelta(hours=1)), suite="warden")
    assert len(stored.report_id) == 32
    assert stored.location.endswith(f"-{stored.report_id}.json"), "a generated id is not encoded"


def test_a_blank_report_id_is_refused(local_store: ReportStore) -> None:
    with pytest.raises(ValueError, match="blank"):
        local_store.put(_report(passed=1, total=1), suite="warden", report_id="  ")
    assert local_store.list() == []


def test_a_generated_report_id_is_32_hex_on_both_local_stores(local_store: ReportStore) -> None:
    stored = local_store.put(_report(passed=1, total=1, at=T0), suite="warden")
    assert len(stored.report_id) == 32
    assert int(stored.report_id, 16) >= 0
    assert local_store.get(stored.report_id) is not None


def test_local_stores_read_the_instant_back_whatever_zone_it_was_written_in(
    local_store: ReportStore,
) -> None:
    paris = timezone(timedelta(hours=1))
    local_store.put(_report(passed=1, total=1, at=T0.astimezone(paris)), suite="warden")
    local_store.put(_report(passed=1, total=1, at=T0 - timedelta(minutes=30)), suite="warden")
    listed = local_store.list(suite="warden")
    assert [item.created_at for item in listed] == [T0, T0 - timedelta(minutes=30)]


# --------------------------------------------------------------------------
# SqliteReportStore: what only a database file has
# --------------------------------------------------------------------------


def test_sqlite_store_keeps_one_table_with_the_report_as_json_and_two_indexes(
    tmp_path: Path,
) -> None:
    store = SqliteReportStore(tmp_path / "evals.db")
    stored = _three_reports(store)
    with sqlite3.connect(tmp_path / "evals.db") as conn:
        objects = conn.execute(
            "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        body = conn.execute(
            "SELECT report FROM locus_eval_reports WHERE report_id = ?", (stored[1].report_id,)
        ).fetchone()[0]
        # The headline figures are columns: a trend is one query, no report read.
        rows = conn.execute(
            "SELECT passed, total_cases FROM locus_eval_reports "
            "WHERE suite = 'warden' ORDER BY created_at"
        ).fetchall()
    conn.close()
    assert objects == [
        ("table", "locus_eval_reports"),
        ("index", "locus_eval_reports_created_at"),
        ("index", "locus_eval_reports_suite"),
    ]
    assert json.loads(body)["passed"] == 2
    assert rows == [(3, 3), (2, 3), (3, 3)]
    assert len(store.ddl()) == 3


def test_sqlite_store_reads_nothing_from_a_file_it_has_not_written(tmp_path: Path) -> None:
    absent = SqliteReportStore(tmp_path / "nowhere" / "evals.db")
    assert absent.list() == []
    assert absent.get("anything") is None
    assert not (tmp_path / "nowhere").exists(), "reading does not create the file"

    # A database another tool made, without this store's table.
    other = tmp_path / "other.db"
    with sqlite3.connect(other) as conn:
        conn.execute("CREATE TABLE unrelated (x)")
    conn.close()
    store = SqliteReportStore(other)
    assert store.list() == []
    assert store.get("anything") is None
    store.put(_report(passed=1, total=1, at=T0), suite="warden", report_id="build-1")
    assert [item.report_id for item in store.list()] == ["build-1"]


def test_sqlite_store_replaces_a_report_put_again_under_the_same_id(tmp_path: Path) -> None:
    store = SqliteReportStore(tmp_path / "evals.db")
    store.put(_report(passed=1, total=3, at=T0), suite="warden", report_id="build-42")
    store.put(
        _report(passed=3, total=3, at=T0 + timedelta(hours=1)), suite="warden", report_id="build-42"
    )
    listed = store.list(suite="warden")
    assert [(item.report_id, item.passed) for item in listed] == [("build-42", 3)]
    loaded = store.get("build-42")
    assert loaded is not None
    assert loaded.passed == 3


def test_sqlite_store_refuses_a_path_or_a_table_it_cannot_use(tmp_path: Path) -> None:
    for path in (":memory:", "", "  "):
        with pytest.raises(ValueError, match="database file path"):
            SqliteReportStore(path)
    with pytest.raises(ValueError, match="invalid table name"):
        SqliteReportStore(tmp_path / "evals.db", table="reports; DROP TABLE x")
    with pytest.raises(ValueError, match="database file path"):
        report_store("sqlite://")


def test_sqlite_store_takes_writers_from_several_threads_at_once(tmp_path: Path) -> None:
    store = SqliteReportStore(tmp_path / "evals.db")
    errors: list[BaseException] = []

    def write(worker: int) -> None:
        try:
            for n in range(10):
                store.put(
                    _report(passed=1, total=1, at=T0 + timedelta(seconds=worker * 100 + n)),
                    suite="warden",
                    report_id=f"w{worker}-{n}",
                )
                assert store.get(f"w{worker}-{n}") is not None
        except BaseException as exc:  # noqa: BLE001 - reported by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    listed = store.list(suite="warden", limit=1000)
    assert len(listed) == 80
    assert len({item.report_id for item in listed}) == 80
    assert listed[0].report_id == "w7-9"


_WRITER = """
import sys
from datetime import UTC, datetime, timedelta
from locus.evaluation import EvalReport, SqliteReportStore

path, worker = sys.argv[1], int(sys.argv[2])
store = SqliteReportStore(path)
start = datetime(2026, 11, 16, 9, 0, tzinfo=UTC)
for n in range(25):
    report = EvalReport(results=[], total_cases=1, passed=1, failed=0, avg_score=1.0)
    report.created_at = start + timedelta(seconds=worker * 100 + n)
    store.put(report, suite="warden", report_id=f"p{worker}-{n}")
"""


def test_sqlite_store_takes_writers_from_several_processes_on_a_new_file(tmp_path: Path) -> None:
    # Four processes race to create the file and the table, then interleave
    # their writes; every row has to be there afterwards.
    path = tmp_path / "shared" / "evals.db"
    writers = [
        subprocess.Popen(  # noqa: S603 - this interpreter, a fixed script
            [sys.executable, "-c", _WRITER, str(path), str(worker)],
            stderr=subprocess.PIPE,
            text=True,
        )
        for worker in range(4)
    ]
    failures = [(w.communicate()[1], w.returncode) for w in writers]
    assert [code for _err, code in failures] == [0, 0, 0, 0], failures
    listed = SqliteReportStore(path).list(suite="warden", limit=1000)
    assert len(listed) == 100
    assert len({item.report_id for item in listed}) == 100
