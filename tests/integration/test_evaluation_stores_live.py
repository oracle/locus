# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Report stores against a real Oracle AI Database and a real OCI bucket.

Gated like the other Oracle integration tests: ``ORACLE_PASSWORD`` in the
environment and a wallet on disk (``ORACLE_WALLET``), with ``ORACLE_DSN`` /
``ORACLE_USER`` overriding the defaults. The bucket test needs
``LOCUS_EVAL_BUCKET`` and ``LOCUS_EVAL_NAMESPACE`` (and ``OCI_PROFILE`` /
``OCI_REGION`` for the client). Both skip cleanly otherwise.

Every run writes under a fresh suite name; the Oracle table it creates is
dropped at the end and the objects it wrote are deleted.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from locus.evaluation import EvalReport, EvalResult, trend


pytestmark = [pytest.mark.integration]


def _oracle_available() -> bool:
    if not os.environ.get("ORACLE_PASSWORD"):
        return False
    wallet = Path(os.environ.get("ORACLE_WALLET", str(Path.home() / ".oci/wallets/deepresearch")))
    return wallet.exists() and (wallet / "tnsnames.ora").exists()


def _bucket_available() -> bool:
    return bool(os.environ.get("LOCUS_EVAL_BUCKET") and os.environ.get("LOCUS_EVAL_NAMESPACE"))


def _report(passed: int, total: int, *, at: datetime) -> EvalReport:
    report = EvalReport(
        results=[
            EvalResult(case_name=f"c{i}", passed=i < passed, score=float(i < passed))
            for i in range(total)
        ],
        total_cases=total,
        passed=passed,
        failed=total - passed,
        avg_score=passed / total,
        fingerprint="abcdefabcdef",
    )
    report.created_at = at
    return report


@pytest.mark.skipif(not _oracle_available(), reason="Oracle ADB credentials not available")
def test_oracle_store_round_trips_and_trends() -> None:
    from locus.evaluation import OracleReportStore

    table = f"LOCUS_EVAL_IT_{uuid.uuid4().hex[:8]}"
    store = OracleReportStore(
        dsn=os.environ.get("ORACLE_DSN", "deepresearch_low"),
        user=os.environ.get("ORACLE_USER", "ADMIN"),
        password=os.environ["ORACLE_PASSWORD"],
        wallet_location=os.environ.get(
            "ORACLE_WALLET", str(Path.home() / ".oci/wallets/deepresearch")
        ),
        wallet_password=os.environ.get("ORACLE_WALLET_PASSWORD") or None,
        table_name=table,
    )
    suite = f"it-{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC).replace(microsecond=0)
    try:
        first = store.put(_report(3, 3, at=now - timedelta(minutes=2)), suite=suite)
        second = store.put(_report(2, 3, at=now - timedelta(minutes=1)), suite=suite)
        listed = store.list(suite=suite)
        assert [item.report_id for item in listed] == [second.report_id, first.report_id]
        assert listed[0].created_at == now - timedelta(minutes=1)
        loaded = store.get(first.report_id)
        assert loaded is not None
        assert loaded.passed == 3
        assert loaded.fingerprint == "abcdefabcdef"
        history = trend(store, suite=suite, last=5)
        assert [p.passed for p in history.points] == [3, 2]
        assert history.regressed
    finally:

        def _drop(conn: object) -> None:
            with conn.cursor() as cur:  # type: ignore[attr-defined]
                cur.execute(f"DROP TABLE {table} PURGE")
            conn.commit()  # type: ignore[attr-defined]

        store._run(_drop)
        store.close()


@pytest.mark.skipif(
    not _bucket_available(), reason="LOCUS_EVAL_BUCKET / LOCUS_EVAL_NAMESPACE not set"
)
def test_oci_bucket_store_round_trips_and_trends() -> None:
    from locus.evaluation import OCIBucketReportStore

    store = OCIBucketReportStore(
        os.environ["LOCUS_EVAL_BUCKET"],
        os.environ["LOCUS_EVAL_NAMESPACE"],
        prefix=f"locus/eval-it/{uuid.uuid4().hex[:8]}/",
        profile_name=os.environ.get("OCI_PROFILE", "DEFAULT"),
        region=os.environ.get("OCI_REGION"),
        auth_type=os.environ.get("OCI_AUTH_TYPE", "api_key"),
    )
    suite = "it"
    now = datetime.now(UTC)
    written: list[str] = []
    try:
        first = store.put(_report(3, 3, at=now - timedelta(minutes=2)), suite=suite)
        second = store.put(_report(2, 3, at=now - timedelta(minutes=1)), suite=suite)
        written = [first.location, second.location]
        listed = store.list(suite=suite)
        assert [item.report_id for item in listed] == [second.report_id, first.report_id]
        loaded = store.get(first.report_id)
        assert loaded is not None
        assert loaded.passed == 3
        history = trend(store, suite=suite, last=5)
        assert [p.passed for p in history.points] == [3, 2]
        assert history.regressed
    finally:
        client = store._get_client()
        for name in written:
            client.delete_object(
                namespace_name=store.namespace, bucket_name=store.bucket_name, object_name=name
            )
