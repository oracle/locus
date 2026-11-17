# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``OracleAuditTrail`` against a real Oracle AI Database.

Gated the same way as the other Oracle integration tests: ``ORACLE_PASSWORD``
in the environment and a wallet on disk (``ORACLE_WALLET``, default
``~/.oci/wallets/deepresearch``), with ``ORACLE_DSN`` / ``ORACLE_USER``
overriding the defaults. Skips cleanly otherwise.

The tables are immutable, so nothing is dropped: every run writes under a
fresh ``trail_id`` into one shared table per kind (``LOCUS_AUDIT_TRAIL_IT``
and ``LOCUS_AUDIT_TRAIL_BC_IT``). Set ``LOCUS_AUDIT_IT_RETENTION_DAYS`` to
shorten the table's retention when it is first created (16 is the minimum).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest


pytestmark = [pytest.mark.integration]


def _oracle_available() -> bool:
    if not os.environ.get("ORACLE_PASSWORD"):
        return False
    wallet = Path(os.environ.get("ORACLE_WALLET", str(Path.home() / ".oci/wallets/deepresearch")))
    return wallet.exists() and (wallet / "tnsnames.ora").exists()


ORACLE_DSN = os.environ.get("ORACLE_DSN", "deepresearch_low")
ORACLE_USER = os.environ.get("ORACLE_USER", "ADMIN")
ORACLE_PASSWORD = os.environ.get("ORACLE_PASSWORD", "")
ORACLE_WALLET = os.environ.get("ORACLE_WALLET", str(Path.home() / ".oci/wallets/deepresearch"))
ORACLE_WALLET_PASSWORD = os.environ.get("ORACLE_WALLET_PASSWORD", "")
RETENTION_DAYS = int(os.environ.get("LOCUS_AUDIT_IT_RETENTION_DAYS", "16"))


def _trail(table_kind: str, trail_id: str):  # noqa: ANN202
    from locus.control import OracleAuditTrail

    return OracleAuditTrail(
        dsn=ORACLE_DSN,
        user=ORACLE_USER,
        password=ORACLE_PASSWORD,
        wallet_location=ORACLE_WALLET,
        wallet_password=ORACLE_WALLET_PASSWORD or None,
        table_name="locus_audit_trail_it"
        if table_kind == "immutable"
        else "locus_audit_trail_bc_it",
        table_kind=table_kind,  # type: ignore[arg-type]
        trail_id=trail_id,
        retention_days=RETENTION_DAYS,
        drop_idle_days=0,
    )


@pytest.mark.skipif(not _oracle_available(), reason="Oracle ADB credentials not available")
@pytest.mark.parametrize("table_kind", ["immutable", "blockchain"])
def test_chain_round_trips_through_the_table(table_kind: str) -> None:
    trail_id = f"it-{uuid.uuid4().hex}"
    trail = _trail(table_kind, trail_id)
    try:
        first = trail.record("gate", {"tool": "bash", "verdict": "allow"})
        second = trail.record("gate", {"tool": "write", "verdict": "deny", "blob": "x" * 40_000})
        assert second.prev_hash == first.hash
        report = trail.check(expected_head=second.hash)
        assert report.ok, report.problem
        assert report.records == 2

        reopened = _trail(table_kind, trail_id)
        assert reopened.head == second.hash
        third = reopened.record("gate", {"n": 3})
        assert third.seq == 2
        assert trail.check(expected_head=third.hash).ok  # check() re-reads the table
        assert len(trail.records()) == 3
    finally:
        trail.close()


@pytest.mark.skipif(not _oracle_available(), reason="Oracle ADB credentials not available")
async def test_async_twins_against_the_database() -> None:
    trail = _trail("immutable", f"it-{uuid.uuid4().hex}")
    try:
        rec = await trail.arecord("gate", {"n": 1})
        assert await trail.averify(expected_head=rec.hash)
    finally:
        trail.close()
