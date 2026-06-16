# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for the workbench ``/api/database/test`` endpoint and the
``_resolve_db`` env-var fallback.

The endpoint is the panel's "Test connection" button: it accepts an
Oracle 26ai connection envelope and opens a pool + runs ``SELECT 1``,
returning the raw oracledb error verbatim on failure. Three regression
gates here:

  1. Empty payload → ``ok: False`` with a "fields required" message,
     no exception bubbling out as a 500.
  2. Bad DSN → ``ok: False`` with the oracledb error string surfaced
     verbatim (callers diagnose wallet / DPY-… faults from the UI).
  3. Real ADB (gated on ``ORACLE_DSN`` / ``ORACLE_PASSWORD`` env) →
     ``ok: True``. Skipped in CI runs without creds.

The endpoint exists so the workbench UI can validate before the user
fires off an Oracle-backed pattern run — without it, a misconfigured
connection only surfaces deep inside the pattern coroutine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


WORKBENCH_BACKEND = (Path(__file__).resolve().parents[2] / "workbench" / "backend").resolve()


@pytest.fixture(scope="module")
def client() -> TestClient:
    sys.path.insert(0, str(WORKBENCH_BACKEND))
    try:
        import runner  # type: ignore[import-not-found]
    finally:
        if str(WORKBENCH_BACKEND) in sys.path:
            sys.path.remove(str(WORKBENCH_BACKEND))
    return TestClient(runner.app)


@pytest.fixture(scope="module")
def runner_module():
    sys.path.insert(0, str(WORKBENCH_BACKEND))
    try:
        import runner  # type: ignore[import-not-found]

        yield runner
    finally:
        if str(WORKBENCH_BACKEND) in sys.path:
            sys.path.remove(str(WORKBENCH_BACKEND))


# ---------------------------------------------------------------------------
# /api/database/test — endpoint contract
# ---------------------------------------------------------------------------


class TestDatabaseTestEndpoint:
    def test_empty_config_returns_field_required(self, client: TestClient) -> None:
        r = client.post("/api/database/test", json={})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is False
        assert "required" in body["detail"].lower()

    def test_partial_config_returns_field_required(self, client: TestClient) -> None:
        # Only DSN, no user/password — still "not set" per is_set().
        r = client.post("/api/database/test", json={"dsn": "mydb_low"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is False
        assert "required" in body["detail"].lower()

    def test_bad_dsn_surfaces_oracledb_error_verbatim(self, client: TestClient) -> None:
        r = client.post(
            "/api/database/test",
            json={
                "dsn": "does_not_exist_low",
                "user": "ADMIN",
                "password": "wrong",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is False
        # Any oracledb error class prefixes the detail.
        # We don't pin the exact code — DPY-4027 (no wallet) or
        # DPY-6005 (TNS could not resolve) etc. all qualify — but the
        # prefix is enough to confirm the verbatim surfacing contract.
        assert ":" in body["detail"]
        assert body["dsn"] == "does_not_exist_low"

    @pytest.mark.skipif(
        not (os.environ.get("ORACLE_DSN") and os.environ.get("ORACLE_PASSWORD")),
        reason="ORACLE_DSN / ORACLE_PASSWORD not set — live DB regression skipped",
    )
    def test_live_connection_returns_ok(self, client: TestClient) -> None:
        r = client.post(
            "/api/database/test",
            json={
                "dsn": os.environ["ORACLE_DSN"],
                "user": os.environ.get("ORACLE_USER", "ADMIN"),
                "password": os.environ["ORACLE_PASSWORD"],
                "wallet_location": os.environ.get("ORACLE_WALLET", ""),
                "wallet_password": os.environ.get("ORACLE_WALLET_PASSWORD", ""),
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True, body["detail"]
        assert "SELECT 1" in body["detail"]


# ---------------------------------------------------------------------------
# _resolve_db — env-var fallback semantics
# ---------------------------------------------------------------------------


class TestResolveDb:
    """``_resolve_db`` picks the per-request envelope when set, otherwise
    falls back to ORACLE_* env vars. This is what makes the same backend
    image work for (a) a developer running locally with env vars and
    (b) a hosted workbench where each tab supplies its own creds."""

    def test_request_config_wins(self, runner_module, monkeypatch) -> None:
        monkeypatch.setenv("ORACLE_DSN", "from_env_low")
        monkeypatch.setenv("ORACLE_USER", "env_user")
        monkeypatch.setenv("ORACLE_PASSWORD", "env_pw")
        cfg = runner_module.DatabaseConfig(
            dsn="from_request_low",
            user="req_user",
            password="req_pw",  # noqa: S105, S106
        )
        out = runner_module._resolve_db(cfg)
        assert out.dsn == "from_request_low"
        assert out.user == "req_user"

    def test_blank_config_falls_through_to_env(self, runner_module, monkeypatch) -> None:
        monkeypatch.setenv("ORACLE_DSN", "from_env_low")
        monkeypatch.setenv("ORACLE_USER", "env_user")
        monkeypatch.setenv("ORACLE_PASSWORD", "env_pw")
        monkeypatch.setenv("ORACLE_WALLET", "/tmp/wallets/env")
        out = runner_module._resolve_db(runner_module.DatabaseConfig())
        assert out.dsn == "from_env_low"
        assert out.user == "env_user"
        assert out.wallet_location == "/tmp/wallets/env"

    def test_none_config_falls_through_to_env(self, runner_module, monkeypatch) -> None:
        monkeypatch.setenv("ORACLE_DSN", "from_env_low")
        monkeypatch.setenv("ORACLE_USER", "env_user")
        monkeypatch.setenv("ORACLE_PASSWORD", "env_pw")
        out = runner_module._resolve_db(None)
        assert out.dsn == "from_env_low"

    def test_partial_request_config_falls_through_because_is_set_fails(
        self, runner_module, monkeypatch
    ) -> None:
        # Only DSN — is_set() returns False, so env wins.
        monkeypatch.setenv("ORACLE_DSN", "from_env_low")
        monkeypatch.setenv("ORACLE_USER", "env_user")
        monkeypatch.setenv("ORACLE_PASSWORD", "env_pw")
        cfg = runner_module.DatabaseConfig(dsn="partial_low")
        out = runner_module._resolve_db(cfg)
        assert out.dsn == "from_env_low"  # env wins because cfg.is_set() is False
