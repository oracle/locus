# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for the OSV malware pre-check in ``locus.integrations.osv``.

The public entry point is ``check_package_for_malware(command, args)``.
All network calls are patched via ``urllib.request.urlopen`` so the
suite is hermetic.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from locus.integrations import osv
from locus.integrations.osv import check_package_for_malware


class _FakeResp(io.BytesIO):
    """Minimal context-manager stand-in for ``urlopen`` return value."""

    def __enter__(self) -> _FakeResp:  # type: ignore[override]
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _patch_osv_response(
    monkeypatch: pytest.MonkeyPatch,
    vulns: list[dict[str, Any]] | None = None,
    raise_exc: BaseException | None = None,
) -> list[dict[str, Any]]:
    """Patch urlopen, return a list that receives each request body (as dict)."""
    captured: list[dict[str, Any]] = []

    def _fake_urlopen(req: Any, timeout: int | None = None) -> _FakeResp:
        if raise_exc is not None:
            raise raise_exc
        try:
            captured.append(json.loads(req.data.decode()))
        except Exception:  # pragma: no cover — defensive
            captured.append({})
        return _FakeResp(json.dumps({"vulns": vulns or []}).encode())

    monkeypatch.setattr(osv.urllib.request, "urlopen", _fake_urlopen)
    return captured


# ---------------------------------------------------------------------------
# Clean + malware.
# ---------------------------------------------------------------------------


class TestMalwareDetection:
    def test_clean_npm_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_osv_response(monkeypatch, vulns=[])
        assert check_package_for_malware("npx", ["@modelcontextprotocol/server-fs"]) is None

    def test_malware_npm_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_osv_response(
            monkeypatch,
            vulns=[{"id": "MAL-2024-00042", "summary": "crypto-stealer in preinstall hook"}],
        )
        reason = check_package_for_malware("npx", ["evil-mcp-server"])
        assert reason is not None
        assert "MAL-2024-00042" in reason
        assert "crypto-stealer" in reason

    def test_cve_not_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Regular CVEs are ignored — only MAL-* triggers.
        _patch_osv_response(
            monkeypatch,
            vulns=[{"id": "GHSA-xxxx-yyyy-zzzz", "summary": "ReDoS"}],
        )
        assert check_package_for_malware("npx", ["some-lib"]) is None

    def test_malware_pypi_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_osv_response(
            monkeypatch, vulns=[{"id": "MAL-2024-0001", "summary": "typo-squatted"}]
        )
        reason = check_package_for_malware("uvx", ["reqeusts"])
        assert reason is not None
        assert "MAL-2024-0001" in reason


# ---------------------------------------------------------------------------
# Ecosystem inference.
# ---------------------------------------------------------------------------


class TestEcosystemInference:
    def test_node_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_osv_response(monkeypatch, vulns=[{"id": "MAL-X"}])
        # Direct `node` invocation isn't a supply-chain launcher — no check.
        assert check_package_for_malware("node", ["server.js"]) is None

    def test_python_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_osv_response(monkeypatch, vulns=[{"id": "MAL-X"}])
        assert check_package_for_malware("python", ["server.py"]) is None

    def test_absolute_path_to_npx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_osv_response(monkeypatch, vulns=[])
        check_package_for_malware("/usr/local/bin/npx", ["my-pkg"])
        assert captured == [{"package": {"name": "my-pkg", "ecosystem": "npm"}}]

    def test_windows_cmd_extension(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_osv_response(monkeypatch, vulns=[])
        check_package_for_malware("C:\\npm\\npx.cmd", ["my-pkg"])
        assert captured == [{"package": {"name": "my-pkg", "ecosystem": "npm"}}]

    def test_bunx_maps_to_npm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_osv_response(monkeypatch, vulns=[])
        check_package_for_malware("bunx", ["cowsay"])
        assert captured == [{"package": {"name": "cowsay", "ecosystem": "npm"}}]

    def test_pipx_maps_to_pypi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_osv_response(monkeypatch, vulns=[])
        check_package_for_malware("pipx", ["httpie"])
        assert captured == [{"package": {"name": "httpie", "ecosystem": "PyPI"}}]


# ---------------------------------------------------------------------------
# Argument parsing (skipping flags, versions, scoped names).
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    def test_skips_leading_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_osv_response(monkeypatch, vulns=[])
        check_package_for_malware("npx", ["--yes", "--quiet", "pkg-name"])
        assert captured[0]["package"]["name"] == "pkg-name"

    def test_skips_value_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_osv_response(monkeypatch, vulns=[])
        check_package_for_malware("npx", ["-p", "other-pkg", "actual-pkg"])
        assert captured[0]["package"]["name"] == "actual-pkg"

    def test_npm_scoped_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_osv_response(monkeypatch, vulns=[])
        check_package_for_malware("npx", ["@modelcontextprotocol/server-git"])
        assert captured[0]["package"]["name"] == "@modelcontextprotocol/server-git"

    def test_npm_scoped_with_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_osv_response(monkeypatch, vulns=[])
        check_package_for_malware("npx", ["@scope/name@1.2.3"])
        assert captured[0] == {
            "package": {"name": "@scope/name", "ecosystem": "npm"},
            "version": "1.2.3",
        }

    def test_npm_version_stripped_when_latest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_osv_response(monkeypatch, vulns=[])
        check_package_for_malware("npx", ["foo@latest"])
        assert "version" not in captured[0]

    def test_pypi_version_specifier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_osv_response(monkeypatch, vulns=[])
        check_package_for_malware("uvx", ["black==24.3.0"])
        assert captured[0] == {
            "package": {"name": "black", "ecosystem": "PyPI"},
            "version": "24.3.0",
        }

    def test_pypi_extras_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_osv_response(monkeypatch, vulns=[])
        check_package_for_malware("uvx", ["httpx[socks]==0.27.0"])
        assert captured[0]["package"]["name"] == "httpx"

    def test_empty_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_osv_response(monkeypatch, vulns=[{"id": "MAL-X"}])
        assert check_package_for_malware("npx", []) is None


# ---------------------------------------------------------------------------
# Fail-open behaviour.
# ---------------------------------------------------------------------------


class TestFailOpen:
    def test_timeout_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import socket

        _patch_osv_response(monkeypatch, raise_exc=socket.timeout("slow"))
        assert check_package_for_malware("npx", ["pkg"]) is None

    def test_http_error_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import urllib.error

        _patch_osv_response(
            monkeypatch,
            raise_exc=urllib.error.HTTPError("x", 503, "down", {}, None),  # type: ignore[arg-type]
        )
        assert check_package_for_malware("npx", ["pkg"]) is None

    def test_json_parse_error_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fake_urlopen(req: Any, timeout: int | None = None) -> _FakeResp:
            return _FakeResp(b"not json")

        monkeypatch.setattr(osv.urllib.request, "urlopen", _fake_urlopen)
        assert check_package_for_malware("npx", ["pkg"]) is None


# ---------------------------------------------------------------------------
# Global opt-out.
# ---------------------------------------------------------------------------


class TestGlobalOptOut:
    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "YES"])
    def test_env_var_disables_check(self, val: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCUS_MCP_SKIP_OSV", val)
        # If the check ran despite opt-out, this patched response would
        # force a malware verdict. We assert it does not.
        _patch_osv_response(monkeypatch, vulns=[{"id": "MAL-WOULD-FAIL"}])
        assert check_package_for_malware("npx", ["would-be-blocked"]) is None

    def test_env_unset_enables_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOCUS_MCP_SKIP_OSV", raising=False)
        _patch_osv_response(monkeypatch, vulns=[{"id": "MAL-2025-0001", "summary": "proof"}])
        assert check_package_for_malware("npx", ["pkg"]) is not None
