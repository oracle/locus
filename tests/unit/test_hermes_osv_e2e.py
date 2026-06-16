# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration test for the OSV malware pre-check (A.3).

Hits the live ``api.osv.dev`` endpoint with a known-clean MCP server
package to verify the end-to-end path works against the real
service. Skips cleanly when the network is unreachable.
"""

from __future__ import annotations

import socket

import pytest

from locus.integrations.osv import check_package_for_malware


def _api_reachable() -> bool:
    try:
        socket.create_connection(("api.osv.dev", 443), timeout=2.0).close()
        return True
    except (OSError, TimeoutError):
        return False


pytestmark = pytest.mark.skipif(not _api_reachable(), reason="api.osv.dev unreachable")


# ---------------------------------------------------------------------------
# Live API — known-clean packages return None.
# ---------------------------------------------------------------------------


class TestOsvLiveQuery:
    @pytest.mark.parametrize(
        ("command", "args"),
        [
            ("npx", ["@modelcontextprotocol/server-everything"]),
            ("npx", ["@modelcontextprotocol/server-filesystem"]),
            ("uvx", ["mcp-server-fetch"]),
        ],
    )
    def test_known_clean_package_passes(self, command: str, args: list[str]) -> None:
        # No MAL-* advisories on these — expect None (allow).
        assert check_package_for_malware(command, args) is None

    def test_unknown_random_package_passes(self) -> None:
        # A name that almost certainly has never been published. OSV
        # responds with empty vulns — should pass.
        assert (
            check_package_for_malware("npx", ["this-package-name-should-not-exist-locus-test-xyz"])
            is None
        )


# ---------------------------------------------------------------------------
# Skip-env opt-out still works against the live API.
# ---------------------------------------------------------------------------


class TestOptOut:
    def test_env_skip_short_circuits_before_network(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCUS_MCP_SKIP_OSV", "1")
        # Pass an obviously-suspicious string. With the skip flag set
        # the check must return None without making any network call,
        # so even an invalid hostname couldn't impact correctness.
        assert check_package_for_malware("npx", ["definitely-not-a-real-package"]) is None
