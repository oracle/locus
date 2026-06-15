# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tests for the security integration contract + conformance kit.

Verifies the langchain-core-style boundary: the `SecurityAdapter` protocol,
the `ToolAdapter` concrete, `security_toolset(extra=...)` merging external
tools, and the `tulip.security.testing` conformance assertions.
"""

from __future__ import annotations

import pytest

from tulip.security import (
    SecurityAdapter,
    ToolAdapter,
    enrich_indicator_tool,
    enrich_to_finding,
    security_toolset,
    siem_query_tool,
)
from tulip.security.testing import (
    assert_adapter_conformance,
    assert_grounds_or_abstains,
    assert_tool_returns_json,
)


def test_tooladapter_satisfies_protocol() -> None:
    adapter = ToolAdapter(name="intel-ref", vendor="reference", _tools=[enrich_indicator_tool])
    assert isinstance(adapter, SecurityAdapter)
    assert adapter.tools() == [enrich_indicator_tool]


def test_assert_adapter_conformance_passes_for_bundled() -> None:
    adapter = ToolAdapter(
        name="triage-ref",
        vendor="reference",
        _tools=[enrich_indicator_tool, siem_query_tool],
    )
    assert_adapter_conformance(adapter)  # must not raise


def test_assert_adapter_conformance_rejects_duplicate_tool_names() -> None:
    dup = ToolAdapter(name="dup", vendor="x", _tools=[enrich_indicator_tool, enrich_indicator_tool])
    with pytest.raises(AssertionError):
        assert_adapter_conformance(dup)


def test_non_adapter_object_is_rejected() -> None:
    class NotAnAdapter:
        name = "x"
        vendor = "y"
        # no tools() method

    with pytest.raises(AssertionError):
        assert_adapter_conformance(NotAnAdapter())  # type: ignore[arg-type]


async def test_assert_tool_returns_json_runs_offline_path() -> None:
    payload = await assert_tool_returns_json(enrich_indicator_tool, "198.51.100.23")
    assert payload["indicator"] == "198.51.100.23"


def test_assert_grounds_or_abstains() -> None:
    assert_grounds_or_abstains(enrich_to_finding("198.51.100.23"))  # ships
    assert_grounds_or_abstains(enrich_to_finding("203.0.113.99"))  # abstains


def test_security_toolset_extra_merges_external_tools() -> None:
    external = ToolAdapter(name="ext", vendor="v", _tools=[]).tools()  # empty external
    base = {t.name for t in security_toolset(siem=False)}
    assert "query_siem" not in base
    # An explicitly-imported external tool is merged via extra=
    merged = {
        t.name
        for t in security_toolset(
            threat_intel=False,
            siem=False,
            edr=False,
            scanner=False,
            fingerprint=False,
            extra=[siem_query_tool],
        )
    }
    assert merged == {"query_siem"}
    assert external == []
