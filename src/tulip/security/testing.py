# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Conformance kit for security integrations — the ``langchain-tests`` analog.

An integration in `tulip-integrations` (or any third-party package) imports
these assertions in its own test suite to prove its adapter satisfies the
:class:`~tulip.security.adapter.SecurityAdapter` contract and the toolkit
conventions (JSON-returning tools, grounded-or-abstain findings). Keeping the
kit in core means the long tail stays compatible as core evolves.

Example (in an integration's test)::

    import pytest
    from tulip.security.testing import (
        assert_adapter_conformance,
        assert_tool_returns_json,
    )
    from tulip_integrations.siem.splunk import splunk_adapter


    def test_conforms():
        assert_adapter_conformance(splunk_adapter())


    @pytest.mark.asyncio
    async def test_tool_json():
        (tool,) = splunk_adapter().tools()
        await assert_tool_returns_json(tool, query="failed login")
"""

from __future__ import annotations

import json
from typing import Any

from tulip.security.adapter import SecurityAdapter
from tulip.security.findings import Evidence
from tulip.security.grounded import Abstention, GroundedFinding
from tulip.tools.decorator import Tool


def assert_adapter_conformance(adapter: SecurityAdapter) -> None:
    """Assert ``adapter`` satisfies the :class:`SecurityAdapter` contract.

    Checks the protocol shape, a non-empty unique tool list, and that every
    entry is a :class:`~tulip.tools.decorator.Tool`.
    """
    assert isinstance(adapter, SecurityAdapter), "does not satisfy the SecurityAdapter protocol"
    assert isinstance(adapter.name, str), "adapter.name must be a str"
    assert adapter.name, "adapter.name must be non-empty"
    assert isinstance(adapter.vendor, str), "adapter.vendor must be a str"
    assert adapter.vendor, "adapter.vendor must be non-empty"
    tools = adapter.tools()
    assert isinstance(tools, list), "tools() must return a list"
    assert tools, "tools() must return a non-empty list"
    assert all(isinstance(t, Tool) for t in tools), "tools() must return Tool objects"
    names = [t.name for t in tools]
    assert len(names) == len(set(names)), f"duplicate tool names: {names}"


async def assert_tool_returns_json(tool: Tool, *args: Any, **kwargs: Any) -> Any:
    """Call ``tool`` and assert it returns a JSON string; return the parsed value.

    Security tools return a JSON string for the agent loop to read — this runs
    the tool's offline-sample path (no credentials) and checks the contract.
    """
    assert isinstance(tool, Tool), "expected a Tool"
    out = await tool(*args, **kwargs)
    assert isinstance(out, str), "a security tool must return a JSON string"
    return json.loads(out)  # raises on non-JSON — the assertion


def assert_grounds_or_abstains(result: GroundedFinding) -> GroundedFinding:
    """Assert a ``*_to_finding`` result is a grounded :class:`Evidence` or an :class:`Abstention`.

    A shipped finding must carry a score in ``[0, 1]`` and at least one evidence
    ref — the abstain-by-construction guarantee.
    """
    assert isinstance(result, (Evidence, Abstention)), "must be a Evidence or an Abstention"
    if isinstance(result, Evidence):
        assert 0.0 <= result.gsar_score <= 1.0, "gsar_score out of range"
        assert result.evidence_refs, "a shipped Evidence must carry evidence refs"
    return result


__all__ = [
    "assert_adapter_conformance",
    "assert_grounds_or_abstains",
    "assert_tool_returns_json",
]
