# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""#177 — deferred tools: invisible until searched, governed the whole time."""

from __future__ import annotations

import json

import pytest

from tulip.tools.decorator import tool
from tulip.tools.registry import create_registry
from tulip.tools.tool_search import create_tool_search_tool, deferred_catalog_note


@tool
def eager_lookup(q: str) -> str:
    """Look up an order."""
    return f"order {q}"


@tool(deferred=True, labels={"payment"})
def issue_refund(order_id: str, amount: float) -> str:
    """Issue a refund for an order — moves money."""
    return f"refunded {amount} on {order_id}"


@tool(deferred=True)
def dns_lookup(hostname: str) -> str:
    """Resolve a hostname to addresses (DNS)."""
    return f"1.2.3.4 for {hostname}"


class TestVisibility:
    def test_deferred_schema_absent_until_activated(self) -> None:
        reg = create_registry(eager_lookup, issue_refund)
        names = [s["function"]["name"] for s in reg.to_openai_schemas()]
        assert names == ["eager_lookup"]
        reg.activate("issue_refund")
        names = [s["function"]["name"] for s in reg.to_openai_schemas()]
        assert set(names) == {"eager_lookup", "issue_refund"}

    def test_deferred_tool_is_still_registered_and_executable(self) -> None:
        """Deferral is visibility only — the tool (and any gate wrapped
        around it) is live from registration."""
        reg = create_registry(issue_refund)
        assert reg.get("issue_refund") is not None
        assert reg.get("issue_refund") is issue_refund

    def test_catalog_note_present_only_with_pending_deferred(self) -> None:
        reg = create_registry(eager_lookup)
        assert deferred_catalog_note(reg) == ""
        reg.register(issue_refund)
        assert "tool_search" in deferred_catalog_note(reg)
        reg.activate("issue_refund")
        assert deferred_catalog_note(reg) == ""


class TestSearch:
    def test_search_ranks_and_activates(self) -> None:
        reg = create_registry(eager_lookup, issue_refund, dns_lookup)
        search_tool = create_tool_search_tool(reg)
        result = json.loads(search_tool.fn("refund payment"))
        assert [m["name"] for m in result["matches"]] == ["issue_refund"]
        assert "issue_refund" in reg.activated
        assert "dns_lookup" not in reg.activated

    def test_no_match_reports_remaining(self) -> None:
        reg = create_registry(issue_refund, dns_lookup)
        search_tool = create_tool_search_tool(reg)
        result = json.loads(search_tool.fn("quantum teleport"))
        assert result["matches"] == []
        assert "2 unloaded" in result["note"]

    def test_activated_tools_leave_the_catalog(self) -> None:
        reg = create_registry(issue_refund, dns_lookup)
        reg.activate("dns_lookup")
        assert [t.name for t in reg.deferred_pending()] == ["issue_refund"]
        assert reg.search("dns hostname") == []


class _StubModel:
    async def complete(self, messages, **kwargs):  # pragma: no cover — never called
        raise AssertionError("construction-only stub")


class TestAgentWiring:
    def test_tool_search_auto_registered_only_when_deferred_exist(self) -> None:
        from tulip.agent import Agent

        with_deferred = Agent(model=_StubModel(), tools=[eager_lookup, issue_refund])
        assert "tool_search" in with_deferred.tools

        without = Agent(model=_StubModel(), tools=[eager_lookup])
        assert "tool_search" not in without.tools


class TestGatingSurvivesDeferral:
    @pytest.mark.asyncio
    async def test_gated_deferred_tool_stays_gated_after_activation(self) -> None:
        """The point of the design: activation must not widen authority."""
        from tulip.control import ControlPolicy, gate_tool

        gated = gate_tool(issue_refund, policy=ControlPolicy(deny_for={"payment"}))
        gated_deferred = gated.model_copy(update={"deferred": True})
        reg = create_registry(gated_deferred)
        reg.activate(gated_deferred.name)
        out = await reg.get_or_raise(gated_deferred.name).execute(order_id="4471", amount=10.0)
        text = str(out)
        assert "refunded" not in text, "the gate must hold after activation"
