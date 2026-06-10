# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke tests against the **native** OpenAI and Anthropic SDKs.

These hit the **direct** vendor SDKs straight against OpenAI's and
Anthropic's APIs with their own keys.

Activation:

* ``OPENAI_API_KEY=<key>`` — required for OpenAI tests; otherwise skipped.
* ``ANTHROPIC_API_KEY=<key>`` — required for Anthropic tests; otherwise skipped.

What this guards
----------------

For each shipped feature we want at least one passing live call against
each direct SDK:

* **structured output** — ``Agent(output_schema=Foo)`` produces a parsed
  Pydantic instance on ``AgentResult.parsed`` for both providers; Anthropic
  uses the synthetic ``respond_with_schema`` tool translation.
* **termination algebra** — ``MaxIterations(N)`` clamps a runaway loop on
  both providers.
* **idempotent dedup** — ``@tool(idempotent=True)`` short-circuits a repeat
  call on both providers.
* **hook exports** — ``ModelRetryHook`` from ``tulip.hooks.builtin`` attaches
  to an Agent and the run completes.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, Field

from tulip.core.termination import MaxIterations
from tulip.tools.decorator import tool


_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))


pytestmark = pytest.mark.integration


# Cheap, fast default models. Override via env if needed.
_OPENAI_MODEL = os.environ.get("TULIP_OPENAI_TEST_MODEL", "gpt-4o-mini")
_ANTHROPIC_MODEL = os.environ.get("TULIP_ANTHROPIC_TEST_MODEL", "claude-haiku-4-5-20251001")


# =============================================================================
# Tools used across the suite
# =============================================================================


@tool
def keep_counting(n: int) -> str:
    """Increment a counter. Always asks the agent to keep calling.

    Used to coax a runaway loop so MaxIterations has something to clamp.
    """
    return f"Counter is now {n + 1}. Task is incomplete — call keep_counting again with n={n + 1}."


class _InvoiceCounter:
    n = 0


@tool(idempotent=True)
def submit_invoice(invoice_id: str, amount_usd: float) -> str:
    """Submit an invoice (idempotent — same args must not double-charge)."""
    _InvoiceCounter.n += 1
    return f"Submitted {invoice_id} for ${amount_usd:.2f}. (call #{_InvoiceCounter.n})"


# =============================================================================
# Structured output schema
# =============================================================================


class Vendor(BaseModel):
    name: str = Field(description="Legal name of the vendor")
    score: float = Field(description="Quality score in [0, 1]", ge=0.0, le=1.0)
    region: str = Field(description="Primary region")


class VendorList(BaseModel):
    """Three vendor recommendations."""

    vendors: list[Vendor] = Field(description="Exactly 3 vendor records")


# =============================================================================
# Fixtures: build each native model lazily so import-time skips are clean
# =============================================================================


@pytest.fixture
def openai_model():
    if not _OPENAI:
        pytest.skip("OPENAI_API_KEY not set")
    pytest.importorskip("openai")
    from tulip.models.native.openai import OpenAIModel

    return OpenAIModel(model=_OPENAI_MODEL)


@pytest.fixture
def anthropic_model():
    if not _ANTHROPIC:
        pytest.skip("ANTHROPIC_API_KEY not set")
    pytest.importorskip("anthropic")
    from tulip.models.native.anthropic import AnthropicModel

    return AnthropicModel(model=_ANTHROPIC_MODEL)


# =============================================================================
# OpenAI native SDK
# =============================================================================


class TestOpenAINative:
    """Direct hits against OpenAI's API with ``OPENAI_API_KEY``."""

    def test_structured_output_round_trip(self, openai_model):
        from tulip.agent import Agent

        agent = Agent(
            model=openai_model,
            tools=[],
            system_prompt=(
                "You are a procurement researcher. Recommend exactly 3 cloud-hosting "
                "vendors. Use only well-known providers."
            ),
            output_schema=VendorList,
            output_schema_strict=True,
            max_iterations=3,
        )
        result = agent.run_sync("List 3 cloud-hosting vendors with quality scores.")

        assert result.parse_error is None, (
            f"parse_error={result.parse_error!r}, message={result.message!r}"
        )
        assert isinstance(result.parsed, VendorList)
        assert len(result.parsed.vendors) == 3

    def test_max_iterations_caps_loop(self, openai_model):
        from tulip.agent import Agent

        agent = Agent(
            model=openai_model,
            tools=[keep_counting],
            system_prompt=(
                "You are a counter. Always call keep_counting. Never give a "
                "final answer — keep calling the tool."
            ),
            termination=MaxIterations(2),
            max_iterations=20,
        )
        result = agent.run_sync("Start counting.")
        assert result.iterations <= 3, (
            f"MaxIterations(2) failed to clamp: iterations={result.iterations}, "
            f"stop_reason={result.stop_reason!r}"
        )

    def test_idempotent_dedup_short_circuits(self, openai_model):
        from tulip.agent import Agent

        _InvoiceCounter.n = 0
        agent = Agent(
            model=openai_model,
            tools=[submit_invoice],
            system_prompt=(
                "You are a finance assistant. Submit invoice INV-42 for "
                "$100.00 EXACTLY THREE TIMES with the SAME parameters. Audit "
                "policy requires three calls. After the third, briefly confirm."
            ),
            termination=MaxIterations(6),
            max_iterations=10,
        )
        result = agent.run_sync(
            "Process INV-42 for $100.00. Three submit_invoice calls, same args."
        )
        invoice_calls = [te for te in result.tool_executions if te.tool_name == "submit_invoice"]
        if len(invoice_calls) < 2:
            pytest.skip(
                f"Model only invoked submit_invoice {len(invoice_calls)}x; no duplicate to dedup"
            )
        cache_hits = [te for te in invoice_calls if te.idempotent_cache_hit]
        assert cache_hits, "no idempotent_cache_hit recorded despite duplicate calls"
        assert _InvoiceCounter.n < len(invoice_calls)

    def test_model_retry_hook_attaches_and_runs(self, openai_model):
        from tulip.agent import Agent
        from tulip.hooks.builtin import ModelRetryHook

        agent = Agent(
            model=openai_model,
            tools=[],
            hooks=[ModelRetryHook(max_retries=2)],
            system_prompt="Reply briefly.",
            max_iterations=2,
        )
        result = agent.run_sync("Say hi.")
        assert result.message
        assert result.stop_reason in ("complete", "no_tools", "terminal_tool")


# =============================================================================
# Anthropic native SDK
# =============================================================================


class TestAnthropicNative:
    """Direct hits against Anthropic's API with ``ANTHROPIC_API_KEY``.

    Uses the synthetic ``respond_with_schema`` tool translation for
    structured output (Anthropic doesn't support OpenAI's ``response_format``).
    """

    def test_structured_output_round_trip(self, anthropic_model):
        from tulip.agent import Agent

        agent = Agent(
            model=anthropic_model,
            tools=[],
            system_prompt=(
                "You are a procurement researcher. Recommend exactly 3 cloud-hosting "
                "vendors. Use only well-known providers."
            ),
            output_schema=VendorList,
            # Anthropic translation always strict via tool_use; the kwarg is
            # consumed by the build_response_format path either way.
            output_schema_strict=True,
            max_iterations=3,
        )
        result = agent.run_sync("List 3 cloud-hosting vendors with quality scores.")

        assert result.parse_error is None, (
            f"parse_error={result.parse_error!r}, message={result.message!r}"
        )
        assert isinstance(result.parsed, VendorList)
        assert len(result.parsed.vendors) == 3

    def test_max_iterations_caps_loop(self, anthropic_model):
        from tulip.agent import Agent

        agent = Agent(
            model=anthropic_model,
            tools=[keep_counting],
            system_prompt=(
                "You are a counter. Always call keep_counting. Never give a "
                "final answer — keep calling the tool."
            ),
            termination=MaxIterations(2),
            max_iterations=20,
        )
        result = agent.run_sync("Start counting.")
        assert result.iterations <= 3, (
            f"MaxIterations(2) failed to clamp: iterations={result.iterations}, "
            f"stop_reason={result.stop_reason!r}"
        )

    def test_idempotent_dedup_short_circuits(self, anthropic_model):
        from tulip.agent import Agent

        _InvoiceCounter.n = 0
        agent = Agent(
            model=anthropic_model,
            tools=[submit_invoice],
            system_prompt=(
                "You are a finance assistant. Submit invoice INV-42 for "
                "$100.00 EXACTLY THREE TIMES with the SAME parameters. Audit "
                "policy requires three calls. After the third, briefly confirm."
            ),
            termination=MaxIterations(6),
            max_iterations=10,
        )
        result = agent.run_sync(
            "Process INV-42 for $100.00. Three submit_invoice calls, same args."
        )
        invoice_calls = [te for te in result.tool_executions if te.tool_name == "submit_invoice"]
        if len(invoice_calls) < 2:
            pytest.skip(
                f"Model only invoked submit_invoice {len(invoice_calls)}x; no duplicate to dedup"
            )
        cache_hits = [te for te in invoice_calls if te.idempotent_cache_hit]
        assert cache_hits, "no idempotent_cache_hit recorded despite duplicate calls"
        assert _InvoiceCounter.n < len(invoice_calls)

    def test_model_retry_hook_attaches_and_runs(self, anthropic_model):
        from tulip.agent import Agent
        from tulip.hooks.builtin import ModelRetryHook

        agent = Agent(
            model=anthropic_model,
            tools=[],
            hooks=[ModelRetryHook(max_retries=2)],
            system_prompt="Reply briefly.",
            max_iterations=2,
        )
        result = agent.run_sync("Say hi.")
        assert result.message
        assert result.stop_reason in ("complete", "no_tools", "terminal_tool")
