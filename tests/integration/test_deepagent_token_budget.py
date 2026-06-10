# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the 0.2.0b23 ``total_token_budget`` rename.

These tests reproduce the empty-output bug that motivated the rename
and verify the fix end-to-end against a stub model:

  - **Reproduce the bug shape**: an explicit small ``total_token_budget``
    plus a long system prompt fires ``TokenLimit`` termination on
    iteration 1, exiting with empty (or near-empty) output. This was
    happening silently before the rename because callers passed
    ``max_tokens=65536`` expecting the per-completion meaning every
    LLM SDK uses, but Tulip interpreted it as the run-level cap.
  - **Verify the fix**: with the default ``total_token_budget=None``
    a long-prompt run terminates only via ``MaxIterations`` / the
    submit_tool path — no token-based termination interferes.
  - **Verify the loud rejection**: passing ``max_tokens=`` to the
    new factory raises ``TypeError`` instead of silently flowing
    through to the per-completion field.

The tests use Tulip's stub model surface so no provider
credentials are required.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from tulip import create_deepagent
from tulip.core.termination import (
    AndCondition,
    MaxIterations,
    OrCondition,
    TokenLimit,
)
from tulip.tools.decorator import tool


class _Echo(BaseModel):
    text: str
    confidence: float = 0.0


@tool
def submit_research(text: str, confidence: float) -> str:
    """Final-answer tool the deepagent terminates on."""
    return f"submitted: {text}"


def _stub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide just enough config to satisfy the model string parser
    without making any real network calls. Tests below only inspect
    the agent's configured termination + kwargs — they never invoke
    the model."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Bug-shape reproduction
# ---------------------------------------------------------------------------


def _walk_leaves(node):
    leaves: list = []

    def _walk(n):
        if isinstance(n, (OrCondition, AndCondition)):
            for child in n._conditions:
                _walk(child)
        else:
            leaves.append(n)

    _walk(node)
    return leaves


def test_explicit_small_budget_attaches_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit small ``total_token_budget`` attaches a
    ``TokenLimit`` term to the termination algebra — this is what
    the legacy ``max_tokens=80_000`` default silently did. The fix
    surfaces it explicitly so callers know they're opting in."""
    _stub_env(monkeypatch)
    agent = create_deepagent(
        model="openai:gpt-4o-mini",
        tools=[submit_research],
        system_prompt="be helpful",
        output_schema=_Echo,
        reflexion=False,
        grounding=False,
        total_token_budget=8_000,  # explicit small cap
        max_iterations=40,
    )
    leaves = _walk_leaves(agent.config.termination)
    token_terms = [leaf for leaf in leaves if isinstance(leaf, TokenLimit)]
    assert len(token_terms) == 1, (
        f"An explicit total_token_budget must attach exactly one TokenLimit term; got {token_terms}"
    )
    # The MaxIterations leaf is always present as the safety net.
    assert any(isinstance(leaf, MaxIterations) for leaf in leaves)


def test_default_budget_none_removes_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default ``total_token_budget=None`` — the fix's main payload.

    Reproduces the empty-output bug shape by inspecting the
    termination tree: under the old API a caller would pass
    ``max_tokens=65536`` expecting the per-completion cap from every
    other LLM SDK; the old code attached
    ``TokenLimit(65_536)`` to termination, and a long system prompt
    on a multi-iteration run would exceed it before the model wrote
    a single completion-token.

    With the rename, the default has ZERO TokenLimit — so the same
    caller, passing nothing, doesn't get silently killed by a
    cumulative token cap."""
    _stub_env(monkeypatch)
    agent = create_deepagent(
        model="openai:gpt-4o-mini",
        tools=[submit_research],
        system_prompt="a " * 5_000,  # long-ish system prompt
        output_schema=_Echo,
        reflexion=False,
        grounding=False,
        # NOTE: NO total_token_budget — using the fixed default.
        max_iterations=12,
    )
    leaves = _walk_leaves(agent.config.termination)
    token_terms = [leaf for leaf in leaves if isinstance(leaf, TokenLimit)]
    assert token_terms == [], (
        "Default deepagent termination must not include a TokenLimit "
        "term — that was the silent-failure default before 0.2.0b23. "
        f"Got: {token_terms}"
    )


def test_long_prompt_with_no_budget_still_runs_via_max_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the default budget=None, a long-prompt run is bounded
    only by MaxIterations + the submit_tool path. This is the
    behavioral contract that fixes the empty-output bug for
    long-narrative deep research."""
    _stub_env(monkeypatch)
    agent = create_deepagent(
        model="openai:gpt-4o-mini",
        tools=[submit_research],
        # Simulate a graph-grounded research prompt
        system_prompt="\n".join([f"line {i}" for i in range(2_000)]),
        output_schema=_Echo,
        reflexion=False,
        grounding=False,
        max_iterations=15,
    )
    leaves = _walk_leaves(agent.config.termination)
    # Termination = (ToolCalled & ConfidenceMet) | MaxIterations
    max_iter_leaves = [leaf for leaf in leaves if isinstance(leaf, MaxIterations)]
    assert len(max_iter_leaves) == 1
    # No token limit — the run isn't bounded by cumulative tokens.
    assert not any(isinstance(leaf, TokenLimit) for leaf in leaves)


# ---------------------------------------------------------------------------
# Loud rejection of the removed kwarg
# ---------------------------------------------------------------------------


def test_legacy_max_tokens_kwarg_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing the removed ``max_tokens=`` raises TypeError with a
    clear message pointing to the new names. This is the load-bearing
    breaking change of 0.2.0b23 — callers either migrate or fail
    loud, never silently get wrong behavior."""
    _stub_env(monkeypatch)
    with pytest.raises(TypeError) as excinfo:
        create_deepagent(
            model="openai:gpt-4o-mini",
            tools=[submit_research],
            system_prompt="be helpful",
            output_schema=_Echo,
            reflexion=False,
            grounding=False,
            max_tokens=65_536,
        )
    msg = str(excinfo.value)
    assert "total_token_budget" in msg, "TypeError must point callers to the new run-level kwarg"
    assert "max_output_tokens" in msg, (
        "TypeError must also point callers to the per-completion kwarg "
        "so the migration choice is explicit"
    )
    assert "0.2.0b23" in msg, "TypeError should cite the version for migration tracking"


def test_max_output_tokens_lands_on_agent_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_output_tokens`` flows to ``AgentConfig.max_tokens`` —
    that's the per-completion cap field the model provider
    forwards to every LLM request. Verifies the wiring stays correct
    after the rename so callers who set the per-completion knob via
    the new name don't accidentally land it elsewhere."""
    _stub_env(monkeypatch)
    agent = create_deepagent(
        model="openai:gpt-4o-mini",
        tools=[submit_research],
        system_prompt="be helpful",
        output_schema=_Echo,
        reflexion=False,
        grounding=False,
        max_output_tokens=65_536,
    )
    assert agent.config.max_tokens == 65_536, (
        "max_output_tokens must land on AgentConfig.max_tokens (the "
        "per-completion field on every LLM request) — that's the "
        "knob callers want when they set the new name."
    )
