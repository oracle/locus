# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage tests for the public facade in ``locus.agent.agent``.

The end-to-end agent loop is exercised by other test files. This file
targets the small public/utility surface that the loop tests don't hit:

- ``_normalize_stop_reason`` (the copy in ``locus.agent.agent``)
- ``Agent.cancel`` initialising the signal lazily
- ``Agent.is_cancelled`` reflecting the signal
- ``Agent.resume`` raising when no interrupt is pending
- ``Agent.invoke`` aliasing run_sync
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from locus.agent.agent import Agent, _normalize_stop_reason
from locus.agent.config import AgentConfig


# ---------------------------------------------------------------------------
# _normalize_stop_reason (the agent.agent copy)
# ---------------------------------------------------------------------------


class TestAgentNormalizeStopReason:
    def test_empty_returns_complete(self) -> None:
        assert _normalize_stop_reason(None) == "complete"
        assert _normalize_stop_reason("") == "complete"

    def test_known_passes_through(self) -> None:
        assert _normalize_stop_reason("max_iterations") == "max_iterations"

    def test_tool_called_routes_to_terminal(self) -> None:
        assert _normalize_stop_reason("tool_called:save_state") == "terminal_tool"

    def test_text_mention_routes_to_complete(self) -> None:
        assert _normalize_stop_reason("text_mention:DONE") == "complete"

    def test_substring_known_returns_match(self) -> None:
        assert _normalize_stop_reason("error talking to model") == "error"

    def test_unknown_falls_back_to_complete(self) -> None:
        assert _normalize_stop_reason("nonsense reason") == "complete"


# ---------------------------------------------------------------------------
# Agent.cancel + is_cancelled
# ---------------------------------------------------------------------------


def _make_agent() -> Agent:
    """Build an Agent with a mocked model (skip real provider construction)."""
    cfg = AgentConfig(model="anthropic:claude-sonnet-4-20250514")
    agent = Agent(config=cfg)
    # Avoid touching the real Anthropic SDK during attribute access.
    agent._model = MagicMock()  # type: ignore[assignment]
    return agent


class TestCancel:
    def test_cancel_initialises_signal_lazily(self) -> None:
        agent = _make_agent()
        assert agent._cancel_signal is None
        agent.cancel()
        assert agent._cancel_signal is not None
        assert agent._cancel_signal.is_set()

    def test_is_cancelled_reflects_signal(self) -> None:
        agent = _make_agent()
        assert agent.is_cancelled is False
        agent.cancel()
        assert agent.is_cancelled is True


# ---------------------------------------------------------------------------
# Agent.resume — guard when no interrupt
# ---------------------------------------------------------------------------


class TestResumeGuard:
    @pytest.mark.asyncio
    async def test_resume_without_interrupt_raises(self) -> None:
        agent = _make_agent()
        gen = agent.resume("answer")
        with pytest.raises(RuntimeError, match="No interrupt"):
            await gen.__anext__()


# ---------------------------------------------------------------------------
# Agent.invoke
# ---------------------------------------------------------------------------


class TestInvokeAliasesRunSync:
    def test_invoke_delegates_to_run_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = _make_agent()
        called = {}

        def _fake_run_sync(self_: Agent, prompt: str, **kw: object) -> str:
            called["prompt"] = prompt
            called["kw"] = kw
            return "ok"

        monkeypatch.setattr(Agent, "run_sync", _fake_run_sync, raising=True)
        out = agent.invoke("hi", thread_id="t1")
        assert out == "ok"
        assert called["prompt"] == "hi"
        assert called["kw"] == {"thread_id": "t1", "metadata": None}
