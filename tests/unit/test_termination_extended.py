# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for the rest of ``tulip.core.termination``.

The existing ``test_agent_termination.py`` covers the most commonly used
conditions. This file fills the remaining gaps:

- ``OrCondition`` short-circuits on first match
- ``TimeLimit`` first-call sets baseline + reset path
- ``TextMention`` falling back to messages and case-sensitive matching
- ``ConfidenceMet`` triggering at the threshold
- ``NoToolCalls`` reading the context flag
"""

from __future__ import annotations

import time

from tulip.core.messages import Message
from tulip.core.state import AgentState, ToolExecution
from tulip.core.termination import (
    ConfidenceMet,
    MaxIterations,
    NoToolCalls,
    OrCondition,
    TextMention,
    TimeLimit,
    ToolCalled,
)


def _state(
    *, iteration: int = 0, confidence: float = 0.0, messages: list[Message] | None = None
) -> AgentState:
    return AgentState(
        agent_id="a",
        iteration=iteration,
        confidence=confidence,
        messages=tuple(messages or []),
    )


# ---------------------------------------------------------------------------
# OrCondition
# ---------------------------------------------------------------------------


class TestOrCondition:
    def test_returns_first_matching_reason(self) -> None:
        # First child matches → OrCondition reports its reason
        cond = OrCondition(MaxIterations(1), ToolCalled("never"))
        stop, reason = cond.check(_state(iteration=2))
        assert stop is True
        assert reason == "max_iterations"

    def test_no_children_match(self) -> None:
        cond = OrCondition(MaxIterations(99), ToolCalled("absent"))
        stop, reason = cond.check(_state(iteration=1))
        assert stop is False
        assert reason is None


# ---------------------------------------------------------------------------
# TimeLimit
# ---------------------------------------------------------------------------


class TestTimeLimit:
    def test_first_call_sets_baseline(self) -> None:
        cond = TimeLimit(seconds=60.0)
        # First check sets ``_start`` to now and returns False (no time elapsed yet).
        stop, reason = cond.check(_state())
        assert stop is False
        assert reason is None

    def test_elapsed_triggers_time_budget(self) -> None:
        cond = TimeLimit(seconds=0.0)
        # 0-second budget — second check after a tick triggers stop.
        cond.check(_state())
        time.sleep(0.001)
        stop, reason = cond.check(_state())
        assert stop is True
        assert reason == "time_budget"

    def test_reset_clears_baseline(self) -> None:
        cond = TimeLimit(seconds=10.0)
        cond.check(_state())
        cond.reset()
        # After reset, ``_start`` is None again — verify by inspecting the
        # private attr; can't directly observe but a fresh check shouldn't
        # have a stale baseline.
        assert cond._start is None


# ---------------------------------------------------------------------------
# TextMention
# ---------------------------------------------------------------------------


class TestTextMention:
    def test_extracts_from_messages_when_no_context(self) -> None:
        cond = TextMention("DONE")
        msgs = [Message.assistant("the work is DONE!")]
        stop, reason = cond.check(_state(messages=msgs))
        assert stop is True
        assert reason == "text_mention:DONE"

    def test_no_messages_no_content(self) -> None:
        cond = TextMention("DONE")
        stop, _ = cond.check(_state())
        assert stop is False

    def test_case_sensitive_no_match(self) -> None:
        cond = TextMention("DONE", case_sensitive=True)
        msgs = [Message.assistant("the work is done!")]
        stop, _ = cond.check(_state(messages=msgs))
        assert stop is False

    def test_case_sensitive_match(self) -> None:
        cond = TextMention("DONE", case_sensitive=True)
        msgs = [Message.assistant("the work is DONE")]
        stop, reason = cond.check(_state(messages=msgs))
        assert stop is True
        assert reason == "text_mention:DONE"


# ---------------------------------------------------------------------------
# ConfidenceMet
# ---------------------------------------------------------------------------


class TestConfidenceMet:
    def test_triggers_at_threshold(self) -> None:
        cond = ConfidenceMet(threshold=0.8)
        stop, reason = cond.check(_state(confidence=0.9))
        assert stop is True
        assert reason == "confidence_met"

    def test_does_not_trigger_below(self) -> None:
        cond = ConfidenceMet(threshold=0.9)
        stop, _ = cond.check(_state(confidence=0.5))
        assert stop is False


# ---------------------------------------------------------------------------
# NoToolCalls
# ---------------------------------------------------------------------------


class TestNoToolCalls:
    def test_triggers_when_flag_true(self) -> None:
        cond = NoToolCalls()
        stop, reason = cond.check(_state(), no_tool_calls=True)
        assert stop is True
        assert reason == "no_tools"

    def test_no_trigger_when_flag_missing(self) -> None:
        cond = NoToolCalls()
        stop, _ = cond.check(_state())
        assert stop is False


# ---------------------------------------------------------------------------
# ToolCalled(require_success=True) — a rejected submit must not terminate
# ---------------------------------------------------------------------------


class TestToolCalledRequireSuccess:
    @staticmethod
    def _exec(error: str | None) -> ToolExecution:
        return ToolExecution(tool_name="submit_tool", tool_call_id="c1", arguments={}, error=error)

    def test_errored_call_does_not_terminate(self) -> None:
        cond = ToolCalled("submit_tool", require_success=True)
        state = _state().with_tool_execution(self._exec("claims rejected"))
        stop, _ = cond.check(state)
        assert stop is False

    def test_successful_call_terminates(self) -> None:
        cond = ToolCalled("submit_tool", require_success=True)
        state = _state().with_tool_execution(self._exec(None))
        stop, reason = cond.check(state)
        assert stop is True
        assert reason == "tool_called:submit_tool"

    def test_rejection_then_success_terminates(self) -> None:
        cond = ToolCalled("submit_tool", require_success=True)
        state = (
            _state()
            .with_tool_execution(self._exec("rejected"))
            .with_tool_execution(self._exec(None))
        )
        stop, _ = cond.check(state)
        assert stop is True

    def test_default_still_counts_errored_calls(self) -> None:
        # Back-compat: without the flag, any call terminates (old semantics).
        cond = ToolCalled("submit_tool")
        state = _state().with_tool_execution(self._exec("boom"))
        stop, _ = cond.check(state)
        assert stop is True
