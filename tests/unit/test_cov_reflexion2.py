# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage-gap tests for tulip.reasoning.reflexion.

Targets the specific uncovered lines:
  - 175 (dead code — unreachable after the line 168 guard)
  - 188 (_get_recent_executions break when >= 5 executions)
  - 211 (_detect_loop return None fallback path)
  - 219-220 (unreachable — inside reasoning_steps < threshold guard)
  - 263 (_detect_loop return None when tool_sets varied)
  - 270-287 (_detect_alternating_pattern method — never called internally)
  - 406 (_summarize_findings return "" for empty list — dead code from caller)
"""

from __future__ import annotations

from tulip.core.messages import ToolCall
from tulip.core.state import AgentState, ReasoningStep, ToolExecution
from tulip.reasoning.reflexion import (
    AssessmentCategory,
    ReflectionResult,
    Reflector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exec(tool: str, *, error: str | None = None) -> ToolExecution:
    return ToolExecution(
        tool_name=tool,
        tool_call_id=f"cid_{tool}",
        arguments={},
        error=error,
    )


def _make_step(tool: str, iteration: int) -> ReasoningStep:
    return ReasoningStep(
        iteration=iteration,
        tool_calls=[ToolCall(name=tool, arguments={})],
    )


# ---------------------------------------------------------------------------
# _get_recent_executions — break when >= 5 items (line 188)
# ---------------------------------------------------------------------------


class TestGetRecentExecutions:
    def test_caps_at_five_executions(self) -> None:
        """_get_recent_executions stops at 5 (hits `break` on line 188)."""
        reflector = Reflector()
        state = AgentState()
        for i in range(7):
            state = state.with_tool_execution(_make_exec(f"tool_{i}"))

        recent = reflector._get_recent_executions(state)
        assert len(recent) <= 5

    def test_returns_last_five_in_order(self) -> None:
        """The last-5 slice is returned in chronological order."""
        reflector = Reflector()
        state = AgentState()
        for i in range(6):
            state = state.with_tool_execution(_make_exec(f"t{i}"))

        recent = reflector._get_recent_executions(state)
        # Chronological — last item should be the 6th tool
        assert recent[-1].tool_name == "t5"


# ---------------------------------------------------------------------------
# _detect_loop — fallback path returns None (line 211)
# ---------------------------------------------------------------------------


class TestDetectLoopFallback:
    def test_returns_none_when_iteration_ge_threshold_but_few_steps_and_history(
        self,
    ) -> None:
        """When iteration >= threshold but reasoning_steps < threshold AND
        tool_history < threshold, _detect_loop returns None (line 211)."""
        reflector = Reflector(loop_threshold=3)

        # Build a state with iteration=3 but no reasoning_steps and
        # only 1 entry in tool_history (< loop_threshold=3).
        state = AgentState(iteration=3)
        state = state.with_tool_execution(_make_exec("search"))
        # Only 1 tool in history; no reasoning steps.

        result = reflector._detect_loop(state)
        assert result is None

    def test_returns_none_when_few_steps_and_history_not_all_same(self) -> None:
        """Fallback path: few reasoning steps, tool_history >= threshold
        but not all-same → returns None (end of fallback branch)."""
        reflector = Reflector(loop_threshold=3)

        state = AgentState(iteration=3)
        # 3 tool executions (tool_history will have 3 items) but all different
        for tool in ("search", "read", "calc"):
            state = state.with_tool_execution(_make_exec(tool))
        # No reasoning_steps → len(state.reasoning_steps) < 3

        result = reflector._detect_loop(state)
        # Different tools → no loop → returns None
        assert result is None


# ---------------------------------------------------------------------------
# _detect_loop — returns None at end when tool_sets are varied (line 263)
# ---------------------------------------------------------------------------


class TestDetectLoopVariedSets:
    def test_returns_none_when_tool_sets_are_varied(self) -> None:
        """_detect_loop returns None when recent tool sets differ and no
        alternating pattern — falls through to the final `return None` (line 263)."""
        reflector = Reflector(loop_threshold=3)

        state = AgentState()
        # Three steps with DIFFERENT tool sets — not a loop
        for i, tool in enumerate(("search", "read", "calc")):
            step = _make_step(tool, i + 1)
            state = state.with_reasoning_step(step)
            state = state.with_tool_execution(_make_exec(tool))
            state = state.next_iteration()

        result = reflector._detect_loop(state)
        assert result is None

    def test_returns_none_when_tool_sets_present_but_not_looping(self) -> None:
        """Varied tool sets across iterations produce no loop detection."""
        reflector = Reflector(loop_threshold=3)
        state = AgentState()

        tools_per_step = [["search", "read"], ["calc"], ["write"]]
        for i, tools in enumerate(tools_per_step):
            step = ReasoningStep(
                iteration=i + 1,
                tool_calls=[ToolCall(name=t, arguments={}) for t in tools],
            )
            state = state.with_reasoning_step(step)
            state = state.next_iteration()

        result = reflector._detect_loop(state)
        assert result is None


# ---------------------------------------------------------------------------
# _detect_alternating_pattern — direct calls (lines 270-287)
# ---------------------------------------------------------------------------


class TestDetectAlternatingPattern:
    def test_returns_none_when_fewer_than_four_tools(self) -> None:
        """Returns None immediately when tool_history has < 4 entries (line 272)."""
        reflector = Reflector()
        state = AgentState()
        for t in ("a", "b", "a"):
            state = state.with_tool_execution(_make_exec(t))

        result = reflector._detect_alternating_pattern(state)
        assert result is None

    def test_detects_abab_pattern(self) -> None:
        """Returns LOOP_DETECTED for a classic A-B-A-B tool pattern (lines 275-277)."""
        reflector = Reflector()
        state = AgentState()
        for t in ("alpha", "beta", "alpha", "beta"):
            state = state.with_tool_execution(_make_exec(t))

        result = reflector._detect_alternating_pattern(state)
        assert result is not None
        assert result.assessment == AssessmentCategory.LOOP_DETECTED
        assert result.loop_pattern is not None
        assert "alpha" in result.loop_pattern
        assert "beta" in result.loop_pattern
        assert result.confidence_delta < 0

    def test_returns_none_when_not_alternating(self) -> None:
        """Returns None when tool_history has 4 items but no A-B-A-B pattern (line 287)."""
        reflector = Reflector()
        state = AgentState()
        for t in ("a", "b", "c", "d"):
            state = state.with_tool_execution(_make_exec(t))

        result = reflector._detect_alternating_pattern(state)
        assert result is None

    def test_returns_none_when_all_same(self) -> None:
        """Returns None when A==B (same tool repeated, not alternating)."""
        reflector = Reflector()
        state = AgentState()
        for _ in range(4):
            state = state.with_tool_execution(_make_exec("search"))

        result = reflector._detect_alternating_pattern(state)
        # recent[0] == recent[1] so recent[0] != recent[1] is False
        assert result is None


# ---------------------------------------------------------------------------
# _summarize_findings — empty list returns "" (line 406)
# ---------------------------------------------------------------------------


class TestSummarizeFindings:
    def test_empty_list_returns_empty_string(self) -> None:
        """_summarize_findings([]) returns '' (line 406)."""
        reflector = Reflector()
        result = reflector._summarize_findings([])
        assert result == ""

    def test_short_content_returned_verbatim(self) -> None:
        """Content under 200 chars is returned as-is."""
        reflector = Reflector()
        result = reflector._summarize_findings(["hello", "world"])
        assert result == "hello world"

    def test_long_content_truncated(self) -> None:
        """Content over 200 chars is truncated with '...'."""
        reflector = Reflector()
        long_str = "x" * 300
        result = reflector._summarize_findings([long_str])
        assert len(result) == 200
        assert result.endswith("...")


# ---------------------------------------------------------------------------
# _detect_loop — 4+ varied tool sets where inner alternating check is False
# (covers branch 248->263: inside the loop_threshold>=4 / len>=4 block but
# the A-B-A-B condition is False, falling through to return None)
# ---------------------------------------------------------------------------


class TestDetectLoopAlternatingFalse:
    def test_four_steps_with_varied_non_alternating_sets(self) -> None:
        """loop_threshold>=4 + 4 tool_sets but NOT alternating → return None (248->263)."""
        reflector = Reflector(loop_threshold=4)
        state = AgentState()
        # Four DIFFERENT tools per step → not a loop and not alternating
        for i, tool in enumerate(("alpha", "beta", "gamma", "delta")):
            step = _make_step(tool, i + 1)
            state = state.with_reasoning_step(step)
            state = state.with_tool_execution(_make_exec(tool))
            state = state.next_iteration()

        result = reflector._detect_loop(state)
        assert result is None

    def test_step_with_no_tool_calls_is_skipped(self) -> None:
        """A ReasoningStep with empty tool_calls does not contribute a tool_set (232->231)."""
        reflector = Reflector(loop_threshold=3)
        state = AgentState()
        # Two steps WITH tool calls, one step WITHOUT
        for i, tool in enumerate(("search", "read")):
            step = _make_step(tool, i + 1)
            state = state.with_reasoning_step(step)
            state = state.next_iteration()
        # A step with no tool_calls — hits the `if step.tool_calls:` False branch
        empty_step = ReasoningStep(iteration=3, tool_calls=[])
        state = state.with_reasoning_step(empty_step)
        state = state.next_iteration()

        result = reflector._detect_loop(state)
        # Only 2 tool_sets out of 3 steps → can't be a loop of 3
        assert result is None


# ---------------------------------------------------------------------------
# _analyze_executions — successful execution with result=None (305->302)
# ---------------------------------------------------------------------------


class TestAnalyzeExecutionsNullResult:
    def test_success_with_no_result_doesnt_append_content(self) -> None:
        """A successful execution with result=None is counted as success but
        no content is added to results_content (covers branch 305->302)."""
        reflector = Reflector()
        # result=None, error=None → success=True but result is falsy
        executions = [
            ToolExecution(
                tool_name="ping",
                tool_call_id="c1",
                arguments={},
                result=None,
            )
        ]
        success, error, contents = reflector._analyze_executions(executions)
        assert success == 1
        assert error == 0
        assert contents == []


# ---------------------------------------------------------------------------
# _generate_loop_guidance — with error in recent executions (line 427)
# ---------------------------------------------------------------------------


class TestGenerateLoopGuidance:
    def test_includes_error_detail_when_present(self) -> None:
        """Guidance appends the latest error message when errors exist."""
        reflector = Reflector(loop_threshold=3)
        state = AgentState()
        state = state.with_tool_execution(_make_exec("search", error="timeout"))
        guidance = reflector._generate_loop_guidance("search", state)
        assert "timeout" in guidance

    def test_no_error_no_extra_step(self) -> None:
        """Guidance without errors doesn't include error step."""
        reflector = Reflector(loop_threshold=3)
        state = AgentState()
        state = state.with_tool_execution(_make_exec("search"))
        guidance = reflector._generate_loop_guidance("search", state)
        assert "search" in guidance


# ---------------------------------------------------------------------------
# _generate_stuck_guidance — with / without errors (covers assess paths)
# ---------------------------------------------------------------------------


class TestGenerateStuckGuidance:
    def test_with_errors_cites_last_error(self) -> None:
        """Guidance for stuck state with errors cites the last error."""
        reflector = Reflector()
        state = AgentState()
        state = state.with_tool_execution(_make_exec("fetch", error="404 not found"))
        guidance = reflector._generate_stuck_guidance(1, state)
        assert "404 not found" in guidance

    def test_without_errors_returns_generic_guidance(self) -> None:
        """Guidance when error_count=0 and no tool errors."""
        reflector = Reflector()
        state = AgentState()
        guidance = reflector._generate_stuck_guidance(0, state)
        assert "stalled" in guidance.lower() or "progress" in guidance.lower()


# ---------------------------------------------------------------------------
# reflect — misc branches that aren't hit yet
# ---------------------------------------------------------------------------


class TestReflectMiscBranches:
    def test_reflect_uses_state_executions_when_none_passed(self) -> None:
        """When iteration_executions=None, uses _get_recent_executions."""
        reflector = Reflector()
        state = AgentState()
        state = state.with_tool_execution(
            ToolExecution(
                tool_name="search",
                tool_call_id="c1",
                arguments={},
                result="a" * 200,
            )
        )
        result = reflector.reflect(state)
        # Should run without error; exact assessment depends on implementation
        assert isinstance(result, ReflectionResult)

    def test_reflect_on_track_with_slow_progress(self) -> None:
        """Iteration with small positive delta but no findings → ON_TRACK with hint."""
        reflector = Reflector(
            success_weight=0.01,  # tiny delta
            min_progress_delta=0.05,
        )
        executions = [
            ToolExecution(
                tool_name="search",
                tool_call_id="c1",
                arguments={},
                result="ok",  # < 100 chars, so no NEW_FINDINGS
            )
        ]
        result = reflector.reflect(AgentState(), executions)
        # With tiny success_weight=0.01 and min_progress_delta=0.05,
        # delta < min_progress_delta, so falls through to default ON_TRACK
        assert result.assessment == AssessmentCategory.ON_TRACK

    def test_create_guidance_message_on_track_no_guidance(self) -> None:
        """create_guidance_message returns None when guidance is None."""
        reflector = Reflector()
        result = ReflectionResult(
            assessment=AssessmentCategory.ON_TRACK,
            guidance=None,
        )
        assert reflector.create_guidance_message(result) is None

    def test_create_guidance_message_with_loop_pattern(self) -> None:
        """create_guidance_message includes loop_pattern when present."""
        reflector = Reflector()
        result = ReflectionResult(
            assessment=AssessmentCategory.LOOP_DETECTED,
            guidance="Try something else",
            loop_pattern="search repeated 3 times",
        )
        msg = reflector.create_guidance_message(result)
        assert msg is not None
        assert "search repeated 3 times" in msg

    def test_create_guidance_message_with_findings(self) -> None:
        """create_guidance_message includes findings_summary when present."""
        reflector = Reflector()
        result = ReflectionResult(
            assessment=AssessmentCategory.NEW_FINDINGS,
            guidance="Continue",
            findings_summary="key datum found",
        )
        msg = reflector.create_guidance_message(result)
        assert msg is not None
        assert "key datum found" in msg
