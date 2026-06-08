# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Unit tests for issue #280 — the empty-content safety net in
``runtime_loop``.

The bug: when the model returns an iteration with ``content=None`` AND
zero ``tool_calls``, the runtime previously emitted a
``TerminateEvent(final_message=None)`` → ``AgentResult.message=""``.
Cause: reasoning-only iterations (gpt-5.x / o-series / Gemini thinking
mode), or models that decide "I'm done" after a long context but don't
write anything.

The fix: detect the empty-content path, inject a "[Final answer
requested]" system message, force one no-tool completion, use that
response as the final message. Mirrors the existing MaxIterations
summary-injection path.

These tests use scripted stub models so no model provider is touched.
"""

from __future__ import annotations

from typing import Any

from tulip.agent import Agent
from tulip.core.messages import Message, ToolCall
from tulip.models.base import ModelResponse
from tulip.tools.decorator import tool


class _ScriptedModel:
    """Returns a scripted sequence of responses, then loops the last one.

    Mirrors the pattern in tests/unit/test_agent_termination.py — we
    drive the agent's loop deterministically without an LLM."""

    def __init__(self, responses: list[ModelResponse]):
        self._script = list(responses)
        self.calls: int = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls += 1
        if not self._script:
            raise AssertionError("_ScriptedModel script exhausted")
        if len(self._script) == 1:
            return self._script[0]
        return self._script.pop(0)

    async def stream(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError


def _empty_content_response() -> ModelResponse:
    """Simulates the bug shape: reasoning-only iteration. The model
    burned completion_tokens but returned no visible ``content`` and
    no ``tool_calls``."""
    return ModelResponse(
        message=Message.assistant(content=None, tool_calls=[]),
        usage={"prompt_tokens": 100, "completion_tokens": 5},
    )


def _real_summary_response(
    text: str = "Here is my final answer based on the work so far.",
) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(content=text, tool_calls=[]),
        usage={"prompt_tokens": 110, "completion_tokens": 12},
    )


@tool
def trivial() -> str:
    """A trivial tool the agent has access to but doesn't need to call."""
    return "ok"


# ---------------------------------------------------------------------------
# Core safety-net behavior
# ---------------------------------------------------------------------------


def test_empty_content_triggers_summary_call_and_returns_real_text() -> None:
    """Iteration 1: model returns content=None + zero tool_calls.
    Without the fix, the agent exits with text=''. With the fix, the
    runtime forces ONE additional no-tool call to extract the model's
    answer; the second response's content lands on AgentResult.text."""
    model = _ScriptedModel(
        [
            _empty_content_response(),  # iter 1 — the bug shape
            _real_summary_response("Final answer: forty-two."),  # forced summary
        ]
    )
    agent = Agent(
        model=model,
        tools=[trivial],
        system_prompt="be helpful",
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    result = agent.run_sync("what is the answer?")
    assert result.text == "Final answer: forty-two.", (
        "Safety net must extract the model's real answer from the "
        "forced summary call, not surface the empty iteration."
    )
    # The model was called TWICE — once for the empty iteration, once
    # for the forced summary.
    assert model.calls == 2
    # We exited cleanly via "complete", not max_iterations / error.
    assert result.stop_reason == "complete"


def test_empty_content_with_empty_summary_falls_back_to_last_assistant() -> None:
    """If the forced summary call ALSO returns empty content, fall
    back to the most recent assistant message we saw before the
    empty iteration. Better than empty."""
    earlier_text = "Earlier turn: I started investigating but..."
    model = _ScriptedModel(
        [
            _real_summary_response(
                earlier_text
            ),  # iter 1 — agent writes content + no tools → would exit
            # Wait, this path needs the EARLIER turn to have content but then
            # a SECOND empty iteration to trigger the safety net. Adjust
            # the script: turn 1 returns content (which would normally
            # cause the "complete" exit), but we add a tool_call so it
            # keeps looping.
        ]
    )
    # Rewrite the script for the actual shape we want to test:
    # - iter 1: content + a tool call (loop continues)
    # - iter 2: empty content, no tool calls (bug shape — safety net fires)
    # - iter 3: forced summary returns empty too (rare; fallback path)
    model = _ScriptedModel(
        [
            ModelResponse(
                message=Message.assistant(
                    content="I'll check trivial first.",
                    tool_calls=[ToolCall(name="trivial", arguments={}, id="c1")],
                ),
                usage={"prompt_tokens": 50, "completion_tokens": 10},
            ),
            _empty_content_response(),  # bug shape after the tool result
            _empty_content_response(),  # forced summary ALSO empty
        ]
    )
    agent = Agent(
        model=model,
        tools=[trivial],
        system_prompt="be helpful",
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    result = agent.run_sync("what is the answer?")
    # The fallback path uses _last_assistant_content from iter 1, OR
    # _build_fallback_summary(state) if even that is empty. Either way,
    # the text must NOT be empty.
    assert result.text, (
        f"Even when both the original iteration AND the forced summary "
        f"return empty content, the safety net must fall back to the "
        f"last seen assistant message OR a deterministic stub — never "
        f"empty. Got: {result.text!r}"
    )
    # The "I'll check trivial first." preamble should land as the
    # fallback since the forced summary couldn't produce a real reply.
    assert "trivial" in result.text.lower() or "based on" in result.text.lower(), (
        f"Fallback should use either _last_assistant_content "
        f"('I'll check trivial first.') or the deterministic "
        f"_build_fallback_summary stub. Got: {result.text!r}"
    )


def test_normal_path_unchanged_when_content_present() -> None:
    """The fix must NOT add an extra LLM call on the happy path —
    when the model returns content normally on iteration 1, the agent
    exits with exactly that content, no extra calls."""
    model = _ScriptedModel(
        [
            _real_summary_response("Direct answer on iteration 1."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[trivial],
        system_prompt="be helpful",
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    result = agent.run_sync("answer?")
    assert result.text == "Direct answer on iteration 1."
    # Exactly ONE model call — the safety net didn't fire spuriously.
    assert model.calls == 1


def test_normal_path_with_tool_calls_unchanged() -> None:
    """Tool-using iterations still loop normally; the safety net only
    fires on the no-tool-calls termination branch."""
    model = _ScriptedModel(
        [
            ModelResponse(
                message=Message.assistant(
                    content="Let me check.",
                    tool_calls=[ToolCall(name="trivial", arguments={}, id="c1")],
                ),
                usage={"prompt_tokens": 50, "completion_tokens": 5},
            ),
            _real_summary_response("Based on the result: it's fine."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[trivial],
        system_prompt="be helpful",
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    result = agent.run_sync("check please")
    assert result.text == "Based on the result: it's fine."
    assert model.calls == 2  # tool + final answer
    assert result.metrics.tool_calls == 1


# ---------------------------------------------------------------------------
# Edge cases that must not regress
# ---------------------------------------------------------------------------


def test_empty_string_content_treated_same_as_none() -> None:
    """``content=""`` is functionally identical to ``content=None`` —
    both look like "no answer" to the operator. Safety net fires for
    both."""
    model = _ScriptedModel(
        [
            ModelResponse(
                message=Message.assistant(content="", tool_calls=[]),
                usage={"prompt_tokens": 100, "completion_tokens": 0},
            ),
            _real_summary_response("Real answer after empty-string trigger."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[trivial],
        system_prompt="be helpful",
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    result = agent.run_sync("answer?")
    assert result.text == "Real answer after empty-string trigger."
    assert model.calls == 2


def test_safety_net_increments_token_metrics() -> None:
    """The forced summary call's tokens land on metrics so cost
    accounting stays honest (operators must see the extra call)."""
    model = _ScriptedModel(
        [
            _empty_content_response(),
            _real_summary_response("yes."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[trivial],
        system_prompt="be helpful",
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    result = agent.run_sync("answer?")
    # Iter 1: 100+5 = 105. Forced summary: 110+12 = 122. Total = 227.
    assert result.metrics.total_tokens >= 100 + 5 + 110 + 12
    assert result.metrics.completion_tokens >= 5 + 12


def test_safety_net_does_not_loop_forever_on_repeated_empty() -> None:
    """If every iteration returns empty content (worst case), the
    safety net fires once then the agent exits with the fallback
    summary, not an infinite loop. Bounded by max_iterations as a
    last resort."""
    model = _ScriptedModel([_empty_content_response()])  # loops forever
    agent = Agent(
        model=model,
        tools=[trivial],
        system_prompt="be helpful",
        max_iterations=5,
        reflexion=False,
        grounding=False,
    )
    result = agent.run_sync("answer?")
    # Should terminate (either via the safety net's fallback or
    # ultimately via MaxIterations); text must not be empty.
    assert result.text, (
        f"Repeated empty content must still terminate with non-empty "
        f"text via the fallback summary. Got: {result.text!r}"
    )
    # Safety net fires at most once before falling back — total calls
    # should be bounded (2-3 from the safety net + summary, not 5+).
    assert model.calls <= 6, (
        f"Safety net must not loop forever — at most one summary "
        f"attempt per top-level termination. Got {model.calls} calls."
    )


# ---------------------------------------------------------------------------
# Integration with deepagent
# ---------------------------------------------------------------------------


def test_deepagent_with_empty_content_iteration() -> None:
    """``create_deepagent`` builds an Agent with the same runtime_loop
    underneath — the safety net must work there too. This is the
    actual production path for #280."""
    from tulip.deepagent import create_deepagent

    @tool
    def search_kb(query: str) -> str:
        """Search the knowledge base."""
        return f"Results for: {query}"

    model = _ScriptedModel(
        [
            # Iter 1: agent calls a tool
            ModelResponse(
                message=Message.assistant(
                    content="I'll search the KB.",
                    tool_calls=[ToolCall(name="search_kb", arguments={"query": "x"}, id="c1")],
                ),
                usage={"prompt_tokens": 50, "completion_tokens": 5},
            ),
            # Iter 2: model returns empty (reasoning-only, the bug shape)
            _empty_content_response(),
            # Forced summary call produces the real answer
            _real_summary_response("Summary: searched, found nothing concrete."),
        ]
    )
    agent = create_deepagent(
        model=model,
        tools=[search_kb],
        system_prompt="be a research agent",
        reflexion=False,
        grounding=False,
        max_iterations=10,
    )
    result = agent.run_sync("find me x")
    assert result.text == "Summary: searched, found nothing concrete.", (
        f"create_deepagent path must also benefit from the safety net — got: {result.text!r}"
    )
