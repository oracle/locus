# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for the main ``Agent.run`` ReAct loop in
``tulip.agent.runtime_loop``.

Drives the error / interrupt / budget / replan / checkpoint branches that
the happy-path end-to-end tests don't reach. All tests use scripted stub
models — no provider is touched.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from tulip.agent import Agent, GroundingConfig
from tulip.agent.runtime_loop import AgentRuntimeMixin
from tulip.core.events import (
    GroundingEvent,
    InterruptEvent,
    TerminateEvent,
    ToolCompleteEvent,
)
from tulip.core.messages import Message, ToolCall
from tulip.core.state import AgentState
from tulip.hooks.provider import AfterToolCallEvent, HookPriority, HookProvider
from tulip.models.base import ModelResponse
from tulip.tools.decorator import tool
from tulip.tools.executor import SequentialExecutor


# ---------------------------------------------------------------------------
# Scripted model + helpers
# ---------------------------------------------------------------------------


class _ScriptedModel:
    """Replay a fixed list of responses; loop the last one when ``loop_last``."""

    def __init__(self, responses: list[ModelResponse], *, loop_last: bool = False):
        self._responses = list(responses)
        self.loop_last = loop_last
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls += 1
        if not self._responses:
            return ModelResponse(message=Message.assistant("done"), usage={})
        if len(self._responses) == 1 and self.loop_last:
            return self._responses[0]
        return self._responses.pop(0)

    async def stream(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def _tc(
    name: str, args: dict[str, Any], *, tc_id: str = "c1", content: str = "thinking"
) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(
            content=content, tool_calls=[ToolCall(id=tc_id, name=name, arguments=args)]
        ),
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )


def _text(content: str | None) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(content=content),
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )


def _empty() -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(content=None, tool_calls=[]),
        usage={"prompt_tokens": 5, "completion_tokens": 5},
    )


@tool
def trivial() -> str:
    """A trivial tool."""
    return "ok"


class _SyncOnlyManager:
    """A conversation manager with only a sync ``apply`` (no ``async_apply``)."""

    def apply(self, messages: list[Message]) -> list[Message]:
        return list(messages)


async def _collect(agent: Agent, prompt: str) -> list[Any]:
    events: list[Any] = []
    async for ev in agent.run(prompt):
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# Long-term memory hooks (lines 217 + 1096)
# ---------------------------------------------------------------------------


class _FakeMemory:
    def __init__(self) -> None:
        self.started = False
        self.ended = False

    async def on_session_start(self, state: AgentState) -> AgentState:
        self.started = True
        return state

    async def on_session_end(self, state: AgentState) -> None:
        self.ended = True


async def test_memory_manager_session_hooks_run() -> None:
    """``on_session_start`` / ``on_session_end`` fire around the loop."""
    mem = _FakeMemory()
    agent = Agent(
        model=_ScriptedModel([_text("hello")]),
        memory_manager=mem,
        reflexion=False,
        grounding=False,
    )
    events = await _collect(agent, "hi")
    assert mem.started is True
    assert mem.ended is True
    assert any(isinstance(e, TerminateEvent) for e in events)
    # initializer wired the manager onto the private attr.
    assert agent._memory_manager is mem


# ---------------------------------------------------------------------------
# External cancellation (lines 237-244 + finally clear at 1075)
# ---------------------------------------------------------------------------


async def test_external_cancellation_terminates() -> None:
    agent = Agent(model=_ScriptedModel([_text("hi")], loop_last=True))
    agent.cancel()
    assert agent._cancel_signal is not None
    events = await _collect(agent, "go")
    term = next(e for e in events if isinstance(e, TerminateEvent))
    assert term.reason == "cancelled"
    # finally-clause cleared the signal.
    assert agent._cancel_signal is not None
    assert not agent._cancel_signal.is_set()


# ---------------------------------------------------------------------------
# max_iterations summary call w/ conversation manager (lines 284-287)
# ---------------------------------------------------------------------------


async def test_max_iterations_summary_with_async_conversation_manager() -> None:
    """The summary call applies an async conversation manager (284-285)."""
    from tulip.memory.conversation import SlidingWindowManager

    agent = Agent(
        model=_ScriptedModel([_tc("trivial", {})], loop_last=True),
        tools=[trivial],
        max_iterations=1,
        conversation_manager=SlidingWindowManager(window_size=10),
        reflexion=False,
        grounding=False,
    )
    events = await _collect(agent, "loop please")
    term = next(e for e in events if isinstance(e, TerminateEvent))
    assert term.reason == "max_iterations"


async def test_max_iterations_summary_with_sync_conversation_manager() -> None:
    """The summary call falls back to the sync ``apply`` branch (287)."""
    agent = Agent(
        model=_ScriptedModel([_tc("trivial", {})], loop_last=True),
        tools=[trivial],
        max_iterations=1,
        conversation_manager=_SyncOnlyManager(),
        reflexion=False,
        grounding=False,
    )
    events = await _collect(agent, "loop please")
    term = next(e for e in events if isinstance(e, TerminateEvent))
    assert term.reason == "max_iterations"


# ---------------------------------------------------------------------------
# Explicit-mode budget warnings (lines 355 + 363)
# ---------------------------------------------------------------------------


async def test_explicit_mode_budget_warnings_injected() -> None:
    """A 3-iteration explicit run hits both the ``remaining == 2`` and
    ``remaining == 0`` budget-warning branches."""
    agent = Agent(
        model=_ScriptedModel([_tc("trivial", {})], loop_last=True),
        tools=[trivial],
        completion_mode="explicit",
        max_iterations=3,
        reflexion=False,
        grounding=False,
    )
    events = await _collect(agent, "do work")
    term = next(e for e in events if isinstance(e, TerminateEvent))
    assert term.reason == "max_iterations"


# ---------------------------------------------------------------------------
# Grounding replan path (lines 442-454)
# ---------------------------------------------------------------------------


async def test_grounding_failure_triggers_replan(monkeypatch: pytest.MonkeyPatch) -> None:
    """When grounding requires a replan, guidance is injected and the loop
    re-enters (442-454)."""

    async def _forced_replan(self: Any, state: AgentState, final_response: str) -> Any:
        return (
            GroundingEvent(
                score=0.1,
                claims_evaluated=1,
                ungrounded_claims=["ungrounded"],
                requires_replan=True,
            ),
            state,
        )

    monkeypatch.setattr(AgentRuntimeMixin, "_apply_grounding", _forced_replan)

    agent = Agent(
        model=_ScriptedModel([_tc("trivial", {}), _text("My grounded answer.")], loop_last=True),
        tools=[trivial],
        grounding=GroundingConfig(enabled=True, threshold=0.5),
        max_iterations=10,
        reflexion=False,
    )
    events = await _collect(agent, "research")
    assert any(isinstance(e, GroundingEvent) for e in events)
    assert any(isinstance(e, TerminateEvent) for e in events)


# ---------------------------------------------------------------------------
# Empty-content safety net: conversation-manager branches (495-498)
# ---------------------------------------------------------------------------


async def test_empty_content_safety_net_async_conversation_manager() -> None:
    from tulip.memory.conversation import SlidingWindowManager

    agent = Agent(
        model=_ScriptedModel([_empty(), _text("recovered answer")]),
        conversation_manager=SlidingWindowManager(window_size=10),
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    events = await _collect(agent, "answer?")
    term = next(e for e in events if isinstance(e, TerminateEvent))
    assert term.final_message == "recovered answer"


async def test_empty_content_safety_net_sync_conversation_manager() -> None:
    agent = Agent(
        model=_ScriptedModel([_empty(), _text("recovered answer")]),
        conversation_manager=_SyncOnlyManager(),
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    events = await _collect(agent, "answer?")
    term = next(e for e in events if isinstance(e, TerminateEvent))
    assert term.final_message == "recovered answer"


# ---------------------------------------------------------------------------
# Empty-content safety net: summary call raises -> fallback (525 + 529)
# ---------------------------------------------------------------------------


class _RaiseOnSecond:
    """First call returns empty content; the forced-summary call raises."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            return _empty()
        raise RuntimeError("summary blew up")

    async def stream(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


async def test_empty_content_summary_call_failure_falls_back() -> None:
    agent = Agent(
        model=_RaiseOnSecond(),
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    events = await _collect(agent, "answer?")
    term = next(e for e in events if isinstance(e, TerminateEvent))
    # Deterministic fallback summary, never empty.
    assert term.final_message
    assert term.reason == "complete"


# ---------------------------------------------------------------------------
# Malformed interrupt marker (lines 883-884)
# ---------------------------------------------------------------------------


@tool
def bad_interrupt() -> str:
    """Return a string that *contains* the interrupt marker but isn't JSON."""
    return 'oops "__interrupt__": true but not valid json at all'


async def test_malformed_interrupt_marker_is_swallowed() -> None:
    agent = Agent(
        model=_ScriptedModel([_tc("bad_interrupt", {}), _text("done")]),
        tools=[bad_interrupt],
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    events = await _collect(agent, "go")
    # No interrupt is surfaced — the malformed marker is ignored.
    assert not any(isinstance(e, InterruptEvent) for e in events)
    term = next(e for e in events if isinstance(e, TerminateEvent))
    assert term.reason == "complete"


# ---------------------------------------------------------------------------
# After-tool retry whose re-execution raises (lines 967-968)
# ---------------------------------------------------------------------------


class _RetryRaisingExecutor(SequentialExecutor):
    """Streaming works; the retry path (``execute``) raises."""

    async def execute(self, tool_calls: Any, registry: Any, ctx_factory: Any = None) -> Any:
        raise RuntimeError("retry exec boom")


class _RetryHook(HookProvider):
    @property
    def priority(self) -> int:
        return HookPriority.OBSERVABILITY_DEFAULT

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        event.retry = True


async def test_after_tool_retry_failure_becomes_error_result() -> None:
    agent = Agent(
        model=_ScriptedModel([_tc("trivial", {}), _text("done")]),
        tools=[trivial],
        hooks=[_RetryHook()],
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    agent._executor = _RetryRaisingExecutor()
    events = await _collect(agent, "go")
    # The retry's re-execution raised, but the loop swallows it (the tool's
    # result is replaced with an error ToolResult) and keeps running to a
    # clean completion rather than crashing.
    assert any(isinstance(e, ToolCompleteEvent) for e in events)
    term = next(e for e in events if isinstance(e, TerminateEvent))
    assert term.reason == "complete"


# ---------------------------------------------------------------------------
# Executor.execute_streaming itself raises (lines 753-757)
# ---------------------------------------------------------------------------


class _StreamRaisingExecutor(SequentialExecutor):
    async def execute_streaming(
        self, tool_calls: Any, registry: Any, ctx_factory: Any = None
    ) -> Any:
        if False:  # pragma: no cover - makes this an async generator
            yield (0, None)
        raise RuntimeError("stream boom")


async def test_executor_streaming_failure_synthesizes_error_results() -> None:
    agent = Agent(
        model=_ScriptedModel([_tc("trivial", {}), _text("done")]),
        tools=[trivial],
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    agent._executor = _StreamRaisingExecutor()
    events = await _collect(agent, "go")
    complete = next(e for e in events if isinstance(e, ToolCompleteEvent))
    assert complete.error is not None
    assert "stream boom" in complete.error


# ---------------------------------------------------------------------------
# In-loop checkpoint every N iterations (lines 1043-1053)
# ---------------------------------------------------------------------------


async def test_in_loop_checkpoint_saved_and_emitted() -> None:
    cp = AsyncMock()
    cp.save = AsyncMock()
    cp.load = AsyncMock(return_value=None)
    agent = Agent(
        model=_ScriptedModel([_tc("trivial", {}), _text("done")]),
        tools=[trivial],
        checkpointer=cp,
        checkpoint_every_n_iterations=1,
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    # No thread_id -> the in-loop checkpoint uses state.run_id.
    await _collect(agent, "go")
    assert cp.save.called


# ---------------------------------------------------------------------------
# output_key persistence in the finally block (lines 1079-1085)
# ---------------------------------------------------------------------------


async def test_output_key_persisted_to_state_metadata() -> None:
    agent = Agent(
        model=_ScriptedModel([_text("the final answer")]),
        output_key="result",
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    await _collect(agent, "go")
    state = agent._last_run_state
    assert state is not None
    assert state.metadata.get("result") == "the final answer"
