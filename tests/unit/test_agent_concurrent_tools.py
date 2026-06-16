# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for ``AgentConfig.tool_execution="concurrent"`` in ``Agent.run()``.

Issue #210: the runtime loop used to feed ``ConcurrentExecutor`` one tool
call at a time inside a ``for`` loop, so ``asyncio.gather`` never saw more
than a singleton — concurrent was silently sequential. These tests pin the
batched behavior end-to-end (wall-time, hook ordering, mixed cancel/cache
paths, executor exceptions, ordering).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from locus.agent import Agent
from locus.core.messages import Message, ToolCall
from locus.core.termination import MaxIterations
from locus.hooks.provider import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookPriority,
    HookProvider,
)
from locus.models.base import ModelResponse
from locus.tools.decorator import tool


class _ScriptedModel:
    """Replay a fixed list of ``ModelResponse`` objects across iterations."""

    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        if not self._responses:
            # Once the script is exhausted, return a plain text reply so the
            # loop terminates cleanly (no tool calls => `auto`-mode complete).
            return ModelResponse(
                message=Message.assistant(content="done"),
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )
        return self._responses.pop(0)

    async def stream(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def _assistant_with_tool_calls(calls: list[ToolCall]) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(content=None, tool_calls=calls),
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )


async def _run_collect(agent: Agent, prompt: str) -> tuple[float, Any]:
    """Drive ``agent.run`` to completion; return (wall_seconds, final_state)."""
    t0 = time.perf_counter()
    async for _ev in agent.run(prompt):
        pass
    elapsed = time.perf_counter() - t0
    return elapsed, agent._last_run_state


# Sleep duration per slow tool call. Chosen large enough that the
# concurrent-vs-sequential gap survives CI jitter, but short enough not to
# slow the suite. Ten parallel sleeps of 100 ms ~= 100 ms; ten serial ones
# ~= 1000 ms.
_SLEEP_MS = 0.1


@tool(name="slow")
async def slow_tool(idx: int) -> str:
    """Sleep ``_SLEEP_MS`` then echo the index."""
    await asyncio.sleep(_SLEEP_MS)
    return f"done {idx}"


@pytest.mark.asyncio
async def test_concurrent_tool_calls_run_in_parallel() -> None:
    """tool_execution='concurrent' must batch a multi-tool-call response.

    Regression test for #210: previously the runtime loop iterated and
    submitted each call separately, so wall time was N * _SLEEP_MS even
    with ``concurrent`` + ``max_concurrency=10``.
    """
    n = 10
    response = _assistant_with_tool_calls(
        [ToolCall(id=f"c{i}", name="slow", arguments={"idx": i}) for i in range(n)]
    )

    agent = Agent(
        model=_ScriptedModel([response]),
        tools=[slow_tool],
        tool_execution="concurrent",
        max_concurrency=n,
        termination=MaxIterations(2),
        max_iterations=5,
    )

    elapsed, state = await _run_collect(agent, "do many slows")

    executions = list(state.tool_executions)
    assert len(executions) == n
    assert {e.result for e in executions} == {f"done {i}" for i in range(n)}

    # Sequential floor = n * _SLEEP_MS. The fix should land us close to
    # _SLEEP_MS (one round of sleep), well under half the serial floor.
    sequential_floor = n * _SLEEP_MS
    assert elapsed < sequential_floor / 2, (
        f"wall time {elapsed:.3f}s suggests serial execution "
        f"(sequential floor {sequential_floor:.3f}s)"
    )


@pytest.mark.asyncio
async def test_sequential_mode_still_serial() -> None:
    """tool_execution='sequential' must keep its per-call serial semantics."""
    n = 5
    response = _assistant_with_tool_calls(
        [ToolCall(id=f"c{i}", name="slow", arguments={"idx": i}) for i in range(n)]
    )

    agent = Agent(
        model=_ScriptedModel([response]),
        tools=[slow_tool],
        tool_execution="sequential",
        termination=MaxIterations(2),
        max_iterations=5,
    )

    elapsed, state = await _run_collect(agent, "do many slows serially")

    assert len(list(state.tool_executions)) == n
    # Sequential should take at least ~80% of the n * _SLEEP_MS floor.
    assert elapsed >= 0.8 * n * _SLEEP_MS, (
        f"sequential wall time {elapsed:.3f}s collapsed below the serial floor"
    )


@pytest.mark.asyncio
async def test_results_preserve_tool_call_order() -> None:
    """Concurrent execution must preserve the model's tool_call order in the
    recorded executions, regardless of which task finishes first."""

    @tool(name="vsleep")
    async def vsleep(idx: int, ms: float) -> str:
        await asyncio.sleep(ms / 1000.0)
        return f"v{idx}"

    calls = [
        ToolCall(id="c0", name="vsleep", arguments={"idx": 0, "ms": 200}),
        ToolCall(id="c1", name="vsleep", arguments={"idx": 1, "ms": 50}),
        ToolCall(id="c2", name="vsleep", arguments={"idx": 2, "ms": 100}),
    ]
    response = _assistant_with_tool_calls(calls)

    agent = Agent(
        model=_ScriptedModel([response]),
        tools=[vsleep],
        tool_execution="concurrent",
        max_concurrency=5,
        termination=MaxIterations(2),
        max_iterations=5,
    )
    _elapsed, state = await _run_collect(agent, "ordered")

    executions = list(state.tool_executions)
    # Order must match tool_call order, even though c1 finishes first wall-clock.
    assert [e.result for e in executions] == ["v0", "v1", "v2"]
    assert [e.tool_call_id for e in executions] == ["c0", "c1", "c2"]


class _RecordingHook(HookProvider):
    """Append a chronological log of every before/after hook fire."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []  # (phase, tool_call_id)

    @property
    def priority(self) -> int:
        return HookPriority.OBSERVABILITY_DEFAULT

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        self.events.append(("before", event.tool_call_id))

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        self.events.append(("after", event.tool_call_id))


@pytest.mark.asyncio
async def test_hooks_fire_for_every_concurrent_call() -> None:
    """Every parallel call still gets its before+after hook pair, and all
    before-hooks complete before any after-hook (batched semantics)."""
    hook = _RecordingHook()
    calls = [ToolCall(id=f"c{i}", name="slow", arguments={"idx": i}) for i in range(4)]
    agent = Agent(
        model=_ScriptedModel([_assistant_with_tool_calls(calls)]),
        tools=[slow_tool],
        hooks=[hook],
        tool_execution="concurrent",
        max_concurrency=4,
        termination=MaxIterations(2),
        max_iterations=5,
    )
    await _run_collect(agent, "hooked")

    before_ids = [tcid for phase, tcid in hook.events if phase == "before"]
    after_ids = [tcid for phase, tcid in hook.events if phase == "after"]
    assert sorted(before_ids) == ["c0", "c1", "c2", "c3"]
    assert sorted(after_ids) == ["c0", "c1", "c2", "c3"]
    first_after = next(i for i, (p, _) in enumerate(hook.events) if p == "after")
    last_before = max(i for i, (p, _) in enumerate(hook.events) if p == "before")
    assert last_before < first_after, hook.events


class _CancelSecondHook(HookProvider):
    """Cancel the second tool call via the before-hook ``cancel`` field."""

    @property
    def priority(self) -> int:
        return HookPriority.SECURITY_DEFAULT

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        if event.tool_call_id == "c1":
            event.cancel = "cancelled by hook"


@pytest.mark.asyncio
async def test_cancel_via_before_hook_skips_executor_for_that_call() -> None:
    """A hook-cancelled call short-circuits but the rest still execute in parallel."""
    calls = [ToolCall(id=f"c{i}", name="slow", arguments={"idx": i}) for i in range(3)]
    agent = Agent(
        model=_ScriptedModel([_assistant_with_tool_calls(calls)]),
        tools=[slow_tool],
        hooks=[_CancelSecondHook()],
        tool_execution="concurrent",
        max_concurrency=5,
        termination=MaxIterations(2),
        max_iterations=5,
    )
    _elapsed, state = await _run_collect(agent, "mixed cancel")

    executions = list(state.tool_executions)
    assert [e.tool_call_id for e in executions] == ["c0", "c1", "c2"]
    assert executions[0].result == "done 0"
    assert executions[1].result == "cancelled by hook"
    assert executions[2].result == "done 2"


@tool(idempotent=True)
async def cached_echo(value: str) -> str:
    """Idempotent echo — second call with same args must reuse the first."""
    return f"echo:{value}"


@pytest.mark.asyncio
async def test_idempotent_cache_short_circuits_in_concurrent_mode() -> None:
    """Same-args idempotent recall inside one concurrent batch hits the cache."""
    iter1 = _assistant_with_tool_calls(
        [ToolCall(id="c0", name="cached_echo", arguments={"value": "x"})]
    )
    iter2 = _assistant_with_tool_calls(
        [
            ToolCall(id="c1", name="cached_echo", arguments={"value": "x"}),
            ToolCall(id="c2", name="cached_echo", arguments={"value": "y"}),
            ToolCall(id="c3", name="cached_echo", arguments={"value": "x"}),
        ]
    )

    agent = Agent(
        model=_ScriptedModel([iter1, iter2]),
        tools=[cached_echo],
        tool_execution="concurrent",
        max_concurrency=5,
        termination=MaxIterations(3),
        max_iterations=5,
    )
    _elapsed, state = await _run_collect(agent, "dedup")

    executions = list(state.tool_executions)
    assert [e.tool_call_id for e in executions] == ["c0", "c1", "c2", "c3"]
    assert executions[0].idempotent_cache_hit is False
    assert executions[1].idempotent_cache_hit is True
    assert executions[2].idempotent_cache_hit is False
    assert executions[3].idempotent_cache_hit is True
    assert [e.result for e in executions] == ["echo:x", "echo:x", "echo:y", "echo:x"]


@tool(idempotent=True)
async def counted_echo(value: str) -> str:
    """Body increments a module counter so we can assert how many times it fires."""
    _BodyCount.n += 1
    return f"echo:{value} (n={_BodyCount.n})"


class _BodyCount:
    n = 0


@pytest.mark.asyncio
async def test_within_batch_idempotent_dedup_in_concurrent_mode() -> None:
    """Same-args idempotent calls in ONE assistant response fire the body once.

    This is the README contract — ``@tool(idempotent=True)`` must not
    double-side-effect just because the model emitted duplicates in one turn.
    Regression guard for a near-miss while fixing #210: the first pass of the
    three-phase split lost the implicit serial dedup the old per-call loop
    had via mid-loop state updates.
    """
    _BodyCount.n = 0
    calls = [
        ToolCall(id="c0", name="counted_echo", arguments={"value": "X"}),
        ToolCall(id="c1", name="counted_echo", arguments={"value": "X"}),
        ToolCall(id="c2", name="counted_echo", arguments={"value": "Y"}),
        ToolCall(id="c3", name="counted_echo", arguments={"value": "X"}),
    ]
    agent = Agent(
        model=_ScriptedModel([_assistant_with_tool_calls(calls)]),
        tools=[counted_echo],
        tool_execution="concurrent",
        max_concurrency=5,
        termination=MaxIterations(2),
        max_iterations=5,
    )
    _elapsed, state = await _run_collect(agent, "within-batch dedup")

    executions = list(state.tool_executions)
    assert [e.tool_call_id for e in executions] == ["c0", "c1", "c2", "c3"]
    # Body ran exactly twice — once for "X" (c0) and once for "Y" (c2).
    assert _BodyCount.n == 2, f"body fired {_BodyCount.n} times, expected 2"
    # c1 and c3 must be marked as cache hits; c0 and c2 are fresh.
    assert executions[0].idempotent_cache_hit is False
    assert executions[1].idempotent_cache_hit is True
    assert executions[2].idempotent_cache_hit is False
    assert executions[3].idempotent_cache_hit is True
    # All "X" results share the same body output (the first fresh call's).
    assert executions[0].result == executions[1].result == executions[3].result


@tool(name="flaky")
async def flaky_tool(idx: int) -> str:
    if idx == 1:
        raise RuntimeError("boom")
    return f"ok-{idx}"


@pytest.mark.asyncio
async def test_executor_exception_isolated_to_one_call() -> None:
    """An exception from one parallel tool body becomes that call's ``error``
    field; siblings still produce successful results."""
    calls = [ToolCall(id=f"c{i}", name="flaky", arguments={"idx": i}) for i in range(3)]
    agent = Agent(
        model=_ScriptedModel([_assistant_with_tool_calls(calls)]),
        tools=[flaky_tool],
        tool_execution="concurrent",
        max_concurrency=5,
        termination=MaxIterations(2),
        max_iterations=5,
    )
    _elapsed, state = await _run_collect(agent, "mixed errors")

    executions = list(state.tool_executions)
    assert [e.tool_call_id for e in executions] == ["c0", "c1", "c2"]
    assert executions[0].result == "ok-0"
    assert executions[0].error is None
    assert executions[1].result is None
    assert executions[1].error is not None
    assert "boom" in executions[1].error
    assert executions[2].result == "ok-2"
    assert executions[2].error is None


# =============================================================================
# tool_event_order — completion vs sequential streaming of ToolCompleteEvent
# =============================================================================


async def _run_collect_events(agent: Agent, prompt: str) -> tuple[list[Any], Any]:
    """Drive ``agent.run`` to completion; return (events, final_state)."""
    events: list[Any] = []
    async for ev in agent.run(prompt):
        events.append(ev)
    return events, agent._last_run_state


@tool(name="vsleep_tag")
async def vsleep_tag(tag: str, ms: float) -> str:
    """Async sleep tool — used to construct deterministic completion-order tests."""
    await asyncio.sleep(ms / 1000.0)
    return f"done {tag}"


@pytest.mark.asyncio
async def test_completion_mode_streams_events_in_finish_order() -> None:
    """``tool_event_order='completion'`` surfaces ``ToolCompleteEvent``s in
    finish order. ``state.tool_executions`` stays in tool_call order.

    Regression guard for the streaming follow-up to #210.
    """
    from locus.core.events import ToolCompleteEvent

    calls = [
        ToolCall(id="c-slow", name="vsleep_tag", arguments={"tag": "slow", "ms": 200}),
        ToolCall(id="c-fast", name="vsleep_tag", arguments={"tag": "fast", "ms": 20}),
        ToolCall(id="c-med", name="vsleep_tag", arguments={"tag": "med", "ms": 100}),
    ]
    agent = Agent(
        model=_ScriptedModel([_assistant_with_tool_calls(calls)]),
        tools=[vsleep_tag],
        tool_execution="concurrent",
        max_concurrency=5,
        tool_event_order="completion",
        termination=MaxIterations(2),
        max_iterations=5,
    )
    events, state = await _run_collect_events(agent, "stream")

    # Event order = completion order (fast → med → slow).
    completes = [ev for ev in events if isinstance(ev, ToolCompleteEvent)]
    assert [ev.tool_call_id for ev in completes] == ["c-fast", "c-med", "c-slow"]

    # State order is unchanged (tool_call order).
    assert [e.tool_call_id for e in state.tool_executions] == ["c-slow", "c-fast", "c-med"]

    # And each ToolCompleteEvent fires exactly once (no double-emit between
    # Phase 2 streaming and Phase 3 fold).
    assert len(completes) == 3


@pytest.mark.asyncio
async def test_sequential_mode_events_in_tool_call_order() -> None:
    """Default ``tool_event_order='sequential'`` keeps the original behaviour:
    events arrive in tool_call order after the whole batch completes."""
    from locus.core.events import ToolCompleteEvent

    calls = [
        ToolCall(id="c-slow", name="vsleep_tag", arguments={"tag": "slow", "ms": 100}),
        ToolCall(id="c-fast", name="vsleep_tag", arguments={"tag": "fast", "ms": 20}),
    ]
    agent = Agent(
        model=_ScriptedModel([_assistant_with_tool_calls(calls)]),
        tools=[vsleep_tag],
        tool_execution="concurrent",
        max_concurrency=5,
        # tool_event_order defaults to "sequential" — not passing it.
        termination=MaxIterations(2),
        max_iterations=5,
    )
    events, _state = await _run_collect_events(agent, "ordered")
    completes = [ev for ev in events if isinstance(ev, ToolCompleteEvent)]
    assert [ev.tool_call_id for ev in completes] == ["c-slow", "c-fast"]
    assert len(completes) == 2


# =============================================================================
# Interrupt cancels in-flight siblings (TaskGroup-equivalent semantics via
# the executor's finally-cancel pattern)
# =============================================================================


class _SiblingBodyCount:
    """Module-level counters so the closure can observe body execution."""

    interrupt_calls = 0
    sibling_calls = 0


@tool(name="interrupting")
async def interrupting_tool() -> str:
    """Returns the ``__interrupt__`` marker the runtime loop watches for."""
    import json as _json

    _SiblingBodyCount.interrupt_calls += 1
    # Tiny wait so the executor's other tasks have a real chance to start
    # before this one finishes and the break fires.
    await asyncio.sleep(0.02)
    return _json.dumps({"__interrupt__": True, "question": "?", "options": None})


@tool(name="slow_sibling")
async def slow_sibling_tool(idx: int) -> str:
    """Slow tool — should be cancelled before its body actually runs."""
    await asyncio.sleep(0.5)
    _SiblingBodyCount.sibling_calls += 1
    return f"sibling-{idx}"


@pytest.mark.asyncio
async def test_interrupt_cancels_in_flight_siblings() -> None:
    """An interrupt mid-batch cancels in-flight siblings before they execute.

    Pre-#210-follow-up behaviour: siblings completed in parallel under
    ``gather`` and only the post-batch fold was halted — sibling side
    effects still landed. The streaming refactor uses the executor's
    finally-cancel to propagate the interrupt to in-flight tasks.
    """
    _SiblingBodyCount.interrupt_calls = 0
    _SiblingBodyCount.sibling_calls = 0

    calls = [
        ToolCall(id="c0", name="interrupting", arguments={}),
        ToolCall(id="c1", name="slow_sibling", arguments={"idx": 1}),
        ToolCall(id="c2", name="slow_sibling", arguments={"idx": 2}),
    ]
    agent = Agent(
        model=_ScriptedModel([_assistant_with_tool_calls(calls)]),
        tools=[interrupting_tool, slow_sibling_tool],
        tool_execution="concurrent",
        max_concurrency=5,
        termination=MaxIterations(2),
        max_iterations=5,
    )
    await _run_collect(agent, "interrupt cancels siblings")

    assert _SiblingBodyCount.interrupt_calls == 1
    # The siblings were running (their sleep had started) but got cancelled
    # before completing — their body's increment must not have fired.
    assert _SiblingBodyCount.sibling_calls == 0, (
        f"sibling bodies executed despite interrupt: {_SiblingBodyCount.sibling_calls}"
    )
