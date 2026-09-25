# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``AgentConfig.tool_execution="concurrent"`` in ``Agent.run()``.

Issue #210: the runtime loop used to feed ``ConcurrentExecutor`` one tool
call at a time inside a ``for`` loop, so ``asyncio.gather`` never saw more
than a singleton — concurrent was silently sequential. These tests pin the
batched behavior end-to-end (overlap, hook ordering, mixed cancel/cache
paths, executor exceptions, ordering). Overlap and ordering are proven with
barriers and events, never wall-clock thresholds.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from tulip.agent import Agent
from tulip.core.messages import Message, ToolCall
from tulip.core.termination import MaxIterations
from tulip.hooks.provider import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookPriority,
    HookProvider,
)
from tulip.models.base import ModelResponse
from tulip.tools.decorator import tool


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


async def _run_collect(agent: Agent, prompt: str) -> Any:
    """Drive ``agent.run`` to completion; return the final state."""
    async for _ev in agent.run(prompt):
        pass
    return agent._last_run_state


# Parallelism is proven structurally, never by wall-clock: a threshold such as
# "elapsed < serial floor / 2" fails on a loaded CI runner (PR #49 run
# 36146751358 measured 0.541s against a 0.500s cut-off) and says nothing
# about *why* it was slow. Instead every tool in a batch parks on a barrier
# that only opens once all of them have started. Parallel execution opens it
# at once; serial execution can never open it, and the first call times out
# with an explicit message instead of the suite hanging.
_BARRIER_TIMEOUT_S = 10.0


class _Barrier:
    """Opens once ``parties`` callers have arrived; records peak concurrency."""

    def __init__(self, parties: int) -> None:
        self.parties = parties
        self.arrived = 0
        self.in_flight = 0
        self.peak_in_flight = 0
        self.broken = False
        self._open = asyncio.Event()

    async def wait(self) -> None:
        if self.broken:
            # One caller already timed out; don't make every later call wait too.
            raise AssertionError("barrier broken by an earlier timeout")
        self.arrived += 1
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            if self.arrived >= self.parties:
                self._open.set()
            try:
                await asyncio.wait_for(self._open.wait(), _BARRIER_TIMEOUT_S)
            except TimeoutError:
                self.broken = True
                raise AssertionError(
                    f"only {self.arrived}/{self.parties} tool calls were ever in flight "
                    "together — the batch ran serially"
                ) from None
        finally:
            self.in_flight -= 1


class _Gauge:
    """Counts concurrently running bodies without making any of them wait."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak_in_flight = 0

    async def hold(self) -> None:
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            # Yield a few times so an overlapping sibling would get scheduled.
            for _ in range(5):
                await asyncio.sleep(0)
        finally:
            self.in_flight -= 1


class _Sync:
    """Per-test synchronisation object the module-level tools reach through."""

    barrier: _Barrier | None = None
    gauge: _Gauge | None = None


@tool(name="slow")
async def slow_tool(idx: int) -> str:
    """Wait on the current barrier (or gauge), then echo the index."""
    if _Sync.barrier is not None:
        await _Sync.barrier.wait()
    elif _Sync.gauge is not None:
        await _Sync.gauge.hold()
    return f"done {idx}"


@pytest.fixture(autouse=True)
def _reset_sync() -> Any:
    _Sync.barrier = None
    _Sync.gauge = None
    yield
    _Sync.barrier = None
    _Sync.gauge = None


class _Chain:
    """Completion order made explicit: a tag finishes only after ``after`` did."""

    done: dict[str, asyncio.Event] = {}

    @classmethod
    def event(cls, tag: str) -> asyncio.Event:
        return cls.done.setdefault(tag, asyncio.Event())


@tool(name="chained")
async def chained_tool(tag: str, after: str = "") -> str:
    """Finish only once the call tagged ``after`` has finished."""
    if after:
        try:
            await asyncio.wait_for(_Chain.event(after).wait(), _BARRIER_TIMEOUT_S)
        except TimeoutError:
            raise AssertionError(f"{tag!r} waited for {after!r}, which never finished") from None
    _Chain.event(tag).set()
    return f"done {tag}"


def _chain_call(call_id: str, tag: str, after: str = "") -> ToolCall:
    return ToolCall(id=call_id, name="chained", arguments={"tag": tag, "after": after})


@pytest.fixture(autouse=True)
def _reset_chain() -> Any:
    _Chain.done = {}
    yield
    _Chain.done = {}


@pytest.mark.asyncio
async def test_concurrent_tool_calls_run_in_parallel() -> None:
    """tool_execution='concurrent' must batch a multi-tool-call response.

    Regression test for #210: previously the runtime loop iterated and
    submitted each call separately, so the calls never overlapped even
    with ``concurrent`` + ``max_concurrency=10``. All ``n`` calls must be in
    flight at the same time for the barrier to open.
    """
    n = 10
    _Sync.barrier = _Barrier(n)
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

    state = await _run_collect(agent, "do many slows")

    executions = list(state.tool_executions)
    assert [e.error for e in executions] == [None] * n
    assert {e.result for e in executions} == {f"done {i}" for i in range(n)}
    assert _Sync.barrier.peak_in_flight == n


@pytest.mark.asyncio
async def test_sequential_mode_still_serial() -> None:
    """tool_execution='sequential' must keep its per-call serial semantics."""
    n = 5
    _Sync.gauge = _Gauge()
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

    state = await _run_collect(agent, "do many slows serially")

    assert len(list(state.tool_executions)) == n
    assert _Sync.gauge.peak_in_flight == 1, (
        f"{_Sync.gauge.peak_in_flight} sequential tool bodies overlapped"
    )


@pytest.mark.asyncio
async def test_results_preserve_tool_call_order() -> None:
    """Concurrent execution must preserve the model's tool_call order in the
    recorded executions, regardless of which task finishes first."""

    # Completion order is c1, c2, c0 — the reverse of what order-by-finish
    # would record for c0.
    calls = [
        _chain_call("c0", "v0", after="v2"),
        _chain_call("c1", "v1"),
        _chain_call("c2", "v2", after="v1"),
    ]
    response = _assistant_with_tool_calls(calls)

    agent = Agent(
        model=_ScriptedModel([response]),
        tools=[chained_tool],
        tool_execution="concurrent",
        max_concurrency=5,
        termination=MaxIterations(2),
        max_iterations=5,
    )
    state = await _run_collect(agent, "ordered")

    executions = list(state.tool_executions)
    # Order must match tool_call order, even though c1 finishes first.
    assert [e.result for e in executions] == ["done v0", "done v1", "done v2"]
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
    state = await _run_collect(agent, "mixed cancel")

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
    state = await _run_collect(agent, "dedup")

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
    state = await _run_collect(agent, "within-batch dedup")

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
    state = await _run_collect(agent, "mixed errors")

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


@pytest.mark.asyncio
async def test_completion_mode_streams_events_in_finish_order() -> None:
    """``tool_event_order='completion'`` surfaces ``ToolCompleteEvent``s in
    finish order. ``state.tool_executions`` stays in tool_call order.

    Regression guard for the streaming follow-up to #210.
    """
    from tulip.core.events import ToolCompleteEvent

    calls = [
        _chain_call("c-slow", "slow", after="med"),
        _chain_call("c-fast", "fast"),
        _chain_call("c-med", "med", after="fast"),
    ]
    agent = Agent(
        model=_ScriptedModel([_assistant_with_tool_calls(calls)]),
        tools=[chained_tool],
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
    from tulip.core.events import ToolCompleteEvent

    calls = [
        _chain_call("c-slow", "slow", after="fast"),
        _chain_call("c-fast", "fast"),
    ]
    agent = Agent(
        model=_ScriptedModel([_assistant_with_tool_calls(calls)]),
        tools=[chained_tool],
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
    siblings_started = 0
    all_siblings_started: asyncio.Event | None = None


@tool(name="interrupting")
async def interrupting_tool() -> str:
    """Returns the ``__interrupt__`` marker once both siblings are in flight."""
    import json as _json

    _SiblingBodyCount.interrupt_calls += 1
    assert _SiblingBodyCount.all_siblings_started is not None
    # Interrupt only after the siblings are provably running, so the test
    # exercises cancelling in-flight work rather than never-started work.
    await asyncio.wait_for(_SiblingBodyCount.all_siblings_started.wait(), _BARRIER_TIMEOUT_S)
    return _json.dumps({"__interrupt__": True, "question": "?", "options": None})


@tool(name="slow_sibling")
async def slow_sibling_tool(idx: int) -> str:
    """Blocks until cancelled; the increment below must never run."""
    _SiblingBodyCount.siblings_started += 1
    if _SiblingBodyCount.siblings_started == 2 and _SiblingBodyCount.all_siblings_started:
        _SiblingBodyCount.all_siblings_started.set()
    # Never set: only cancellation (or the timeout, on a regression) ends this.
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(asyncio.Event().wait(), _BARRIER_TIMEOUT_S)
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
    _SiblingBodyCount.siblings_started = 0
    _SiblingBodyCount.all_siblings_started = asyncio.Event()

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
    assert _SiblingBodyCount.siblings_started == 2
    # The siblings were running (their sleep had started) but got cancelled
    # before completing — their body's increment must not have fired.
    assert _SiblingBodyCount.sibling_calls == 0, (
        f"sibling bodies executed despite interrupt: {_SiblingBodyCount.sibling_calls}"
    )
