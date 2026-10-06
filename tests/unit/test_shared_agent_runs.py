# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""One Agent, many runs: the hooks that used to be bound to the whole Agent.

A chat backend builds one Agent and runs every user's turn on it, often
concurrently. ``EventBusHook`` must tag each run's events with that run's id,
read when the event fires, and ``PlaybookEnforcerHook`` must keep one plan per
run, so one user's progress (or violations) never counts for another's.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tulip.agent import Agent
from tulip.core.events import RunInfo
from tulip.observability.bus_hook import EventBusHook
from tulip.observability.context import run_context
from tulip.observability.event_bus import StreamEvent, get_event_bus, reset_event_bus
from tulip.playbooks.hook import PlaybookEnforcerHook
from tulip.playbooks.models import Playbook, PlaybookStep
from tulip.testing import FunctionModel, text, tool_call
from tulip.tools.decorator import tool


@tool
async def search(query: str) -> str:
    """Search."""
    await asyncio.sleep(0)
    return f"results for {query}"


@tool
async def classify(severity: str) -> str:
    """Classify."""
    await asyncio.sleep(0)
    return f"classified {severity}"


def _playbook() -> Playbook:
    return Playbook(
        id="triage",
        name="Triage",
        steps=[
            PlaybookStep(id="find", description="Find context", expected_tools=["search"]),
            PlaybookStep(id="classify", description="Classify", expected_tools=["classify"]),
        ],
    )


def _user_text(messages: list[Any]) -> str:
    return next(
        (m.content or "" for m in messages if getattr(m.role, "value", m.role) == "user"), ""
    )


def _model() -> FunctionModel:
    """Calls the tools the prompt names, in order, one per turn, then answers."""

    def handler(messages: list[Any], tools: list[dict[str, Any]]) -> Any:
        wanted = _user_text(messages).split()
        done = sum(1 for m in messages if getattr(m.role, "value", m.role) == "tool")
        if done < len(wanted):
            name = wanted[done]
            args = {"query": "q"} if name == "search" else {"severity": "high"}
            return tool_call(name, call_id=f"call_{done}", **args)
        return text("done")

    return FunctionModel(handler)


async def _events(run_id: str) -> list[StreamEvent]:
    bus = get_event_bus()
    await bus.close_stream(run_id)
    return [e async for e in bus.subscribe(run_id)]


def _hook_tools(events: list[StreamEvent]) -> list[str]:
    """Tools the hook reported starting. (The loop's own ``emit()`` publishes an
    ``agent.tool.started`` of its own inside a run context; the hook's carries
    ``argument_keys``.)"""
    return [
        e.data["tool_name"]
        for e in events
        if e.event_type == "agent.tool.started" and "argument_keys" in e.data
    ]


@pytest.fixture(autouse=True)
def _bus() -> Any:
    reset_event_bus()
    yield
    reset_event_bus()


# =============================================================================
# EventBusHook
# =============================================================================


class TestEventBusHookFollowsTheRun:
    async def test_concurrent_runs_in_run_contexts_keep_their_own_ids(self) -> None:
        agent = Agent(model=_model(), tools=[search, classify], hooks=[EventBusHook()])

        async def turn(run_id: str, prompt: str) -> None:
            async with run_context(run_id):
                await agent.arun(prompt)

        await asyncio.gather(turn("turn-a", "search"), turn("turn-b", "classify search"))

        a, b = await _events("turn-a"), await _events("turn-b")
        assert _hook_tools(a) == ["search"]
        assert _hook_tools(b) == ["classify", "search"]
        for found in (a, b):
            kinds = {e.event_type for e in found}
            assert {"agent.invocation.started", "agent.invocation.completed"} <= kinds
            assert {"agent.iteration.started", "agent.model.started"} <= kinds

    async def test_without_a_run_context_each_run_uses_its_own_run_id(self) -> None:
        agent = Agent(model=_model(), tools=[search, classify], hooks=[EventBusHook()])

        first, second = await asyncio.gather(agent.arun("search"), agent.arun("classify"))

        assert first.state.run_id != second.state.run_id
        a = await _events(first.state.run_id)
        b = await _events(second.state.run_id)
        assert _hook_tools(a) == ["search"]
        assert _hook_tools(b) == ["classify"]
        assert any(e.event_type == "agent.invocation.started" for e in a)

    async def test_a_fixed_run_id_still_tags_every_event(self) -> None:
        hook = EventBusHook(run_id="fixed")
        agent = Agent(model=_model(), tools=[search], hooks=[hook])

        async with run_context("ignored"):
            await agent.arun("search")

        assert hook.run_id == "fixed"
        assert _hook_tools(await _events("fixed")) == ["search"]
        # The loop's own events follow the context; none of the hook's do.
        assert _hook_tools(await _events("ignored")) == []

    async def test_run_id_property_follows_the_context(self) -> None:
        hook = EventBusHook()
        assert hook.run_id is None
        async with run_context("now"):
            assert hook.run_id == "now"

    def test_empty_run_id_is_still_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            EventBusHook(run_id="")


# =============================================================================
# PlaybookEnforcerHook
# =============================================================================


class TestPlaybookPerRun:
    async def test_sequential_runs_each_start_the_plan_over(self) -> None:
        hook = PlaybookEnforcerHook(_playbook())
        agent = Agent(model=_model(), tools=[search, classify], hooks=[hook])

        first = await agent.arun("search classify")
        assert hook.enforcer.is_complete
        assert hook.enforcer.violations == []

        # Under one plan for the whole agent this run would start at a
        # finished playbook; per run it starts at step one, and an
        # out-of-order call is caught again.
        second = await agent.arun("classify")
        assert hook.enforcer_for(second.state.run_id) is hook.enforcer
        assert len(hook.enforcer.violations) == 1
        done = hook.enforcer_for(first.state.run_id)
        assert done is not None
        assert done.is_complete
        assert done.violations == []

    async def test_concurrent_runs_do_not_share_progress(self) -> None:
        hook = PlaybookEnforcerHook(_playbook())
        agent = Agent(model=_model(), tools=[search, classify], hooks=[hook])

        good, bad = await asyncio.gather(agent.arun("search classify"), agent.arun("classify"))

        on_good = hook.enforcer_for(good.state.run_id)
        on_bad = hook.enforcer_for(bad.state.run_id)
        assert on_good is not None
        assert on_bad is not None
        assert on_good.is_complete
        assert on_good.violations == []
        assert [v.tool_name for v in on_bad.violations] == ["classify"]
        cancelled = [t for t in bad.state.tool_executions if t.tool_name == "classify"]
        assert cancelled
        assert "PlaybookEnforcer blocked" in str(cancelled[0].result)

    async def test_select_enables_a_playbook_for_one_run_only(self) -> None:
        asked: list[RunInfo] = []

        def select(run: RunInfo) -> Playbook | None:
            asked.append(run)
            return _playbook() if run.metadata.get("playbook") == "triage" else None

        hook = PlaybookEnforcerHook(select=select)
        agent = Agent(model=_model(), tools=[search, classify], hooks=[hook])

        enforced = await agent.arun("classify", metadata={"playbook": "triage"})
        free = await agent.arun("classify")

        assert hook.enforcer_for(free.state.run_id) is None
        assert [t.result for t in free.state.tool_executions] == ["classified high"]
        on = hook.enforcer_for(enforced.state.run_id)
        assert on is not None
        assert len(on.violations) == 1
        # Asked once per run, not once per call.
        assert [r.run_id for r in asked] == [enforced.state.run_id, free.state.run_id]

    async def test_a_selector_that_raises_fails_closed(self) -> None:
        def select(run: RunInfo) -> Playbook | None:
            raise RuntimeError("registry down")

        agent = Agent(model=_model(), tools=[search], hooks=[PlaybookEnforcerHook(select=select)])
        result = await agent.arun("search")

        [call] = result.state.tool_executions
        assert "could not be chosen" in str(call.result)

    async def test_thread_scope_carries_the_plan_across_turns(self) -> None:
        hook = PlaybookEnforcerHook(_playbook(), scope="thread")
        agent = Agent(model=_model(), tools=[search, classify], hooks=[hook])

        await agent.arun("search", thread_id="t1")
        await agent.arun("classify", thread_id="t1")

        plan = hook.enforcer_for("thread:t1")
        assert plan is not None
        assert plan.is_complete
        assert plan.violations == []

    async def test_the_map_of_runs_is_bounded(self) -> None:
        hook = PlaybookEnforcerHook(_playbook(), max_runs=2)
        agent = Agent(model=_model(), tools=[search], hooks=[hook])

        runs = [await agent.arun("search") for _ in range(3)]

        assert hook.enforcer_for(runs[0].state.run_id) is None
        assert hook.enforcer_for(runs[2].state.run_id) is not None

    def test_playbook_or_select_exactly_one(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            PlaybookEnforcerHook()
        with pytest.raises(ValueError, match="exactly one"):
            PlaybookEnforcerHook(_playbook(), select=lambda run: None)

    def test_select_has_no_enforcer_before_a_run(self) -> None:
        with pytest.raises(LookupError):
            _ = PlaybookEnforcerHook(select=lambda run: None).enforcer
