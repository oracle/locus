# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The ``task`` tool and what it rests on: live child events, shared budgets,
resumable children.

Each test guards a way delegation could lie or leak:

- a child whose events never reach the parent's stream is a frozen UI over a
  busy agent, and one whose events arrive *unwrapped* makes the child's
  ``TerminateEvent`` read as the parent finishing;
- a child that ignores the parent's time or token budget is how a budgeted
  run overspends;
- a resumed child that starts cold re-reads everything it already read;
- a subagent type that can name tools the parent lacks, or that escapes the
  parent's hooks, turns delegation into a way around the gate.
"""

from __future__ import annotations

import asyncio
import threading
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from tulip.agent import (
    Agent,
    AgentSpec,
    Subagent,
    SubagentResult,
    TaskRegistry,
    run_subagent,
    task_tool,
)
from tulip.agent.subagent import (
    _child_budgets,
    enter_parent_run,
    exit_parent_run,
    parent_hooks,
)
from tulip.agent.tasks import DEFAULT_SUBAGENT_PROMPT, task_depth
from tulip.core.events import (
    SubagentEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
    TulipEvent,
)
from tulip.hooks.provider import BeforeToolCallEvent, HookProvider
from tulip.models.metadata import ModelMetadata, register_metadata
from tulip.testing import FunctionModel, ScriptedModel, text, tool_call
from tulip.tools.context import (
    ToolContext,
    bind_tool_context,
    forward_event,
    forwarding_events,
)
from tulip.tools.decorator import tool


@tool
def read(path: str) -> str:
    """Read a file."""
    return f"contents of {path}"


@tool
def write(path: str, content: str) -> str:
    """Write a file."""
    return f"wrote {path}"


EXPLORE = AgentSpec(
    name="explore",
    description="Read-only search.",
    prompt="You explore.",
    tools=("read",),
    mode="subagent",
)
GENERAL = AgentSpec(name="general", description="Everything.", mode="subagent")


async def _collect(agent: Agent, prompt: str) -> list[TulipEvent]:
    return [event async for event in agent.run(prompt)]


def _task_call(call_id: str = "call_task", **arguments: Any) -> Any:
    arguments.setdefault("description", "look around")
    return tool_call("task", call_id=call_id, **arguments)


# ---------------------------------------------------------------------------
# Live events on the parent's stream
# ---------------------------------------------------------------------------


async def test_child_events_stream_live_wrapped_and_attributed() -> None:
    child = ScriptedModel([tool_call("read", path="a.py"), text("found it in a.py")])
    task = task_tool([EXPLORE], tools=[read, write], model=child)
    parent = Agent(
        model=ScriptedModel(
            [_task_call(prompt="find it", subagent_type="explore"), text("parent done")]
        ),
        tools=[read, write, task],
    )

    events = await _collect(parent, "go")

    wrapped = [e for e in events if isinstance(e, SubagentEvent)]
    inner = [w.event for w in wrapped]
    assert {type(e) for e in inner} >= {ThinkEvent, ToolStartEvent, ToolCompleteEvent}
    assert any(isinstance(e, TerminateEvent) for e in inner)
    assert all(w.tool_call_id == "call_task" and w.tool_name == "task" for w in wrapped)
    assert all(w.agent_name == "explore" for w in wrapped)
    assert len({w.task_id for w in wrapped}) == 1
    # Between the delegating call's start and its result.
    kinds = [type(e).__name__ for e in events]
    first, last = kinds.index("SubagentEvent"), len(kinds) - 1 - kinds[::-1].index("SubagentEvent")
    assert kinds.index("ToolStartEvent") < first
    task_done = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, ToolCompleteEvent) and e.tool_name == "task"
    )
    assert last < task_done
    # The child's TerminateEvent is wrapped; the only bare one is the parent's.
    bare = [e for e in events if isinstance(e, TerminateEvent)]
    assert len(bare) == 1
    assert bare[0].final_message == "parent done"


async def test_wrapped_events_serialize_with_the_child_event_inside() -> None:
    wrapped = SubagentEvent(
        tool_call_id="c1", tool_name="task", event=ThinkEvent(iteration=1, reasoning="hm")
    )
    data = wrapped.model_dump()
    assert data["event_type"] == "subagent"
    assert data["event"]["event_type"] == "think"
    assert data["event"]["reasoning"] == "hm"


async def test_a_grandchild_arrives_wrapped_twice() -> None:
    grandchild = ScriptedModel([text("deepest answer")])
    child = ScriptedModel(
        [_task_call(call_id="inner", prompt="deeper", subagent_type="general"), text("mid")]
    )
    models = {"child": child, "grandchild": grandchild}
    general = GENERAL.model_copy(update={"model": "child"})
    # The grandchild is spawned by the child's task tool with the same spec,
    # so route the second spawn to the grandchild model.
    order = iter(["child", "grandchild"])
    task = task_tool(
        [general],
        tools=lambda: [read, task],
        model=None,
        resolve_model=lambda _name: models[next(order)],
    )
    parent = Agent(
        model=ScriptedModel([_task_call(prompt="go deep", subagent_type="general"), text("top")]),
        tools=[read, task],
    )
    events = await _collect(parent, "go")

    nested = [
        e.event
        for e in events
        if isinstance(e, SubagentEvent) and isinstance(e.event, SubagentEvent)
    ]
    assert nested
    assert all(n.tool_call_id == "inner" for n in nested)
    assert any(
        isinstance(n.event, TerminateEvent) and n.event.final_message == "deepest answer"
        for n in nested
    )


async def test_a_plain_tool_running_a_child_does_not_stream_it() -> None:
    """Only a tool that declares ``emits_progress`` has a live channel."""

    @tool
    async def quiet(request: str) -> str:
        """Delegate without streaming."""
        return (await run_subagent(request, model=ScriptedModel([text("hidden")]))).text

    parent = Agent(
        model=ScriptedModel([tool_call("quiet", request="x"), text("done")]), tools=[quiet]
    )
    events = await _collect(parent, "go")
    assert not any(isinstance(e, SubagentEvent) for e in events)


def test_forward_event_outside_a_tool_reaches_nobody() -> None:
    assert forwarding_events() is False
    assert forward_event(ThinkEvent(iteration=1)) is False


# ---------------------------------------------------------------------------
# Budgets are the parent's
# ---------------------------------------------------------------------------


def test_child_time_budget_is_capped_by_the_parent_deadline() -> None:
    token = enter_parent_run(threading.Event(), time_budget_seconds=30)
    try:
        capped = _child_budgets("m", {"time_budget_seconds": 600})
        assert isinstance(capped, dict)
        assert 0 < capped["time_budget_seconds"] <= 30
        own = _child_budgets("m", {"time_budget_seconds": 5})
        assert isinstance(own, dict)
        assert own["time_budget_seconds"] == 5
    finally:
        exit_parent_run(token)


async def test_a_child_started_past_the_deadline_never_calls_its_model() -> None:
    model = ScriptedModel([])  # any call would raise
    token = enter_parent_run(threading.Event(), time_budget_seconds=0.01)
    try:
        await asyncio.sleep(0.02)
        result = await run_subagent("late", model=model, name="late")
        sub = await Subagent(model=model, name="later").send("later still")
    finally:
        exit_parent_run(token)
    assert result.stop_reason == "time_budget"
    assert result.agent_name == "late"
    assert sub.stop_reason == "time_budget"
    assert sub.task_id is not None
    assert model.call_count == 0


def test_without_a_parent_run_the_child_keeps_its_own_budgets() -> None:
    assert _child_budgets("m", {"token_budget": 7}) == {"token_budget": 7}


async def test_child_token_budget_is_what_the_parent_has_left() -> None:
    seen: dict[str, Any] = {}

    @tool
    async def probe(request: str) -> str:
        """Report the budgets a child would get."""
        budgets = _child_budgets("m", {"token_budget": 10_000})
        seen.update(budgets if isinstance(budgets, dict) else {"exhausted": budgets})
        return "ok"

    parent = Agent(
        model=ScriptedModel([tool_call("probe", request="x"), text("done")]),
        tools=[probe],
        token_budget=100,
    )
    await parent.arun("go")
    # The parent's first model call (30 tokens) is already counted.
    assert seen["token_budget"] == 70


async def test_a_parent_out_of_tokens_starts_no_child() -> None:
    child = ScriptedModel([])

    @tool
    async def delegate(request: str) -> str:
        """Delegate."""
        return (await run_subagent(request, model=child)).stop_reason

    parent = Agent(
        model=ScriptedModel([tool_call("delegate", request="x"), text("done")]),
        tools=[delegate],
        token_budget=30,
    )
    result = await parent.arun("go")
    assert child.call_count == 0
    assert result.tool_executions[0].result == "token_budget"


def _priced_model(model_id: str, input_price: str, output_price: str) -> ScriptedModel:
    register_metadata(
        ModelMetadata(
            model_id=model_id,
            family="test",
            context_length=100_000,
            max_output_tokens=4_000,
            input_price_per_mtok=Decimal(input_price),
            output_price_per_mtok=Decimal(output_price),
        )
    )
    return ScriptedModel([])


async def test_child_spend_is_counted_at_the_childs_own_prices() -> None:
    _priced_model("tasks-test-parent", "1", "1")
    _priced_model("tasks-test-child", "1000", "1000")
    child = ScriptedModel([text("child answer")])
    child.model = "tasks-test-child"  # type: ignore[attr-defined]  # how model_id_of names it
    seen: dict[str, Any] = {}

    @tool
    async def delegate(request: str) -> str:
        """Delegate."""
        seen.update(_child_budgets(child, {}))  # type: ignore[arg-type]
        return (await run_subagent(request, model=child)).text

    parent_model = ScriptedModel([tool_call("delegate", request="x"), text("done")])
    parent_model.model = "tasks-test-parent"  # type: ignore[attr-defined]
    parent = Agent(model=parent_model, tools=[delegate], max_cost_usd=10.0)
    await parent.arun("go")

    # Parent: 2 calls x 30 tokens at $1/M; child: 30 tokens at $1000/M.
    state = parent._last_run_state
    assert state is not None
    assert state.cost_usd_used == pytest.approx(60 / 1e6 + 30 * 1000 / 1e6)
    # The priced child was capped at what the parent had left.
    assert 0 < seen["max_cost_usd"] < 10.0


async def test_an_unpriced_child_gets_no_cost_cap_but_still_counts() -> None:
    _priced_model("tasks-test-parent-2", "1", "1")
    seen: dict[str, Any] = {}

    @tool
    async def delegate(request: str) -> str:
        """Delegate."""
        model = ScriptedModel([text("child answer")])
        seen.update(_child_budgets(model, {}))  # type: ignore[arg-type]
        return (await run_subagent(request, model=model)).text

    parent_model = ScriptedModel([tool_call("delegate", request="x"), text("done")])
    parent_model.model = "tasks-test-parent-2"  # type: ignore[attr-defined]
    parent = Agent(model=parent_model, tools=[delegate], max_cost_usd=10.0)
    await parent.arun("go")
    assert "max_cost_usd" not in seen
    state = parent._last_run_state
    assert state is not None
    # Child tokens priced at the parent's rate: 3 x 30 tokens at $1/M.
    assert state.cost_usd_used == pytest.approx(90 / 1e6)


async def test_a_parent_out_of_money_starts_no_child() -> None:
    child = ScriptedModel([])
    spent = SimpleNamespace(
        token_budget=None, total_tokens_used=0, cost_budget_usd=1.0, cost_usd_used=1.5
    )
    call = ToolContext(tool_call_id="c", tool_name="t", run_id="r", iteration=1, state=spent)
    token = enter_parent_run(threading.Event())
    try:
        with bind_tool_context(call):
            result = await run_subagent("x", model=child)
    finally:
        exit_parent_run(token)
    assert child.call_count == 0
    assert result.stop_reason == "cost_budget"


# ---------------------------------------------------------------------------
# Resumable children
# ---------------------------------------------------------------------------


async def test_a_subagent_keeps_its_conversation_between_turns() -> None:
    model = ScriptedModel([text("first answer"), text("second answer")])
    sub = Subagent(model=model, system_prompt="You help.", task_id="t-1")

    one = await sub.send("first question")
    two = await sub.send("follow-up")

    assert (one.text, two.text) == ("first answer", "second answer")
    assert one.task_id == two.task_id == "t-1"
    assert sub.turns == 2
    second_call = [m.content for m in model.received_messages[1]]
    assert second_call == ["You help.", "first question", "first answer", "follow-up"]


async def test_each_turn_reports_only_its_own_usage() -> None:
    model = ScriptedModel([text("a"), text("b")])
    sub = Subagent(model=model)
    await sub.send("one")
    second = await sub.send("two")
    assert second.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


def test_a_subagent_refuses_a_checkpointer() -> None:
    with pytest.raises(ValueError, match="own conversation"):
        Subagent(model=ScriptedModel([]), checkpointer=object())


# ---------------------------------------------------------------------------
# The task tool
# ---------------------------------------------------------------------------


async def test_explore_gets_only_its_tools_from_the_pool() -> None:
    child = ScriptedModel([text("nothing to see")])
    task = task_tool([EXPLORE, GENERAL], tools=[read, write], model=child)
    out = await task.execute(description="d", prompt="look", subagent_type="explore")

    assert out.startswith("nothing to see")
    assert child.offered_tools[0] == ["read"]
    assert child.received_messages[0][0].content == "You explore."


async def test_a_spec_cannot_grant_a_tool_the_pool_lacks() -> None:
    child = ScriptedModel([text("ok")])
    greedy = AgentSpec(name="greedy", tools=("read", "write", "deploy"), mode="subagent")
    task = task_tool([greedy], tools=[read], model=child)
    await task.execute(description="d", prompt="p", subagent_type="greedy")
    assert child.offered_tools[0] == ["read"]


async def test_general_gets_the_whole_pool_and_the_default_prompt() -> None:
    child = ScriptedModel([text("ok")])
    cool = GENERAL.model_copy(update={"temperature": 0.1})
    task = task_tool([cool], tools=[read, write], model=child)
    await task.execute(description="d", prompt="p")
    assert sorted(child.offered_tools[0]) == ["read", "write"]
    assert child.received_messages[0][0].content == DEFAULT_SUBAGENT_PROMPT


async def test_the_result_carries_a_task_id_that_resumes_the_conversation() -> None:
    child = ScriptedModel([text("first"), text("second")])
    registry = TaskRegistry()
    task = task_tool([GENERAL], tools=[read], model=child, registry=registry)

    first = await task.execute(description="d", prompt="q1")
    task_id = first.split("task_id=", 1)[1].split(",", 1)[0]
    assert registry.ids() == [task_id]
    second = await task.execute(description="d", prompt="q2", task_id=task_id)

    assert second.startswith("second")
    assert f"task_id={task_id}" in second
    contents = [m.content for m in child.received_messages[1]]
    assert contents[1:] == ["q1", "first", "q2"]


async def test_an_unknown_task_id_or_type_is_a_message_not_a_crash() -> None:
    task = task_tool([GENERAL], tools=[read], model=ScriptedModel([]))
    assert "no subagent with task_id" in await task.execute(
        description="d", prompt="p", task_id="task_nope"
    )
    out = await task.execute(description="d", prompt="p", subagent_type="wizard")
    assert out == "unknown subagent_type 'wizard'; available: general"


async def test_an_unfinished_child_says_why_it_stopped() -> None:
    child = ScriptedModel([tool_call("read", path="a")], repeat_last=True)
    limited = GENERAL.model_copy(update={"max_turns": 2})
    task = task_tool([limited], tools=[read], model=child)
    out = await task.execute(description="d", prompt="p")
    assert "stopped=max_iterations" in out
    assert "turns=2" in out


async def test_an_empty_answer_still_reads_as_one(monkeypatch: pytest.MonkeyPatch) -> None:
    async def empty(self: Subagent, prompt: str, **_: Any) -> SubagentResult:
        return SubagentResult(text="  ", stop_reason="complete", iterations=1, task_id="t")

    monkeypatch.setattr(Subagent, "send", empty)
    task = task_tool([GENERAL], tools=[], model=None)
    out = await task.execute(description="d", prompt="p")
    assert out.startswith("(the subagent finished without a final message)")


async def test_resolve_model_maps_a_specs_model_and_none_inherits() -> None:
    fast = ScriptedModel([text("fast answer")])
    default = ScriptedModel([text("default answer")])
    quick = AgentSpec(name="quick", model="fast-alias", mode="subagent")
    task = task_tool(
        [quick, GENERAL],
        tools=[],
        model=default,
        resolve_model=lambda alias: {"fast-alias": fast}[alias],
    )
    assert (await task.execute(description="d", prompt="p", subagent_type="quick")).startswith(
        "fast answer"
    )
    assert (await task.execute(description="d", prompt="p", subagent_type="general")).startswith(
        "default answer"
    )


async def test_a_spec_model_is_used_as_is_without_a_resolver() -> None:
    task = task_tool(
        [AgentSpec(name="raw", model="nope:model", mode="subagent")], tools=[], model=None
    )
    with pytest.raises(Exception):  # noqa: B017, PT011 — whatever the registry raises for a bad id
        await task.execute(description="d", prompt="p")


async def test_parallel_task_calls_run_concurrently() -> None:
    both_started = asyncio.Event()
    started = 0

    @tool
    async def rendezvous(who: str) -> str:
        """Wait until both subagents are inside a tool at the same time."""
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=5)
        return f"met {who}"

    def child_turn(messages: Any, _tools: Any) -> Any:
        if any(getattr(m.role, "value", m.role) == "tool" for m in messages):
            return text("met")
        return tool_call("rendezvous", who=messages[1].content)

    task = task_tool([GENERAL], tools=[rendezvous], model=FunctionModel(child_turn))
    parent = Agent(
        model=ScriptedModel(
            [
                _parallel(
                    _task_call(call_id="a", prompt="alpha"),
                    _task_call(call_id="b", prompt="beta"),
                ),
                text("both done"),
            ]
        ),
        tools=[task],
    )
    result = await parent.arun("go")
    assert result.text == "both done"
    assert started == 2
    assert all(e.result and e.result.startswith("met") for e in result.tool_executions)


def _parallel(*turns: Any) -> Any:
    first = turns[0]
    calls = [c for t in turns for c in t.message.tool_calls]
    return first.model_copy(
        update={"message": first.message.model_copy(update={"tool_calls": calls})}
    )


async def test_the_last_level_is_not_given_the_task_tool() -> None:
    child = ScriptedModel([text("leaf")])
    holder: list[Any] = []
    task = task_tool([GENERAL], tools=lambda: [read, *holder], model=child, max_depth=1)
    holder.append(task)
    await task.execute(description="d", prompt="p")
    assert child.offered_tools[0] == ["read"]


async def test_nesting_past_the_limit_is_refused() -> None:
    task = task_tool([GENERAL], tools=[], model=ScriptedModel([]), max_depth=1)
    from tulip.agent.tasks import _TASK_DEPTH

    token = _TASK_DEPTH.set(1)
    try:
        assert task_depth() == 1
        out = await task.execute(description="d", prompt="p")
    finally:
        _TASK_DEPTH.reset(token)
    assert out.startswith("refused: task calls may nest 1 deep")


async def test_the_parents_hooks_gate_the_child() -> None:
    ran: list[str] = []

    @tool
    def deploy(target: str) -> str:
        """Deploy somewhere."""
        ran.append(target)
        return "deployed"

    class NoDeploy(HookProvider):
        @property
        def priority(self) -> int:
            return 0

        async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
            if event.tool_name == "deploy":
                event.cancel = "deploys are blocked"

    child = ScriptedModel([tool_call("deploy", target="prod"), text("blocked, stopping")])
    task = task_tool([GENERAL], tools=[deploy], model=child)
    parent = Agent(
        model=ScriptedModel([_task_call(prompt="ship it"), text("ok")]),
        tools=[deploy, task],
        hooks=[NoDeploy()],
    )
    await parent.arun("go")
    assert ran == []
    child_result = child.received_messages[1][-1].content or ""
    assert "deploys are blocked" in child_result


async def test_inherit_hooks_off_leaves_only_the_explicit_ones() -> None:
    seen: list[str] = []

    class Watch(HookProvider):
        @property
        def priority(self) -> int:
            return 0

        async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
            seen.append(event.tool_name)

    child = ScriptedModel([tool_call("read", path="x"), text("done")])
    task = task_tool([GENERAL], tools=[read], model=child, inherit_hooks=False)
    parent = Agent(
        model=ScriptedModel([_task_call(prompt="p"), text("ok")]),
        tools=[read, task],
        hooks=[Watch()],
    )
    await parent.arun("go")
    assert seen == ["task"]  # the parent's own call only


def test_parent_hooks_outside_a_run_are_empty() -> None:
    assert parent_hooks() == ()


def test_task_tool_validates_its_configuration() -> None:
    primary = AgentSpec(name="build", mode="primary")
    with pytest.raises(ValueError, match="at least one agent spec"):
        task_tool([primary], tools=[], model=None)
    with pytest.raises(ValueError, match="max_depth"):
        task_tool([GENERAL], tools=[], model=None, max_depth=0)
    with pytest.raises(ValueError, match="default_type"):
        task_tool([GENERAL], tools=[], model=None, default_type="nope")


def test_the_tool_describes_its_types_and_streams() -> None:
    task = task_tool({"explore": EXPLORE, "general": GENERAL}, tools=[], model=None)
    assert task.emits_progress is True
    assert "- explore: Read-only search." in task.description
    assert "- general: Everything." in task.description
    assert task.parameters["properties"]["subagent_type"]["default"] == "explore"


def test_the_registry_forgets_the_least_recently_used() -> None:
    registry = TaskRegistry(max_tasks=2)
    subs = [Subagent(model=None, task_id=f"t{i}") for i in range(3)]
    registry.add("general", subs[0])
    registry.add("general", subs[1])
    assert registry.get("t0") is not None  # t0 is now the most recent
    registry.add("general", subs[2])
    assert registry.ids() == ["t0", "t2"]
    assert len(registry) == 2
