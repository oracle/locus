# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""First-class subagents: isolation, rollup, cancellation, attribution.

The SDK could always spawn an ``Agent`` inside a tool body — the deepagent
``task_tool`` did exactly that. What it could not do was account for it:
the child's tokens vanished from the parent's budget, cancelling the parent
left the child running, and the child's events went nowhere. These tests
pin the contract of ``run_subagent`` — the pieces a hand-rolled child loop
silently lacks — because each one guards a way the feature could lie:

- a parent whose ``TerminateEvent.usage`` omits delegated spend under-reports
  cost to whoever meters it;
- a child that inherits the parent's toolset turns "delegate a narrow task"
  into "hand over everything";
- a child that survives ``parent.cancel()`` is a runaway; a child whose
  wind-down *un*-cancels the parent is worse;
- a silent child renders as a frozen UI over a busy agent.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from tulip.agent import Agent, SubagentResult, run_subagent
from tulip.agent.subagent import _LinkedCancelSignal
from tulip.core.events import (
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    TulipEvent,
)
from tulip.core.messages import Message
from tulip.hooks.provider import BeforeToolCallEvent, HookProvider
from tulip.models.base import ModelResponse
from tulip.testing import FunctionModel, ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


# ---------------------------------------------------------------------------
# The result itself
# ---------------------------------------------------------------------------


async def test_child_returns_result() -> None:
    """A one-turn child comes back with text, usage, and a stop reason."""
    result = await run_subagent(
        "What color is the sky?",
        model=ScriptedModel([text("blue")]),
    )

    assert isinstance(result, SubagentResult)
    assert result.text == "blue"
    assert result.stop_reason == "complete"
    assert result.success
    assert result.iterations == 1
    assert result.agent_name == "subagent"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


async def test_unmetered_child_reports_usage_none() -> None:
    """A provider that reports no usage yields ``usage=None`` — unmetered,
    which a consumer must never read as free."""
    silent = ScriptedModel(
        [ModelResponse(message=Message.assistant("done"), usage={}, stop_reason="end_turn")]
    )
    result = await run_subagent("task", model=silent)

    assert result.text == "done"
    assert result.usage is None


async def test_child_runs_its_own_system_prompt_and_conversation() -> None:
    """Isolation: the child sees ITS system prompt and the delegated prompt —
    nothing of any parent conversation."""
    model = ScriptedModel([text("ack")])
    await run_subagent("the delegated task", model=model, system_prompt="You verify claims.")

    [messages] = model.received_messages
    roles = [getattr(m.role, "value", m.role) for m in messages]
    assert roles == ["system", "user"]
    assert messages[0].content == "You verify claims."
    assert messages[1].content == "the delegated task"


# ---------------------------------------------------------------------------
# Tool allowlist
# ---------------------------------------------------------------------------


@tool
def allowed_lookup(key: str) -> str:
    """Look up a value the child is allowed to see."""
    return f"value-for-{key}"


async def test_tools_are_an_explicit_allowlist() -> None:
    """The child can call what it was handed and NOTHING else — a tool the
    parent owns but did not pass simply does not exist for the child."""
    model = ScriptedModel(
        [
            tool_call("parent_secret_tool", target="prod"),
            tool_call("allowed_lookup", key="k1", call_id="call_2"),
            text("finished"),
        ]
    )
    events: list[TulipEvent] = []
    result = await run_subagent(
        "do the task",
        model=model,
        tools=[allowed_lookup],
        on_event=events.append,
    )

    completions = {e.tool_name: e for e in events if isinstance(e, ToolCompleteEvent)}
    assert "Unknown tool" in (completions["parent_secret_tool"].error or "")
    assert completions["allowed_lookup"].result == "value-for-k1"
    assert result.text == "finished"
    # And the allowlist is what the model was OFFERED, too.
    assert model.offered_tools[0] == ["allowed_lookup"]


async def test_agent_method_child_does_not_inherit_parent_tools() -> None:
    """``Agent.run_subagent`` defaults the child to the parent's MODEL, never
    to the parent's toolset."""
    parent = Agent(
        model=ScriptedModel([text("hi from the parent's model")]), tools=[allowed_lookup]
    )

    result = await parent.run_subagent("delegated")

    assert result.text == "hi from the parent's model"
    # The child offered the model no tools at all.
    assert parent.model.offered_tools[0] == []


# ---------------------------------------------------------------------------
# Usage rollup
# ---------------------------------------------------------------------------


async def test_child_usage_rolls_up_into_parent_counters() -> None:
    """Delegated spend is spend: the parent's TerminateEvent.usage and
    AgentResult metrics include the child's tokens."""

    @tool
    async def delegate(request: str) -> str:
        """Delegate a subproblem to a subagent."""
        sub = await run_subagent(request, model=ScriptedModel([text("child answer")]))
        return sub.text

    parent = Agent(
        model=ScriptedModel([tool_call("delegate", request="sub-task"), text("parent answer")]),
        tools=[delegate],
    )

    terminate: TerminateEvent | None = None
    async for event in parent.run("big task"):
        if isinstance(event, TerminateEvent):
            terminate = event

    # Parent made 2 model calls (30 tokens each), the child 1 (30).
    assert terminate is not None
    assert terminate.usage == {
        "prompt_tokens": 20 + 10,
        "completion_tokens": 40 + 20,
        "total_tokens": 60 + 30,
    }


async def test_rollup_reaches_agent_result_metrics() -> None:
    """The same rollup shows on ``AgentResult.metrics`` via ``arun``."""

    @tool
    async def delegate(request: str) -> str:
        """Delegate a subproblem to a subagent."""
        sub = await run_subagent(request, model=ScriptedModel([text("child answer")]))
        return sub.text

    parent = Agent(
        model=ScriptedModel([tool_call("delegate", request="sub-task"), text("parent answer")]),
        tools=[delegate],
    )
    result = await parent.arun("big task")

    assert result.metrics.total_tokens == 90


async def test_grandchild_usage_counts_exactly_once() -> None:
    """A child that itself delegates reports its FINAL counters — which
    already include the grandchild — so the parent counts them once."""

    @tool
    async def go_deeper(request: str) -> str:
        """Delegate further down."""
        sub = await run_subagent(request, model=ScriptedModel([text("grandchild answer")]))
        return sub.text

    @tool
    async def delegate(request: str) -> str:
        """Delegate a subproblem to a subagent."""
        sub = await run_subagent(
            request,
            model=ScriptedModel([tool_call("go_deeper", request="deeper"), text("child answer")]),
            tools=[go_deeper],
        )
        return sub.text

    parent = Agent(
        model=ScriptedModel([tool_call("delegate", request="sub"), text("parent answer")]),
        tools=[delegate],
    )
    result = await parent.arun("big task")

    # parent 2×30, child 2×30, grandchild 1×30 — and NOT 30 more from
    # the grandchild being counted again at the parent.
    assert result.metrics.total_tokens == 150


async def test_child_usage_counts_against_parent_token_budget() -> None:
    """A parent budget is enforced against delegated spend too."""

    @tool
    async def delegate(request: str) -> str:
        """Delegate a subproblem to a subagent."""
        sub = await run_subagent(request, model=ScriptedModel([text("child answer")]))
        return sub.text

    parent = Agent(
        model=ScriptedModel([tool_call("delegate", request="sub")], repeat_last=True),
        tools=[delegate],
        token_budget=80,  # parent's own 2×30 would not trip this; +30/child does
    )
    result = await parent.arun("big task")

    assert result.stop_reason == "token_budget"
    assert result.metrics.total_tokens >= 80


# ---------------------------------------------------------------------------
# Parallel children
# ---------------------------------------------------------------------------


async def test_parallel_children_via_gather() -> None:
    """Several children fan out with plain ``asyncio.gather``; every one's
    usage lands in the parent."""

    @tool
    async def fan_out(request: str) -> str:
        """Fan a task out to three subagents."""
        results = await asyncio.gather(
            *(
                run_subagent(
                    f"{request} #{i}",
                    model=ScriptedModel([text(f"child {i}")]),
                    name=f"child-{i}",
                )
                for i in range(3)
            )
        )
        assert [r.text for r in results] == ["child 0", "child 1", "child 2"]
        assert [r.agent_name for r in results] == ["child-0", "child-1", "child-2"]
        return " / ".join(r.text for r in results)

    parent = Agent(
        model=ScriptedModel([tool_call("fan_out", request="survey"), text("merged")]),
        tools=[fan_out],
    )
    result = await parent.arun("survey the field")

    assert result.message == "merged"
    # parent 2×30 + three children at 30 each.
    assert result.metrics.total_tokens == 150


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


async def test_parent_cancel_stops_running_child() -> None:
    """``parent.cancel()`` reaches a child mid-run — and the child's own
    wind-down does not un-cancel the parent."""
    started = asyncio.Event()

    @tool
    async def spin(n: int) -> str:
        """Busy-work so the child stays mid-run."""
        started.set()
        await asyncio.sleep(0)
        return f"spun {n}"

    calls = 0

    def keep_spinning(messages: list[Message], tools: object) -> ModelResponse:
        nonlocal calls
        calls += 1
        return tool_call("spin", n=calls, call_id=f"call_{calls}")

    parent = Agent(model=ScriptedModel([text("unused")]))
    child_run = asyncio.create_task(
        parent.run_subagent(
            "spin until told otherwise",
            model=FunctionModel(keep_spinning),
            tools=[spin],
            max_iterations=200,
        )
    )

    await asyncio.wait_for(started.wait(), timeout=5)
    parent.cancel()
    result = await asyncio.wait_for(child_run, timeout=5)

    assert result.stop_reason == "cancelled"
    assert not result.success
    # The child's finally cleared ITS signal, not the parent's.
    assert parent.is_cancelled


def test_linked_signal_clear_never_clears_the_parent() -> None:
    """The linkage in both directions: parent's set is visible, child's
    clear is local."""
    parent_signal = threading.Event()
    linked = _LinkedCancelSignal(parent_signal)

    assert not linked.is_set()
    parent_signal.set()
    assert linked.is_set()

    linked.clear()  # what the child loop's finally does
    assert parent_signal.is_set()
    assert linked.is_set()  # still cancelled — the parent says so

    parent_signal.clear()
    linked.set()
    assert linked.is_set()
    assert not parent_signal.is_set()  # child's own set never cancels the parent


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


async def test_child_events_are_observable_and_attributed() -> None:
    """No silent children: every child event reaches ``on_event``, stamped
    with the child's name so a merged stream stays legible."""
    events: list[TulipEvent] = []

    async def collect(event: TulipEvent) -> None:  # async callbacks are awaited
        events.append(event)

    result = await run_subagent(
        "check the claim",
        model=ScriptedModel([tool_call("allowed_lookup", key="claim"), text("verified")]),
        tools=[allowed_lookup],
        name="verifier",
        on_event=collect,
    )

    assert result.agent_name == "verifier"
    kinds = {type(e) for e in events}
    assert ThinkEvent in kinds
    assert ToolCompleteEvent in kinds
    assert TerminateEvent in kinds
    assert all(e.agent_name == "verifier" for e in events)


# ---------------------------------------------------------------------------
# Gating — a subagent is not a policy bypass
# ---------------------------------------------------------------------------


async def test_hooks_gate_child_tool_calls() -> None:
    """A policy hook handed to the child refuses the child's tool calls the
    same way it refuses the parent's."""
    executed: list[str] = []

    @tool
    def dangerous(target: str) -> str:
        """A tool policy must be able to stop."""
        executed.append(target)
        return "boom"

    class DenyDangerous(HookProvider):
        @property
        def priority(self) -> int:
            return 0

        async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
            if event.tool_name == "dangerous":
                event.cancel = "Blocked by policy"

    events: list[TulipEvent] = []
    result = await run_subagent(
        "wreak havoc",
        model=ScriptedModel([tool_call("dangerous", target="prod"), text("could not")]),
        tools=[dangerous],
        hooks=[DenyDangerous()],
        on_event=events.append,
    )

    assert executed == []  # the body never ran
    blocked = [e for e in events if isinstance(e, ToolCompleteEvent)]
    assert any("Blocked by policy" in (e.result or "") for e in blocked)
    assert result.text == "could not"


# ---------------------------------------------------------------------------
# The deepagent task tool now rides on the primitive
# ---------------------------------------------------------------------------


async def test_deepagent_task_tool_usage_rolls_up() -> None:
    """The pre-existing ``task_tool`` gains the rollup for free: what used to
    vanish from the parent's meter now lands on it."""
    from tulip.deepagent.subagent import SubAgentDef, task_tool

    task = task_tool(
        [
            SubAgentDef(
                name="researcher",
                description="Researches.",
                system_prompt="You research.",
                model=ScriptedModel([text("research findings")]),
            )
        ],
        parent_model=None,
    )

    parent = Agent(
        model=ScriptedModel(
            [tool_call("task", subagent_type="researcher", description="dig in"), text("done")]
        ),
        tools=[task],
    )
    result = await parent.arun("investigate")

    assert result.message == "done"
    assert result.metrics.total_tokens == 90  # 2×30 parent + 30 child


# ---------------------------------------------------------------------------
# Harness shape: outside any run
# ---------------------------------------------------------------------------


async def test_run_subagent_outside_any_run_with_explicit_cancel_signal() -> None:
    """A harness with no running loop still gets the linkage by passing its
    own signal; usage travels on the result alone."""
    signal = threading.Event()
    signal.set()  # already cancelled before the child begins

    result = await run_subagent(
        "never really starts",
        model=ScriptedModel([text("unreachable")], repeat_last=True),
        cancel_signal=signal,
    )

    assert result.stop_reason == "cancelled"
    assert signal.is_set()  # the child's wind-down left the harness signal alone


async def test_child_error_propagates_after_an_error_terminate_event() -> None:
    """A crashing child is loud twice over: the loop's error TerminateEvent
    reaches ``on_event``, and the exception reaches the caller."""

    def explode(messages: list[Message], tools: object) -> ModelResponse:
        raise RuntimeError("provider fell over")

    events: list[TulipEvent] = []
    with pytest.raises(RuntimeError, match="provider fell over"):
        await run_subagent("task", model=FunctionModel(explode), on_event=events.append)

    terminates = [e for e in events if isinstance(e, TerminateEvent)]
    assert [e.reason for e in terminates] == ["error"]


async def test_child_that_pauses_reads_as_interrupted() -> None:
    """A child has no one to resume it: a run that pauses for input comes
    back ``stop_reason="interrupted"``, never a fake success."""

    @tool
    def ask_user(question: str) -> str:
        """Pause and ask the human — which a subagent cannot reach."""
        return '{"__interrupt__": true, "question": "which env?"}'

    result = await run_subagent(
        "deploy it",
        model=ScriptedModel([tool_call("ask_user", question="which env?")]),
        tools=[ask_user],
    )

    assert result.stop_reason == "interrupted"
    assert not result.success
    assert result.text == ""
