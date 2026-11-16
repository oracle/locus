# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Durability modes and the tool-effect ledger on the in-memory checkpointer.

The cases that matter: the three modes end a turn in the same state and
differ only in when they write; a process killed after a tool ran but before
any checkpoint landed (``async`` and ``exit`` have that window by design) is
picked up by ``continue_turn`` / ``resume(invocation_id=...)`` with the tool
run exactly once overall; a replayed call hits the ledger when it is the same
call and misses when it is not; and the graph runner obeys the same modes.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from locus.agent import Agent
from locus.agent.durability import CheckpointWriter, check_durability
from locus.core.events import InterruptEvent, TerminateEvent, ToolCompleteEvent, ToolStartEvent
from locus.core.messages import Message, Role, ToolCall
from locus.core.state import AgentState
from locus.durable.segments import run_agent_segment
from locus.memory.backends.memory import MemoryCheckpointer
from locus.memory.ledger import ToolEffectLedger, ledger_thread_id
from locus.models.base import ModelResponse
from locus.multiagent.graph import END, START, StateGraph
from locus.tools.decorator import tool


class _KilledError(Exception):
    """Stands in for the process dying under the run."""


class _Model:
    """Replays scripted responses; a callable entry is invoked instead (to kill the run)."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.seen: list[list[Message]] = []

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **_: Any
    ) -> ModelResponse:
        self.seen.append(list(messages))
        step = self.script.pop(0)
        if callable(step) and not isinstance(step, ModelResponse):
            step()
        response: ModelResponse = step
        return response


class _KillableCheckpointer(MemoryCheckpointer):
    """A memory checkpointer whose writes stop once the 'process' dies.

    ``latency`` makes every write take a turn of the loop first, so a write
    scheduled by ``durability="async"`` is still in flight when the kill
    lands — and is lost, as a real in-flight write would be. ``die_on``
    names a thread whose first save kills the process in the middle of the
    write itself.
    """

    def __init__(self, *, latency: float = 0.0, die_on: str | None = None) -> None:
        super().__init__()
        self.dead = False
        self.latency = latency
        self.die_on = die_on
        self.saves: dict[str, int] = {}

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if self.latency:
            await asyncio.sleep(self.latency)
        if self.dead:
            return "lost"
        if thread_id == self.die_on:
            self.dead = True
            raise _KilledError("killed during the checkpoint write")
        self.saves[thread_id] = self.saves.get(thread_id, 0) + 1
        return await super().save(state, thread_id, checkpoint_id, metadata)

    async def delete(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        if self.dead:
            return False
        return await super().delete(thread_id, checkpoint_id)

    def checkpoints_of(self, thread_id: str) -> int:
        return len(self._storage.get(thread_id, {}))


calls: dict[str, int] = {}


@tool
def charge(order_id: str) -> str:
    """Charge an order. Not idempotent: a second call charges twice."""
    calls[order_id] = calls.get(order_id, 0) + 1
    return f"charged {order_id}"


@tool(idempotent=True)
def lookup(order_id: str) -> str:
    """Look an order up."""
    calls[f"lookup:{order_id}"] = calls.get(f"lookup:{order_id}", 0) + 1
    return f"order {order_id}: open"


@tool
def ask(question: str) -> str:
    """Pause the run on a question (the interrupt marker ``ask_user`` returns)."""
    return json.dumps({"__interrupt__": True, "question": question})


def _call(name: str, call_id: str, **arguments: Any) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(
            content=None, tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)]
        ),
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )


def _answer(text: str) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(text), usage={"prompt_tokens": 1, "completion_tokens": 1}
    )


def _agent(model: Any, checkpointer: Any, **kwargs: Any) -> Agent:
    return Agent(
        model=model,
        tools=[charge, lookup, ask],
        checkpointer=checkpointer,
        max_iterations=10,
        reflexion=False,
        grounding=False,
        **kwargs,
    )


async def _drain(gen: Any) -> list[Any]:
    return [event async for event in gen]


def _kill(store: _KillableCheckpointer) -> Any:
    def die() -> None:
        store.dead = True
        raise _KilledError("killed")

    return die


def _shape(state: AgentState) -> list[tuple[str, str | None, list[str]]]:
    return [(m.role.value, m.content, [tc.name for tc in m.tool_calls]) for m in state.messages]


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    calls.clear()


# --- the modes ---------------------------------------------------------------


def test_durability_values_are_checked() -> None:
    assert check_durability("sync") == "sync"
    with pytest.raises(ValueError, match="sync, async, exit"):
        check_durability("eventually")
    with pytest.raises(ValueError):
        Agent(model=_Model([]), durability="eventually")  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["sync", "async", "exit"])
async def test_every_mode_ends_the_turn_in_the_same_state(mode: str) -> None:
    store = _KillableCheckpointer()
    script = [
        _call("charge", "c1", order_id="o1"),
        _call("lookup", "c2", order_id="o1"),
        _answer("ok"),
    ]
    reference = _agent(_Model(list(script)), MemoryCheckpointer())
    await _drain(reference.run("charge o1", thread_id="ref"))
    expected = await reference.config.checkpointer.load("ref")
    assert expected is not None

    agent = _agent(_Model(list(script)), store, durability=mode)  # type: ignore[arg-type]
    events = await _drain(agent.run("charge o1", thread_id="t"))

    final = await store.load("t")
    assert final is not None
    assert _shape(final) == _shape(expected)
    assert [e.tool_name for e in final.tool_executions] == ["charge", "lookup"]
    assert final.iteration == expected.iteration
    assert [e for e in events if isinstance(e, TerminateEvent)][-1].reason == "complete"
    # Two iteration saves plus the final one, or the final one alone.
    assert store.saves["t"] == (1 if mode == "exit" else 3)
    # The iteration saves were deleted once the final save superseded them.
    assert store.checkpoints_of("t") == 1
    # The ledger (on for async and exit) was cleared by the final save.
    assert ledger_thread_id("t") not in store.get_thread_ids()


async def test_async_writes_land_before_the_run_returns() -> None:
    store = _KillableCheckpointer(latency=0.005)
    agent = _agent(
        _Model([_call("charge", "c1", order_id="o1"), _answer("ok")]), store, durability="async"
    )
    await _drain(agent.run("charge o1", thread_id="t"))
    assert store.saves["t"] == 2
    final = await store.load("t")
    assert final is not None
    assert final.messages[-1].content == "ok"


async def test_per_run_override_beats_the_agent_default() -> None:
    store = _KillableCheckpointer()
    script = [_call("charge", "c1", order_id="o1"), _answer("ok")]
    agent = _agent(_Model([*script, *script]), store)
    assert agent.config.durability == "sync"
    await _drain(agent.run("charge o1", thread_id="t", durability="exit"))
    assert store.saves["t"] == 1
    result = await agent.arun("charge o1 again", thread_id="t2", durability="exit")
    assert result.message == "ok"
    assert store.saves["t2"] == 1


async def test_exit_mode_still_saves_before_an_interrupt() -> None:
    store = _KillableCheckpointer()
    agent = _agent(_Model([_call("ask", "c1", question="sure?")]), store, durability="exit")
    events = await _drain(agent.run("go", thread_id="t"))
    assert isinstance(events[-1], InterruptEvent)
    paused = await store.load("t")
    assert paused is not None
    assert paused.messages[-1].tool_calls[0].name == "ask"


async def test_writer_flush_logs_a_failed_scheduled_write(caplog: pytest.LogCaptureFixture) -> None:
    class _Failing(MemoryCheckpointer):
        async def save(self, *args: Any, **kwargs: Any) -> str:
            raise OSError("disk gone")

    writer = CheckpointWriter(_Failing(), "async")
    assert await writer.save_iteration(AgentState(), "t")
    with caplog.at_level("WARNING"):
        await writer.flush()
    assert "scheduled checkpoint write" in caplog.text
    assert writer.pending == 0
    assert not await CheckpointWriter(MemoryCheckpointer(), "exit").save_iteration(
        AgentState(), "t"
    )


# --- crash and resume --------------------------------------------------------


@pytest.mark.parametrize("mode", ["async", "exit"])
@pytest.mark.parametrize("how", ["continue_turn", "resume"])
async def test_killed_after_the_tool_ran_the_tool_runs_once_overall(mode: str, how: str) -> None:
    store = _KillableCheckpointer(latency=0.005)
    first = _Model([_call("charge", "c1", order_id="o1"), _kill(store)])
    agent = _agent(first, store, durability=mode)  # type: ignore[arg-type]
    with pytest.raises(_KilledError):
        await _drain(agent.run("charge o1", thread_id="t", invocation_id="job-7"))

    # The tool ran, nothing of the turn was checkpointed, the ledger has it.
    assert calls == {"o1": 1}
    assert await store.load("t") is None
    entries = await ToolEffectLedger(store).entries("t")
    assert [(e.step, e.tool, e.invocation_id) for e in entries] == [(1, "charge", "job-7")]

    # A new process: fresh agent, same store (alive again). The model asks for
    # the same call again (a new call id, as a model would); the ledger answers.
    store.dead = False
    second = _Model([_call("charge", "c9", order_id="o1"), _answer("charged once")])
    agent2 = _agent(second, store, durability=mode)  # type: ignore[arg-type]
    if how == "continue_turn":
        events = await _drain(agent2.continue_turn("t"))
    else:
        events = await _drain(agent2.resume(invocation_id="job-7", thread_id="t"))

    assert calls == {"o1": 1}, "the recorded call is replayed, not run again"
    started = [e for e in events if isinstance(e, ToolStartEvent)]
    completed = [e for e in events if isinstance(e, ToolCompleteEvent)]
    assert [e.tool_name for e in started] == ["charge"]
    assert completed[0].result == "charged o1"
    assert [e for e in events if isinstance(e, TerminateEvent)][-1].final_message == "charged once"

    final = await store.load("t")
    assert final is not None
    assert final.run_id == "job-7"
    assert [m.content for m in final.messages if m.role == Role.USER] == ["charge o1"]
    assert [m.content for m in final.messages if m.role == Role.TOOL] == ["charged o1"]
    assert final.messages[-1].content == "charged once"
    assert ledger_thread_id("t") not in store.get_thread_ids(), "the ledger is cleared"


async def test_sync_mode_with_the_ledger_on_survives_a_kill_during_the_checkpoint_write() -> None:
    store = _KillableCheckpointer(die_on="t")
    agent = _agent(_Model([_call("charge", "c1", order_id="o1")]), store, tool_ledger=True)
    with pytest.raises(_KilledError):
        await _drain(agent.run("charge o1", thread_id="t"))
    assert calls == {"o1": 1}
    assert await store.load("t") is None

    store.die_on = None
    store.dead = False
    agent2 = _agent(
        _Model([_call("charge", "c2", order_id="o1"), _answer("done")]), store, tool_ledger=True
    )
    await _drain(agent2.continue_turn("t"))
    assert calls == {"o1": 1}
    final = await store.load("t")
    assert final is not None
    assert final.messages[-1].content == "done"


async def test_a_rerun_under_the_same_invocation_id_replays() -> None:
    store = _KillableCheckpointer()
    with pytest.raises(_KilledError):
        await _drain(
            _agent(
                _Model([_call("charge", "c1", order_id="o1"), _kill(store)]),
                store,
                durability="exit",
            ).run("charge o1", thread_id="t", invocation_id="job-1")
        )
    store.dead = False
    second = _Model([_call("charge", "c2", order_id="o1"), _answer("done")])
    result = await _agent(second, store, durability="exit").arun(
        "charge o1", thread_id="t", invocation_id="job-1"
    )
    assert calls == {"o1": 1}
    assert result.message == "done"
    assert result.state.run_id == "job-1"
    assert [m.content for m in result.state.messages if m.role == Role.USER] == ["charge o1"]


async def test_a_fresh_turn_never_reads_a_crashed_turns_records() -> None:
    store = _KillableCheckpointer()
    with pytest.raises(_KilledError):
        await _drain(
            _agent(
                _Model([_call("charge", "c1", order_id="o1"), _kill(store)]),
                store,
                durability="exit",
            ).run("charge o1", thread_id="t")
        )
    # A new run on the thread (no invocation id) is a new turn: at-least-once.
    store.dead = False
    second = _Model([_call("charge", "c2", order_id="o1"), _answer("done")])
    await _drain(_agent(second, store, durability="exit").run("charge o1", thread_id="t"))
    assert calls == {"o1": 2}
    assert ledger_thread_id("t") not in store.get_thread_ids()


async def test_ledger_miss_runs_the_call_and_a_hit_does_not() -> None:
    store = _KillableCheckpointer()
    first = _Model([_call("charge", "c1", order_id="o1"), _kill(store)])
    with pytest.raises(_KilledError):
        await _drain(_agent(first, store, durability="exit").run("charge", thread_id="t"))
    assert calls == {"o1": 1}

    store.dead = False
    # The continued run asks for a different order (a miss) and then the
    # recorded one (a hit), in one batch.
    second = _Model(
        [
            ModelResponse(
                message=Message.assistant(
                    content=None,
                    tool_calls=[
                        ToolCall(id="c2", name="charge", arguments={"order_id": "o2"}),
                        ToolCall(id="c3", name="charge", arguments={"order_id": "o1"}),
                    ],
                ),
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            ),
            _answer("done"),
        ]
    )
    events = await _drain(_agent(second, store, durability="exit").continue_turn("t"))
    assert calls == {"o1": 1, "o2": 1}
    completed = {e.tool_call_id: e.result for e in events if isinstance(e, ToolCompleteEvent)}
    assert completed == {"c2": "charged o2", "c3": "charged o1"}


async def test_a_call_at_another_step_is_another_call() -> None:
    store = _KillableCheckpointer()
    first = _Model([_call("charge", "c1", order_id="o1"), _kill(store)])
    with pytest.raises(_KilledError):
        await _drain(_agent(first, store, durability="exit").run("charge", thread_id="t"))

    store.dead = False
    # Step 1 looks up, step 2 charges o1 again: the step differs, so it runs.
    second = _Model(
        [
            _call("lookup", "c2", order_id="o1"),
            _call("charge", "c3", order_id="o1"),
            _answer("done"),
        ]
    )
    await _drain(_agent(second, store, durability="exit").continue_turn("t"))
    assert calls == {"o1": 2, "lookup:o1": 1}


@pytest.mark.parametrize("rerun", [False, True])
async def test_idempotent_tools_may_rerun_on_replay_when_the_developer_prefers(rerun: bool) -> None:
    store = _KillableCheckpointer()
    first = _Model([_call("lookup", "c1", order_id="o1"), _kill(store)])
    with pytest.raises(_KilledError):
        await _drain(_agent(first, store, durability="exit").run("look", thread_id="t"))
    assert calls == {"lookup:o1": 1}

    store.dead = False
    second = _Model([_call("lookup", "c2", order_id="o1"), _answer("done")])
    agent2 = _agent(second, store, durability="exit", rerun_idempotent_on_replay=rerun)
    await _drain(agent2.continue_turn("t"))
    assert calls == {"lookup:o1": 2 if rerun else 1}


async def test_resume_with_an_invocation_id_checks_it_and_needs_a_thread() -> None:
    store = _KillableCheckpointer()
    with pytest.raises(_KilledError):
        await _drain(
            _agent(
                _Model([_call("charge", "c1", order_id="o1"), _kill(store)]),
                store,
                durability="exit",
            ).run("charge", thread_id="t", invocation_id="job-1")
        )
    store.dead = False
    agent = _agent(_Model([_answer("done")]), store, durability="exit")
    with pytest.raises(RuntimeError, match="needs thread_id"):
        await _drain(agent.resume(invocation_id="job-1"))
    with pytest.raises(RuntimeError, match="takes no response"):
        await _drain(agent.resume("yes", invocation_id="job-1", thread_id="t"))
    with pytest.raises(RuntimeError, match="not in flight"):
        await _drain(agent.resume(invocation_id="job-2", thread_id="t"))
    assert calls == {"o1": 1}


async def test_continue_turn_still_refuses_a_finished_turn_and_an_unknown_thread() -> None:
    store = _KillableCheckpointer()
    agent = _agent(_Model([_answer("done")]), store, durability="exit")
    with pytest.raises(RuntimeError, match="No checkpoint"):
        await _drain(agent.continue_turn("nowhere"))
    await _drain(agent.run("hi", thread_id="t"))
    with pytest.raises(RuntimeError, match="already finished"):
        await _drain(agent.continue_turn("t"))


async def test_a_hook_retry_runs_the_call_and_refreshes_the_record() -> None:
    from locus.hooks import HookPriority, HookProvider
    from locus.hooks.provider import AfterToolCallEvent

    class _RetryOnce(HookProvider):
        def __init__(self) -> None:
            self.retried = False

        @property
        def priority(self) -> int:
            return HookPriority.OBSERVABILITY_DEFAULT

        async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
            if not self.retried:
                self.retried = True
                event.retry = True

    store = _KillableCheckpointer()
    agent = _agent(
        _Model([_call("charge", "c1", order_id="o1"), _answer("done")]),
        store,
        durability="exit",
        hooks=[_RetryOnce()],
    )
    await _drain(agent.run("charge", thread_id="t"))
    assert calls == {"o1": 2}, "the retry ran the body, not the ledger record"


# --- the graph runner ----------------------------------------------------------


def _graph(store: Any, thread_id: str, mode: str) -> StateGraph:
    graph = StateGraph()

    async def a(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"count": inputs.get("count", 0) + 1}

    async def b(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"count": inputs["count"] + 1}

    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    graph.compile(checkpointer=store)
    graph.config.thread_id = thread_id
    graph.config.durability = mode  # type: ignore[assignment]
    return graph


@pytest.mark.parametrize("mode", ["sync", "async", "exit"])
async def test_graph_runner_obeys_the_mode(mode: str) -> None:
    store = _KillableCheckpointer(latency=0.002)
    result = await _graph(store, "g", mode).execute({"count": 0})
    assert result.success
    assert result.final_state["count"] == 2
    # One save per completed step, or one save at the end.
    assert store.saves["g"] == (1 if mode == "exit" else 2)
    saved = await store.load("g")
    assert saved is not None
    assert saved.metadata["graph_state"]["count"] == 2


# --- segments ------------------------------------------------------------------


async def test_a_segment_can_be_continued_after_the_process_died() -> None:
    store = _KillableCheckpointer()
    models = iter(
        [
            _Model([_call("charge", "c1", order_id="o1"), _kill(store)]),
            _Model([_call("charge", "c2", order_id="o1"), _answer("charged once")]),
        ]
    )
    agents = {"billing": lambda: _agent(next(models), store, durability="exit")}
    with pytest.raises(_KilledError):
        await run_agent_segment(agents, "billing", thread_id="t", kind="run", prompt="charge o1")
    outcome = await run_agent_segment(agents, "billing", thread_id="t", kind="continue")
    assert outcome.status == "done"
    assert outcome.final_message == "charged once"
    assert calls == {"o1": 1}
