# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""What every checkpointer's time-travel surface must do, whatever it is backed by.

A test class mixes :class:`CheckpointerTimeTravelConformance` in and provides a
``make_checkpointer`` fixture that builds a checkpointer on a fresh backing;
one that bounds its history also mixes in :class:`BoundedHistoryConformance`
and provides ``make_bounded(keep)``. The unit suite runs them over the
in-memory, file, adapter-wrapped, S3 (mocked) and OCI bucket (faked)
checkpointers and over a minimal subclass that has only the base defaults;
the integration suite runs them over live PostgreSQL, Redis, Oracle and an
OCI bucket behind the usual service markers.

``settle_seconds`` is the pause between saves whose order the tests read
back: zero where the backend stamps writes to the microsecond, about a
second for an object store whose listing keeps modification times to the
second.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from locus.agent import Agent
from locus.core.events import InterruptEvent
from locus.core.messages import Message, Role, ToolCall, ToolResult
from locus.core.state import AgentState
from locus.memory.ledger import RunLedger, ToolEffectLedger
from locus.memory.time_travel import CheckpointInfo
from locus.testing import FunctionModel, text, tool_call
from locus.tools.decorator import tool


def _state(iteration: int) -> AgentState:
    """A state ``iteration`` steps in, with one more message per step."""
    state = AgentState(agent_id="tt").with_iteration(iteration)
    for step in range(iteration + 1):
        state = state.with_message(Message.user(f"message {step}"))
    return state


def _user_texts(messages: list[Message]) -> list[str]:
    return [str(m.content) for m in messages if m.role == Role.USER]


def _tool_texts(messages: list[Message]) -> list[str]:
    return [str(m.content) for m in messages if m.role == Role.TOOL]


@tool
def ask(question: str) -> str:
    """Pause the run on a question (the interrupt marker ``ask_user`` returns)."""
    return json.dumps({"__interrupt__": True, "question": question})


def _agent(
    checkpointer: Any, seen: list[list[Message]] | None = None, *, asks: bool = False
) -> Agent:
    """An agent that answers ``ok``; with ``asks`` it first pauses on a question each turn."""

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        if seen is not None:
            seen.append(list(messages))
        if asks and not _tool_texts(messages):
            return tool_call("ask", question="Which option?")
        return text("ok")

    # ``exit`` durability: one checkpoint per turn (or pause), so the saves
    # the tests read back are the turns themselves on every backend.
    return Agent(
        model=FunctionModel(handler),
        tools=[ask] if asks else [],
        checkpointer=checkpointer,
        max_iterations=5,
        reflexion=False,
        grounding=False,
        durability="exit",
    )


async def _drain(stream: Any) -> list[Any]:
    return [event async for event in stream]


async def _record_calls(
    ledger: ToolEffectLedger, thread_id: str, steps: range
) -> dict[int, ToolCall]:
    run = RunLedger(ledger, thread_id=thread_id, invocation_id="inv-1", replay=False, prompt="go")
    calls: dict[int, ToolCall] = {}
    for step in steps:
        call = ToolCall(id=f"call-{step}", name="lookup", arguments={"step": step})
        await run.record(
            call,
            ToolResult(tool_call_id=call.id, name="lookup", content=f"result {step}"),
            step=step,
        )
        calls[step] = call
    return calls


class _Settles:
    settle_seconds: float = 0.0

    async def settle(self) -> None:
        """Let the next save get a later timestamp than the last."""
        await asyncio.sleep(self.settle_seconds or 0.002)


class CheckpointerTimeTravelConformance(_Settles):
    """Mix in; provide a ``make_checkpointer`` fixture."""

    @pytest.fixture
    def cp(self, make_checkpointer: Any) -> Any:
        return make_checkpointer()

    async def _three(self, cp: Any, thread_id: str = "t") -> list[str]:
        ids: list[str] = []
        for step in range(3):
            ids.append(
                await cp.save(_state(step), thread_id, f"cp-{step}", metadata={"note": step})
            )
            await self.settle()
        return ids

    # -- list ----------------------------------------------------------------

    async def test_list_describes_each_checkpoint_newest_first(self, cp: Any) -> None:
        await self._three(cp)

        infos = await cp.list("t")

        assert all(isinstance(i, CheckpointInfo) for i in infos)
        assert [i.checkpoint_id for i in infos] == ["cp-2", "cp-1", "cp-0"]
        assert [i.iteration for i in infos] == [2, 1, 0]
        assert [i.parent_checkpoint_id for i in infos] == ["cp-1", "cp-0", None]
        assert [i.forked_from for i in infos] == [None, None, None]
        assert [i.metadata["note"] for i in infos] == [2, 1, 0]
        assert all(i.thread_id == "t" for i in infos)
        assert all(i.created_at is not None for i in infos)
        assert infos[0].created_at >= infos[-1].created_at
        sizes = [i.size for i in infos]
        assert all(isinstance(s, int) and s > 0 for s in sizes)
        assert sizes[0] > sizes[-1], "a state with more messages serialises larger"

    async def test_list_honours_the_limit_and_an_unknown_thread(self, cp: Any) -> None:
        await self._three(cp)

        assert [i.checkpoint_id for i in await cp.list("t", limit=2)] == ["cp-2", "cp-1"]
        assert await cp.list("no-such-thread") == []

    async def test_list_names_the_parent_a_checkpoint_recorded(self, cp: Any) -> None:
        await self._three(cp)
        edited = await cp.save(
            _state(0).with_message(Message.user("edited")),
            "t",
            metadata={"source": "update_state", "parent_checkpoint_id": "cp-0"},
        )

        infos = await cp.list("t")

        assert infos[0].checkpoint_id == edited
        assert infos[0].parent_checkpoint_id == "cp-0", "an explicit parent wins over position"
        assert infos[0].source == "update_state"
        assert infos[1].parent_checkpoint_id == "cp-1", "the rest still follow the line"

    # -- fork ----------------------------------------------------------------

    async def test_fork_copies_the_state_at_a_checkpoint_and_nothing_after(self, cp: Any) -> None:
        await self._three(cp)

        forked = await cp.fork("t", "cp-1", new_thread_id="t-alt")

        assert forked == "t-alt"
        fork_state = await cp.load("t-alt")
        assert fork_state is not None
        assert _user_texts(fork_state.messages) == ["message 0", "message 1"]
        assert fork_state.iteration == 1
        [first] = await cp.list("t-alt")
        assert first.forked_from == "t"
        assert first.parent_checkpoint_id == "cp-1"
        assert first.source == "fork"
        assert first.iteration == 1
        assert [i.checkpoint_id for i in await cp.list("t")] == ["cp-2", "cp-1", "cp-0"]

    async def test_a_fork_diverges_without_touching_its_parent(self, cp: Any) -> None:
        await self._three(cp)
        await cp.fork("t", "cp-1", new_thread_id="t-alt")
        await self.settle()

        await cp.save(_state(1).with_message(Message.user("elsewhere")), "t-alt", "alt-2")

        parent = await cp.load("t")
        assert parent is not None
        assert _user_texts(parent.messages) == ["message 0", "message 1", "message 2"]
        assert len(await cp.list("t")) == 3
        alt = await cp.list("t-alt")
        assert [i.checkpoint_id for i in alt][:1] == ["alt-2"]
        assert alt[0].parent_checkpoint_id == alt[1].checkpoint_id
        assert alt[1].forked_from == "t"

    async def test_fork_defaults_to_the_latest_checkpoint_and_a_fresh_id(self, cp: Any) -> None:
        await self._three(cp)

        forked = await cp.fork("t")

        assert forked.startswith("t-fork-")
        assert forked != "t"
        [first] = await cp.list(forked)
        assert first.parent_checkpoint_id == "cp-2"
        assert first.forked_from == "t"
        latest = await cp.load(forked)
        assert latest is not None
        assert latest.iteration == 2

    async def test_fork_refuses_a_missing_checkpoint_and_the_same_thread(self, cp: Any) -> None:
        await self._three(cp)

        with pytest.raises(LookupError):
            await cp.fork("t", "no-such-checkpoint")
        with pytest.raises(LookupError):
            await cp.fork("no-such-thread")
        with pytest.raises(ValueError, match="different"):
            await cp.fork("t", "cp-1", new_thread_id="t")

    # -- the agent: run and resume from a past step --------------------------

    async def test_run_from_checkpoint_starts_the_turn_at_a_past_step(self, cp: Any) -> None:
        seen: list[list[Message]] = []
        agent = _agent(cp, seen)
        await agent.arun("first", thread_id="t")
        await self.settle()
        await agent.arun("second", thread_id="t")
        await self.settle()
        history = await cp.list("t")
        assert len(history) == 2
        oldest = history[-1]

        await agent.arun("alternative second", thread_id="t", from_checkpoint=oldest.checkpoint_id)

        assert _user_texts(seen[-1]) == ["first", "alternative second"]
        newest, middle, first = await cp.list("t")
        assert first.checkpoint_id == oldest.checkpoint_id
        assert newest.parent_checkpoint_id == oldest.checkpoint_id, "recorded: not the previous one"
        assert middle.parent_checkpoint_id == oldest.checkpoint_id
        latest = await cp.load("t")
        assert latest is not None
        assert "alternative second" in _user_texts(latest.messages)
        assert "second" not in _user_texts(latest.messages)

    async def test_run_from_checkpoint_needs_a_real_checkpoint_on_a_thread(self, cp: Any) -> None:
        agent = _agent(cp)
        await agent.arun("first", thread_id="t")

        with pytest.raises(LookupError):
            await agent.arun("again", thread_id="t", from_checkpoint="no-such-checkpoint")
        with pytest.raises(RuntimeError, match="thread_id"):
            await agent.arun("again", from_checkpoint="whatever")

    async def test_resume_from_checkpoint_answers_a_past_question_again(self, cp: Any) -> None:
        seen: list[list[Message]] = []
        agent = _agent(cp, seen, asks=True)
        events = await _drain(agent.run("pick one", thread_id="t"))
        assert any(isinstance(e, InterruptEvent) for e in events), "the run paused"
        paused = (await cp.list("t"))[0]
        await self.settle()
        await _drain(agent.resume("option A", thread_id="t"))
        await self.settle()
        assert _tool_texts(seen[-1]) == ["option A"]
        answered_a = (await cp.list("t"))[0]
        assert answered_a.checkpoint_id != paused.checkpoint_id

        await _drain(agent.resume("option B", thread_id="t", from_checkpoint=paused.checkpoint_id))

        assert _tool_texts(seen[-1]) == ["option B"], "the question is answered afresh"
        newest = (await cp.list("t"))[0]
        assert newest.checkpoint_id != answered_a.checkpoint_id
        assert newest.parent_checkpoint_id == paused.checkpoint_id, "the branch is recorded"
        latest = await cp.load("t")
        assert latest is not None
        assert _tool_texts(latest.messages) == ["option B"]
        with pytest.raises(LookupError):
            await _drain(agent.resume("x", thread_id="t", from_checkpoint="no-such-checkpoint"))
        with pytest.raises(RuntimeError, match="thread_id"):
            await _drain(agent.resume("x", from_checkpoint=paused.checkpoint_id))

    # -- the ledger goes with the fork, up to the fork point ------------------

    async def test_a_fork_carries_the_ledger_only_up_to_the_forked_step(self, cp: Any) -> None:
        ledger = ToolEffectLedger(cp)
        calls = await _record_calls(ledger, "t", range(1, 4))
        await cp.save(_state(2), "t", "cp-2")

        await cp.fork("t", "cp-2", new_thread_id="t-alt")

        copied = await ledger.entries("t-alt")
        assert [e.step for e in copied] == [1, 2], "step 3 is past the fork point"
        assert all(e.thread_id == "t-alt" for e in copied)
        manifest = await ledger.manifest("t-alt")
        assert manifest is not None
        assert (manifest.thread_id, manifest.invocation_id) == ("t-alt", "inv-1")
        assert len(await ledger.entries("t")) == 3, "the source keeps its ledger"

        replaying = RunLedger(ledger, thread_id="t-alt", invocation_id="inv-1", replay=True)
        assert await replaying.replay(calls[3], step=3) is None
        served = await replaying.replay(calls[2], step=2)
        assert served is not None
        assert served.replayed is True
        assert served.content == "result 2"

    async def test_a_fork_of_a_thread_without_a_ledger_has_none(self, cp: Any) -> None:
        await cp.save(_state(0), "t", "cp-0")

        await cp.fork("t", "cp-0", new_thread_id="t-alt")

        assert await ToolEffectLedger(cp).entries("t-alt") == []
        assert await ToolEffectLedger(cp).manifest("t-alt") is None


class BoundedHistoryConformance(_Settles):
    """Mix in; provide a ``make_bounded(keep)`` fixture."""

    @pytest.fixture
    def bounded(self, make_bounded: Any) -> Any:
        return make_bounded(2)

    async def test_the_bound_keeps_the_newest_checkpoints(self, bounded: Any) -> None:
        for step in range(4):
            await bounded.save(_state(step), "t", f"cp-{step}")
            await self.settle()

        infos = await bounded.list("t")

        assert [i.checkpoint_id for i in infos] == ["cp-3", "cp-2"]
        assert infos[-1].parent_checkpoint_id is None, "history begins where the bound cut it"
        assert await bounded.load("t", "cp-0") is None
        latest = await bounded.load("t")
        assert latest is not None
        assert latest.iteration == 3

    async def test_the_bound_never_trims_the_ledger(self, bounded: Any) -> None:
        ledger = ToolEffectLedger(bounded)

        await _record_calls(ledger, "t", range(1, 5))

        assert [e.step for e in await ledger.entries("t")] == [1, 2, 3, 4]
