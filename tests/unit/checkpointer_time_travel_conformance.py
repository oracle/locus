# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""What every checkpointer's time-travel surface must do, whatever it is backed by.

A test class mixes :class:`CheckpointerTimeTravelConformance` in and provides a
``make_checkpointer`` fixture that builds a checkpointer on a fresh backing;
one that bounds its history also mixes in :class:`BoundedHistoryConformance`
and provides ``make_bounded(keep)``; one that takes the production settings
(``ttl``, ``encryption``, ``scope``) mixes in
:class:`ProductionSettingsConformance` and provides ``make_production(**settings)``,
whose every call in one test builds a checkpointer over the *same* backing.
The unit suite runs them over the in-memory, file, adapter-wrapped, S3
(mocked) and OCI bucket (faked) checkpointers and over a minimal subclass
that has only the base defaults; the integration suite runs them over live
PostgreSQL, Redis, Oracle and an OCI bucket behind the usual service markers.

Between saves whose order the tests read back there is a pause of a few
milliseconds and no more: every backend orders checkpoints by a time it
writes itself to the microsecond, an object store included, so several
checkpoints saved within one second must list in the order they were
saved (:meth:`CheckpointerTimeTravelConformance.test_saves_within_one_second_keep_their_order`).
"""

from __future__ import annotations

import asyncio
import json
import secrets
from typing import Any

import pytest

from locus.agent import Agent
from locus.core.events import InterruptEvent
from locus.core.messages import Message, Role, ToolCall, ToolResult
from locus.core.state import AgentState
from locus.memory.encryption import CheckpointEncryptionError, EncryptedSerializer
from locus.memory.ledger import RunLedger, ToolEffectLedger
from locus.memory.scope import CheckpointScope
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
    async def settle(self) -> None:
        """Let the next save get a later timestamp than the last."""
        await asyncio.sleep(0.002)


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

        window = await cp.list("t", limit=2)

        assert [i.checkpoint_id for i in window] == ["cp-2", "cp-1"]
        assert window[-1].parent_checkpoint_id == "cp-0", "the window's edge is not a root"
        assert [i.parent_checkpoint_id for i in await cp.list("t", limit=1)] == ["cp-1"]
        assert await cp.list("no-such-thread") == []

    async def test_saves_within_one_second_keep_their_order(self, cp: Any) -> None:
        for step in range(4):
            await cp.save(_state(step), "t", f"cp-{step}")

        ids = [i.checkpoint_id for i in await cp.list("t")]

        assert ids == ["cp-3", "cp-2", "cp-1", "cp-0"]
        assert await cp.list_checkpoints("t", limit=2) == ["cp-3", "cp-2"]
        latest = await cp.load("t")
        assert latest is not None
        assert latest.iteration == 3
        forked = await cp.fork("t", new_thread_id="t-alt")
        [first] = await cp.list(forked)
        fork_state = await cp.load(forked)
        assert fork_state is not None
        assert (first.parent_checkpoint_id, fork_state.iteration) == ("cp-3", 3)

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
        # No pause between saves: the bound must cut the oldest even when a
        # store's own clock cannot tell the four apart.
        for step in range(4):
            await bounded.save(_state(step), "t", f"cp-{step}")

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


#: The two scopes the production cases write under (and a live suite cleans up).
SCOPE_A = CheckpointScope(app_id="billing", user_id="u-1")
SCOPE_B = CheckpointScope(app_id="billing", user_id="u-2")
#: A ttl short enough to wait out in a test on a local store; Redis rounds it
#: up to one second for its own ``EXPIRE``, the recorded expiry stays exact. A
#: class over a remote store (an object-storage bucket) sets ``short_ttl``
#: to a few seconds: its saves take longer than this.
SHORT_TTL = 0.2


async def _thread_ids(cp: Any, **kwargs: Any) -> set[str]:
    """The thread ids ``list_threads`` names, as a set.

    The adapter over a key-value store lists its keys (``t:latest``,
    ``t:<checkpoint>``, ``t:_checkpoints``) rather than thread ids, as it
    always has; the part before the first colon is the thread, and the ids
    these cases use carry none.
    """
    return {t.split(":", 1)[0] for t in await cp.list_threads(limit=1000, **kwargs)}


class ProductionSettingsConformance(_Settles):
    """Mix in; provide a ``make_production`` fixture (``make(**settings)``, one backing per test)."""

    #: The ttl the expiry cases save with; a class over a slow store raises it.
    short_ttl: float = SHORT_TTL

    @pytest.fixture
    def production(self, make_production: Any) -> Any:
        return make_production

    async def outlive_the_ttl(self) -> None:
        """Wait until anything saved with ``short_ttl`` has expired."""
        await asyncio.sleep(self.short_ttl * 1.75)

    # -- ttl -----------------------------------------------------------------

    async def test_an_expired_checkpoint_is_not_loaded_listed_or_forked(
        self, production: Any
    ) -> None:
        cp = production(ttl=self.short_ttl)
        await cp.save(_state(0), "t", "cp-0", metadata={"note": 0})
        await self.settle()
        await cp.save(_state(1), "t", "cp-1", metadata={"note": 1})
        live = await cp.load("t")
        assert live is not None
        assert live.iteration == 1
        assert [i.checkpoint_id for i in await cp.list("t")] == ["cp-1", "cp-0"]
        assert await cp.exists("t")
        assert await cp.exists("t", "cp-0")

        await self.outlive_the_ttl()

        assert await cp.load("t") is None
        assert await cp.load("t", "cp-0") is None
        assert await cp.list("t") == []
        assert await cp.list_checkpoints("t") == []
        assert await cp.exists("t") is False
        assert await cp.exists("t", "cp-0") is False
        with pytest.raises(LookupError):
            await cp.fork("t")
        with pytest.raises(LookupError):
            await cp.fork("t", "cp-0")

    async def test_expire_removes_what_has_run_out_and_counts_it(self, production: Any) -> None:
        cp = production(ttl=self.short_ttl)
        await cp.save(_state(0), "t", "cp-0")
        await cp.save(_state(1), "u", "cp-1")
        assert await cp.expire() == 0, "nothing has run out yet"
        assert await cp.load("u") is not None

        await self.outlive_the_ttl()

        assert await cp.expire() == 2
        assert await cp.expire() == 0
        assert await cp.load("t") is None
        assert await cp.load("u") is None
        await cp.save(_state(2), "t", "cp-2")
        fresh = await cp.load("t")
        assert fresh is not None
        assert fresh.iteration == 2

    async def test_without_a_ttl_nothing_expires(self, production: Any) -> None:
        cp = production()
        assert cp.ttl is None
        await cp.save(_state(0), "t", "cp-0")
        await self.outlive_the_ttl()
        assert await cp.expire() == 0
        assert await cp.load("t") is not None

    async def test_a_ttl_bounds_what_the_agent_remembers(self, production: Any) -> None:
        cp = production(ttl=self.short_ttl)
        seen: list[list[Message]] = []
        agent = _agent(cp, seen)
        await agent.arun("first", thread_id="t")
        await agent.arun("second", thread_id="t")
        assert _user_texts(seen[-1]) == ["first", "second"]

        await self.outlive_the_ttl()
        await agent.arun("third", thread_id="t")

        assert _user_texts(seen[-1]) == ["third"], "the expired turns are gone"

    # -- encryption ----------------------------------------------------------

    async def test_encrypted_state_round_trips_lists_and_forks(self, production: Any) -> None:
        cp = production(encryption=EncryptedSerializer(secrets.token_bytes(32)))
        await self._three(cp)

        loaded = await cp.load("t")
        assert loaded is not None
        assert _user_texts(loaded.messages) == ["message 0", "message 1", "message 2"]
        infos = await cp.list("t")
        assert [i.iteration for i in infos] == [2, 1, 0], "the iteration stays readable"
        assert [i.metadata["note"] for i in infos] == [2, 1, 0]
        forked = await cp.fork("t", "cp-1", new_thread_id="t-alt")
        fork_state = await cp.load(forked)
        assert fork_state is not None
        assert _user_texts(fork_state.messages) == ["message 0", "message 1"]

    async def test_the_key_is_needed_to_load_but_not_to_list(self, production: Any) -> None:
        key, other, new = (secrets.token_bytes(32) for _ in range(3))
        writer = production(encryption=EncryptedSerializer(key))
        await self._three(writer)

        without_key = production()
        infos = await without_key.list("t")
        assert [i.checkpoint_id for i in infos] == ["cp-2", "cp-1", "cp-0"]
        assert [i.iteration for i in infos] == [2, 1, 0]
        assert [i.parent_checkpoint_id for i in infos] == ["cp-1", "cp-0", None]
        assert await without_key.list_checkpoints("t") == ["cp-2", "cp-1", "cp-0"]
        with pytest.raises(CheckpointEncryptionError, match="encrypted"):
            await without_key.load("t")
        with pytest.raises(CheckpointEncryptionError, match="no configured key"):
            await production(encryption=EncryptedSerializer(other)).load("t")

        rotated = production(encryption=EncryptedSerializer(new, previous_keys=[key]))
        old = await rotated.load("t", "cp-2")
        assert old is not None
        assert old.iteration == 2, "the previous key still reads what it wrote"
        await rotated.save(_state(3), "t", "cp-3")
        with pytest.raises(CheckpointEncryptionError):
            await writer.load("t", "cp-3")

    async def test_plaintext_checkpoints_load_after_encryption_is_turned_on(
        self, production: Any
    ) -> None:
        plain = production()
        await plain.save(_state(0), "t", "cp-0")
        await self.settle()

        encrypted = production(encryption=EncryptedSerializer(secrets.token_bytes(32)))
        before = await encrypted.load("t")
        assert before is not None
        assert before.iteration == 0
        await encrypted.save(_state(1), "t", "cp-1")

        assert [i.checkpoint_id for i in await encrypted.list("t")] == ["cp-1", "cp-0"]
        still_plain = await plain.load("t", "cp-0")
        assert still_plain is not None
        with pytest.raises(CheckpointEncryptionError):
            await plain.load("t")

    # -- scope ---------------------------------------------------------------

    async def test_a_scope_keeps_the_same_thread_id_apart(self, production: Any) -> None:
        a, b, none = production(scope=SCOPE_A), production(scope=SCOPE_B), production()
        await a.save(_state(1), "t", "cp-a")
        await b.save(_state(2), "t", "cp-b")
        await none.save(_state(3), "t", "cp-n")

        for cp, iteration in ((a, 1), (b, 2), (none, 3)):
            loaded = await cp.load("t")
            assert loaded is not None
            assert loaded.iteration == iteration
        assert await a.load("t", "cp-b") is None, "another scope's checkpoint is a miss"
        assert await none.load("t", "cp-a") is None
        assert await a.exists("t", "cp-b") is False
        assert [i.checkpoint_id for i in await a.list("t")] == ["cp-a"]
        assert [i.checkpoint_id for i in await none.list("t")] == ["cp-n"]

        assert await _thread_ids(a) == {"t"}
        assert await _thread_ids(a, scope=SCOPE_B) == {"t"}
        assert await _thread_ids(none) == {"t"}, "no scope lists the unscoped threads only"
        assert await _thread_ids(none, scope=SCOPE_A) == {"t"}
        assert await _thread_ids(production(scope=CheckpointScope(app_id="other"))) == set()

    async def test_a_scoped_fork_stays_in_its_scope(self, production: Any) -> None:
        a = production(scope=SCOPE_A)
        await a.save(_state(0), "t", "cp-0")

        forked = await a.fork("t", new_thread_id="t-alt")

        assert await a.load(forked) is not None
        assert await production().load(forked) is None
        assert await production(scope=SCOPE_B).load(forked) is None

    async def test_a_scoped_run_resumes_on_its_scope_and_misses_on_another(
        self, production: Any
    ) -> None:
        cp = production()
        seen: list[list[Message]] = []
        agent = _agent(cp, seen)
        await agent.arun("first", thread_id="t", scope=SCOPE_A)
        await self.settle()
        await agent.arun("second", thread_id="t", scope=SCOPE_A)
        assert _user_texts(seen[-1]) == ["first", "second"]

        await agent.arun("other", thread_id="t", scope=SCOPE_B)
        assert _user_texts(seen[-1]) == ["other"], "another scope is another thread"
        await agent.arun("plain", thread_id="t")
        assert _user_texts(seen[-1]) == ["plain"], "no scope is a third thread"

        assert len(await production(scope=SCOPE_A).list("t")) == 2
        assert len(await production(scope=SCOPE_B).list("t")) == 1
        assert len(await cp.list("t")) == 1
        with SCOPE_A.active():
            assert len(await cp.list("t")) == 2, "an active scope beats the instance's"
        events = [e async for e in agent.run("streamed", thread_id="t", scope=SCOPE_A)]
        assert events
        assert _user_texts(seen[-1]) == ["first", "second", "streamed"]
