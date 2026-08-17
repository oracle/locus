# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for ``AgentRuntimeMixin._run_from_state`` (the resume
loop) and ``Agent.resume`` in ``tulip.agent``.

``_run_from_state`` is driven directly with constructed states and scripted
models so each branch (time/termination/should_terminate stops, no-tool
completion, explicit-mode continue, tool execution, interrupt + plain tool
error) is reached deterministically. ``resume`` is exercised end-to-end via
an ``ask_user`` interrupt.
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip.agent import Agent
from tulip.core.events import InterruptEvent, TerminateEvent, ToolCompleteEvent
from tulip.core.interrupt import InterruptException, InterruptValue
from tulip.core.messages import Message, Role, ToolCall
from tulip.core.state import AgentState
from tulip.core.termination import MaxIterations
from tulip.memory.checkpointer import BaseCheckpointer
from tulip.models.base import ModelResponse
from tulip.tools.decorator import tool
from tulip.tools.executor import SequentialExecutor


class _ScriptedModel:
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


def _tc(name: str, args: dict[str, Any], *, tc_id: str = "c1") -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(
            content="thinking", tool_calls=[ToolCall(id=tc_id, name=name, arguments=args)]
        ),
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )


def _text(content: str) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(content=content),
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )


@tool
def trivial() -> str:
    """A trivial tool."""
    return "ok"


@tool
def needs_input() -> str:
    """Return the runtime's interrupt marker (like ask_user does)."""
    import json

    return json.dumps({"__interrupt__": True, "question": "Which option?", "options": ["a", "b"]})


async def _run_from_state(agent: Agent, state: AgentState, prompt: str = "p") -> list[Any]:
    events: list[Any] = []
    async for ev in agent._run_from_state(state, prompt, None, None):
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# Time budget stop (lines 1135-1157)
# ---------------------------------------------------------------------------


async def test_run_from_state_time_budget_stop() -> None:
    agent = Agent(
        model=_ScriptedModel([_text("x")], loop_last=True),
        time_budget_seconds=1e-9,
        termination=MaxIterations(100),
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    state = await agent._create_initial_state("p", None, None)
    # An assistant message exercises the "last assistant content" extractor.
    state = state.with_message(Message.assistant("prior answer"))
    events = await _run_from_state(agent, state)
    term = next(e for e in events if isinstance(e, TerminateEvent))
    assert term.reason == "time_budget"


# ---------------------------------------------------------------------------
# User-supplied termination stop (lines 1159-1173)
# ---------------------------------------------------------------------------


async def test_run_from_state_user_termination_stop() -> None:
    agent = Agent(
        model=_ScriptedModel([_text("x")], loop_last=True),
        termination=MaxIterations(0),
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    state = await agent._create_initial_state("p", None, None)
    events = await _run_from_state(agent, state)
    term = next(e for e in events if isinstance(e, TerminateEvent))
    assert term.reason == "max_iterations"


# ---------------------------------------------------------------------------
# state.should_terminate stop (lines 1175-1184)
# ---------------------------------------------------------------------------


async def test_run_from_state_should_terminate_stop() -> None:
    agent = Agent(
        model=_ScriptedModel([_text("x")], loop_last=True),
        reflexion=False,
        grounding=False,
    )
    # iteration already at the cap -> should_terminate fires immediately.
    state = AgentState(max_iterations=1, iteration=1)
    events = await _run_from_state(agent, state)
    term = next(e for e in events if isinstance(e, TerminateEvent))
    assert term.reason == "max_iterations"


# ---------------------------------------------------------------------------
# No-tool completion (lines 1186-1216)
# ---------------------------------------------------------------------------


async def test_run_from_state_no_tool_completion() -> None:
    agent = Agent(
        model=_ScriptedModel([_text("final answer")]),
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    state = await agent._create_initial_state("p", None, None)
    events = await _run_from_state(agent, state)
    term = next(e for e in events if isinstance(e, TerminateEvent))
    assert term.reason == "complete"
    assert term.final_message == "final answer"


# ---------------------------------------------------------------------------
# Explicit-mode continue when no tool calls (lines 1218-1219)
# ---------------------------------------------------------------------------


async def test_run_from_state_explicit_no_tools_continues() -> None:
    agent = Agent(
        model=_ScriptedModel([_text("still thinking")], loop_last=True),
        completion_mode="explicit",
        max_iterations=2,
        reflexion=False,
        grounding=False,
    )
    state = await agent._create_initial_state("p", None, None)
    events = await _run_from_state(agent, state)
    # Explicit mode never auto-completes on empty tool calls; it loops until
    # the hard iteration cap.
    term = next(e for e in events if isinstance(e, TerminateEvent))
    assert term.reason == "max_iterations"


# ---------------------------------------------------------------------------
# Normal tool execution (lines 1222-1290)
# ---------------------------------------------------------------------------


async def test_run_from_state_tool_execution() -> None:
    agent = Agent(
        model=_ScriptedModel([_tc("trivial", {}), _text("done")]),
        tools=[trivial],
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    state = await agent._create_initial_state("p", None, None)
    events = await _run_from_state(agent, state)
    complete = next(e for e in events if isinstance(e, ToolCompleteEvent))
    assert complete.tool_name == "trivial"
    assert complete.result == "ok"


# ---------------------------------------------------------------------------
# Interrupt raised from the executor (lines 1240-1263)
# ---------------------------------------------------------------------------


class _InterruptExecutor(SequentialExecutor):
    async def execute(self, tool_calls: Any, registry: Any, ctx_factory: Any = None) -> Any:
        raise InterruptException(
            InterruptValue(payload={"question": "Proceed?", "options": ["yes", "no"]})
        )


async def test_run_from_state_interrupt_yields_interrupt_event() -> None:
    agent = Agent(
        model=_ScriptedModel([_tc("trivial", {})]),
        tools=[trivial],
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    agent._executor = _InterruptExecutor()
    state = await agent._create_initial_state("p", None, None)
    events = await _run_from_state(agent, state)
    interrupt = next(e for e in events if isinstance(e, InterruptEvent))
    assert interrupt.question == "Proceed?"
    assert interrupt.options == ["yes", "no"]
    # Interrupt bookkeeping was stored for a subsequent resume.
    assert agent._interrupt_state is not None


# ---------------------------------------------------------------------------
# Plain tool error from the executor (lines 1264-1290)
# ---------------------------------------------------------------------------


class _PlainErrorExecutor(SequentialExecutor):
    async def execute(self, tool_calls: Any, registry: Any, ctx_factory: Any = None) -> Any:
        raise RuntimeError("exec boom")


async def test_run_from_state_plain_tool_error() -> None:
    agent = Agent(
        model=_ScriptedModel([_tc("trivial", {}), _text("done")]),
        tools=[trivial],
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    agent._executor = _PlainErrorExecutor()
    state = await agent._create_initial_state("p", None, None)
    events = await _run_from_state(agent, state)
    complete = next(e for e in events if isinstance(e, ToolCompleteEvent))
    assert complete.error is not None
    assert "exec boom" in complete.error


# ---------------------------------------------------------------------------
# End-to-end interrupt + resume (agent.py 505-529 + _run_from_state)
# ---------------------------------------------------------------------------


async def test_interrupt_then_resume_round_trip() -> None:
    model = _ScriptedModel(
        [
            _tc("needs_input", {}),
            _text("resumed and finished"),
        ]
    )
    agent = Agent(
        model=model,
        tools=[needs_input],
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )

    first: list[Any] = []
    async for ev in agent.run("start"):
        first.append(ev)
    assert any(isinstance(e, InterruptEvent) for e in first)
    assert agent._interrupt_state is not None

    second: list[Any] = []
    async for ev in agent.resume("the user's answer"):
        second.append(ev)
    term = next(e for e in second if isinstance(e, TerminateEvent))
    assert term.reason == "complete"
    assert term.final_message == "resumed and finished"
    # resume() cleared the interrupt bookkeeping.
    assert agent._interrupt_state is None


async def test_second_ask_user_after_resume_re_pauses() -> None:
    """A SECOND ``ask_user`` in a RESUMED run must pause again — the spine of a
    multi-turn clarification thread (ask → answer → ask again → answer → done).

    The resume loop only paused on ``InterruptException`` before; a marker-
    returning tool (``ask_user``) called on resume folded its marker in as an
    ordinary tool result and the run carried on. This asserts the second ask
    re-pauses instead of finishing.
    """
    model = _ScriptedModel(
        [
            _tc("needs_input", {}, tc_id="q1"),  # ask #1  (first pass)
            _tc("needs_input", {}, tc_id="q2"),  # ask #2  (after the first resume)
            _text("all done"),  # only reached after the second answer
        ]
    )
    agent = Agent(
        model=model,
        tools=[needs_input],
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )

    first: list[Any] = []
    async for ev in agent.run("start"):
        first.append(ev)
    assert any(isinstance(e, InterruptEvent) for e in first)  # ask #1

    # First answer → the model asks AGAIN. Must re-pause, not run to completion.
    second: list[Any] = []
    async for ev in agent.resume("first answer"):
        second.append(ev)
    assert any(isinstance(e, InterruptEvent) for e in second), "second ask_user did not re-pause"
    assert not any(isinstance(e, TerminateEvent) for e in second), "run finished instead of asking"
    assert agent._interrupt_state is not None  # parked for the next answer

    # Second answer → the run actually finishes.
    third: list[Any] = []
    async for ev in agent.resume("second answer"):
        third.append(ev)
    term = next(e for e in third if isinstance(e, TerminateEvent))
    assert term.reason == "complete"
    assert term.final_message == "all done"
    assert agent._interrupt_state is None


# ---------------------------------------------------------------------------
# Cross-process resume — rehydrate the interrupt from a checkpointer
# ---------------------------------------------------------------------------


class _DictCheckpointer(BaseCheckpointer):
    """Checkpointer over a shared dict — two Agent instances see one store."""

    def __init__(self, store: dict[str, AgentState]) -> None:
        self.store = store
        self.saves = 0

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.saves += 1
        self.store[thread_id] = state
        return thread_id

    async def load(self, thread_id: str, checkpoint_id: str | None = None) -> AgentState | None:
        return self.store.get(thread_id)

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[Any]:
        return [thread_id] if thread_id in self.store else []


async def test_resume_rehydrates_from_checkpoint_in_fresh_process() -> None:
    """A run pauses in one Agent; a FRESH Agent resumes it via the checkpointer."""
    store: dict[str, AgentState] = {}

    first_agent = Agent(
        model=_ScriptedModel([_tc("needs_input", {})]),
        tools=[needs_input],
        checkpointer=_DictCheckpointer(store),
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    first: list[Any] = []
    async for ev in first_agent.run("start", thread_id="t-crosspod"):
        first.append(ev)
    assert any(isinstance(e, InterruptEvent) for e in first)
    # The pause-time state was persisted (run()'s final checkpoint).
    assert "t-crosspod" in store

    # A fresh Agent — no in-memory interrupt, only the shared checkpointer.
    second_agent = Agent(
        model=_ScriptedModel([_text("resumed on a new pod")]),
        tools=[needs_input],
        checkpointer=_DictCheckpointer(store),
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    assert second_agent._interrupt_state is None
    second: list[Any] = []
    async for ev in second_agent.resume("approve", thread_id="t-crosspod"):
        second.append(ev)
    term = next(e for e in second if isinstance(e, TerminateEvent))
    assert term.reason == "complete"
    assert term.final_message == "resumed on a new pod"
    # The rehydrated state carries the decision for the model — as the TOOL
    # RESULT of the call it decides, not as a system note beside it.
    #
    # This assertion used to expect "[User Response] approve". That was the
    # system-note fallback, which fired here because the fold was restricted to
    # calls named `ask_user` and this one is `needs_input`. Note what this test
    # actually covers: a cross-pod resume carrying an APPROVAL. The fallback
    # breaks the call->result rhythm, and a model that does not recover from the
    # break silently never performs the approved action while the run reports
    # success. The assertion followed the bug; it now follows the fix.
    msgs = second_agent._last_run_state.messages
    assert any(m.role == Role.TOOL and "approve" in str(m.content or "") for m in msgs), (
        "the decision should arrive as the dangling call's tool result"
    )
    assert not any(
        m.role == Role.SYSTEM and "[User Response]" in str(m.content or "") for m in msgs
    )


async def test_interrupt_checkpoints_before_yield_when_consumer_parks() -> None:
    """The pause-time state persists even if the consumer stops at the interrupt.

    An HTTP layer parks the run the moment it sees the InterruptEvent and never
    drives the generator further — its finally (the final checkpoint) would only
    run at GC. The interrupt site must save BEFORE yielding.
    """
    store: dict[str, AgentState] = {}
    agent = Agent(
        model=_ScriptedModel([_tc("needs_input", {})]),
        tools=[needs_input],
        checkpointer=_DictCheckpointer(store),
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    gen = agent.run("start", thread_id="t-parked")
    async for ev in gen:
        if isinstance(ev, InterruptEvent):
            break  # park: stop consuming, like the gateway's SSE layer does
    assert "t-parked" in store  # persisted before the yield, not in finally
    await gen.aclose()


async def test_resume_from_state_saves_final_checkpoint() -> None:
    """A resumed run re-persists its state — durability survives resume."""
    store: dict[str, AgentState] = {}
    ckpt = _DictCheckpointer(store)
    agent = Agent(
        model=_ScriptedModel([_tc("needs_input", {}), _text("done")]),
        tools=[needs_input],
        checkpointer=ckpt,
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    async for _ in agent.run("start", thread_id="t-durable"):
        pass
    saves_at_pause = ckpt.saves
    assert saves_at_pause >= 1

    async for _ in agent.resume("go ahead"):
        pass
    # The in-memory resume path also checkpoints its final state.
    assert ckpt.saves > saves_at_pause
    final = store["t-durable"]
    # Folded as the dangling call's tool result rather than a system note —
    # see the cross-pod case above for why the fallback is the wrong channel.
    assert any(m.role == Role.TOOL and "go ahead" in str(m.content or "") for m in final.messages)
    assert not any(
        m.role == Role.SYSTEM and "[User Response]" in str(m.content or "") for m in final.messages
    )


@tool(name="ask_user")
def ask_user(question: str, options: str = "") -> str:
    """Ask the user and pause — returns the runtime's interrupt marker."""
    import json

    opts = [o.strip() for o in options.split(",") if o.strip()] if options else None
    return json.dumps({"__interrupt__": True, "question": question, "options": opts})


async def test_ask_answer_ask_again_parks_each_time_and_folds_tool_results() -> None:
    """ask → answer → ask-again works for N turns, and each answer lands as the
    dangling ask_user call's TOOL result (not a system note) — the rhythm that
    keeps a live model asking through the tool instead of in its final text."""
    agent = Agent(
        model=_ScriptedModel(
            [
                _tc("ask_user", {"question": "Payment id?"}, tc_id="q1"),
                _tc("ask_user", {"question": "And the reason?"}, tc_id="q2"),
                _text("refund filed"),
            ]
        ),
        tools=[ask_user],
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    first: list[Any] = []
    async for ev in agent.run("refund please"):
        first.append(ev)
    assert any(isinstance(e, InterruptEvent) for e in first)

    # First answer: the resumed loop must pause AGAIN on the second ask_user.
    second: list[Any] = []
    async for ev in agent.resume("pi_123"):
        second.append(ev)
    parked_again = [e for e in second if isinstance(e, InterruptEvent)]
    assert len(parked_again) == 1
    assert parked_again[0].question == "And the reason?"
    assert not any(isinstance(e, TerminateEvent) for e in second)

    # Second answer completes the run.
    third: list[Any] = []
    async for ev in agent.resume("damaged-goods"):
        third.append(ev)
    term = next(e for e in third if isinstance(e, TerminateEvent))
    assert term.final_message == "refund filed"

    # Both answers were folded as tool results of their OWN calls, and no
    # "[User Response]" system note was injected for either.
    msgs = agent._last_run_state.messages
    folded = {m.tool_call_id: m.content for m in msgs if m.name == "ask_user" and m.content}
    assert folded.get("q1") == "pi_123"
    assert folded.get("q2") == "damaged-goods"
    assert not any("[User Response]" in (m.content or "") for m in msgs)


async def test_resumed_loop_fires_tool_hooks_and_honors_cancel() -> None:
    """The resume loop runs the SAME before/after tool hook seam as run() —
    a playbook tracker or admit()-style gate must not go blind after an
    answer, and a hook cancel must stop the body from executing."""
    from tulip.hooks.provider import AfterToolCallEvent, BeforeToolCallEvent, HookProvider

    seen: dict[str, list[str]] = {"before": [], "after": []}

    class _Gate(HookProvider):
        @property
        def priority(self) -> int:
            return 100

        async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
            seen["before"].append(event.tool_name)
            if event.tool_name == "trivial" and len(seen["before"]) > 1:
                event.cancel = "blocked on resume by the gate"

        async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
            seen["after"].append(event.tool_name)

    agent = Agent(
        model=_ScriptedModel(
            [
                _tc("ask_user", {"question": "go on?"}, tc_id="q1"),
                _tc("trivial", {}, tc_id="t1"),
                _text("finished"),
            ]
        ),
        tools=[ask_user, trivial],
        hooks=[_Gate()],
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    async for _ in agent.run("start"):
        pass
    resumed: list[Any] = []
    async for ev in agent.resume("yes"):
        resumed.append(ev)
    # The hook saw the resumed call, and its cancel stood in for the body.
    assert "trivial" in seen["before"]
    complete = next(
        e for e in resumed if isinstance(e, ToolCompleteEvent) and e.tool_name == "trivial"
    )
    assert complete.result == "blocked on resume by the gate"
    term = next(e for e in resumed if isinstance(e, TerminateEvent))
    assert term.final_message == "finished"


async def test_interrupt_fields_survive_park_and_repark() -> None:
    """A structured input request ({name,label,type} field specs) rides the
    interrupt marker to the InterruptEvent — on the FIRST park and on an
    ask-again park after resume — so a console can render a form."""
    import json as _json

    from tulip.tools.decorator import tool as _tool

    @_tool(name="ask_form")
    def ask_form(question: str) -> str:
        """Ask with a structured field spec."""
        return _json.dumps(
            {
                "__interrupt__": True,
                "question": question,
                "options": None,
                "fields": [{"name": "payment_intent", "label": "Payment ID", "type": "text"}],
            }
        )

    agent = Agent(
        model=_ScriptedModel(
            [
                _tc("ask_form", {"question": "Details?"}, tc_id="f1"),
                _tc("ask_form", {"question": "More?"}, tc_id="f2"),
                _text("done"),
            ]
        ),
        tools=[ask_form],
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )
    first = [ev async for ev in agent.run("go")]
    park1 = next(e for e in first if isinstance(e, InterruptEvent))
    assert park1.fields == [{"name": "payment_intent", "label": "Payment ID", "type": "text"}]
    second = [ev async for ev in agent.resume("pi_1")]
    park2 = next(e for e in second if isinstance(e, InterruptEvent))
    assert park2.fields is not None
    assert park2.fields[0]["name"] == "payment_intent"


async def test_resume_without_interrupt_or_thread_id_raises() -> None:
    agent = Agent(
        model=_ScriptedModel([_text("x")]),
        checkpointer=_DictCheckpointer({}),
        reflexion=False,
        grounding=False,
    )
    with pytest.raises(RuntimeError, match="No interrupt to resume from"):
        async for _ in agent.resume("hello"):
            pass  # pragma: no cover


async def test_resume_without_checkpointer_raises_even_with_thread_id() -> None:
    agent = Agent(model=_ScriptedModel([_text("x")]), reflexion=False, grounding=False)
    with pytest.raises(RuntimeError, match="No interrupt to resume from"):
        async for _ in agent.resume("hello", thread_id="t-missing"):
            pass  # pragma: no cover


async def test_resume_with_missing_checkpoint_raises() -> None:
    agent = Agent(
        model=_ScriptedModel([_text("x")]),
        checkpointer=_DictCheckpointer({}),
        reflexion=False,
        grounding=False,
    )
    with pytest.raises(RuntimeError, match="No checkpoint found for thread 't-ghost'"):
        async for _ in agent.resume("hello", thread_id="t-ghost"):
            pass  # pragma: no cover


async def test_resume_answers_a_dangling_non_ask_user_call_as_its_tool_result() -> None:
    """A human's reply must arrive as the TOOL RESULT of whatever call is
    dangling — not only when that call happens to be named ``ask_user``.

    The fold exists because a bare system note breaks the call->result rhythm
    the model is pattern-matching on. That reasoning is name-independent, but
    the search was restricted to ``ask_user``, so an APPROVAL hold — which
    suspends on the governed call itself — always took the system-note path.
    Observed consequence: on resume the model returns an empty turn, the loop
    reads that as "finished", and an action a human approved is never performed
    while the run reports success.

    Asserted on the conversation the model is handed, because that is the thing
    the behaviour depends on; whether a given model recovers from a broken
    rhythm varies by model and is not a property this repo can test.
    """
    model = _ScriptedModel([_tc("needs_input", {}), _text("done")])
    agent = Agent(
        model=model, tools=[needs_input], max_iterations=10, reflexion=False, grounding=False
    )

    async for _ in agent.run("start"):
        pass

    seen: list[Message] = []
    original = model.complete

    async def _capture(messages: list[Message], *a: Any, **k: Any) -> ModelResponse:
        seen.clear()
        seen.extend(messages)
        return await original(messages, *a, **k)

    model.complete = _capture  # type: ignore[method-assign]
    async for _ in agent.resume("APPROVED — issue the call again to perform it"):
        pass

    call = next(tc for m in seen if m.role == Role.ASSISTANT for tc in (m.tool_calls or []))
    assert call.name == "needs_input"

    results = [m for m in seen if m.role == Role.TOOL and m.tool_call_id == call.id]
    assert results, (
        "the dangling call was left unanswered and the reply went in as a bare "
        "system note — the rhythm break this fold exists to prevent"
    )
    assert "APPROVED" in str(results[-1].content)
    assert not [
        m for m in seen if m.role == Role.SYSTEM and "[User Response]" in str(m.content or "")
    ], "the system-note fallback should not fire when a dangling call exists"
