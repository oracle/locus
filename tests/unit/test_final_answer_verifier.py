# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``AgentConfig.final_answer_verifier``: a pluggable check on every final answer.

The grounding evaluator only runs after a tool call and cannot be swapped, and
an after-model hook's ``retry`` re-called the model blind. These tests pin a
verifier that runs whether or not tools were called, feeds its verdict back for
a bounded number of replans, keeps rejected drafts out of what is persisted, and
(optionally) out of what is streamed.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from tulip.agent import Agent
from tulip.agent.verification import (
    EPHEMERAL_MESSAGE_KEY,
    FinalAnswerContext,
    is_ephemeral_message,
    mark_ephemeral,
)
from tulip.control import Action, ControlPolicy, InMemoryApprovals, gate_tool
from tulip.core.events import (
    FinalAnswerVerificationEvent,
    InterruptEvent,
    ModelChunkEvent,
    TerminateEvent,
    TulipEvent,
)
from tulip.core.messages import Message
from tulip.hooks.provider import AfterModelCallEvent, HookProvider
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.testing import FunctionModel, ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


@tool
def get_rate(hotel: str) -> str:
    """Look up tonight's rate."""
    return f"{hotel}: 420 EUR"


async def _collect(stream: Any) -> list[TulipEvent]:
    return [e async for e in stream]


def _agent(model: Any, verifier: Any, **kwargs: Any) -> Agent:
    return Agent(
        model=model,
        tools=kwargs.pop("tools", []),
        final_answer_verifier=verifier,
        reflexion=False,
        grounding=False,
        **kwargs,
    )


class _Verifier:
    """Rejects any draft containing ``bad`` with ``feedback``; records calls."""

    def __init__(self, bad: str = "WRONG", feedback: str = "Fix the price.") -> None:
        self.bad = bad
        self.feedback = feedback
        self.seen: list[tuple[str, FinalAnswerContext]] = []

    async def __call__(self, draft: str, ctx: FinalAnswerContext) -> str | None:
        self.seen.append((draft, ctx))
        return self.feedback if self.bad in draft else None


def _verdicts(events: list[TulipEvent]) -> list[FinalAnswerVerificationEvent]:
    return [e for e in events if isinstance(e, FinalAnswerVerificationEvent)]


def _terminate(events: list[TulipEvent]) -> TerminateEvent:
    return next(e for e in events if isinstance(e, TerminateEvent))


# ---------------------------------------------------------------------------
# Off by default
# ---------------------------------------------------------------------------


async def test_off_by_default_nothing_changes() -> None:
    model = ScriptedModel([text("WRONG answer")])
    agent = Agent(model=model, reflexion=False, grounding=False)
    events = await _collect(agent.run("hi"))
    assert not _verdicts(events)
    assert _terminate(events).final_message == "WRONG answer"
    assert agent.config.final_answer_verifier_max_replans == 1
    assert agent.config.hold_final_answer_tokens is False


def test_a_non_callable_verifier_is_refused() -> None:
    with pytest.raises(TypeError, match="final_answer_verifier"):
        Agent(model=ScriptedModel([text("x")]), final_answer_verifier="nope")


# ---------------------------------------------------------------------------
# Runs with and without tools; replans with feedback
# ---------------------------------------------------------------------------


async def test_runs_without_any_tool_call_and_replans_with_feedback() -> None:
    model = ScriptedModel([text("It is 100 WRONG"), text("It is 420 EUR")])
    verifier = _Verifier()
    events = await _collect(_agent(model, verifier).run("price?"))

    verdicts = _verdicts(events)
    assert [(v.passed, v.replanning, v.attempt) for v in verdicts] == [
        (False, True, 0),
        (True, False, 1),
    ]
    assert verdicts[0].feedback == "Fix the price."
    assert _terminate(events).final_message == "It is 420 EUR"
    # The second call saw the draft and the feedback.
    second = model.received_messages[1]
    assert second[-2].content == "It is 100 WRONG"
    assert second[-1].role == "user"
    assert "not from the user" in (second[-1].content or "")
    assert "Fix the price." in (second[-1].content or "")
    # The context carried the run and the attempt number.
    assert [ctx.attempt for _, ctx in verifier.seen] == [0, 1]
    assert verifier.seen[0][1].prompt == "price?"
    assert verifier.seen[0][1].tool_executions == ()


async def test_runs_after_tool_calls_with_their_results_in_context() -> None:
    model = ScriptedModel(
        [tool_call("get_rate", hotel="Aman"), text("Aman is 420 EUR")],
    )
    verifier = _Verifier()
    events = await _collect(_agent(model, verifier, tools=[get_rate]).run("rate?"))

    assert [v.passed for v in _verdicts(events)] == [True]
    (_, ctx) = verifier.seen[0]
    assert [e.tool_name for e in ctx.tool_executions] == ["get_rate"]
    assert ctx.messages[-1].content == "Aman is 420 EUR"


async def test_replans_are_bounded_and_the_last_draft_is_returned() -> None:
    model = ScriptedModel([text("WRONG 1"), text("WRONG 2"), text("WRONG 3")])
    events = await _collect(
        _agent(model, _Verifier(), final_answer_verifier_max_replans=1).run("q")
    )

    verdicts = _verdicts(events)
    assert [(v.passed, v.replanning) for v in verdicts] == [(False, True), (False, False)]
    assert _terminate(events).final_message == "WRONG 2"
    assert model.call_count == 2


async def test_zero_replans_judges_only() -> None:
    model = ScriptedModel([text("WRONG")])
    events = await _collect(
        _agent(model, _Verifier(), final_answer_verifier_max_replans=0).run("q")
    )
    (verdict,) = _verdicts(events)
    assert (verdict.passed, verdict.replanning) == (False, False)
    assert model.call_count == 1


async def test_a_raising_verifier_fails_open_and_reports_it() -> None:
    async def broken(draft: str, ctx: FinalAnswerContext) -> str | None:
        raise RuntimeError("judge offline")

    events = await _collect(_agent(ScriptedModel([text("fine")]), broken).run("q"))
    (verdict,) = _verdicts(events)
    assert verdict.passed is False
    assert verdict.replanning is False
    assert verdict.error == "RuntimeError: judge offline"
    assert _terminate(events).final_message == "fine"


async def test_an_empty_final_answer_is_verified_after_the_summary_call() -> None:
    """The empty-content safety net's summary is the answer, so it is verified."""
    replies = iter([text(""), text("WRONG summary"), text("good answer")])

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        return next(replies)

    model = FunctionModel(handler)
    verifier = _Verifier()
    agent = _agent(model, verifier)
    result = await agent.arun("q")

    assert result.message == "good answer"
    assert verifier.seen[0][0] == "WRONG summary"
    # The summary draft was not in state; the replan added it (turn-only).
    assert all("WRONG" not in (m.content or "") for m in result.state.messages)


# ---------------------------------------------------------------------------
# Rejected drafts and feedback are not persisted
# ---------------------------------------------------------------------------


async def test_rejected_draft_and_feedback_never_reach_checkpoint_or_result() -> None:
    checkpointer = MemoryCheckpointer()
    model = ScriptedModel([text("It is WRONG"), text("It is right")])
    agent = _agent(model, _Verifier(), checkpointer=checkpointer)

    result = await agent.arun("q", thread_id="t1")

    assert result.message == "It is right"
    for messages in (result.state.messages, (await checkpointer.load("t1")).messages):
        contents = [m.content or "" for m in messages]
        assert not any("WRONG" in c for c in contents)
        assert not any("Answer check" in c for c in contents)
        assert contents[-1] == "It is right"


def test_mark_ephemeral_keeps_existing_metadata() -> None:
    msg = Message(role="system", content="x", metadata={"a": 1})
    marked = mark_ephemeral(msg, "why")
    assert marked.metadata == {"a": 1, EPHEMERAL_MESSAGE_KEY: "why"}
    assert is_ephemeral_message(marked)
    assert not is_ephemeral_message(msg)
    assert not is_ephemeral_message(object())


# ---------------------------------------------------------------------------
# Streaming: hold the final answer until it passes
# ---------------------------------------------------------------------------


def _streamed_text(events: list[TulipEvent]) -> str:
    return "".join(e.content or "" for e in events if isinstance(e, ModelChunkEvent))


async def test_without_hold_the_rejected_draft_streams() -> None:
    model = ScriptedModel([text("WRONG draft"), text("good answer")])
    events = await _collect(_agent(model, _Verifier()).run("q", stream_tokens=True))
    assert _streamed_text(events) == "WRONG draftgood answer"


async def test_hold_drops_the_rejected_draft_and_releases_the_accepted_one() -> None:
    model = ScriptedModel([text("WRONG draft"), text("good answer")])
    agent = _agent(model, _Verifier(), hold_final_answer_tokens=True)
    events = await _collect(agent.run("q", stream_tokens=True))

    assert _streamed_text(events) == "good answer"
    # Released after the verdict, before termination.
    kinds = [type(e).__name__ for e in events]
    last_verdict = max(i for i, k in enumerate(kinds) if k == "FinalAnswerVerificationEvent")
    first_chunk = min(i for i, k in enumerate(kinds) if k == "ModelChunkEvent")
    assert first_chunk > last_verdict
    assert kinds[-1] == "TerminateEvent"


async def test_hold_releases_tool_steps_and_exhausted_drafts() -> None:
    model = ScriptedModel(
        [tool_call("get_rate", hotel="Aman"), text("WRONG 1"), text("WRONG 2")],
    )
    agent = _agent(
        model,
        _Verifier(),
        tools=[get_rate],
        hold_final_answer_tokens=True,
        final_answer_verifier_max_replans=1,
    )
    events = await _collect(agent.run("q", stream_tokens=True))

    chunks = [e for e in events if isinstance(e, ModelChunkEvent)]
    assert any(c.tool_calls for c in chunks), "the tool step's chunks were released"
    # The rejected first draft is dropped; the returned (exhausted) one shows.
    assert _streamed_text(events) == "WRONG 2"
    assert _terminate(events).final_message == "WRONG 2"


async def test_hold_lets_reasoning_stream_live() -> None:
    class _Thinking(ScriptedModel):
        async def stream(self, messages: Any, tools: Any = None, **kwargs: Any) -> Any:
            yield ModelChunkEvent(reasoning="thinking")
            async for chunk in super().stream(messages, tools, **kwargs):
                yield chunk

    agent = _agent(_Thinking([text("fine")]), _Verifier(), hold_final_answer_tokens=True)
    events = await _collect(agent.run("q", stream_tokens=True))
    kinds = [
        ("reasoning" if isinstance(e, ModelChunkEvent) and e.reasoning else type(e).__name__)
        for e in events
    ]
    assert kinds.index("reasoning") < kinds.index("FinalAnswerVerificationEvent")


# ---------------------------------------------------------------------------
# Resume path
# ---------------------------------------------------------------------------


booked: list[str] = []


@tool
def book(hotel: str) -> str:
    """Book a hotel."""
    booked.append(hotel)
    return f"booked {hotel}"


async def test_verifier_runs_on_the_answer_after_an_approval_resume() -> None:
    store = InMemoryApprovals()
    gated = gate_tool(
        book,
        policy=ControlPolicy(
            require_verification_score=0.0, require_human_for=frozenset({"production"})
        ),
        action=lambda n, kw: Action(name=n, asset=kw["hotel"], environment="production"),
        approval=store,
        on_refusal="interrupt",
        principal="u1",
    )
    model = ScriptedModel([tool_call("book", call_id="c1", hotel="Aman")])
    verifier = _Verifier()
    agent = _agent(model, verifier, tools=[gated], checkpointer=MemoryCheckpointer())
    paused = await _collect(agent.run("book it", thread_id="t"))
    assert isinstance(paused[-1], InterruptEvent)

    for record in store.pending():
        store.decide(record.approval_id, "approved", by="advisor")
    model._turns = [
        tool_call("book", call_id="c1", hotel="Aman"),
        text("Booked, WRONG dates"),
        text("Booked Aman."),
    ]
    events = await _collect(agent.resume("ok", thread_id="t", perform_dangling=True))

    assert [(v.passed, v.replanning) for v in _verdicts(events)] == [(False, True), (True, False)]
    assert _terminate(events).final_message == "Booked Aman."


# ---------------------------------------------------------------------------
# AfterModelCallEvent.retry_feedback
# ---------------------------------------------------------------------------


class _RetryOnce(HookProvider):
    def __init__(self) -> None:
        self.fired = False

    @property
    def priority(self) -> int:
        return 0

    async def on_after_model_call(self, event: AfterModelCallEvent) -> None:
        if not self.fired:
            self.fired = True
            event.retry = True
            event.retry_feedback = "Answer in French."


async def test_after_model_retry_feedback_reaches_the_re_call_only() -> None:
    model = ScriptedModel([text("Hello"), text("Bonjour")])
    agent = Agent(model=model, hooks=[_RetryOnce()], reflexion=False, grounding=False)
    result = await agent.arun("greet")

    assert result.message == "Bonjour"
    retry_call = model.received_messages[1]
    assert retry_call[-1].role == "user"
    assert (retry_call[-1].content or "").endswith("discarded: Answer in French.")
    assert all("Retry feedback" not in (m.content or "") for m in result.state.messages)


async def test_hold_drops_the_chunks_of_a_call_a_hook_discarded() -> None:
    model = ScriptedModel([text("Hello"), text("Bonjour")])
    agent = Agent(
        model=model,
        hooks=[_RetryOnce()],
        final_answer_verifier=_Verifier(),
        hold_final_answer_tokens=True,
        reflexion=False,
        grounding=False,
    )
    events = await _collect(agent.run("greet", stream_tokens=True))
    assert _streamed_text(events) == "Bonjour"


def test_retry_feedback_defaults_to_none() -> None:
    assert AfterModelCallEvent(response=None, messages=[]).retry_feedback is None


# ---------------------------------------------------------------------------
# Structured-output repair stays out of the result state, and is metered
# ---------------------------------------------------------------------------


class _Answer(BaseModel):
    city: str


async def test_schema_repair_messages_are_not_in_the_result_state() -> None:
    replies = iter(
        [
            text("Paris, I think"),
            text('{"town": "Paris"}'),  # still invalid
            text('{"city": "Paris"}'),
        ]
    )

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        response = next(replies)
        response.usage = {"prompt_tokens": 10, "completion_tokens": 5}
        return response

    agent = Agent(
        model=FunctionModel(handler),
        output_schema=_Answer,
        reflexion=False,
        grounding=False,
    )
    result = await agent.arun("capital of France?")

    assert result.parsed == _Answer(city="Paris")
    contents = [m.content or "" for m in result.state.messages]
    assert not any("Schema Repair" in c for c in contents)
    assert not any("town" in c for c in contents)
    # The two repair calls are metered on top of the run's one call.
    assert result.metrics.prompt_tokens == 30
    assert result.metrics.completion_tokens == 15
