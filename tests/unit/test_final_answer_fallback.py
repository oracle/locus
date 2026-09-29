# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``AgentConfig.final_answer_fallback``: what the run says when the last
draft still fails the final-answer verifier.

Without it the loop returns the failed draft once replans run out — and with
``hold_final_answer_tokens`` releases it to the stream, so a user sees exactly
the answer the product's own check refused, and the checkpoint keeps it as the
conversation's history.
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip.agent import Agent
from tulip.agent.verification import FinalAnswerContext
from tulip.core.events import (
    FinalAnswerVerificationEvent,
    ModelChunkEvent,
    TerminateEvent,
    TulipEvent,
)
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.testing import ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


SAFE = "I could not confirm that. Here is what I can say for sure."


@tool
def get_rate(hotel: str) -> str:
    """Look up tonight's rate."""
    return f"{hotel}: 420 EUR"


async def _reject_wrong(draft: str, ctx: FinalAnswerContext) -> str | None:
    return "It says WRONG." if "WRONG" in draft else None


class _Fallback:
    def __init__(self, reply: str | None = SAFE, boom: bool = False) -> None:
        self.reply = reply
        self.boom = boom
        self.calls: list[tuple[str, str, int]] = []

    async def __call__(self, draft: str, ctx: FinalAnswerContext, feedback: str) -> str | None:
        self.calls.append((draft, feedback, ctx.attempt))
        if self.boom:
            raise RuntimeError("fallback broke")
        return self.reply


def _agent(model: Any, fallback: Any, **kwargs: Any) -> Agent:
    return Agent(
        model=model,
        tools=kwargs.pop("tools", []),
        final_answer_verifier=_reject_wrong,
        final_answer_fallback=fallback,
        reflexion=False,
        grounding=False,
        **kwargs,
    )


async def _collect(stream: Any) -> list[TulipEvent]:
    return [e async for e in stream]


def _streamed(events: list[TulipEvent]) -> str:
    return "".join(e.content or "" for e in events if isinstance(e, ModelChunkEvent))


def _terminate(events: list[TulipEvent]) -> TerminateEvent:
    return next(e for e in events if isinstance(e, TerminateEvent))


def _verdicts(events: list[TulipEvent]) -> list[FinalAnswerVerificationEvent]:
    return [e for e in events if isinstance(e, FinalAnswerVerificationEvent)]


async def test_an_exhausted_draft_is_replaced_everywhere() -> None:
    checkpointer = MemoryCheckpointer()
    fallback = _Fallback()
    model = ScriptedModel([tool_call("get_rate", hotel="Aman"), text("WRONG 1"), text("WRONG 2")])
    agent = _agent(
        model,
        fallback,
        tools=[get_rate],
        checkpointer=checkpointer,
        hold_final_answer_tokens=True,
        final_answer_verifier_max_replans=1,
    )
    events = await _collect(agent.run("q", thread_id="t1", stream_tokens=True))

    assert "WRONG" not in _streamed(events), "the failed draft never streams"
    assert _streamed(events).endswith(SAFE)
    assert _terminate(events).final_message == SAFE
    verdicts = _verdicts(events)
    assert [v.replanning for v in verdicts] == [True, False]
    assert verdicts[-1].replaced is True
    assert verdicts[-1].passed is False
    assert fallback.calls == [("WRONG 2", "It says WRONG.", 1)]
    saved = await checkpointer.load("t1")
    contents = [m.content or "" for m in saved.messages]
    assert not any("WRONG" in c for c in contents)
    assert contents[-1] == SAFE


async def test_zero_replans_goes_straight_to_the_fallback() -> None:
    fallback = _Fallback()
    agent = _agent(ScriptedModel([text("WRONG")]), fallback, final_answer_verifier_max_replans=0)
    result = await agent.arun("q")
    assert result.message == SAFE
    assert len(fallback.calls) == 1


async def test_a_passing_draft_never_calls_the_fallback() -> None:
    fallback = _Fallback()
    agent = _agent(ScriptedModel([text("WRONG"), text("fine")]), fallback)
    events = await _collect(agent.run("q"))
    assert _terminate(events).final_message == "fine"
    assert fallback.calls == []
    assert not any(v.replaced for v in _verdicts(events))


async def test_a_fallback_returning_none_keeps_the_draft() -> None:
    agent = _agent(
        ScriptedModel([text("WRONG")]), _Fallback(reply=None), final_answer_verifier_max_replans=0
    )
    events = await _collect(agent.run("q"))
    assert _terminate(events).final_message == "WRONG"
    assert _verdicts(events)[-1].replaced is False


async def test_a_raising_fallback_keeps_the_draft() -> None:
    agent = _agent(
        ScriptedModel([text("WRONG")]), _Fallback(boom=True), final_answer_verifier_max_replans=0
    )
    result = await agent.arun("q")
    assert result.message == "WRONG"


async def test_a_raising_verifier_is_not_replaced() -> None:
    async def broken(draft: str, ctx: FinalAnswerContext) -> str | None:
        raise RuntimeError("verifier broke")

    fallback = _Fallback()
    agent = Agent(
        model=ScriptedModel([text("anything")]),
        final_answer_verifier=broken,
        final_answer_fallback=fallback,
        reflexion=False,
        grounding=False,
    )
    result = await agent.arun("q")
    assert result.message == "anything", "a verifier error fails open, never to the fallback"
    assert fallback.calls == []


async def test_without_hold_the_replacement_is_not_streamed_twice() -> None:
    agent = _agent(ScriptedModel([text("WRONG")]), _Fallback(), final_answer_verifier_max_replans=0)
    events = await _collect(agent.run("q", stream_tokens=True))
    # Without hold the draft already streamed; the answer of record is the fallback.
    assert _streamed(events) == "WRONG"
    assert _terminate(events).final_message == SAFE


def test_a_non_callable_fallback_is_refused() -> None:
    with pytest.raises(TypeError, match="final_answer_fallback"):
        Agent(model=ScriptedModel([text("x")]), final_answer_fallback="nope")
