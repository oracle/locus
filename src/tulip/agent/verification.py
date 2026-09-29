# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Pluggable final-answer verification.

``AgentConfig.final_answer_verifier`` is an async callable the loop runs on
every final answer — whether or not a tool was called — before the run
terminates::

    async def verify(draft: str, ctx: FinalAnswerContext) -> str | None:
        if "€" in draft and not any(
            e.tool_name == "get_rates" for e in ctx.tool_executions
        ):
            return "You quoted a price without calling get_rates. Call it, or drop the price."
        return None  # accept


    agent = Agent(model=..., final_answer_verifier=verify)

Returning ``None`` (or an empty string) accepts the draft. Returning text
rejects it: the text is appended as an automated user-role note and the model
is called again, up to ``final_answer_verifier_max_replans`` times. When the
replans run out, the last draft is returned as the answer and the
:class:`~tulip.core.events.FinalAnswerVerificationEvent` says it did not pass.

The rejected draft and the feedback are *ephemeral*: the model sees them for
the rest of the turn, but they are dropped from every checkpoint and from the
run's result state, so the persisted conversation holds only the answer that
was returned.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from tulip.core.events import RunInfo
    from tulip.core.messages import Message
    from tulip.core.state import ToolExecution


__all__ = [
    "EPHEMERAL_MESSAGE_KEY",
    "FinalAnswerContext",
    "FinalAnswerVerifier",
    "is_ephemeral_message",
    "mark_ephemeral",
]


#: ``Message.metadata`` key marking a message the loop adds for one turn only
#: (a rejected draft, verifier feedback). The runtime strips such messages
#: before every checkpoint save and from the run's result state.
EPHEMERAL_MESSAGE_KEY = "tulip_ephemeral"


@dataclass(frozen=True)
class FinalAnswerContext:
    """What a final-answer verifier sees besides the draft.

    Attributes:
        run: The run's identity (``run_id``, ``thread_id``, invocation
            metadata, agent name) — the same object hooks see as ``event.run``.
        prompt: The user prompt of this turn.
        messages: The conversation as the model saw it when it wrote the
            draft (the draft is the last assistant message).
        tool_executions: Every tool call of the run so far, with results.
        attempt: 0 for the first draft; ``n`` after ``n`` rejected drafts.
        max_replans: ``AgentConfig.final_answer_verifier_max_replans``. When
            ``attempt == max_replans`` a rejection no longer triggers a
            replan — the draft is returned regardless.
    """

    run: RunInfo
    prompt: str
    messages: tuple[Message, ...]
    tool_executions: tuple[ToolExecution, ...]
    attempt: int
    max_replans: int


#: ``async (draft_text, ctx) -> feedback | None``.
FinalAnswerVerifier = Callable[[str, FinalAnswerContext], Awaitable[str | None]]


def is_ephemeral_message(message: Any) -> bool:
    """Whether ``message`` is marked turn-only (see :data:`EPHEMERAL_MESSAGE_KEY`)."""
    metadata = getattr(message, "metadata", None)
    return isinstance(metadata, dict) and bool(metadata.get(EPHEMERAL_MESSAGE_KEY))


def mark_ephemeral(message: Message, reason: str) -> Message:
    """A copy of ``message`` marked turn-only, with ``reason`` as the marker."""
    return message.model_copy(
        update={"metadata": {**message.metadata, EPHEMERAL_MESSAGE_KEY: reason}}
    )
