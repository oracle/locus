# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The grounding verdict, which the agent computed and then dropped.

`AgentResult.grounding_score` and `.ungrounded_claims` are documented fields
that were always `None` and `[]`. The grounding loop itself ran — it can
trigger replans — and emitted a `GroundingEvent` saying what it found. Nothing
carried that into the result, so the one place a caller looks never learned it.

A second thing turned up while fixing it, and is pinned here because the issue
report got it wrong in a way a reader would repeat: grounding only runs when
the agent actually used a tool. Evidence comes from tool results, so an agent
with no tools has nothing to ground *against*, and the guard skips it. The
issue's own reproduction used a tool-less agent, which would have returned
`None` even with the plumbing fixed.
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip.agent import Agent
from tulip.core.messages import Message, ToolCall
from tulip.models.base import ModelResponse
from tulip.tools.decorator import tool


@tool
def headcount(office: str) -> str:
    """Look up an office headcount."""
    return "Lisbon: 42 staff"


class _Contradicting:
    """Calls the tool, then answers with a number the tool did not support."""

    async def complete(
        self, messages: list[Message], tools: Any = None, **kwargs: Any
    ) -> ModelResponse:
        if _has_tool_result(messages):
            return ModelResponse(
                message=Message.assistant("The Lisbon office has 900 staff."), usage={}
            )
        return ModelResponse(
            message=Message.assistant(
                content=None,
                tool_calls=[ToolCall(id="c1", name="headcount", arguments={"office": "Lisbon"})],
            ),
            usage={},
        )

    async def stream(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def _has_tool_result(messages: list[Message]) -> bool:
    return any(getattr(m.role, "value", m.role) == "tool" for m in messages)


# --------------------------------------------------------------------------
# The regression
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_grounding_score_reaches_the_result() -> None:
    """Always ``None`` before this, however ungrounded the answer was."""
    agent = Agent(model=_Contradicting(), tools=[headcount], grounding=True)

    result = await agent.arun("How many staff in Lisbon?")

    assert result.grounding_score is not None


@pytest.mark.asyncio
async def test_the_ungrounded_claims_reach_the_result() -> None:
    """A score with no claims tells you something is wrong and not what."""
    agent = Agent(model=_Contradicting(), tools=[headcount], grounding=True)

    result = await agent.arun("How many staff in Lisbon?")

    assert result.ungrounded_claims
    assert any("900" in claim for claim in result.ungrounded_claims)


@pytest.mark.asyncio
async def test_a_contradicted_answer_does_not_score_as_grounded() -> None:
    """The tool said 42 and the answer says 900; a passing score here would
    make the whole field decorative."""
    agent = Agent(model=_Contradicting(), tools=[headcount], grounding=True)

    result = await agent.arun("How many staff in Lisbon?")

    assert result.grounding_score < 1.0


# --------------------------------------------------------------------------
# When it deliberately does not run
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grounding_is_absent_when_it_is_not_enabled() -> None:
    """Off means off — the field stays ``None`` rather than defaulting to 1.0,
    which would read as "checked and fine"."""
    agent = Agent(model=_Contradicting(), tools=[headcount])

    result = await agent.arun("How many staff in Lisbon?")

    assert result.grounding_score is None
    assert result.ungrounded_claims == []


@pytest.mark.asyncio
async def test_an_agent_with_no_tools_has_nothing_to_ground_against() -> None:
    """Documented here because the issue report got it wrong.

    Evidence comes from tool results. An agent that called no tool has none,
    so grounding is skipped and the score stays ``None`` — not because the
    plumbing is broken, but because there was nothing to check the answer
    against. A reader copying the issue's reproduction would conclude the fix
    had not worked.
    """

    class _NoTools:
        async def complete(self, messages: list[Message], tools: Any = None, **kw: Any) -> Any:
            return ModelResponse(
                message=Message.assistant("The Lisbon office has 900 staff."), usage={}
            )

        async def stream(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
            raise NotImplementedError

    agent = Agent(model=_NoTools(), grounding=True)

    result = await agent.arun("How many staff in Lisbon?")

    assert result.grounding_score is None
