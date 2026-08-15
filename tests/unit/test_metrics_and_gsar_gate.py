# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Two more settings that existed, were documented, and did nothing.

Both are the shape that keeps turning up in this codebase: the work is done,
and the result never reaches the caller.

``ExecutionMetrics.reflexion_evaluations`` and ``.grounding_evaluations`` were
always ``0``. The runtime counts both — ``_reflexion_evals`` and
``_grounding_evals`` — but they are locals in a generator, and metrics is built
from state. The only other mention of either field in the repo was a test that
constructed one by hand, which proves an int can hold 3.

``GSARConfig.fail_on_low_score`` was documented as raising
``GSARValidationError`` when a judgment does not clear the bar. Nothing read
the flag, and ``GSARValidationError`` existed only inside the sentence
promising it. So an agent explicitly configured to refuse un-grounded output
shipped it silently — the one outcome the setting exists to prevent, which is
what makes a no-op setting worse than a missing one.
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip.agent import Agent
from tulip.agent.config import GSARConfig
from tulip.core.errors import GSARValidationError
from tulip.core.messages import Message, ToolCall
from tulip.models.base import ModelResponse
from tulip.reasoning.gsar_judge import BaseGSARJudge, JudgeOutput
from tulip.tools.decorator import tool


@tool
def headcount(office: str) -> str:
    """Look up an office headcount."""
    return "Lisbon: 42 staff"


class _ToolThenAnswer:
    async def complete(
        self, messages: list[Message], tools: Any = None, **kwargs: Any
    ) -> ModelResponse:
        if any(getattr(m.role, "value", m.role) == "tool" for m in messages):
            return ModelResponse(message=Message.assistant("42 staff."), usage={})
        return ModelResponse(
            message=Message.assistant(
                content=None,
                tool_calls=[ToolCall(id="c1", name="headcount", arguments={"office": "Lisbon"})],
            ),
            usage={},
        )

    async def stream(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------
# The evaluation counters
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_evaluation_counters_report_what_ran() -> None:
    """Both were structurally always 0, whatever the run did."""
    agent = Agent(model=_ToolThenAnswer(), tools=[headcount], reflexion=True, grounding=True)

    result = await agent.arun("staff in Lisbon?")

    assert result.metrics.reflexion_evaluations > 0
    assert result.metrics.grounding_evaluations > 0


@pytest.mark.asyncio
async def test_the_counters_stay_zero_when_nothing_evaluated() -> None:
    """Zero has to keep meaning "none ran", or the fix just moves the lie."""
    agent = Agent(model=_ToolThenAnswer(), tools=[headcount])

    result = await agent.arun("staff in Lisbon?")

    assert result.metrics.reflexion_evaluations == 0
    assert result.metrics.grounding_evaluations == 0


@pytest.mark.asyncio
async def test_reflexion_alone_does_not_count_groundings() -> None:
    """Two counters that move together would be one counter with a typo."""
    agent = Agent(model=_ToolThenAnswer(), tools=[headcount], reflexion=True)

    result = await agent.arun("staff in Lisbon?")

    assert result.metrics.reflexion_evaluations > 0
    assert result.metrics.grounding_evaluations == 0


# --------------------------------------------------------------------------
# fail_on_low_score
# --------------------------------------------------------------------------


class _Abstaining(BaseGSARJudge):
    """A judge that will not vouch for the answer."""

    async def judge(
        self, *, report_synthesis: str, evidence_corpus: str, **kwargs: Any
    ) -> JudgeOutput:
        return JudgeOutput(
            grounding_score=0.1,
            is_grounded=False,
            explanation="nothing in the evidence supports this",
            decision_status="abstain",
            abstain_reason="no evidence",
        )


class _Vouching(BaseGSARJudge):
    """A judge whose claims are all tool-matched.

    The claims matter, not ``grounding_score``: the agent recomputes S from
    the claim partition under the configured weights, so a JudgeOutput with
    ``grounding_score=1.0`` and no claims still scores 0.5 and decides
    ``replan``. I got that wrong first and this test caught it.
    """

    async def judge(
        self, *, report_synthesis: str, evidence_corpus: str, **kwargs: Any
    ) -> JudgeOutput:
        from tulip.reasoning.gsar import Claim, EvidenceType

        return JudgeOutput(
            grounding_score=1.0,
            is_grounded=True,
            grounded_claims=[
                Claim(
                    text="Lisbon has 42 staff", type=EvidenceType.TOOL_MATCH, evidence_refs=["t1"]
                )
            ],
            explanation="fully supported",
            decision_status="resolved",
        )


class _Plain:
    async def complete(
        self, messages: list[Message], tools: Any = None, **kw: Any
    ) -> ModelResponse:
        return ModelResponse(message=Message.assistant("An ungrounded summary."), usage={})

    async def stream(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


@pytest.mark.asyncio
async def test_a_low_judgment_stops_the_run_when_asked() -> None:
    """The documented behaviour, which nothing implemented."""
    agent = Agent(model=_Plain(), gsar=GSARConfig(judge=_Abstaining(), fail_on_low_score=True))

    with pytest.raises(GSARValidationError) as caught:
        await agent.arun("summarise")

    assert caught.value.decision == "abstain"
    assert caught.value.score is not None


@pytest.mark.asyncio
async def test_the_default_still_returns_the_answer_and_the_judgment() -> None:
    """Off by default: a judgment is information most callers want to weigh."""
    agent = Agent(model=_Plain(), gsar=GSARConfig(judge=_Abstaining()))

    result = await agent.arun("summarise")

    assert result.message
    assert result.gsar_decision == "abstain"


@pytest.mark.asyncio
async def test_a_judgment_that_clears_the_bar_does_not_raise() -> None:
    """A gate that stops everything is as useless as one that stops nothing."""
    agent = Agent(model=_Plain(), gsar=GSARConfig(judge=_Vouching(), fail_on_low_score=True))

    result = await agent.arun("summarise")

    assert result.gsar_decision == "proceed"


@pytest.mark.asyncio
async def test_without_gsar_the_flag_is_irrelevant() -> None:
    """No judgment means nothing to fail on, not an implicit failure."""
    agent = Agent(model=_Plain())

    assert (await agent.arun("summarise")).message


def test_the_error_is_part_of_the_public_hierarchy() -> None:
    """A caller catching TulipError should catch this too, and the kind is what
    structured logging groups on."""
    from tulip.core.errors import TulipError

    assert issubclass(GSARValidationError, TulipError)
    assert GSARValidationError.kind == "gsar_validation_error"
