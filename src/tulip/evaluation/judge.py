# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""LLM-as-judge scoring, and trajectory assertions.

Structural checks — did it call this tool, does the answer contain this
string — verify shape, not quality. Substring matching in particular fails in
both directions: it passes on an answer that happens to contain the word and
fails on a correct answer phrased differently. For anything where the right
answer is not one exact string, the check has to read the answer.

:class:`LLMJudge` does that against a written rubric and returns a typed
:class:`Verdict` with the reason it decided:

    judge = LLMJudge(model=get_model("openai:gpt-4o"))
    verdict = await judge.score(
        prompt="Is ord-4821 refundable?",
        output="Yes — it was delivered damaged and is within the window.",
        rubric="Passes if it states a decision and cites a reason for it.",
    )
    assert verdict.passed, verdict.reason

Two things this deliberately does not do. It does not retry until it gets a
pass — a judge you re-roll is not a judge. And it does not silently score
zero when the model is unreachable: an unusable judge raises, because a
"failing" eval that actually means "the judge was down" is worse than no eval.

:func:`check_trajectory` covers the other half. ``expected_tools`` asks only
whether a tool appeared somewhere in the run, which cannot tell "looked the
order up, then refunded it" from "refunded it, then looked it up" — and for
an agent that acts, order is most of what correctness means.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from tulip.core.messages import Message


__all__ = [
    "LLMJudge",
    "Verdict",
    "check_trajectory",
]


_PROMPT = """\
You are grading one response against a rubric. Be strict and literal: grade \
only what the rubric asks, not whether you would have answered differently.

RUBRIC
{rubric}

THE REQUEST THAT WAS MADE
{prompt}

THE RESPONSE TO GRADE
{output}

Reply with only a JSON object, no prose and no code fence:
{{"passed": true or false, "score": 0.0 to 1.0, "reason": "one sentence"}}
"""

#: First balanced ``{...}`` in the reply. Models wrap JSON in fences or
#: preamble often enough that requiring a bare object would fail on answers
#: that are otherwise perfectly good.
_JSON = re.compile(r"\{.*\}", re.DOTALL)


class Verdict(BaseModel):
    """One judge decision."""

    passed: bool = Field(description="Whether the response satisfies the rubric")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Graded 0-1")
    reason: str = Field(default="", description="Why the judge decided that")

    #: Set when the reply could not be parsed. The verdict then fails closed
    #: and says so, rather than being counted as a genuine judgement.
    unparseable: bool = Field(
        default=False,
        description="The judge replied with something that was not a verdict",
    )


class LLMJudge:
    """Grade a response against a rubric using a model.

    Args:
        model: Any object satisfying ``ModelProtocol``. Use a *different*
            model from the one under test where you can — a model grading its
            own output is measuring self-consistency, not correctness.
        max_tokens: Budget for the verdict. The reply is one small JSON
            object, but the default is deliberately generous: a reasoning
            model spends the allowance thinking first and returns **empty
            content** if it runs out, which parses as a failed verdict
            rather than an error. Measured on a local Qwen3.6-35B —
            measured against a local Qwen3.6-35B, the reply was empty at
            256 and 512, and a clean verdict from 1024 up. 1024 sits on
            the boundary — reasoning length varies per call, so the same
            budget that worked once returned empty on the next run. The
            default is well clear of it.
    """

    def __init__(self, model: Any, *, max_tokens: int = 2048) -> None:
        self.model = model
        self.max_tokens = max_tokens

    async def score(self, *, prompt: str, output: str, rubric: str) -> Verdict:
        """Grade ``output`` against ``rubric``.

        Raises:
            RuntimeError: If the model call fails. An eval that reports
                failure when the judge was simply unreachable is worse than
                one that stops and says so.
        """
        request = _PROMPT.format(rubric=rubric.strip(), prompt=prompt, output=output)
        try:
            response = await self.model.complete(
                [Message.user(request)], max_tokens=self.max_tokens
            )
        except Exception as exc:
            raise RuntimeError(f"LLM judge could not be reached: {exc}") from exc

        return self._parse(response.content or "")

    @staticmethod
    def _parse(reply: str) -> Verdict:
        """Turn a reply into a Verdict, failing closed on anything unusable."""
        if not reply.strip():
            # Almost always a reasoning model that spent its whole budget
            # thinking. It arrives as an empty string rather than an error,
            # so without this it reads as a content failure and the rubric
            # gets blamed for a budget problem.
            return Verdict(
                passed=False,
                reason=(
                    "judge returned no content — usually a reasoning model that "
                    "exhausted max_tokens before answering; raise LLMJudge("
                    "max_tokens=...)"
                ),
                unparseable=True,
            )

        match = _JSON.search(reply)
        if not match:
            return Verdict(
                passed=False,
                reason=f"judge did not return a verdict: {reply[:120]!r}",
                unparseable=True,
            )
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return Verdict(
                passed=False,
                reason=f"judge returned invalid JSON: {match.group(0)[:120]!r}",
                unparseable=True,
            )

        passed = bool(data.get("passed"))
        raw_score = data.get("score")
        try:
            # A judge that says passed but omits a score gets 1.0, rather than
            # a 0.0 that would contradict its own verdict.
            score = float(raw_score) if raw_score is not None else float(passed)
        except (TypeError, ValueError):
            score = float(passed)

        return Verdict(
            passed=passed,
            score=min(1.0, max(0.0, score)),
            reason=str(data.get("reason", "")),
        )


def check_trajectory(
    actual: Sequence[str],
    expected: Sequence[str],
    *,
    exact: bool = False,
) -> tuple[bool, str]:
    """Check the order tools were called in, not merely that they were.

    ``expected_tools`` asks whether a tool appeared anywhere in the run, which
    cannot distinguish "looked the order up, then refunded it" from "refunded
    it, then looked it up". For an agent that acts, that ordering is most of
    what correctness means.

    Args:
        actual: Tool names in the order they were called.
        expected: The order required.
        exact: When ``True`` the sequences must match exactly. The default
            requires only that ``expected`` appears in order, so unrelated
            extra calls — a retry, a lookup the agent added — do not fail a
            test about ordering.

    Returns:
        ``(passed, reason)``, where the reason names what was missing or out
        of order so a failure is actionable without a debugger.
    """
    actual = list(actual)
    expected = list(expected)

    if exact:
        if actual == expected:
            return True, "trajectory matches exactly"
        return False, f"expected exactly {expected}, got {actual}"

    remaining = iter(actual)
    for step in expected:
        if not any(call == step for call in remaining):
            missing = expected[expected.index(step) :]
            return False, (f"expected {expected} in order; {missing} did not follow, got {actual}")
    return True, "trajectory contains the expected order"
