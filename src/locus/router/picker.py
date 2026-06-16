# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Opt-in LLM protocol picker — the second selection mode.

The default routing path is :meth:`ProtocolRegistry.select`, a
deterministic tuple-rank in :func:`_rank_key`. This module adds a
second mode where the model picks the protocol from an
already-filtered candidate set, returning a typed
:class:`PickedProtocol` (id + rationale).

The picker is strictly opt-in. Pass an instance to
``CognitiveCompiler(protocol_picker=...)``; default callers see no
change. PolicyGate still runs after selection, and the candidate set
is filtered by the same three gates the rule-based path uses — the
LLM only resolves the *last-mile* ambiguity when multiple protocols
qualify.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from locus.router.goal_frame import GoalFrame
from locus.router.protocol import Protocol


class PickerError(RuntimeError):
    """Raised by :class:`LLMProtocolPicker` when the model output is
    unusable (parse failure or returned id not in the candidate set).

    The compiler catches this and falls back to the rule-based ranker
    after emitting a ``router.protocol.picker_fallback`` event. Users
    do not normally see this exception.
    """


class PickedProtocol(BaseModel):
    """Typed output schema for :class:`LLMProtocolPicker`.

    Used as ``output_schema=`` on the picker agent so the model is
    forced to return a structured pick rather than free-form text.
    """

    protocol_id: str = Field(
        description="One of the candidate protocol ids, verbatim.",
    )
    rationale: str = Field(
        default="",
        description="One-sentence justification for the pick.",
    )

    model_config = {"frozen": True}


_DEFAULT_SYSTEM_PROMPT = (
    "You are a protocol picker for an agentic router. Given a typed "
    "GoalFrame describing the user's request and a list of candidate "
    "protocols that already passed risk + capability filtering, pick the "
    "single best protocol for the request and explain why in one short "
    "sentence.\n\n"
    "Rules:\n"
    "1. Return the chosen protocol's id exactly as listed — no paraphrase.\n"
    "2. Prefer protocols whose `primary_for` matches the frame's primary_goal "
    "when the choice is otherwise even.\n"
    "3. Prefer lower-cost protocols when they fit; reach for high-cost "
    "shapes (debate, specialist_fanout) only when the task genuinely needs "
    "their structure.\n"
    "4. If you are uncertain, return the canonical protocol for the goal.\n"
)


def _format_candidates(candidates: list[Protocol]) -> str:
    """Render the candidate list for the picker prompt.

    One numbered entry per candidate with id + description + cost +
    latency + primary_for so the model can rank intelligently. Kept
    short — every byte is a token at picker time.
    """
    lines: list[str] = []
    for i, p in enumerate(candidates, start=1):
        primary = ", ".join(t.value for t in p.primary_for) or "(none)"
        lines.append(
            f"{i}. id={p.id}  cost={p.cost}  latency={p.latency}  "
            f"primary_for=[{primary}]\n   {p.description}"
        )
    return "\n".join(lines)


def _format_frame(frame: GoalFrame) -> str:
    """Render the goal frame for the picker prompt."""
    caps = ", ".join(sorted(frame.required_capabilities)) or "(none)"
    return (
        f"primary_goal: {frame.primary_goal.value}\n"
        f"risk: {frame.risk.value}\n"
        f"complexity: {frame.complexity.value}\n"
        f"domain: {frame.domain or '(unspecified)'}\n"
        f"required_capabilities: {caps}"
    )


class LLMProtocolPicker:
    """Picks a protocol from a pre-filtered candidate set using an LLM.

    Construct with a locus model (any provider) and an optional
    ``system_prompt`` override. The picker reuses the same Agent +
    ``output_schema`` machinery the router's extractor uses — one
    structured-output call per pick.

    Parameters
    ----------
    model:
        Any locus model instance (an :class:`~locus.Agent`-compatible
        model object or model string).
    system_prompt:
        Optional override for the picker's system prompt. The default
        instructs the model on the rule hierarchy (canonical match,
        cost-fit, conservative fallback).

    Example::

        picker = LLMProtocolPicker(model=my_model)
        compiler = CognitiveCompiler(
            protocols=registry,
            capabilities=capabilities,
            policy=PolicyGate(),
            model=my_model,
            protocol_picker=picker,
        )
    """

    def __init__(
        self,
        *,
        model: Any,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    async def pick(
        self,
        frame: GoalFrame,
        candidates: list[Protocol],
    ) -> tuple[Protocol, str]:
        """Pick one protocol from ``candidates`` for ``frame``.

        Returns a ``(protocol, rationale)`` tuple. The compiler threads
        the rationale into the ``router.protocol.selected`` event so
        the LLM's reasoning is observable.

        Raises :class:`PickerError` when the model output is unparseable
        or names an id not in ``candidates`` — the compiler catches this
        and falls back to the rule-based ranker.

        The caller is responsible for the candidate set being non-empty
        and already filtered; this method does not re-validate handles
        / risk / capabilities.
        """
        # Import here so the picker module doesn't pull Agent into the
        # import graph of anyone who only uses ProtocolRegistry.
        from locus.agent.agent import Agent

        prompt = (
            "Goal frame:\n"
            f"{_format_frame(frame)}\n\n"
            "Candidate protocols (one of these ids must be picked):\n"
            f"{_format_candidates(candidates)}\n\n"
            "Pick exactly one protocol id and explain in one sentence."
        )

        picker_agent = Agent(
            model=self.model,
            system_prompt=self.system_prompt,
            output_schema=PickedProtocol,
        )
        result: Any = await asyncio.to_thread(picker_agent.invoke, prompt)
        parsed = result.parsed
        if not isinstance(parsed, PickedProtocol):
            raise PickerError(
                f"Picker did not return PickedProtocol. "
                f"parse_error={result.parse_error!r}, "
                f"message={result.message!r}"
            )

        by_id = {p.id: p for p in candidates}
        picked = by_id.get(parsed.protocol_id)
        if picked is None:
            raise PickerError(
                f"Picker returned unknown protocol_id={parsed.protocol_id!r}. "
                f"Candidates were {sorted(by_id.keys())}."
            )

        return picked, parsed.rationale


__all__ = ["LLMProtocolPicker", "PickedProtocol", "PickerError"]
