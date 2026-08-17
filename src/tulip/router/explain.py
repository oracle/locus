# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Why the router chose what it chose.

Routing already computes everything a reader needs to audit a decision — the
three gates in :meth:`ProtocolRegistry.filter_candidates`, the four-part rank
in :func:`_rank_key`, the :class:`PolicyVerdict` — and then throws it away.
The only trace is a fire-and-forget event on the bus, which means a developer
holding a surprising decision has no way to ask *why*.

This module returns that evidence instead of publishing it. Nothing here runs
a model beyond the one extraction call, and nothing executes: it is the same
work the compiler does, reported rather than discarded.

**Structured evidence, not chain-of-thought.** Everything below is a fact
about the registry, the frame, and the policy — which gate rejected a
protocol, how the survivors ranked and on which term, what the policy said.
The model's private reasoning is deliberately not exposed; when an opt-in
LLM picker ran, only the short rationale it returned as data is carried.

    explanation = await router.explain("Investigate the checkout latency spike")

    print(explanation)                      # human-readable
    explanation.rejected                    # every protocol that was ruled out, and why
    explanation.candidates                  # survivors, best first, with their rank terms

The name is :class:`RoutingExplanation` rather than ``RoutingDecision``
because ``RoutingDecision`` is already taken by the unrelated
:class:`~tulip.multiagent.orchestrator.RoutingDecision`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from tulip.router.goal_frame import Complexity, GoalFrame
from tulip.router.protocol import Protocol, _rank_key


if TYPE_CHECKING:  # pragma: no cover — typing only
    from tulip.router.policy import PolicyVerdict


#: Which of the three gates in ``filter_candidates`` ruled a protocol out.
Gate = Literal["goal", "risk", "capabilities"]

#: How the survivor was picked. Mirrors the ``method`` already reported on
#: ``router.protocol.selected``.
Method = Literal["rule_based", "single_candidate", "llm_picked", "rule_based_fallback"]

_COST_RANK = {"low": 0, "medium": 1, "high": 2}
_COMPLEXITY_RANK = {Complexity.LOW: 0, Complexity.MEDIUM: 1, Complexity.HIGH: 2}


@dataclass(frozen=True)
class RejectedProtocol:
    """A protocol that never reached ranking, and the gate that stopped it."""

    protocol_id: str
    gate: Gate
    reason: str

    def __str__(self) -> str:
        return f"{self.protocol_id} — {self.reason}"


@dataclass(frozen=True)
class RankedProtocol:
    """A survivor, with the rank terms that ordered it.

    ``rank`` is the raw tuple from :func:`_rank_key`, lower being better, and
    the named fields spell out what each term meant so a reader does not have
    to decode a 4-tuple.
    """

    protocol_id: str
    rank: tuple[int, int, int, int]
    complexity_distance: int
    is_canonical: bool
    cost: str
    handles_count: int
    selected: bool = False

    def __str__(self) -> str:
        mark = "→" if self.selected else " "
        canon = "canonical" if self.is_canonical else "not canonical"
        return (
            f"{mark} {self.protocol_id:26s} cost={self.cost:6s} "
            f"distance={self.complexity_distance} {canon}"
        )


@dataclass(frozen=True)
class RoutingExplanation:
    """The full decision record for one routing call. No execution."""

    goal: str
    goal_frame: GoalFrame
    selected_protocol: str | None
    method: Method | None
    candidates: tuple[RankedProtocol, ...]
    rejected: tuple[RejectedProtocol, ...]
    available_capabilities: tuple[str, ...]
    policy: PolicyVerdict | None = None
    #: Only present when an opt-in :class:`LLMProtocolPicker` ran. The short
    #: rationale it returned as structured data — never hidden reasoning.
    picker_rationale: str | None = None

    @property
    def allowed(self) -> bool:
        """Whether the policy let the selection through."""
        return self.policy is None or bool(self.policy.allow)

    @property
    def requires_approval(self) -> bool:
        return bool(self.policy is not None and self.policy.require_approval)

    def __str__(self) -> str:
        frame = self.goal_frame
        lines = [
            f"goal: {self.goal}",
            "",
            "GoalFrame",
            f"  intent      {frame.primary_goal.value}",
            f"  complexity  {frame.complexity.value}",
            f"  risk        {frame.risk.value}",
            f"  domain      {frame.domain}",
        ]
        if frame.required_capabilities:
            lines.append(f"  capabilities {list(frame.required_capabilities)}")

        lines += ["", f"Selected protocol  {self.selected_protocol or '(none)'}"]
        if self.method:
            lines.append(f"  chosen by        {self.method}")
        if self.picker_rationale:
            lines.append(f"  rationale        {self.picker_rationale}")

        if self.candidates:
            lines += ["", f"Considered ({len(self.candidates)}, best first)"]
            lines += [f"  {c}" for c in self.candidates]

        if self.rejected:
            lines += ["", f"Rejected ({len(self.rejected)})"]
            lines += [f"  {r}" for r in self.rejected]

        if self.policy is not None:
            state = "allow" if self.policy.allow else "deny"
            if self.requires_approval:
                state += " (human approval required)"
            lines += ["", f"Policy  {state}", f"  {self.policy.reason}"]

        return "\n".join(lines)


def rejection_for(
    protocol: Protocol, frame: GoalFrame, available: set[str]
) -> RejectedProtocol | None:
    """Which gate, if any, rules ``protocol`` out for ``frame``.

    Gates are evaluated in the same order as ``filter_candidates`` and the
    first failure is reported — a protocol that fails two gates is described
    by the first one, which is the one a reader needs to fix.
    """
    if frame.primary_goal not in protocol.handles:
        return RejectedProtocol(
            protocol.id,
            "goal",
            f"does not handle {frame.primary_goal.value!r} "
            f"(handles {sorted(g.value for g in protocol.handles)})",
        )
    if not protocol.risk_max >= frame.risk:
        return RejectedProtocol(
            protocol.id,
            "risk",
            f"caps risk at {protocol.risk_max.value!r}, frame is {frame.risk.value!r}",
        )
    missing = set(protocol.requires_capabilities) - available
    if missing:
        return RejectedProtocol(
            protocol.id,
            "capabilities",
            f"needs {sorted(missing)}, not available",
        )
    return None


def rank_candidates(
    candidates: list[Protocol], frame: GoalFrame, selected: str | None
) -> tuple[RankedProtocol, ...]:
    """Order survivors exactly as the rule-based ranker does, best first."""
    ranked = [
        RankedProtocol(
            protocol_id=p.id,
            rank=_rank_key(p, frame),
            complexity_distance=abs(_COST_RANK[p.cost] - _COMPLEXITY_RANK[frame.complexity]),
            is_canonical=frame.primary_goal in p.primary_for,
            cost=p.cost,
            handles_count=len(p.handles),
            selected=(p.id == selected),
        )
        for p in candidates
    ]
    return tuple(sorted(ranked, key=lambda r: r.rank))
