# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Every advertised shape can actually be selected by something.

Eight orchestration shapes are advertised. Seven were selectable and
``a2a_delegate`` was not — by anything. It declared ``primary_for=[]`` to stay
opt-in, but nothing supplied an opt-in, so across every
goal x complexity x risk frame the deterministic ranker never returned it. Both of
its goal types have a canonical peer at equal or lower cost, so it lost every
tie it could have won.

A shape nobody can reach is worse than a missing shape: it appears in the
catalogue, in the documentation and in the count, and no test fails.

The fix uses the machinery already there. A configured remote peer is expressed
as a synthetic capability, and the ordinary ``requires_capabilities`` filter does
the gating — so "a peer exists" *is* the opt-in. That is also the honest
condition: ``_build_a2a_delegate`` raises without an endpoint, so a shape
selectable while unconfigured would trade "never selectable" for "selectable and
then fails", which is worse.
"""

from __future__ import annotations

import itertools

import pytest

from tulip.router.goal_frame import Complexity, GoalFrame, Risk, TaskType
from tulip.router.protocol import (
    A2A_PEER_CAPABILITY,
    NoMatchingProtocolError,
    ProtocolRegistry,
    builtin_protocols,
)


def _registry() -> ProtocolRegistry:
    registry = ProtocolRegistry()
    registry.register_many(builtin_protocols())
    return registry


def _sweep(capabilities: set[str]) -> dict[str, int]:
    """Which shape wins, across every frame the goal model can express.

    The whole cross-product rather than a sample: reachability is a property of the
    ranker over the entire input space, and a sampled sweep would have reported
    `a2a_delegate` unreachable for a reason nobody could act on.
    """
    registry = _registry()
    picked: dict[str, int] = {}
    for goal, complexity, risk in itertools.product(TaskType, Complexity, Risk):
        frame = GoalFrame(primary_goal=goal, domain="ops", complexity=complexity, risk=risk)
        try:
            chosen = registry.select(frame, available_capabilities=capabilities)
        except NoMatchingProtocolError:
            continue
        picked[chosen.id] = picked.get(chosen.id, 0) + 1
    return picked


def test_every_shape_is_reachable_in_some_configuration() -> None:
    reachable = set(_sweep(set())) | set(_sweep({A2A_PEER_CAPABILITY}))
    advertised = {p.id for p in builtin_protocols()}
    assert reachable == advertised, f"unreachable: {sorted(advertised - reachable)}"


def test_a2a_delegate_is_unreachable_without_a_peer() -> None:
    """Not an oversight — the builder raises without an endpoint.

    Filtering it out is the difference between "this shape needs configuration"
    and "this shape fails at run time".
    """
    assert "a2a_delegate" not in _sweep(set())


def test_a_configured_peer_makes_it_selectable() -> None:
    assert _sweep({A2A_PEER_CAPABILITY}).get("a2a_delegate", 0) > 0


def test_it_wins_coordinate_and_only_coordinate() -> None:
    """Delegating a coordination task is the reason a peer gets configured.

    Narrow on purpose: it should not quietly capture every goal it merely handles,
    or configuring a peer would reroute unrelated work off the machine.
    """
    registry = _registry()
    caps = {A2A_PEER_CAPABILITY}
    coordinate = registry.select(
        GoalFrame(
            primary_goal=TaskType.COORDINATE,
            domain="ops",
            complexity=Complexity.MEDIUM,
            risk=Risk.LOW,
        ),
        available_capabilities=caps,
    )
    assert coordinate.id == "a2a_delegate"

    for goal in (TaskType.ANSWER, TaskType.GENERATE_CODE, TaskType.REMEDIATE):
        other = registry.select(
            GoalFrame(primary_goal=goal, domain="ops", complexity=Complexity.MEDIUM, risk=Risk.LOW),
            available_capabilities=caps,
        )
        assert other.id != "a2a_delegate", f"{goal} was captured by a2a_delegate"


def test_configuring_a_peer_does_not_disturb_the_other_shapes() -> None:
    """Only COORDINATE moves. Anything else changing would be a regression.

    Asserted as an exact set difference rather than a spot check, because "the
    other shapes still work" is the kind of claim that stays true until it doesn't.
    """
    without, with_peer = _sweep(set()), _sweep({A2A_PEER_CAPABILITY})
    moved = {
        shape
        for shape in set(without) | set(with_peer)
        if without.get(shape, 0) != with_peer.get(shape, 0)
    }
    # a2a_delegate gains; handoff_chain is the canonical peer it takes COORDINATE from.
    assert moved == {"a2a_delegate", "handoff_chain"}, moved


@pytest.mark.parametrize("risk", list(Risk))
def test_it_still_respects_the_risk_ceiling(risk: Risk) -> None:
    """A capability makes a shape *eligible*, never exempt.

    The protocol declares `risk_max=MEDIUM`, so a HIGH-risk coordination goal must
    not reach a remote peer merely because one is configured — otherwise the opt-in
    would have widened the risk envelope as a side effect.
    """
    registry = _registry()
    frame = GoalFrame(
        primary_goal=TaskType.COORDINATE, domain="ops", complexity=Complexity.MEDIUM, risk=risk
    )
    try:
        chosen = registry.select(frame, available_capabilities={A2A_PEER_CAPABILITY})
    except NoMatchingProtocolError:
        return
    if chosen.id == "a2a_delegate":
        assert chosen.risk_max >= risk
