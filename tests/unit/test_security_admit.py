# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Admission control — the runtime's enforcement point.

``admit`` is what turns the composable trust chain into an *enforced* one: a
side-effecting action runs only if it clears ``approve`` (ALLOW), and the
decision is recorded to the audit trail either way.
"""

from __future__ import annotations

import pytest

from tulip.security import (
    Action,
    AdmissionError,
    AuditTrail,
    SecurityContext,
    SecurityPolicy,
    admit,
)
from tulip.security.verify import Verdict


# A verdict strong enough to clear the default policy bar (0.8).
_STRONG = Verdict(survives=True, confidence=0.95, evidence_quality=0.95)


async def test_admitted_action_runs_and_returns_result() -> None:
    ran: list[str] = []

    async def perform() -> str:
        ran.append("did it")
        return "ok"

    result = await admit(
        Action(name="rotate_key", asset="svc-1", environment="staging"),
        perform,
        policy=SecurityPolicy(),
        verdict=_STRONG,
    )
    assert result == "ok"
    assert ran == ["did it"]


async def test_production_requires_human_and_blocks_the_side_effect() -> None:
    ran: list[str] = []

    async def perform() -> str:
        ran.append("should not happen")
        return "ok"

    with pytest.raises(AdmissionError) as exc:
        await admit(
            Action(name="disable_user", asset="mallory@corp", environment="production"),
            perform,
            policy=SecurityPolicy(),
            verdict=_STRONG,
        )
    assert exc.value.decision.outcome == "require_human"
    assert ran == []  # the gate ran BEFORE the side effect


async def test_denied_label_blocks_the_side_effect() -> None:
    async def perform() -> str:
        raise AssertionError("must not run")

    policy = SecurityPolicy(deny_for=frozenset({"prod"}))
    with pytest.raises(AdmissionError) as exc:
        await admit(
            Action(name="wipe_disk", asset="db-1", environment="prod"),
            perform,
            policy=policy,
            verdict=_STRONG,
        )
    assert exc.value.decision.outcome == "deny"


async def test_every_admission_is_recorded_admitted_or_not() -> None:
    trail = AuditTrail()

    async def perform() -> str:
        return "done"

    await admit(
        Action(name="rotate_key", environment="staging"),
        perform,
        policy=SecurityPolicy(),
        verdict=_STRONG,
        trail=trail,
    )
    with pytest.raises(AdmissionError):
        await admit(
            Action(name="disable_user", environment="production"),
            perform,
            policy=SecurityPolicy(),
            verdict=_STRONG,
            trail=trail,
        )

    records = trail.records()
    assert len(records) == 2  # both the allowed and the rejected attempt
    assert all(r.event_type == "action-admission" for r in records)
    assert trail.verify()  # the chain is intact and tamper-evident


async def test_security_context_actions_execute_enforces_the_gate() -> None:
    ctx = SecurityContext()  # default actions provider, default policy

    async def perform() -> str:
        return "ran"

    # Non-production, verified -> admitted.
    assert (
        await ctx.actions.execute(
            Action(name="enrich", environment="staging"), perform, verdict=_STRONG
        )
        == "ran"
    )
    # Production -> held for a human, side effect blocked.
    with pytest.raises(AdmissionError):
        await ctx.actions.execute(
            Action(name="disable_user", environment="production"), perform, verdict=_STRONG
        )
