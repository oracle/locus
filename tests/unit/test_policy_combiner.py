# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The control-model combiner: proof that a model cannot widen authority.

This is the only change to the reference monitor, and the product's central
claim rests on it: *the model cannot argue past the gate*. That claim is
falsifiable, so it is tested exhaustively over the full 3x3 outcome space rather
than sampled -- nine cases is cheap, and "we checked the interesting ones" is
not a security argument.

The three properties:

  Safety            v_effective >= v_policy, always.
  Degradation       no advisor, a broken advisor, or a silent one gives exactly
                    the decision policy alone would have made.
  Idempotent absence
                    removing the model entirely changes no outcome.
"""

from __future__ import annotations

import pytest

from tulip.security.policy import (
    Action,
    ApprovalOutcome,
    ControlPolicy,
    approve,
)


_ORDER = {ApprovalOutcome.ALLOW: 0, ApprovalOutcome.REQUIRE_HUMAN: 1, ApprovalOutcome.DENY: 2}
_ALL = (ApprovalOutcome.ALLOW, ApprovalOutcome.REQUIRE_HUMAN, ApprovalOutcome.DENY)


class _Advisor:
    """A control model that always returns what it was constructed with."""

    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls = 0

    def advise(self, action: Action) -> object:
        self.calls += 1
        return self.outcome


class _Exploding:
    """A control model that fails. Availability is not its job either."""

    def __init__(self) -> None:
        self.calls = 0

    def advise(self, action: Action) -> str | None:
        self.calls += 1
        raise RuntimeError("engine died mid-request")


def _policy_for(outcome: str) -> tuple[Action, ControlPolicy]:
    """An action/policy pair the deterministic engine decides as ``outcome``."""
    permissive = ControlPolicy(require_verification_score=0.0, max_blast_radius=10)
    if outcome == ApprovalOutcome.ALLOW:
        return Action(name="read", environment="dev"), permissive
    if outcome == ApprovalOutcome.REQUIRE_HUMAN:
        return Action(name="deploy", environment="production"), permissive
    return (
        Action(name="wipe", environment="dev", kind="destructive"),
        ControlPolicy(
            require_verification_score=0.0,
            max_blast_radius=10,
            deny_for=frozenset({"destructive"}),
        ),
    )


class TestSafety:
    """v_effective >= v_policy. The whole security argument."""

    @pytest.mark.parametrize("policy_outcome", _ALL)
    @pytest.mark.parametrize("model_outcome", _ALL)
    def test_the_model_never_weakens_a_decision(
        self, policy_outcome: str, model_outcome: str
    ) -> None:
        action, policy = _policy_for(policy_outcome)
        decision = approve(action, policy=policy, advisor=_Advisor(model_outcome))
        assert decision.policy_outcome == policy_outcome, "test fixture drifted"
        assert _ORDER[decision.outcome] >= _ORDER[policy_outcome]

    @pytest.mark.parametrize("policy_outcome", _ALL)
    def test_a_model_screaming_allow_cannot_open_a_closed_gate(
        self, policy_outcome: str
    ) -> None:
        """The adversarial case: a compromised model advising allow, always."""
        action, policy = _policy_for(policy_outcome)
        decision = approve(action, policy=policy, advisor=_Advisor(ApprovalOutcome.ALLOW))
        assert decision.outcome == policy_outcome

    def test_the_model_can_escalate(self) -> None:
        """Escalation must actually work, or the subsystem is decorative."""
        action, policy = _policy_for(ApprovalOutcome.ALLOW)
        decision = approve(action, policy=policy, advisor=_Advisor(ApprovalOutcome.DENY))
        assert decision.outcome == ApprovalOutcome.DENY
        assert decision.escalated_by_model


class TestDegradation:
    """Absent, broken or silent -- all three behave as if no advisor existed."""

    @pytest.mark.parametrize("policy_outcome", _ALL)
    def test_no_advisor_matches_policy_alone(self, policy_outcome: str) -> None:
        action, policy = _policy_for(policy_outcome)
        assert approve(action, policy=policy).outcome == policy_outcome

    @pytest.mark.parametrize("policy_outcome", _ALL)
    def test_a_crashing_advisor_matches_policy_alone(self, policy_outcome: str) -> None:
        """Not in the TCB for authorization, so not in it for availability."""
        action, policy = _policy_for(policy_outcome)
        without = approve(action, policy=policy)
        with_broken = approve(action, policy=policy, advisor=_Exploding())
        assert with_broken.outcome == without.outcome

    def test_a_crash_is_recorded_rather_than_hidden(self) -> None:
        action, policy = _policy_for(ApprovalOutcome.ALLOW)
        decision = approve(action, policy=policy, advisor=_Exploding())
        assert any("unavailable" in c for c in decision.checks)
        assert decision.model_outcome is None

    @pytest.mark.parametrize("policy_outcome", _ALL)
    def test_an_abstaining_advisor_matches_policy_alone(self, policy_outcome: str) -> None:
        action, policy = _policy_for(policy_outcome)
        decision = approve(action, policy=policy, advisor=_Advisor(None))
        assert decision.outcome == policy_outcome
        assert decision.model_outcome is None


class TestMalformedAdvice:
    """Anything outside the three known strings is no opinion, not a verdict."""

    @pytest.mark.parametrize(
        "advice", ["ALLOW", "allow ", "Deny", "probably fine", "", 1, True, ["deny"], {}]
    )
    def test_unrecognised_advice_is_discarded(self, advice: object) -> None:
        action, policy = _policy_for(ApprovalOutcome.ALLOW)
        decision = approve(action, policy=policy, advisor=_Advisor(advice))
        assert decision.outcome == ApprovalOutcome.ALLOW
        assert decision.model_outcome is None

    def test_a_near_miss_is_not_coerced_into_an_escalation_either(self) -> None:
        """Discarding must be symmetric -- 'DENY' is not a deny."""
        action, policy = _policy_for(ApprovalOutcome.ALLOW)
        decision = approve(action, policy=policy, advisor=_Advisor("DENY"))
        assert decision.outcome == ApprovalOutcome.ALLOW


class TestConsultedOnlyOnAllow:
    """If policy already stops the action, the model cannot change the answer."""

    @pytest.mark.parametrize(
        "policy_outcome", [ApprovalOutcome.REQUIRE_HUMAN, ApprovalOutcome.DENY]
    )
    def test_not_consulted_when_policy_already_stops_it(self, policy_outcome: str) -> None:
        action, policy = _policy_for(policy_outcome)
        advisor = _Advisor(ApprovalOutcome.DENY)
        approve(action, policy=policy, advisor=advisor)
        assert advisor.calls == 0, "spent latency on a decision the model cannot affect"

    def test_consulted_when_policy_allows(self) -> None:
        action, policy = _policy_for(ApprovalOutcome.ALLOW)
        advisor = _Advisor(ApprovalOutcome.ALLOW)
        approve(action, policy=policy, advisor=advisor)
        assert advisor.calls == 1


class TestAuditability:
    """An auditor must see what policy decided and what the model changed."""

    def test_agreement_is_recorded_without_changing_the_outcome(self) -> None:
        action, policy = _policy_for(ApprovalOutcome.ALLOW)
        decision = approve(action, policy=policy, advisor=_Advisor(ApprovalOutcome.ALLOW))
        assert decision.outcome == ApprovalOutcome.ALLOW
        assert decision.model_outcome == ApprovalOutcome.ALLOW
        assert not decision.escalated_by_model
        assert any("no escalation" in c for c in decision.checks)

    def test_an_escalation_names_both_ends_of_it(self) -> None:
        action, policy = _policy_for(ApprovalOutcome.ALLOW)
        decision = approve(
            action, policy=policy, advisor=_Advisor(ApprovalOutcome.REQUIRE_HUMAN)
        )
        line = next(c for c in decision.checks if "escalated" in c)
        assert ApprovalOutcome.ALLOW in line
        assert ApprovalOutcome.REQUIRE_HUMAN in line

    def test_the_policy_decision_survives_the_escalation(self) -> None:
        """The audit record keeps what policy said, not only the final answer."""
        action, policy = _policy_for(ApprovalOutcome.ALLOW)
        decision = approve(action, policy=policy, advisor=_Advisor(ApprovalOutcome.DENY))
        assert decision.policy_outcome == ApprovalOutcome.ALLOW
        assert decision.outcome == ApprovalOutcome.DENY
