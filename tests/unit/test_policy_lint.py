# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``lint_policy`` finds the three gap classes; ``explain`` shows the gate's own path.

Lint is only worth running in CI if each class of gap is caught, and explain
is only trustworthy if the outcome it prints is the one ``approve()`` gives.
"""

from __future__ import annotations

import pytest

from locus.control import (
    Action,
    ControlPolicy,
    PolicyDocument,
    Severity,
    approve,
    evaluate_rules,
    explain,
    lint_policy,
)
from locus.control.findings import Evidence
from locus.control.policy_lint import (
    KIND_NEVER_HELD,
    UNCOVERED_ENVIRONMENT,
    UNREACHABLE_RULE,
    LintReport,
    action_from_json,
    action_from_mapping,
    action_to_dict,
    lint_document,
)
from locus.control.verification import VerificationResult


def _open(**overrides: object) -> ControlPolicy:
    base: dict[str, object] = {
        "require_verification_score": 0.0,
        "max_blast_radius": 1,
        "require_human_for": frozenset({"production"}),
    }
    return ControlPolicy(**{**base, **overrides})  # type: ignore[arg-type]


def _codes(report: LintReport) -> list[str]:
    return [finding.code for finding in report.findings]


# --------------------------------------------------------------------- unreachable


def test_a_hold_label_that_deny_also_names_is_unreachable() -> None:
    report = lint_policy(
        _open(require_human_for=frozenset({"production", "iam"}), deny_for=frozenset({"iam"}))
    )

    (finding,) = report.findings
    assert finding.code == UNREACHABLE_RULE
    assert finding.rule == "require_human_for"
    assert finding.label == "iam"
    assert "deny_for denies it first" in finding.message


def test_a_sandbox_label_that_deny_also_names_is_unreachable() -> None:
    report = lint_policy(
        _open(require_sandbox_for=frozenset({"code"}), deny_for=frozenset({"code"}))
    )

    assert [(f.code, f.rule, f.label) for f in report.findings] == [
        (UNREACHABLE_RULE, "require_sandbox_for", "code")
    ]


def test_denying_the_sandboxed_tag_makes_the_sandbox_rule_unsatisfiable() -> None:
    report = lint_policy(
        _open(require_sandbox_for=frozenset({"code"}), deny_for=frozenset({"sandboxed"}))
    )

    (finding,) = report.findings
    assert finding.code == UNREACHABLE_RULE
    assert finding.label == "sandboxed"
    assert "can never be satisfied" in finding.message


def test_a_verification_bar_above_one_holds_everything() -> None:
    report = lint_policy(ControlPolicy(require_verification_score=1.2))

    (finding,) = report.findings
    assert (finding.code, finding.rule) == (UNREACHABLE_RULE, "require_verification_score")
    assert "every action is held" in finding.message
    assert lint_policy(ControlPolicy(require_verification_score=1.0)).ok


def test_a_cost_hold_at_or_above_the_spend_limit_is_unreachable() -> None:
    report = lint_policy(_open(require_human_over_usd=500.0, spend_limit_usd=500.0))

    (finding,) = report.findings
    assert (finding.code, finding.rule) == (UNREACHABLE_RULE, "require_human_over_usd")
    assert "spend_limit_usd" in finding.message
    assert lint_policy(_open(require_human_over_usd=100.0, spend_limit_usd=500.0)).ok


def test_a_zero_blast_radius_cap_leaves_no_allow_path() -> None:
    report = lint_policy(_open(max_blast_radius=0))

    (finding,) = report.findings
    assert (finding.code, finding.rule) == (UNREACHABLE_RULE, "max_blast_radius")


def test_a_sound_policy_has_no_findings() -> None:
    assert lint_policy(ControlPolicy()).ok
    assert lint_policy(_open()).ok
    assert lint_policy(_open(), environments=["production"], kinds=["payment"]).ok


# --------------------------------------------------------------------- coverage


def test_an_environment_nothing_mentions_is_uncovered() -> None:
    report = lint_policy(_open(), environments=["production", "staging", "dev"])

    assert [(f.code, f.label) for f in report.findings] == [
        (UNCOVERED_ENVIRONMENT, "dev"),
        (UNCOVERED_ENVIRONMENT, "staging"),
    ]
    assert "has no rule" in report.findings[0].message


def test_an_environment_a_generic_rule_holds_is_covered() -> None:
    # The default bar holds every unverified action, so staging is covered
    # even though no label rule names it.
    assert lint_policy(ControlPolicy(), environments=["staging"]).ok
    # A deny is a rule too; so is a sandbox requirement.
    assert lint_policy(_open(deny_for=frozenset({"dev"})), environments=["dev"]).ok
    assert lint_policy(_open(require_sandbox_for=frozenset({"lab"})), environments=["lab"]).ok


def test_a_kind_never_put_in_front_of_a_person_is_a_finding() -> None:
    report = lint_policy(_open(), environments=["production", "staging"], kinds=["payment"])

    kinds = [f for f in report.findings if f.code == KIND_NEVER_HELD]
    assert kinds == []  # held in production, so it is held somewhere
    report = lint_policy(
        _open(require_human_for=frozenset({"iam"})),
        environments=["production", "staging"],
        kinds=["payment", "iam"],
    )
    assert [(f.code, f.label) for f in report.findings] == [
        (UNCOVERED_ENVIRONMENT, "production"),
        (UNCOVERED_ENVIRONMENT, "staging"),
        (KIND_NEVER_HELD, "payment"),
    ]
    assert "['production', 'staging']" in report.findings[-1].message


def test_a_kind_is_checked_on_its_own_when_no_environment_is_declared() -> None:
    report = lint_policy(_open(require_human_for=frozenset({"iam"})), kinds=["payment", "iam"])

    assert [(f.code, f.label) for f in report.findings] == [(KIND_NEVER_HELD, "payment")]
    assert "no environments declared" in report.findings[0].message


def test_a_denied_kind_is_not_reported_as_never_held() -> None:
    assert lint_policy(_open(deny_for=frozenset({"wipe"})), kinds=["wipe"]).ok


def test_the_notes_say_what_was_not_checked_and_what_auto_allows() -> None:
    report = lint_policy(_open())

    assert report.ok
    assert any("no environments or kinds declared" in note for note in report.notes)
    assert any("no environment, kind or tags" in note for note in report.notes)
    assert lint_policy(ControlPolicy(), environments=["production"]).notes == ()


def test_lint_document_adds_what_the_file_declares() -> None:
    document = PolicyDocument(
        policy=_open(), environments=frozenset({"staging"}), source="refunds.yaml"
    )

    report = lint_document(document, kinds=["payment"])

    # staging is the only declared environment, so a payment there is never held either.
    assert _codes(report) == [UNCOVERED_ENVIRONMENT, KIND_NEVER_HELD]
    assert report.source == "refunds.yaml"
    assert report.policy_hash == _open().content_hash()
    rendered = report.render()
    assert rendered.startswith("policy refunds.yaml (sha256 ")
    assert "2 finding(s):" in rendered
    assert "uncovered-environment: environment 'staging'" in rendered
    data = report.to_dict()
    assert data["ok"] is False
    assert data["findings"][0]["label"] == "staging"
    assert lint_policy(ControlPolicy()).render().splitlines()[1] == "no findings"


# --------------------------------------------------------------------- explain


_ACTIONS = [
    Action(name="noop"),
    Action(name="refund", environment="production", kind="payment", blast_radius=6),
    Action(name="wipe", tags=frozenset({"prohibited"})),
    Action(name="run", kind="code"),
    Action(name="run", kind="code", tags=frozenset({"sandboxed"})),
    Action(name="pay", kind="payment", cost_usd=900.0),
    Action(name="pay", kind="payment", cost_usd=4_900.0),
]

_POLICIES = [
    ControlPolicy(),
    _open(
        deny_for=frozenset({"prohibited"}),
        require_sandbox_for=frozenset({"code"}),
        require_human_over_usd=500.0,
        spend_limit_usd=5_000.0,
    ),
    ControlPolicy(require_verification_score=0.9, max_blast_radius=2),
]


@pytest.mark.parametrize("policy", _POLICIES)
@pytest.mark.parametrize("action", _ACTIONS)
@pytest.mark.parametrize(
    "verdict",
    [
        None,
        VerificationResult(survives=True, confidence=0.95, evidence_quality=0.9),
        VerificationResult(survives=True, confidence=0.5, evidence_quality=0.9),
        VerificationResult(survives=False, confidence=0.0, evidence_quality=0.0),
    ],
)
def test_explain_shows_the_decision_the_gate_makes(
    policy: ControlPolicy, action: Action, verdict: VerificationResult | None
) -> None:
    decision = approve(action, policy=policy, verdict=verdict, spent_usd=200.0)

    explanation = explain(policy, action, verdict=verdict, spent_usd=200.0)

    assert explanation.decision == decision
    fired = [check.detail for check in explanation.checks if check.fired]
    assert fired == decision.checks
    assert [check.rule for check in explanation.checks] == [
        "deny_for",
        "require_sandbox_for",
        "min_severity",
        "require_verification_score",
        "max_blast_radius",
        "require_human_over_usd",
        "spend_limit_usd",
        "require_human_for",
    ]
    assert all(check.detail for check in explanation.checks)


def test_the_rendering_names_the_rule_that_matched_and_what_happens_next() -> None:
    policy = _open(require_human_for=frozenset({"high_value"}), version="refund-gate-1")
    action = Action(
        name="issue_refund",
        asset="cust:7763",
        blast_radius=6,
        environment="production",
        kind="payment",
        tags=frozenset({"high_value", "refund"}),
    )

    text = explain(policy, action, source="refunds.yaml").render()

    lines = text.splitlines()
    assert lines[0].startswith("policy   refunds.yaml (sha256 ")
    assert lines[0].endswith("version refund-gate-1")
    assert "action   issue_refund environment=production kind=payment blast_radius=6" in lines[1]
    assert lines[2] == "labels   ['high_value', 'payment', 'production', 'refund']"
    assert any(
        "max_blast_radius" in line and "require_human" in line and "exceeds the maximum 1" in line
        for line in lines
    )
    assert any("deny_for" in line and "  -  " in line for line in lines)
    assert lines[-1].startswith("outcome  require_human: held for a person")


def test_the_rendering_for_allow_and_deny() -> None:
    policy = _open(deny_for=frozenset({"prohibited"}))

    allowed = explain(policy, Action(name="noop"))
    denied = explain(policy, Action(name="wipe", tags=frozenset({"prohibited"})))

    assert allowed.requires_approval is False
    assert (
        allowed.render().splitlines()[0]
        == f"policy   <policy> (sha256 {policy.content_hash()[:12]})"
    )
    assert allowed.render().endswith("outcome  allow: the action runs; no approval needed")
    assert denied.requires_approval is False
    assert denied.render().endswith("outcome  deny: denied; no approval overrides it")
    assert explain(policy, Action(name="x", environment="production")).requires_approval


def test_the_json_form_carries_every_rule_and_the_action() -> None:
    policy = _open(version="v1")
    action = Action(name="refund", kind="payment", tags=frozenset({"b", "a"}), cost_usd=2.5)

    data = explain(policy, action, source="p.yaml").to_dict()

    assert data["source"] == "p.yaml"
    assert data["policy_version"] == "v1"
    assert data["policy_hash"] == policy.content_hash()
    assert data["action"] == action_to_dict(action)
    assert data["action"]["tags"] == ["a", "b"]
    assert data["labels"] == ["a", "b", "payment", "unknown"]
    assert len(data["rules"]) == 8
    assert data["outcome"] == "allow"
    assert data["requires_approval"] is False
    assert data["reason"] == "all policy checks passed"


def test_evaluate_rules_reads_a_finding_too() -> None:
    policy = ControlPolicy(min_severity=Severity.HIGH, require_verification_score=0.0)
    low = _evidence(Severity.LOW)
    high = _evidence(Severity.CRITICAL)

    (low_check,) = [
        c
        for c in evaluate_rules(Action(name="a"), policy=policy, finding=low)
        if c.rule == "min_severity"
    ]
    (high_check,) = [
        c
        for c in evaluate_rules(Action(name="a"), policy=policy, finding=high)
        if c.rule == "min_severity"
    ]

    assert low_check.outcome == "deny"
    assert high_check.outcome is None
    assert "meets the minimum high" in high_check.detail


def _evidence(severity: Severity) -> Evidence:
    return Evidence(
        title="t",
        description="d",
        severity=severity,
        asset="host-1",
        remediation="r",
        gsar_score=1.0,
        confidence=1.0,
        evidence_refs=["e1"],
    )


# --------------------------------------------------------------------- the action


def test_an_action_from_json_round_trips() -> None:
    action = action_from_json(
        '{"name": "refund", "asset": "o1", "blast_radius": 3, "environment": "production", '
        '"kind": "payment", "tags": ["b", "a"], "cost_usd": 12}'
    )

    assert action == Action(
        name="refund",
        asset="o1",
        blast_radius=3,
        environment="production",
        kind="payment",
        tags=frozenset({"a", "b"}),
        cost_usd=12.0,
    )
    assert action_from_mapping(action_to_dict(action)) == action
    assert action_from_json('{"name": "x"}') == Action(name="x")


@pytest.mark.parametrize(
    ("text", "error", "fragment"),
    [
        ('{"name": "x", "nmae": 1}', ValueError, "unknown action key 'nmae'"),
        ('{"asset": "o1"}', TypeError, "'name' must be a string"),
        ('{"name": ""}', ValueError, "non-empty 'name'"),
        ('{"name": "x", "tags": "a"}', TypeError, "'tags' must be a list"),
        ('{"name": "x", "tags": [1]}', TypeError, "'tags' must be a list"),
        ('{"name": "x", "blast_radius": 1.5}', TypeError, "'blast_radius' must be an integer"),
        ('{"name": "x", "blast_radius": true}', TypeError, "'blast_radius' must be an integer"),
        ('{"name": "x", "cost_usd": "12"}', TypeError, "'cost_usd' must be a number"),
        ('{"name": "x", "kind": 3}', TypeError, "'kind' must be a string"),
        ("[1]", TypeError, "must be a JSON object"),
        ("{nope", ValueError, "not valid JSON"),
    ],
)
def test_a_bad_action_says_what_is_wrong(text: str, error: type[Exception], fragment: str) -> None:
    with pytest.raises(error, match=fragment):
        action_from_json(text)
