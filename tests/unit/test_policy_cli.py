# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus policy lint`` and ``locus policy explain``, end to end through ``main(argv)``.

The commands exist for a CI job and a reviewer: the exit code is the
verdict, ``--format json`` is for the job, the text is for the person.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from locus.cli import main
from locus.control import ControlPolicy


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def policy_file(tmp_path: Path) -> Path:
    path = tmp_path / "refunds.yaml"
    ControlPolicy(
        require_verification_score=0.0,
        max_blast_radius=1,
        require_human_for=frozenset({"high_value"}),
        version="refund-gate-1",
    ).dump(path)
    return path


def test_lint_passes_a_sound_file_and_exits_zero(
    policy_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["policy", "lint", str(policy_file)])

    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith(f"policy {policy_file} (sha256 ")
    assert "no findings" in out
    assert "note: no environments or kinds declared" in out


def test_lint_reports_gaps_and_exits_one(
    policy_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "policy",
            "lint",
            str(policy_file),
            "--environments",
            "production",
            "staging",
            "--kinds",
            "payment",
        ]
    )

    out = capsys.readouterr().out
    assert code == 1
    assert "3 finding(s):" in out
    assert "uncovered-environment: environment 'production' has no rule" in out
    assert "uncovered-environment: environment 'staging' has no rule" in out
    assert "kind-never-held: kind 'payment' is never held" in out


def test_lint_reads_what_the_file_declares_it_governs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "deploy.yaml"
    path.write_text(
        ControlPolicy(require_verification_score=0.0, max_blast_radius=4).dump()
        + 'governs:\n  environments: ["production", "staging"]\n'
    )

    code = main(["policy", "lint", str(path), "--format", "json"])

    data = json.loads(capsys.readouterr().out)
    assert code == 1
    assert data["ok"] is False
    assert data["source"] == str(path)
    assert [f["label"] for f in data["findings"]] == ["staging"]
    assert data["policy_hash"] == ControlPolicy.load(path).content_hash()


def test_lint_finds_an_unreachable_rule_in_a_json_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "policy.json"
    ControlPolicy(
        require_human_for=frozenset({"production", "iam"}), deny_for=frozenset({"iam"})
    ).dump(path)

    code = main(["policy", "lint", str(path), "--format", "json"])

    data = json.loads(capsys.readouterr().out)
    assert code == 1
    assert data["findings"] == [
        {
            "code": "unreachable-rule",
            "rule": "require_human_for",
            "label": "iam",
            "message": "label 'iam' in require_human_for is unreachable: deny_for denies it first",
        }
    ]


def test_explain_prints_the_decision_path(
    policy_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    action = json.dumps(
        {
            "name": "issue_refund",
            "asset": "cust:7763",
            "environment": "production",
            "kind": "payment",
            "blast_radius": 6,
            "tags": ["high_value", "refund"],
        }
    )

    code = main(["policy", "explain", str(policy_file), "--action", action])

    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith(f"policy   {policy_file} (sha256 ")
    assert "version refund-gate-1" in out.splitlines()[0]
    assert "max_blast_radius            require_human  blast radius 6 exceeds the maximum 1" in out
    assert (
        "require_human_for           require_human  labels ['high_value'] require human approval"
        in out
    )
    assert "deny_for                    -              no label of the action is denied" in out
    assert out.rstrip().endswith(
        "outcome  require_human: held for a person; runs once someone approves "
        "(admit(..., approved_by=...))"
    )


def test_explain_as_json_with_spend_and_verification(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "spend.yaml"
    ControlPolicy(
        require_verification_score=0.8, require_human_for=frozenset(), spend_limit_usd=1_000.0
    ).dump(path)

    code = main(
        [
            "policy",
            "explain",
            str(path),
            "--action",
            '{"name": "pay", "cost_usd": 300}',
            "--spent-usd",
            "800",
            "--verification",
            "0.95",
            "--format",
            "json",
        ]
    )

    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["outcome"] == "deny"
    assert data["requires_approval"] is False
    rules = {rule["rule"]: rule for rule in data["rules"]}
    assert rules["spend_limit_usd"]["outcome"] == "deny"
    assert rules["spend_limit_usd"]["detail"].startswith("spend $800.00 + $300.00 would exceed")
    assert rules["require_verification_score"]["outcome"] is None
    assert "0.95 meets the bar 0.80" in rules["require_verification_score"]["detail"]
    assert data["action"]["cost_usd"] == 300.0


def test_expect_turns_explain_into_a_check(
    policy_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    small = '{"name": "issue_refund", "kind": "payment"}'
    large = '{"name": "issue_refund", "kind": "payment", "tags": ["high_value"]}'

    assert (
        main(["policy", "explain", str(policy_file), "--action", small, "--expect", "allow"]) == 0
    )
    assert (
        main(
            ["policy", "explain", str(policy_file), "--action", large, "--expect", "require_human"]
        )
        == 0
    )
    assert main(["policy", "explain", str(policy_file), "--action", small, "--expect", "deny"]) == 1
    capsys.readouterr()


@pytest.mark.parametrize(
    ("argv", "fragment"),
    [
        (["policy", "lint", "{tmp}/missing.yaml"], "No such file"),
        (["policy", "lint", "{tmp}/typo.yaml"], "did you mean 'max_blast_radius'"),
        (["policy", "explain", "{tmp}/ok.yaml", "--action", "{nope"], "not valid JSON"),
        (
            ["policy", "explain", "{tmp}/ok.yaml", "--action", '{"unknown_key": "x"}'],
            "unknown action key",
        ),
        (
            ["policy", "explain", "{tmp}/ok.yaml", "--action", '{"name": 1}'],
            "'name' must be a string",
        ),
    ],
)
def test_usage_errors_are_reported_without_a_traceback(
    tmp_path: Path, argv: list[str], fragment: str, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "typo.yaml").write_text("max_blast_radious: 2\n")
    ControlPolicy().dump(tmp_path / "ok.yaml")

    code = main([arg.replace("{tmp}", str(tmp_path)) for arg in argv])

    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("locus: error: ")
    assert fragment in err
