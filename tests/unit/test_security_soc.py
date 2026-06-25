# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SOC-analyst factory, controls, and report grounding."""

from __future__ import annotations

from unittest.mock import patch

from tulip.security import (
    Abstention,
    Evidence,
    PostureEvidence,
    PostureFinding,
    PostureReport,
    SecurityControls,
    Severity,
    create_soc_analyst,
    ground_report,
    is_finding,
    submit_posture,
)


def _grounded_finding(**over: object) -> PostureFinding:
    base: dict[str, object] = {
        "title": "Root account has active access keys",
        "description": "The account root user has access keys (CIS 1.4).",
        "severity": Severity.HIGH,
        "asset": "aws-account:000000000000:root",
        "remediation": "Delete the root access keys.",
        "evidence": [
            PostureEvidence(
                statement="GetAccountSummary reports AccountAccessKeysPresent=1",
                ref="aws:iam:GetAccountSummary:AccountAccessKeysPresent",
                grounded=True,
            )
        ],
    }
    base.update(over)
    return PostureFinding(**base)  # type: ignore[arg-type]


def _speculative_finding() -> PostureFinding:
    return PostureFinding(
        title="Speculative lateral-movement risk",
        description="No evidence; speculation only.",
        severity=Severity.MEDIUM,
        asset="aws-account:000000000000",
        remediation="n/a",
        evidence=[
            PostureEvidence(
                statement="An attacker could pivot",
                ref="inference:none",
                grounded=False,
            )
        ],
    )


class TestSecurityControls:
    def test_default_attaches_readonly_aws_tools(self) -> None:
        names = {t.name for t in SecurityControls.default().tools()}
        assert names == {"describe_aws", "use_aws"}

    def test_readonly_aws_disabled_drops_tools(self) -> None:
        assert SecurityControls(readonly_aws=False).tools() == []

    def test_thresholds_are_strictly_ordered(self) -> None:
        th = SecurityControls(min_gsar=0.6).thresholds()
        assert th.proceed == 0.6
        assert th.regenerate < th.proceed

    def test_thresholds_ordered_even_when_min_gsar_tiny(self) -> None:
        th = SecurityControls(min_gsar=0.01).thresholds()
        assert th.regenerate < th.proceed  # the 1e-6 fallback keeps it ordered


class TestGroundReport:
    def test_grounded_finding_ships(self) -> None:
        report = PostureReport(summary="s", confidence=0.9, findings=[_grounded_finding()])
        out = ground_report(report)
        assert len(out) == 1
        assert is_finding(out[0])
        assert isinstance(out[0], Evidence)
        assert out[0].gsar_score >= 0.6

    def test_speculative_finding_abstains(self) -> None:
        report = PostureReport(summary="s", confidence=0.9, findings=[_speculative_finding()])
        out = ground_report(report)
        assert len(out) == 1
        assert not is_finding(out[0])
        assert isinstance(out[0], Abstention)

    def test_order_is_preserved(self) -> None:
        report = PostureReport(
            summary="s",
            confidence=0.9,
            findings=[_grounded_finding(), _speculative_finding()],
        )
        out = ground_report(report)
        assert is_finding(out[0])
        assert not is_finding(out[1])

    def test_empty_report_grounds_to_empty(self) -> None:
        report = PostureReport(summary="clean", confidence=1.0, findings=[])
        assert ground_report(report) == []

    def test_taxonomy_carried_onto_finding(self) -> None:
        from tulip.security import OwaspASI

        finding = _grounded_finding(taxonomy=[OwaspASI.IDENTITY_AND_PRIVILEGE_ABUSE])
        report = PostureReport(summary="s", confidence=0.9, findings=[finding])
        out = ground_report(report)
        assert is_finding(out[0])
        assert OwaspASI.IDENTITY_AND_PRIVILEGE_ABUSE in out[0].taxonomy


def test_submit_posture_summarizes() -> None:
    report = PostureReport(summary="s", confidence=0.83, findings=[_grounded_finding()])
    msg = submit_posture(report)
    assert "1 finding" in msg
    assert "83%" in msg


class TestCreateSocAnalyst:
    def test_wires_aws_tools_schema_and_submit(self) -> None:
        with patch("tulip.deepagent.create_deepagent") as make:
            create_soc_analyst(model="mock:test")
            kwargs = make.call_args.kwargs
        assert kwargs["output_schema"] is PostureReport
        assert kwargs["submit_tool"] == "submit_posture"
        assert kwargs["grounding"] is True
        tool_names = {t.name for t in kwargs["tools"]}
        assert {"describe_aws", "use_aws", "submit_posture"} <= tool_names

    def test_controls_can_drop_aws_tools(self) -> None:
        controls = SecurityControls(readonly_aws=False)
        with patch("tulip.deepagent.create_deepagent") as make:
            create_soc_analyst(model="mock:test", controls=controls)
            tool_names = {t.name for t in make.call_args.kwargs["tools"]}
        assert tool_names == {"submit_posture"}

    def test_min_confidence_defaults_to_controls(self) -> None:
        controls = SecurityControls(min_confidence=0.55)
        with patch("tulip.deepagent.create_deepagent") as make:
            create_soc_analyst(model="mock:test", controls=controls)
            assert make.call_args.kwargs["min_confidence"] == 0.55

    def test_min_confidence_override_wins(self) -> None:
        with patch("tulip.deepagent.create_deepagent") as make:
            create_soc_analyst(model="mock:test", min_confidence=0.95)
            assert make.call_args.kwargs["min_confidence"] == 0.95

    def test_scope_is_appended_to_prompt(self) -> None:
        with patch("tulip.deepagent.create_deepagent") as make:
            create_soc_analyst(model="mock:test", scope="Focus only on IAM and S3.")
            assert "Focus only on IAM and S3." in make.call_args.kwargs["system_prompt"]

    def test_explicit_system_prompt_replaces_default(self) -> None:
        with patch("tulip.deepagent.create_deepagent") as make:
            create_soc_analyst(model="mock:test", system_prompt="custom", scope="ignored")
            assert make.call_args.kwargs["system_prompt"] == "custom"

    def test_extra_tools_are_appended(self) -> None:
        from tulip.tools.decorator import tool

        @tool
        def enrich_ioc(ioc: str) -> str:
            """Enrich an indicator."""
            return ioc

        with patch("tulip.deepagent.create_deepagent") as make:
            create_soc_analyst(model="mock:test", tools=[enrich_ioc])
            tool_names = {t.name for t in make.call_args.kwargs["tools"]}
        assert "enrich_ioc" in tool_names
