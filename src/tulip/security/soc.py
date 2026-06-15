# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""``create_soc_analyst()`` — a grounded cloud-posture agent factory.

Composes the read-only AWS posture tools (:func:`~tulip.security.describe_aws`
/ :func:`~tulip.security.use_aws`) into a SOC-analyst-shaped agent that:

1. discovers the shape of an AWS account from the API spec,
2. gathers evidence with read-only API calls, and
3. proposes posture findings, each citing the exact API facts it stood on.

The proposals are then run through GSAR grounding (:func:`ground_report`):
each proposed finding becomes a typed :class:`~tulip.security.Finding` only if
its cited evidence clears the grounding threshold — otherwise it abstains. The
model gathers and proposes; Python decides what survives grounding. That split
is the point — an analyst's report is trustworthy by construction, not by
trusting the model's say-so.

Example::

    from tulip.security import create_soc_analyst, ground_report, is_finding

    analyst = create_soc_analyst(model="anthropic:claude-sonnet-4-6")
    result = await analyst.run(
        "Review the IAM and account-level posture of this AWS account."
    )
    for grounded in ground_report(result.output):
        if is_finding(grounded):
            print(grounded.severity, grounded.title, grounded.gsar_score)
        else:
            print("withheld:", grounded.reason)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from tulip.reasoning.gsar import Claim, EvidenceType, GSARThresholds, Partition
from tulip.security.aws import describe_aws_tool, use_aws_tool
from tulip.security.edr import (
    fetch_host_timeline_tool,
    isolate_host_tool,
    list_detections_tool,
)
from tulip.security.fingerprint import fingerprint_endpoint_tool
from tulip.security.grounded import GroundedFinding, ground_finding
from tulip.security.intel import enrich_indicator_tool
from tulip.security.scanner import scan_dependencies_tool, scan_endpoint_tool
from tulip.security.siem import siem_query_tool
from tulip.security.taxonomy import Severity, TaxonomyTag
from tulip.tools.decorator import tool


# --- What the analyst submits ----------------------------------------------


class PostureEvidence(BaseModel):
    """One atomic fact the analyst observed, with a pointer to its source."""

    statement: str = Field(
        description="The atomic observation, e.g. 'Account has 1 root access key present'.",
    )
    ref: str = Field(
        description=(
            "Reference to the source read-only call, e.g. "
            "'aws:iam:GetAccountSummary:AccountAccessKeysPresent'."
        ),
    )
    grounded: bool = Field(
        default=True,
        description=(
            "True when this is a direct API observation; False when it is an "
            "inference drawn from observations (inferences carry less weight "
            "and cannot, on their own, ground a finding)."
        ),
    )


class PostureFinding(BaseModel):
    """A posture issue the analyst proposes — grounded later by ``ground_report``."""

    title: str = Field(description="One-line summary of the issue.")
    description: str = Field(description="What was observed and why it matters.")
    severity: Severity = Field(description="Severity band.")
    asset: str = Field(description="Affected AWS resource (ARN / id / service).")
    remediation: str = Field(description="Recommended remediation.")
    evidence: list[PostureEvidence] = Field(
        default_factory=list,
        description=(
            "The API facts this finding stands on. A finding whose evidence "
            "does not clear the grounding threshold will abstain, not ship."
        ),
    )
    taxonomy: list[TaxonomyTag] = Field(
        default_factory=list,
        description="MITRE ATLAS / OWASP tags, when applicable.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Analyst confidence in this specific finding.",
    )


class PostureReport(BaseModel):
    """The structured posture report the agent submits to end the run."""

    summary: str = Field(description="Executive summary of the account's posture.")
    findings: list[PostureFinding] = Field(
        default_factory=list,
        description="Proposed posture findings.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence that the review is complete and correct.",
    )


@tool
def submit_posture(report: PostureReport) -> str:
    """Submit the completed cloud-posture report and end the review.

    Call this once you have gathered enough read-only evidence to stand behind
    each finding. Every finding must cite the exact API facts it relies on in
    its ``evidence`` list — ungrounded findings are discarded downstream.

    Args:
        report: The completed :class:`PostureReport`.
    """
    return f"submitted: {len(report.findings)} finding(s) ({report.confidence:.0%} confidence)"


# --- Controls bundle --------------------------------------------------------


@dataclass(frozen=True)
class SecurityControls:
    """The security posture of a SOC analyst — grounding + tooling, in one bundle.

    Holds the knobs that make a posture agent trustworthy: the grounding
    threshold a finding's evidence must clear, the submission-confidence floor,
    and whether the read-only AWS tools are attached. Passed to
    :func:`create_soc_analyst`; reused by :func:`ground_report` so the agent and
    the grounding step share one source of truth.
    """

    min_gsar: float = 0.6
    """GSAR ``τ_proceed`` — a finding's evidence must score at or above this."""
    min_confidence: float = 0.7
    """Submission-confidence floor wired into the agent's termination."""
    readonly_aws: bool = True
    """Attach the read-only ``describe_aws`` / ``use_aws`` tools."""
    threat_intel: bool = False
    """Attach the IOC-enrichment tool (``enrich_indicator``)."""
    siem: bool = False
    """Attach the SIEM search tool (``query_siem``)."""
    edr: bool = False
    """Attach the EDR forensics tools (``fetch_host_timeline`` / ``list_detections``)."""
    scanner: bool = False
    """Attach the vuln/posture scanners (``scan_dependencies`` / ``scan_endpoint``)."""
    fingerprint: bool = False
    """Attach the inference-fingerprinting tool (``fingerprint_endpoint``)."""
    allow_containment: bool = False
    """Attach the containment write (``isolate_host``). Requires ``edr``; off by default."""
    region: str | None = None
    """Default AWS region hint (informational; tools read ``TULIP_AWS_REGION``)."""

    @classmethod
    def default(cls) -> SecurityControls:
        """The standard read-only cloud-posture control set."""
        return cls()

    @classmethod
    def soc_triage(cls) -> SecurityControls:
        """The read-only SOC triage loop — intel, SIEM, EDR, scanner, fingerprint."""
        return cls(
            readonly_aws=False,
            threat_intel=True,
            siem=True,
            edr=True,
            scanner=True,
            fingerprint=True,
        )

    def tools(self) -> list[Any]:
        """The security toolset implied by these controls."""
        bag: list[Any] = []
        if self.readonly_aws:
            bag.extend([describe_aws_tool, use_aws_tool])
        if self.threat_intel:
            bag.append(enrich_indicator_tool)
        if self.siem:
            bag.append(siem_query_tool)
        if self.edr:
            bag.extend([fetch_host_timeline_tool, list_detections_tool])
            if self.allow_containment:
                bag.append(isolate_host_tool)
        if self.scanner:
            bag.extend([scan_dependencies_tool, scan_endpoint_tool])
        if self.fingerprint:
            bag.append(fingerprint_endpoint_tool)
        return bag

    def thresholds(self) -> GSARThresholds:
        """GSAR thresholds derived from ``min_gsar`` (``τ_regenerate < τ_proceed``)."""
        return GSARThresholds(
            proceed=self.min_gsar,
            regenerate=min(self.min_gsar * 0.6, self.min_gsar - 1e-6),
        )


# --- Grounding the submitted report ----------------------------------------


def ground_report(
    report: PostureReport,
    controls: SecurityControls | None = None,
) -> list[GroundedFinding]:
    """Run every proposed finding through GSAR grounding.

    Each :class:`PostureFinding` becomes a typed
    :class:`~tulip.security.Finding` only if its cited evidence clears the
    grounding threshold; otherwise it yields an
    :class:`~tulip.security.Abstention`. Direct API observations
    (``grounded=True``) become grounded GSAR claims; inferences become
    ungrounded claims, which cannot on their own admit a finding.

    Args:
        report: The :class:`PostureReport` the agent submitted.
        controls: Controls supplying the grounding threshold. Defaults to
            :meth:`SecurityControls.default`.

    Returns:
        One :class:`~tulip.security.GroundedFinding` per proposed finding, in
        the order the agent proposed them.
    """
    controls = controls or SecurityControls.default()
    thresholds = controls.thresholds()

    results: list[GroundedFinding] = []
    for finding in report.findings:
        grounded_claims = [
            Claim(text=ev.statement, type=EvidenceType.TOOL_MATCH, evidence_refs=[ev.ref])
            for ev in finding.evidence
            if ev.grounded
        ]
        ungrounded_claims = [
            Claim(text=ev.statement, type=EvidenceType.INFERENCE, evidence_refs=[ev.ref])
            for ev in finding.evidence
            if not ev.grounded
        ]
        partition = Partition(grounded=grounded_claims, ungrounded=ungrounded_claims)
        results.append(
            ground_finding(
                title=finding.title,
                description=finding.description,
                severity=finding.severity,
                asset=finding.asset,
                remediation=finding.remediation,
                partition=partition,
                taxonomy=finding.taxonomy or None,
                confidence=finding.confidence,
                thresholds=thresholds,
            )
        )
    return results


# --- The factory ------------------------------------------------------------


_DEFAULT_SYSTEM_PROMPT = """\
You are a read-only cloud security posture analyst auditing an AWS account.

You have two spec-driven tools, both read-only by construction:

- `describe_aws(service, operation)` — discover the shape of AWS. With no
  arguments it lists every service; with a service it lists that service's
  read-only operations; with a service and operation it lists the operation's
  parameters. Use it to find out what you can inspect.
- `use_aws(service, operation, parameters)` — run one read-only operation and
  get back the raw API response. The response is your evidence. It refuses any
  non-read operation, so you cannot change anything.

Method:

1. Orient. Start from the account-level and identity surfaces (iam, sts,
   accessanalyzer), then the high-signal services (s3, ec2, cloudtrail, kms,
   rds, lambda). Use `describe_aws` to find the right read-only operations.
2. Gather. Call `use_aws` to collect concrete facts. Compare them against
   well-known baselines (CIS AWS Foundations): root access keys present, root
   or users without MFA, public S3 buckets, security groups open to 0.0.0.0/0,
   CloudTrail disabled or not multi-region, unused over-privileged roles.
3. Report. For each issue, propose a finding and cite the exact API facts it
   rests on in `evidence` — mark direct API observations `grounded=true` and
   anything you inferred `grounded=false`, with a `ref` pointing back to the
   call (e.g. `aws:iam:GetAccountSummary:AccountAccessKeysPresent`).

Rules:

- Claim only what the evidence shows. A finding with no grounded evidence is
  discarded downstream — do not speculate to fill the report.
- Prefer fewer, well-grounded findings over many thin ones.
- You are strictly read-only. Never attempt a write; it will be refused.

When you have gathered enough evidence, call `submit_posture` with the report.
"""


def create_soc_analyst(
    *,
    model: str | Any,
    controls: SecurityControls | None = None,
    tools: list[Any] | None = None,
    system_prompt: str | None = None,
    scope: str | None = None,
    min_confidence: float | None = None,
    max_iterations: int = 30,
    **deepagent_kwargs: Any,
) -> Any:
    """Construct a grounded, read-only AWS cloud-posture agent.

    A thin security-shaped layer over :func:`~tulip.create_deepagent`: it bakes
    in the read-only AWS posture tools, a SOC-analyst system prompt, the
    :class:`PostureReport` output schema, and grounding/reflexion. The agent
    proposes findings; pass its ``result.output`` to :func:`ground_report` to
    get typed :class:`~tulip.security.Finding` / ``Abstention`` results.

    Args:
        model: A tulip model string (``"anthropic:claude-sonnet-4-6"``) or a
            ``ModelProtocol`` instance.
        controls: The :class:`SecurityControls` bundle (grounding threshold,
            confidence floor, AWS tooling). Defaults to
            :meth:`SecurityControls.default`.
        tools: Extra tools to attach alongside the AWS posture tools and the
            ``submit_posture`` tool (e.g. threat-intel or SIEM enrichment).
        system_prompt: Override the default SOC-analyst prompt entirely. When
            set, ``scope`` is ignored.
        scope: Optional one-line narrowing appended to the default prompt
            (e.g. ``"Focus only on IAM and S3."``). Ignored when
            ``system_prompt`` is given.
        min_confidence: Submission-confidence floor for early-exit. Defaults to
            ``controls.min_confidence``.
        max_iterations: Cap on reasoning steps. Default 30.
        **deepagent_kwargs: Forwarded to :func:`~tulip.create_deepagent`
            (checkpointer, datastores, summarization, hooks, …).

    Returns:
        A configured ``tulip.Agent`` ready for ``await agent.run(prompt)``.
    """
    from tulip.deepagent import create_deepagent

    controls = controls or SecurityControls.default()

    if system_prompt is not None:
        prompt = system_prompt
    elif scope:
        prompt = f"{_DEFAULT_SYSTEM_PROMPT}\nScope for this review: {scope}\n"
    else:
        prompt = _DEFAULT_SYSTEM_PROMPT

    all_tools: list[Any] = [*controls.tools(), submit_posture, *(tools or [])]

    return create_deepagent(
        model=model,
        tools=all_tools,
        system_prompt=prompt,
        output_schema=PostureReport,
        submit_tool="submit_posture",
        min_confidence=(min_confidence if min_confidence is not None else controls.min_confidence),
        max_iterations=max_iterations,
        reflexion=True,
        grounding=True,
        **deepagent_kwargs,
    )
