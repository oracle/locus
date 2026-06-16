# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""GSAR LLM-as-judge protocol and structured-output schema.

Implements §6 + Appendix C of `arXiv:2604.23366`. The judge is the
single LLM interface the rest of GSAR depends on; it consumes a
synthesised report and the raw evidence corpus, and emits a typed
``JudgeOutput`` covering the four-way partition, a scalar grounding
score, an abstain channel, and a one-sentence explanation.

This module ships:

- :class:`JudgeOutput` — Pydantic schema matching Appendix C.
- :class:`BaseGSARJudge` — runtime-checkable Protocol any backend can
  implement.
- :class:`StructuredOutputGSARJudge` — concrete reference judge that
  drives any :class:`locus.models.base.BaseModel` via
  ``response_format`` constrained decoding (paper's §6 + §7).
- :func:`safe_default_judge_output` — the structured-output failure
  fallback the paper recommends (`s = 0.5`, partition empty,
  ``a = abstain``) so the outer loop stays sound under judge failure
  (§6 "Robustness").
"""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from locus.reasoning.gsar import (
    NEUTRAL_SCORE_ON_EMPTY,
    Claim,
    EvidenceType,
    Partition,
)


class JudgeOutput(BaseModel):
    """The structured judge verdict, mirroring Appendix C verbatim.

    Field naming intentionally matches the paper so persisted blobs
    round-trip cleanly between the locus implementation and any third
    party that reads the schema off the arXiv paper.
    """

    grounding_score: float = Field(
        ge=0.0,
        le=1.0,
        description="The judge's own scalar in [0, 1].",
    )
    is_grounded: bool = Field(
        description="The judge's binary verdict (kept for audit / cross-check).",
    )

    grounded_claims: list[Claim] = Field(default_factory=list, description="G.")
    ungrounded_claims: list[Claim] = Field(default_factory=list, description="U.")
    contradicted_claims: list[Claim] = Field(default_factory=list, description="X.")
    complementary_claims: list[Claim] = Field(default_factory=list, description="K.")

    gaps: list[str] = Field(
        default_factory=list,
        description="Specific evidence / investigation gaps the judge identified.",
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Explicit contradictions the judge noted (free-form).",
    )

    verification_needed: bool = Field(
        default=False,
        description="Whether the judge believes another tool call would help.",
    )
    verification_reason: str | None = Field(
        default=None,
        description="One-line natural-language reason; None when no verification needed.",
    )

    explanation: str = Field(
        default="",
        description="Single-sentence judge explanation (the ε channel from §6).",
    )

    decision_status: str = Field(
        default="resolved",
        description=(
            "'resolved' or 'abstain'. The paper's a ∈ {resolved, abstain}. "
            "Abstain triggers the same outer-loop branch as δ = replan."
        ),
    )
    abstain_reason: str | None = Field(
        default=None,
        description="Required when decision_status='abstain'; otherwise None.",
    )

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _decision_status_value(self) -> JudgeOutput:
        if self.decision_status not in ("resolved", "abstain"):
            raise ValueError(
                f"decision_status must be 'resolved' or 'abstain', got {self.decision_status!r}"
            )
        if self.decision_status == "abstain" and not self.abstain_reason:
            raise ValueError("abstain_reason is required when decision_status='abstain'")
        return self

    def to_partition(self) -> Partition:
        """Project the four-way claim lists into a :class:`Partition`."""
        return Partition(
            grounded=list(self.grounded_claims),
            ungrounded=list(self.ungrounded_claims),
            contradicted=list(self.contradicted_claims),
            complementary=list(self.complementary_claims),
        )

    @property
    def abstained(self) -> bool:
        return self.decision_status == "abstain"


def safe_default_judge_output(reason: str = "judge produced malformed output") -> JudgeOutput:
    """The §6 'Robustness' fallback.

    Returns ``s = 0.5``, empty partition, ``a = abstain``. The outer
    loop will treat this as ``δ = replan``, escalating to a fresh
    investigation rather than silently emitting a degraded report.
    """
    return JudgeOutput(
        grounding_score=NEUTRAL_SCORE_ON_EMPTY,
        is_grounded=False,
        decision_status="abstain",
        abstain_reason=reason,
        explanation=f"Safe default: {reason}",
    )


@runtime_checkable
class BaseGSARJudge(Protocol):
    """Protocol every GSAR judge backend must implement.

    The judge is the only LLM interface in the GSAR layer; everything
    else (scoring, decision, outer loop) is deterministic on the
    judge's output. See §6 of the paper for the contract.
    """

    async def judge(
        self,
        *,
        report_synthesis: str,
        evidence_corpus: str,
        **kwargs: Any,
    ) -> JudgeOutput:
        """Judge a synthesised report against its evidence corpus.

        Args:
            report_synthesis: The natural-language synthesis ``θ``
                produced by the orchestrator.
            evidence_corpus: The raw evidence ``E`` — concatenated,
                deduplicated tool outputs and structured step outputs.
            **kwargs: Backend-specific options.

        Returns:
            A :class:`JudgeOutput` covering the four-way partition,
            scalar score, abstain channel, and explanation. On any
            unrecoverable error the implementation should return
            :func:`safe_default_judge_output` rather than raise.
        """
        ...


# ---------------------------------------------------------------------------
# Reference judge — drives any BaseModel via constrained decoding
# ---------------------------------------------------------------------------


_DEFAULT_SYSTEM_PROMPT = """\
You are a grounding judge for a multi-agent diagnostic system.

You receive a SYNTHESIS (the candidate report) and an EVIDENCE corpus
(tool outputs, signal fields, structured step results). Your job is
to partition the synthesis into atomic claims and label each claim
according to its relationship to the evidence.

FIRST: enumerate every atomic claim in the synthesis. EVERY atomic
statement must appear in exactly one of the four buckets — none can
be omitted. The four lists together must contain all atomic claims
of the synthesis.

CRITICAL GROUNDING RULE
-----------------------
A claim is `grounded` ONLY when you can point to a specific line or
field of the EVIDENCE that supports it directly. If you cannot quote
that evidence:
- if the claim makes a specific factual assertion (a name, an ID, a
  timestamp, a number, a verb of action) that is not in evidence, it
  goes to ungrounded_claims;
- if a fact in the claim conflicts with the evidence, it goes to
  contradicted_claims;
- if the claim is a high-level non-redundant alternative perspective
  that doesn't conflict with anything in grounded_claims, it goes to
  complementary_claims.

Plausibility is NOT grounding. "Engineer Pat Smith paged at 14:05"
is ungrounded if no evidence row mentions Pat Smith. "Query QID-77321
killed" is ungrounded if no evidence row mentions QID-77321. Putting
such claims in grounded_claims is a labelling error.

Conversely, claims that DO match evidence rows (paraphrases of tool
output, references to signal fields) MUST be placed in grounded_claims
— do not omit them.

Use the following partitions:

- grounded_claims: claims directly supported by the evidence.
- ungrounded_claims: claims that are plausible but not supported.
- contradicted_claims: claims that the evidence directly refutes.
- complementary_claims: NON-REDUNDANT additional perspectives that
  neither restate a grounded claim nor conflict with anything in
  grounded_claims ∪ contradicted_claims. Empty if there are none.

For every claim, attach an `EvidenceType` describing its provenance,
chosen from:
  tool_match, specific_data, signal_match, complementary_finding,
  synthesis, neg_evidence, inference, domain.

Rules:
- A claim that paraphrases a tool output row → tool_match.
- A claim citing a structured field of a step output → specific_data.
- A claim referencing the originating alert / signal → signal_match.
- A non-redundant alternative perspective → complementary_finding.
- A cross-specialist combination → synthesis.
- An "absence of error" / negative finding → neg_evidence.
- A model-internal inference → inference.
- A textbook fact → domain.

Output a single JSON object matching the schema. Be conservative:
if uncertain, prefer ungrounded over grounded. If the evidence is
malformed or insufficient to score, set decision_status="abstain"
and abstain_reason to a one-sentence explanation; in that case the
four claim lists may be empty.
"""


_USER_TEMPLATE = """\
SYNTHESIS:
{synthesis}

EVIDENCE:
{evidence}

Partition the synthesis. Return JSON only.
"""


class StructuredOutputGSARJudge(BaseModel):
    """Reference :class:`BaseGSARJudge` driven by a locus model.

    Drives any :class:`locus.models.base.BaseModel` through
    ``response_format`` constrained decoding. The model produces a
    JSON object conforming to :class:`JudgeOutput`'s schema; we parse
    via :func:`locus.core.structured.parse_structured` and fall back
    to :func:`safe_default_judge_output` on validation failure
    (paper's §6 "Robustness" three-tier fallback: structured first,
    then manual JSON, then safe default).

    Args:
        model: The locus model to drive. The judge model recommended
            by the paper (Table 10) varies by use case — Opus 4.7 for
            strict / safety-first; gpt-5.4 for high-throughput;
            Sonnet 4.6 for balanced default.
        system_prompt: Override the default partition-instruction
            prompt.
        strict: Whether to request provider-enforced strict
            ``json_schema``. Disable for OCI model families that
            reject strict mode.
    """

    model: Any = Field(description="A locus BaseModel-compatible chat model.")
    system_prompt: str = Field(default=_DEFAULT_SYSTEM_PROMPT)
    strict: bool = Field(default=True)

    model_config = {"arbitrary_types_allowed": True}

    async def judge(
        self,
        *,
        report_synthesis: str,
        evidence_corpus: str,
        **kwargs: Any,
    ) -> JudgeOutput:
        """Run the judge — see :class:`BaseGSARJudge` for the contract."""
        from locus.core.messages import Message
        from locus.core.structured import build_response_format, parse_structured

        messages: list[Message] = [
            Message.system(self.system_prompt),
            Message.user(
                _USER_TEMPLATE.format(
                    synthesis=report_synthesis,
                    evidence=evidence_corpus,
                )
            ),
        ]

        try:
            response = await self.model.complete(
                messages=messages,
                tools=None,
                response_format=build_response_format(JudgeOutput, strict=self.strict),
            )
        except Exception as exc:  # noqa: BLE001 — paper §6 "Robustness"
            return safe_default_judge_output(f"judge model call failed: {type(exc).__name__}")

        content = response.message.content or "{}"
        parsed = parse_structured(content, JudgeOutput, strict=False)
        if parsed.success and parsed.parsed is not None:
            # ``parse_structured`` returns ``BaseModel | None``; the schema
            # we passed in pins the dynamic type to ``JudgeOutput``.
            return cast("JudgeOutput", parsed.parsed)
        return safe_default_judge_output(f"judge output parse failed: {parsed.error or 'unknown'}")


# Re-export so callers don't have to import EvidenceType separately when
# they're already importing the judge module.
__all__ = [
    "BaseGSARJudge",
    "Claim",
    "EvidenceType",
    "JudgeOutput",
    "Partition",
    "StructuredOutputGSARJudge",
    "safe_default_judge_output",
]
