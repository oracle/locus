# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the GSAR judge protocol + reference judge.

Covers:

- :class:`JudgeOutput` schema matches Appendix C — every field is
  present, decision_status validation fires, abstain_reason is
  required when abstaining.
- :func:`safe_default_judge_output` returns the exact §6 'Robustness'
  fallback shape (s = 0.5, empty partition, abstain).
- :class:`StructuredOutputGSARJudge` parses well-formed model JSON,
  falls back to safe-default on malformed JSON, and falls back on
  model-call exceptions.
- :class:`BaseGSARJudge` accepts duck-typed fakes via the
  ``runtime_checkable`` Protocol.
"""

from __future__ import annotations

from typing import Any

import pytest

from locus.core.messages import Message
from locus.models.base import ModelResponse
from locus.reasoning.gsar import (
    NEUTRAL_SCORE_ON_EMPTY,
    Claim,
    EvidenceType,
)
from locus.reasoning.gsar_judge import (
    BaseGSARJudge,
    JudgeOutput,
    StructuredOutputGSARJudge,
    safe_default_judge_output,
)


# ---------------------------------------------------------------------------
# Schema (Appendix C)
# ---------------------------------------------------------------------------


class TestJudgeOutputSchema:
    def test_minimal_construction(self) -> None:
        out = JudgeOutput(grounding_score=0.9, is_grounded=True)
        assert out.grounded_claims == []
        assert out.decision_status == "resolved"
        assert out.abstain_reason is None

    def test_full_partition(self) -> None:
        out = JudgeOutput(
            grounding_score=0.7,
            is_grounded=True,
            grounded_claims=[Claim(text="g", type=EvidenceType.TOOL_MATCH)],
            ungrounded_claims=[Claim(text="u", type=EvidenceType.INFERENCE)],
            contradicted_claims=[Claim(text="x", type=EvidenceType.SPECIFIC_DATA)],
            complementary_claims=[Claim(text="k", type=EvidenceType.COMPLEMENTARY_FINDING)],
            explanation="mostly fine",
        )
        partition = out.to_partition()
        assert partition.total_claims == 4
        assert not out.abstained

    def test_score_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            JudgeOutput(grounding_score=1.5, is_grounded=True)
        with pytest.raises(ValueError):
            JudgeOutput(grounding_score=-0.1, is_grounded=False)

    def test_abstain_requires_reason(self) -> None:
        with pytest.raises(ValueError):
            JudgeOutput(
                grounding_score=0.5,
                is_grounded=False,
                decision_status="abstain",
                # missing abstain_reason
            )

    def test_decision_status_value_validation(self) -> None:
        with pytest.raises(ValueError):
            JudgeOutput(
                grounding_score=0.5,
                is_grounded=False,
                decision_status="bogus",
            )

    def test_round_trip(self) -> None:
        out = JudgeOutput(
            grounding_score=0.5,
            is_grounded=False,
            decision_status="abstain",
            abstain_reason="under-evidenced",
        )
        reloaded = JudgeOutput.model_validate_json(out.model_dump_json())
        assert reloaded == out
        assert reloaded.abstained


# ---------------------------------------------------------------------------
# Safe default
# ---------------------------------------------------------------------------


class TestSafeDefault:
    def test_shape(self) -> None:
        out = safe_default_judge_output("test reason")
        assert out.grounding_score == NEUTRAL_SCORE_ON_EMPTY
        assert out.decision_status == "abstain"
        assert out.abstain_reason == "test reason"
        assert out.to_partition().is_empty


# ---------------------------------------------------------------------------
# Protocol runtime-checkability
# ---------------------------------------------------------------------------


class _FakeJudge:
    """Duck-typed BaseGSARJudge — must pass isinstance via runtime_checkable."""

    def __init__(self, output: JudgeOutput) -> None:
        self.output = output
        self.calls: list[tuple[str, str]] = []

    async def judge(
        self,
        *,
        report_synthesis: str,
        evidence_corpus: str,
        **_: Any,
    ) -> JudgeOutput:
        self.calls.append((report_synthesis, evidence_corpus))
        return self.output


class TestProtocol:
    def test_fake_passes_isinstance(self) -> None:
        out = JudgeOutput(grounding_score=1.0, is_grounded=True)
        assert isinstance(_FakeJudge(out), BaseGSARJudge)


# ---------------------------------------------------------------------------
# StructuredOutputGSARJudge — parsing + fallbacks
# ---------------------------------------------------------------------------


class _ScriptedModel:
    """Returns a fixed string content (or raises) — used to drive the judge."""

    def __init__(self, content: str | None = None, *, exc: Exception | None = None) -> None:
        self.content = content
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if self.exc is not None:
            raise self.exc
        return ModelResponse(
            message=Message.assistant(content=self.content),
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )

    async def stream(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError


class TestStructuredOutputJudge:
    @pytest.mark.asyncio
    async def test_parses_well_formed_output(self) -> None:
        payload = JudgeOutput(
            grounding_score=0.92,
            is_grounded=True,
            grounded_claims=[Claim(text="ok", type=EvidenceType.TOOL_MATCH)],
            explanation="fine",
        )
        model = _ScriptedModel(content=payload.model_dump_json())
        judge = StructuredOutputGSARJudge(model=model)
        out = await judge.judge(report_synthesis="report", evidence_corpus="evidence")
        assert out.grounding_score == 0.92
        assert out.grounded_claims[0].text == "ok"

    @pytest.mark.asyncio
    async def test_falls_back_on_malformed_json(self) -> None:
        model = _ScriptedModel(content="not json at all")
        judge = StructuredOutputGSARJudge(model=model)
        out = await judge.judge(report_synthesis="r", evidence_corpus="e")
        assert out.abstained
        assert "parse failed" in (out.abstain_reason or "")

    @pytest.mark.asyncio
    async def test_falls_back_on_model_exception(self) -> None:
        model = _ScriptedModel(exc=RuntimeError("boom"))
        judge = StructuredOutputGSARJudge(model=model)
        out = await judge.judge(report_synthesis="r", evidence_corpus="e")
        assert out.abstained
        assert "model call failed" in (out.abstain_reason or "")
        assert "RuntimeError" in (out.abstain_reason or "")

    @pytest.mark.asyncio
    async def test_passes_response_format(self) -> None:
        payload = JudgeOutput(grounding_score=1.0, is_grounded=True)
        model = _ScriptedModel(content=payload.model_dump_json())
        judge = StructuredOutputGSARJudge(model=model)
        await judge.judge(report_synthesis="r", evidence_corpus="e")
        rf = model.calls[0]["kwargs"].get("response_format")
        assert rf is not None
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "JudgeOutput"

    @pytest.mark.asyncio
    async def test_system_and_user_message_shape(self) -> None:
        payload = JudgeOutput(grounding_score=1.0, is_grounded=True)
        model = _ScriptedModel(content=payload.model_dump_json())
        judge = StructuredOutputGSARJudge(model=model)
        await judge.judge(
            report_synthesis="report-θ",
            evidence_corpus="evidence-E",
        )
        msgs = model.calls[0]["messages"]
        assert msgs[0].role == "system"
        assert "grounding judge" in (msgs[0].content or "").lower()
        assert msgs[1].role == "user"
        # Both report and evidence must reach the model.
        assert "report-θ" in (msgs[1].content or "")
        assert "evidence-E" in (msgs[1].content or "")
