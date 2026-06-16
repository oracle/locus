# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``Agent(gsar=GSARConfig(...))`` integration.

Covers:

- ``GSARConfig`` accepts the documented kwargs and validates threshold
  ordering.
- When ``gsar`` is unset, ``AgentResult.gsar_*`` fields stay ``None``.
- When ``gsar`` is set with a scripted judge, the judge sees the
  agent's final answer + tool-execution history as evidence, and the
  result surfaces ``gsar_judgment``, ``gsar_score``, ``gsar_decision``.
- A judge raising an exception falls back to ``(None, None, None)``
  on the result rather than crashing the agent (paper §6 "Robustness").
- ``contradiction_penalty`` and ``weight_map`` overrides are applied
  when scoring the partition.
"""

from __future__ import annotations

from typing import Any

import pytest

from locus.agent import Agent
from locus.agent.config import AgentConfig, GSARConfig
from locus.core.messages import Message, ToolCall
from locus.models.base import ModelResponse
from locus.reasoning.gsar import Claim, EvidenceType
from locus.reasoning.gsar_judge import JudgeOutput
from locus.tools.decorator import tool


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _ScriptedModel:
    """Returns one or more ModelResponses; tracks how many calls happened."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls += 1
        if not self._responses:
            raise AssertionError("scripted model exhausted")
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)

    async def stream(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError


class _RecordingJudge:
    """Records what the agent passes to the judge and returns a fixed payload."""

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


class _RaisingJudge:
    """Always raises — exercises the §6 'Robustness' fallback."""

    async def judge(self, **_: Any) -> JudgeOutput:
        raise RuntimeError("simulated judge failure")


def _assistant(content: str | None, *, tool_calls: list[ToolCall] | None = None) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(content=content, tool_calls=tool_calls or []),
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )


# ---------------------------------------------------------------------------
# GSARConfig validation
# ---------------------------------------------------------------------------


class TestGSARConfig:
    def test_defaults_match_appendix_b(self) -> None:
        cfg = GSARConfig()
        assert cfg.contradiction_penalty == 0.5
        assert cfg.tau_proceed == 0.80
        assert cfg.tau_regenerate == 0.65
        assert cfg.judge is None
        assert cfg.weight_map is None
        assert cfg.fail_on_low_score is False

    def test_threshold_ordering_enforced(self) -> None:
        with pytest.raises(ValueError):
            GSARConfig(tau_proceed=0.5, tau_regenerate=0.6)
        with pytest.raises(ValueError):
            GSARConfig(tau_proceed=0.5, tau_regenerate=0.5)

    def test_rho_range_validated(self) -> None:
        with pytest.raises(ValueError):
            GSARConfig(contradiction_penalty=-0.1)
        with pytest.raises(ValueError):
            GSARConfig(contradiction_penalty=1.1)


# ---------------------------------------------------------------------------
# Default behaviour: gsar unset → fields stay None
# ---------------------------------------------------------------------------


class TestGSARUnsetDefault:
    def test_run_sync_leaves_gsar_fields_none_when_unset(self) -> None:
        model = _ScriptedModel([_assistant("hello")])
        agent = Agent(model=model, system_prompt="say hello")
        result = agent.run_sync("hi")
        assert result.gsar_judgment is None
        assert result.gsar_score is None
        assert result.gsar_decision is None


# ---------------------------------------------------------------------------
# Happy path: judge runs, AgentResult carries the verdict
# ---------------------------------------------------------------------------


class TestGSARSurfacedOnAgentResult:
    def test_proceed_decision_surfaces(self) -> None:
        model = _ScriptedModel([_assistant("the answer is 42")])
        # All-grounded payload → S=1.0 → δ=proceed.
        judge = _RecordingJudge(
            JudgeOutput(
                grounding_score=1.0,
                is_grounded=True,
                grounded_claims=[Claim(text="the answer is 42", type=EvidenceType.TOOL_MATCH)],
            )
        )
        agent = Agent(
            model=model,
            system_prompt="answer the question",
            gsar=GSARConfig(judge=judge),
        )
        result = agent.run_sync("what is the answer?")
        assert result.gsar_judgment is not None
        assert result.gsar_score == pytest.approx(1.0)
        assert result.gsar_decision == "proceed"

    def test_replan_decision_surfaces(self) -> None:
        model = _ScriptedModel([_assistant("an unsupported claim")])
        # Judge marks the answer as ungrounded only → S=0.0 → δ=replan.
        judge = _RecordingJudge(
            JudgeOutput(
                grounding_score=0.0,
                is_grounded=False,
                ungrounded_claims=[Claim(text="an unsupported claim", type=EvidenceType.INFERENCE)],
            )
        )
        agent = Agent(
            model=model,
            system_prompt="answer",
            gsar=GSARConfig(judge=judge),
        )
        result = agent.run_sync("anything")
        assert result.gsar_decision == "replan"
        assert result.gsar_score == pytest.approx(0.0)

    def test_abstain_decision_surfaces(self) -> None:
        model = _ScriptedModel([_assistant("inscrutable")])
        judge = _RecordingJudge(
            JudgeOutput(
                grounding_score=0.5,
                is_grounded=False,
                decision_status="abstain",
                abstain_reason="under-evidenced",
            )
        )
        agent = Agent(
            model=model,
            system_prompt="answer",
            gsar=GSARConfig(judge=judge),
        )
        result = agent.run_sync("anything")
        assert result.gsar_decision == "abstain"


# ---------------------------------------------------------------------------
# Evidence corpus assembly
# ---------------------------------------------------------------------------


_tool_calls_done: int = 0


@tool(name="fake_lookup")
def _fake_lookup(query: str) -> str:
    """Return a fixed string so the agent has a tool execution to evidence."""
    global _tool_calls_done
    _tool_calls_done += 1
    return f"lookup({query!r}) → 42"


class TestGSAREvidenceCorpusAssembly:
    def test_tool_executions_make_it_into_evidence(self) -> None:
        # Two-step model: first response calls the tool; second response
        # returns the final answer.
        tc = ToolCall(id="tc-1", name="fake_lookup", arguments={"query": "foo"})
        responses = [
            _assistant(content=None, tool_calls=[tc]),
            _assistant("found 42"),
        ]
        model = _ScriptedModel(responses)

        judge = _RecordingJudge(JudgeOutput(grounding_score=1.0, is_grounded=True))
        agent = Agent(
            model=model,
            tools=[_fake_lookup],
            system_prompt="use the tool",
            gsar=GSARConfig(judge=judge),
        )
        agent.run_sync("hi")

        # Judge was called exactly once, and the evidence corpus
        # contains the tool's name + result.
        assert len(judge.calls) == 1
        synthesis, evidence = judge.calls[0]
        assert "found 42" in synthesis
        assert "fake_lookup" in evidence
        assert "42" in evidence

    def test_no_tool_executions_yields_placeholder_evidence(self) -> None:
        model = _ScriptedModel([_assistant("just chatting")])
        judge = _RecordingJudge(JudgeOutput(grounding_score=1.0, is_grounded=True))
        agent = Agent(
            model=model,
            system_prompt="chat",
            gsar=GSARConfig(judge=judge),
        )
        agent.run_sync("hi")
        _, evidence = judge.calls[0]
        assert "no tool executions" in evidence


# ---------------------------------------------------------------------------
# Robustness: judge failure must not crash the agent
# ---------------------------------------------------------------------------


class TestGSARRobustness:
    def test_judge_exception_yields_none_fields(self) -> None:
        model = _ScriptedModel([_assistant("answer")])
        agent = Agent(
            model=model,
            system_prompt="answer",
            gsar=GSARConfig(judge=_RaisingJudge()),
        )
        result = agent.run_sync("anything")
        # Agent still returned a result; GSAR fields are None.
        assert result.message == "answer"
        assert result.gsar_judgment is None
        assert result.gsar_score is None
        assert result.gsar_decision is None


# ---------------------------------------------------------------------------
# Score recomputation honours config overrides
# ---------------------------------------------------------------------------


class TestGSARScoreRecomputation:
    def test_rho_zero_inflates_score_under_contradicted_partition(self) -> None:
        model = _ScriptedModel([_assistant("answer")])
        # Partition with contradicted mass — under default ρ=0.5 the
        # denominator includes 0.5·W(X); under ρ=0 it's 0 (paper P5).
        judge_payload = JudgeOutput(
            grounding_score=0.0,
            is_grounded=True,
            grounded_claims=[Claim(text="g", type=EvidenceType.TOOL_MATCH)],
            contradicted_claims=[Claim(text="x", type=EvidenceType.SPECIFIC_DATA)],
        )

        a_default = Agent(
            model=_ScriptedModel([_assistant("answer")]),
            system_prompt="x",
            gsar=GSARConfig(judge=_RecordingJudge(judge_payload)),
        )
        a_rho_zero = Agent(
            model=_ScriptedModel([_assistant("answer")]),
            system_prompt="x",
            gsar=GSARConfig(judge=_RecordingJudge(judge_payload), contradiction_penalty=0.0),
        )
        r_default = a_default.run_sync("hi")
        r_rho_zero = a_rho_zero.run_sync("hi")
        assert r_rho_zero.gsar_score is not None
        assert r_default.gsar_score is not None
        assert r_rho_zero.gsar_score > r_default.gsar_score

    def test_custom_thresholds_change_decision(self) -> None:
        # Partition: 2 grounded tool_match (W=2.0) + 1 ungrounded inference
        # (W=0.6) → S = 2.0 / 2.6 ≈ 0.769. Above default τ_regenerate=0.65
        # but below default τ_proceed=0.80; with strict τ_proceed=0.95
        # it falls into regenerate; with lenient τ_proceed=0.60 it
        # crosses into proceed.
        judge_payload = JudgeOutput(
            grounding_score=0.0,
            is_grounded=True,
            grounded_claims=[
                Claim(text="g1", type=EvidenceType.TOOL_MATCH),
                Claim(text="g2", type=EvidenceType.TOOL_MATCH),
            ],
            ungrounded_claims=[Claim(text="u", type=EvidenceType.INFERENCE)],
        )

        agent_strict = Agent(
            model=_ScriptedModel([_assistant("answer")]),
            system_prompt="x",
            gsar=GSARConfig(
                judge=_RecordingJudge(judge_payload),
                tau_proceed=0.95,
                tau_regenerate=0.65,
            ),
        )
        agent_lenient = Agent(
            model=_ScriptedModel([_assistant("answer")]),
            system_prompt="x",
            gsar=GSARConfig(
                judge=_RecordingJudge(judge_payload),
                tau_proceed=0.60,
                tau_regenerate=0.40,
            ),
        )
        r_strict = agent_strict.run_sync("hi")
        r_lenient = agent_lenient.run_sync("hi")
        # Same score, different decision tier under different τ.
        assert r_strict.gsar_score == pytest.approx(r_lenient.gsar_score)
        assert r_strict.gsar_decision == "regenerate"
        assert r_lenient.gsar_decision == "proceed"


# ---------------------------------------------------------------------------
# AgentConfig field plumbing
# ---------------------------------------------------------------------------


class TestAgentConfigPlumbing:
    def test_agent_config_accepts_gsar_kwarg(self) -> None:
        cfg = AgentConfig(
            model="openai:gpt-4o-mini",
            gsar=GSARConfig(contradiction_penalty=0.3),
        )
        assert isinstance(cfg.gsar, GSARConfig)
        assert cfg.gsar.contradiction_penalty == 0.3

    def test_agent_init_accepts_gsar_kwarg(self) -> None:
        # The Agent.__init__ kwargs path uses **kwargs → AgentConfig.
        # Make sure it propagates.
        agent = Agent(
            model=_ScriptedModel([_assistant("ok")]),
            system_prompt="hi",
            gsar=GSARConfig(tau_proceed=0.95, tau_regenerate=0.5),
        )
        assert agent.config.gsar is not None
        assert agent.config.gsar.tau_proceed == 0.95
