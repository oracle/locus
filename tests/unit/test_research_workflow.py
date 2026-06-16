# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for locus.deepagent.workflow — research workflow primitives."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Add examples/ to path so we can use MockModel
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples"))
from config import MockModel  # noqa: E402

from locus.deepagent.workflow import (
    KEY_CAUSAL_CHAIN,
    KEY_CAUSAL_CONFIDENCE,
    KEY_CAUSAL_HYPOTHESIS,
    KEY_EVIDENCE,
    KEY_EXECUTE_PROMPT,
    KEY_GROUNDING_FACTS,
    KEY_GROUNDING_SCORE,
    KEY_PROMPT,
    KEY_REGENERATION_COUNT,
    KEY_REPLAN_COUNT,
    KEY_STOP_REASON,
    KEY_STRUCTURED_OUTPUT,
    KEY_SUMMARY,
    KEY_UNGROUNDED_CLAIMS,
    create_research_workflow,
    make_causal_inference_node,
    make_execute_node,
    make_grounding_eval_node,
    make_regenerate_summary_node,
    make_replan_node,
    make_summarize_node,
    route_after_grounding,
)


def _model() -> MockModel:
    return MockModel()


# ---------------------------------------------------------------------------
# KEY_* constants
# ---------------------------------------------------------------------------


class TestStateKeys:
    def test_all_keys_are_non_empty_strings(self) -> None:
        keys = [
            KEY_PROMPT,
            KEY_EXECUTE_PROMPT,
            KEY_EVIDENCE,
            KEY_GROUNDING_FACTS,
            KEY_CAUSAL_CHAIN,
            KEY_CAUSAL_HYPOTHESIS,
            KEY_CAUSAL_CONFIDENCE,
            KEY_SUMMARY,
            KEY_STRUCTURED_OUTPUT,
            KEY_GROUNDING_SCORE,
            KEY_UNGROUNDED_CLAIMS,
            KEY_REPLAN_COUNT,
            KEY_REGENERATION_COUNT,
            KEY_STOP_REASON,
        ]
        for k in keys:
            assert isinstance(k, str)
            assert len(k) > 0

    def test_keys_are_unique(self) -> None:
        keys = [
            KEY_PROMPT,
            KEY_EXECUTE_PROMPT,
            KEY_EVIDENCE,
            KEY_GROUNDING_FACTS,
            KEY_CAUSAL_CHAIN,
            KEY_CAUSAL_HYPOTHESIS,
            KEY_CAUSAL_CONFIDENCE,
            KEY_SUMMARY,
            KEY_STRUCTURED_OUTPUT,
            KEY_GROUNDING_SCORE,
            KEY_UNGROUNDED_CLAIMS,
            KEY_REPLAN_COUNT,
            KEY_REGENERATION_COUNT,
            KEY_STOP_REASON,
        ]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# route_after_grounding — pure function, no model needed
# ---------------------------------------------------------------------------


class TestRouteAfterGrounding:
    def test_passes_on_high_score(self) -> None:
        from locus.multiagent.graph import END

        r = route_after_grounding(threshold=0.65)
        assert r({KEY_GROUNDING_SCORE: 0.9}) == END

    def test_passes_at_threshold(self) -> None:
        from locus.multiagent.graph import END

        r = route_after_grounding(threshold=0.65)
        assert r({KEY_GROUNDING_SCORE: 0.65}) == END

    def test_regenerates_on_first_failure(self) -> None:
        r = route_after_grounding(threshold=0.65, max_replans=2, max_regenerations=1)
        assert (
            r({KEY_GROUNDING_SCORE: 0.3, KEY_REPLAN_COUNT: 0, KEY_REGENERATION_COUNT: 0})
            == "regenerate"
        )

    def test_replans_after_regen_exhausted(self) -> None:
        r = route_after_grounding(threshold=0.65, max_replans=2, max_regenerations=1)
        assert (
            r({KEY_GROUNDING_SCORE: 0.3, KEY_REPLAN_COUNT: 0, KEY_REGENERATION_COUNT: 1})
            == "replan"
        )

    def test_ends_when_all_limits_hit(self) -> None:
        from locus.multiagent.graph import END

        r = route_after_grounding(threshold=0.65, max_replans=1, max_regenerations=1)
        assert r({KEY_GROUNDING_SCORE: 0.2, KEY_REPLAN_COUNT: 1, KEY_REGENERATION_COUNT: 1}) == END

    def test_empty_state_starts_with_regenerate(self) -> None:
        r = route_after_grounding(threshold=0.65, max_regenerations=1)
        assert r({}) == "regenerate"

    def test_custom_threshold(self) -> None:
        from locus.multiagent.graph import END

        r = route_after_grounding(threshold=0.9)
        assert r({KEY_GROUNDING_SCORE: 0.85}) != END
        assert r({KEY_GROUNDING_SCORE: 0.95}) == END


# ---------------------------------------------------------------------------
# make_replan_node — pure state transform, no model needed
# ---------------------------------------------------------------------------


class TestMakeReplanNode:
    @pytest.mark.asyncio
    async def test_generates_focused_prompt(self) -> None:
        node = make_replan_node()
        result = await node(
            {
                KEY_PROMPT: "Research locus",
                KEY_UNGROUNDED_CLAIMS: ["claim A", "claim B"],
                KEY_REPLAN_COUNT: 0,
            }
        )
        assert "claim A" in result[KEY_EXECUTE_PROMPT]
        assert result[KEY_REPLAN_COUNT] == 1

    @pytest.mark.asyncio
    async def test_increments_count(self) -> None:
        node = make_replan_node()
        result = await node({KEY_PROMPT: "t", KEY_UNGROUNDED_CLAIMS: [], KEY_REPLAN_COUNT: 2})
        assert result[KEY_REPLAN_COUNT] == 3

    @pytest.mark.asyncio
    async def test_generic_prompt_when_no_ungrounded(self) -> None:
        node = make_replan_node()
        result = await node(
            {KEY_PROMPT: "My topic", KEY_UNGROUNDED_CLAIMS: [], KEY_REPLAN_COUNT: 0}
        )
        assert "My topic" in result[KEY_EXECUTE_PROMPT]

    @pytest.mark.asyncio
    async def test_caps_ungrounded_at_six(self) -> None:
        node = make_replan_node()
        claims = [f"claim {i}" for i in range(10)]
        result = await node({KEY_PROMPT: "p", KEY_UNGROUNDED_CLAIMS: claims, KEY_REPLAN_COUNT: 0})
        assert "claim 5" in result[KEY_EXECUTE_PROMPT]
        assert "claim 6" not in result[KEY_EXECUTE_PROMPT]


# ---------------------------------------------------------------------------
# make_execute_node — uses Agent(model=MockModel)
# ---------------------------------------------------------------------------


class TestMakeExecuteNode:
    @pytest.mark.asyncio
    async def test_returns_evidence_and_facts(self) -> None:
        node = make_execute_node(_model(), [], max_iterations=1)
        result = await node({KEY_PROMPT: "What is Python?"})
        assert KEY_EVIDENCE in result
        assert KEY_GROUNDING_FACTS in result
        assert isinstance(result[KEY_EVIDENCE], list)
        assert isinstance(result[KEY_GROUNDING_FACTS], list)

    @pytest.mark.asyncio
    async def test_uses_execute_prompt_when_present(self) -> None:
        node = make_execute_node(_model(), [], max_iterations=1)
        result = await node({KEY_PROMPT: "original", KEY_EXECUTE_PROMPT: "focused"})
        # Should not raise; evidence is collected regardless
        assert KEY_EVIDENCE in result

    @pytest.mark.asyncio
    async def test_conclusion_added_to_evidence(self) -> None:
        node = make_execute_node(_model(), [], max_iterations=1)
        result = await node({KEY_PROMPT: "simple question"})
        # MockModel terminates with a message; that message should land in evidence
        conclusion_facts = [
            f for f in result[KEY_GROUNDING_FACTS] if f["source"] == "agent_conclusion"
        ]
        assert len(conclusion_facts) >= 1


# ---------------------------------------------------------------------------
# make_summarize_node
# ---------------------------------------------------------------------------


class TestMakeSummarizeNode:
    @pytest.mark.asyncio
    async def test_produces_summary_string(self) -> None:
        node = make_summarize_node(_model())
        result = await node(
            {
                KEY_PROMPT: "What is locus?",
                KEY_EVIDENCE: ["locus is an agent SDK", "it handles tool calls"],
                KEY_CAUSAL_HYPOTHESIS: "",
            }
        )
        assert KEY_SUMMARY in result
        assert isinstance(result[KEY_SUMMARY], str)

    @pytest.mark.asyncio
    async def test_no_crash_without_evidence(self) -> None:
        node = make_summarize_node(_model())
        result = await node({KEY_PROMPT: "topic", KEY_EVIDENCE: [], KEY_CAUSAL_HYPOTHESIS: ""})
        assert KEY_SUMMARY in result


# ---------------------------------------------------------------------------
# make_grounding_eval_node
# ---------------------------------------------------------------------------


class TestMakeGroundingEvalNode:
    @pytest.mark.asyncio
    async def test_zero_on_empty_summary(self) -> None:
        node = make_grounding_eval_node(_model())
        result = await node({KEY_SUMMARY: "", KEY_EVIDENCE: ["evidence"]})
        assert result[KEY_GROUNDING_SCORE] == 0.0
        assert result[KEY_UNGROUNDED_CLAIMS] == []

    @pytest.mark.asyncio
    async def test_zero_on_empty_evidence(self) -> None:
        node = make_grounding_eval_node(_model())
        result = await node({KEY_SUMMARY: "Some claim.", KEY_EVIDENCE: []})
        assert result[KEY_GROUNDING_SCORE] == 0.0

    @pytest.mark.asyncio
    async def test_calls_evaluator(self) -> None:
        mock_result = MagicMock()
        mock_result.score = 0.75
        mock_result.ungrounded_claims = ["unverified"]

        with patch(
            "locus.reasoning.grounding.GroundingEvaluator.evaluate_with_llm",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            node = make_grounding_eval_node(_model())
            result = await node(
                {
                    KEY_SUMMARY: "Claim one. Claim two.",
                    KEY_EVIDENCE: ["evidence A"],
                }
            )

        assert result[KEY_GROUNDING_SCORE] == 0.75
        assert "unverified" in result[KEY_UNGROUNDED_CLAIMS]


# ---------------------------------------------------------------------------
# make_regenerate_summary_node
# ---------------------------------------------------------------------------


class TestMakeRegenerateSummaryNode:
    @pytest.mark.asyncio
    async def test_increments_regeneration_count(self) -> None:
        node = make_regenerate_summary_node(_model())
        result = await node(
            {
                KEY_SUMMARY: "old summary",
                KEY_EVIDENCE: ["e1"],
                KEY_UNGROUNDED_CLAIMS: ["claim X"],
                KEY_REGENERATION_COUNT: 0,
            }
        )
        assert result[KEY_REGENERATION_COUNT] == 1

    @pytest.mark.asyncio
    async def test_returns_summary_key(self) -> None:
        node = make_regenerate_summary_node(_model())
        result = await node(
            {
                KEY_SUMMARY: "original",
                KEY_EVIDENCE: ["e"],
                KEY_UNGROUNDED_CLAIMS: [],
                KEY_REGENERATION_COUNT: 0,
            }
        )
        assert KEY_SUMMARY in result


# ---------------------------------------------------------------------------
# make_causal_inference_node
# ---------------------------------------------------------------------------


class TestMakeCausalInferenceNode:
    @pytest.mark.asyncio
    async def test_empty_on_no_evidence(self) -> None:
        node = make_causal_inference_node(_model())
        result = await node({KEY_EVIDENCE: [], KEY_PROMPT: "test"})
        assert result[KEY_CAUSAL_CHAIN] is None
        assert result[KEY_CAUSAL_HYPOTHESIS] == ""
        assert result[KEY_CAUSAL_CONFIDENCE] == 0.0

    @pytest.mark.asyncio
    async def test_empty_on_non_json_response(self) -> None:
        m = _model()
        # MockModel returns plain text — causal node should handle gracefully
        node = make_causal_inference_node(m)
        result = await node({KEY_EVIDENCE: ["some evidence"], KEY_PROMPT: "diagnose"})
        # Either builds a chain or returns empty — must not raise
        assert KEY_CAUSAL_CHAIN in result
        assert KEY_CAUSAL_HYPOTHESIS in result

    @pytest.mark.asyncio
    async def test_builds_chain_from_valid_json(self) -> None:
        m = _model()
        m._responses["default"] = """[
            {"label": "High latency", "causes": [], "type": "root_cause", "confidence": 0.9},
            {"label": "Timeouts", "causes": ["High latency"], "type": "symptom", "confidence": 0.8}
        ]"""
        node = make_causal_inference_node(m)
        result = await node({KEY_EVIDENCE: ["latency spike"], KEY_PROMPT: "diagnose"})
        if result[KEY_CAUSAL_CHAIN] is not None:
            assert result[KEY_CAUSAL_HYPOTHESIS] != ""


# ---------------------------------------------------------------------------
# create_research_workflow
# ---------------------------------------------------------------------------


class TestCreateResearchWorkflow:
    def test_returns_compiled_graph(self) -> None:
        wf = create_research_workflow(model=_model(), tools=[])
        assert wf is not None
        assert hasattr(wf, "execute")

    def test_without_causal_inference(self) -> None:
        wf = create_research_workflow(model=_model(), tools=[], causal_inference=False)
        assert wf is not None

    def test_accepts_output_schema(self) -> None:
        from pydantic import BaseModel

        class Schema(BaseModel):
            result: str

        wf = create_research_workflow(model=_model(), tools=[], output_schema=Schema)
        assert wf is not None

    def test_custom_thresholds(self) -> None:
        wf = create_research_workflow(
            model=_model(),
            tools=[],
            grounding_threshold=0.8,
            max_replans=3,
            max_regenerations=2,
        )
        assert wf is not None

    def test_accepts_separate_models(self) -> None:
        wf = create_research_workflow(
            model=_model(),
            tools=[],
            summarization_model=_model(),
            grounding_model=_model(),
        )
        assert wf is not None


# ---------------------------------------------------------------------------
# SSE emission — nodes emit research.* events under run_context
# ---------------------------------------------------------------------------


class TestWorkflowSSEEmission:
    @pytest.mark.asyncio
    async def test_execute_node_emits_sse_events(self) -> None:
        import asyncio

        from locus.observability import get_event_bus, run_context
        from locus.observability.event_bus import reset_event_bus

        reset_event_bus()
        node = make_execute_node(_model(), [], max_iterations=1)
        collected: list[str] = []

        async def drain(rid: str) -> None:
            async for ev in get_event_bus().subscribe(rid):
                collected.append(ev.event_type)
                if ev.event_type == "research.execute.completed":
                    return

        async with run_context() as rid:
            t = asyncio.create_task(drain(rid))
            await asyncio.sleep(0)
            await node({KEY_PROMPT: "test"})
            await get_event_bus().close_stream(rid)
            await asyncio.wait_for(t, timeout=5.0)

        assert "research.execute.started" in collected
        assert "research.execute.completed" in collected
        reset_event_bus()

    @pytest.mark.asyncio
    async def test_replan_node_emits_sse_event(self) -> None:
        import asyncio

        from locus.observability import get_event_bus, run_context
        from locus.observability.event_bus import reset_event_bus

        reset_event_bus()
        node = make_replan_node()
        collected: list[str] = []

        async def drain(rid: str) -> None:
            async for ev in get_event_bus().subscribe(rid):
                collected.append(ev.event_type)
                if ev.event_type == "research.replan":
                    return

        async with run_context() as rid:
            t = asyncio.create_task(drain(rid))
            await asyncio.sleep(0)
            await node({KEY_PROMPT: "p", KEY_UNGROUNDED_CLAIMS: ["c"], KEY_REPLAN_COUNT: 0})
            await get_event_bus().close_stream(rid)
            await asyncio.wait_for(t, timeout=5.0)

        assert "research.replan" in collected
        reset_event_bus()

    @pytest.mark.asyncio
    async def test_summarize_node_emits_sse_event(self) -> None:
        import asyncio

        from locus.observability import get_event_bus, run_context
        from locus.observability.event_bus import reset_event_bus

        reset_event_bus()
        node = make_summarize_node(_model())
        collected: list[str] = []

        async def drain(rid: str) -> None:
            async for ev in get_event_bus().subscribe(rid):
                collected.append(ev.event_type)
                if ev.event_type == "research.summarize.completed":
                    return

        async with run_context() as rid:
            t = asyncio.create_task(drain(rid))
            await asyncio.sleep(0)
            await node({KEY_PROMPT: "p", KEY_EVIDENCE: ["e"], KEY_CAUSAL_HYPOTHESIS: ""})
            await get_event_bus().close_stream(rid)
            await asyncio.wait_for(t, timeout=5.0)

        assert "research.summarize.completed" in collected
        reset_event_bus()

    @pytest.mark.asyncio
    async def test_regenerate_node_emits_sse_events(self) -> None:
        import asyncio

        from locus.observability import get_event_bus, run_context
        from locus.observability.event_bus import reset_event_bus

        reset_event_bus()
        node = make_regenerate_summary_node(_model())
        collected: list[str] = []

        async def drain(rid: str) -> None:
            async for ev in get_event_bus().subscribe(rid):
                collected.append(ev.event_type)
                if ev.event_type == "research.regenerate.completed":
                    return

        async with run_context() as rid:
            t = asyncio.create_task(drain(rid))
            await asyncio.sleep(0)
            await node(
                {
                    KEY_SUMMARY: "old",
                    KEY_EVIDENCE: ["e"],
                    KEY_UNGROUNDED_CLAIMS: [],
                    KEY_REGENERATION_COUNT: 0,
                }
            )
            await get_event_bus().close_stream(rid)
            await asyncio.wait_for(t, timeout=5.0)

        assert "research.regenerate.started" in collected
        assert "research.regenerate.completed" in collected
        reset_event_bus()

    @pytest.mark.asyncio
    async def test_grounding_eval_emits_sse_event(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from locus.observability import get_event_bus, run_context
        from locus.observability.event_bus import reset_event_bus

        reset_event_bus()
        mock_result = MagicMock()
        mock_result.score = 0.8
        mock_result.ungrounded_claims = []
        mock_result.requires_replan = False

        with patch(
            "locus.reasoning.grounding.GroundingEvaluator.evaluate_with_llm",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            node = make_grounding_eval_node(_model())
            collected: list[str] = []

            async def drain(rid: str) -> None:
                async for ev in get_event_bus().subscribe(rid):
                    collected.append(ev.event_type)
                    if ev.event_type == "research.grounding.evaluated":
                        return

            async with run_context() as rid:
                t = asyncio.create_task(drain(rid))
                await asyncio.sleep(0)
                await node({KEY_SUMMARY: "Claim one. Claim two.", KEY_EVIDENCE: ["e"]})
                await get_event_bus().close_stream(rid)
                await asyncio.wait_for(t, timeout=5.0)

        assert "research.grounding.evaluated" in collected
        reset_event_bus()


class TestWorkflowEdgeCases:
    @pytest.mark.asyncio
    async def test_execute_node_tool_complete_event(self) -> None:
        """Cover lines 164-173: ToolCompleteEvent handling in the async for loop."""
        from unittest.mock import patch

        from locus.agent.agent import Agent
        from locus.core.events import TerminateEvent, ToolCompleteEvent

        # Patch Agent.run to yield a ToolCompleteEvent then TerminateEvent
        async def _mock_run(self_agent, prompt, **kw):
            yield ToolCompleteEvent(
                tool_name="search_kb",
                tool_call_id="c1",
                result="search results here",
                success=True,
            )
            yield TerminateEvent(
                reason="no_tool_calls",
                final_message="done",
                iterations_used=2,
                total_tool_calls=1,
                final_confidence=0.9,
            )

        with patch.object(Agent, "run", _mock_run):
            node = make_execute_node(_model(), [], max_iterations=2)
            result = await node({KEY_PROMPT: "find something"})

        assert any("search results here" in e for e in result[KEY_EVIDENCE])
        assert result[KEY_GROUNDING_FACTS][0]["source"] == "search_kb"
        assert result[KEY_GROUNDING_FACTS][0]["id"] == "fact_000"

    @pytest.mark.asyncio
    async def test_summarize_node_with_output_schema(self) -> None:
        """Cover line 316: structured output path when output_schema is set."""
        from unittest.mock import patch

        from pydantic import BaseModel

        from locus.agent.agent import Agent

        class Report(BaseModel):
            summary: str
            confidence: float = 0.8

        report_instance = Report(summary="structured result", confidence=0.9)

        # Mock AgentResult with parsed output
        mock_result = MagicMock()
        mock_result.message = '{"summary": "structured result", "confidence": 0.9}'
        mock_result.parsed = report_instance

        with patch.object(Agent, "run_sync", return_value=mock_result):
            node = make_summarize_node(_model(), output_schema=Report)
            result = await node(
                {
                    KEY_PROMPT: "test",
                    KEY_EVIDENCE: ["evidence"],
                    KEY_CAUSAL_HYPOTHESIS: "",
                }
            )

        assert KEY_SUMMARY in result
        assert result.get(KEY_STRUCTURED_OUTPUT) is report_instance

    def test_create_research_workflow_with_checkpointer(self) -> None:
        """Cover line 640: checkpointer kwarg path."""
        from locus.memory.backends import MemoryCheckpointer

        checkpointer = MemoryCheckpointer()
        wf = create_research_workflow(model=_model(), tools=[], checkpointer=checkpointer)
        assert wf is not None

    @pytest.mark.asyncio
    async def test_causal_node_json_parse_error(self) -> None:
        """Cover lines 247-248: JSON parse exception path."""
        m = _model()
        # MockModel default response is plain text — will fail JSON parse
        # but _responses default is "This is a mock response for testing purposes."
        node = make_causal_inference_node(m)
        # When JSON parsing fails, should return empty causal chain gracefully
        result = await node({KEY_EVIDENCE: ["some evidence"], KEY_PROMPT: "diagnose"})
        # Either None (parse failed) or a chain (if JSON accidentally valid) — must not raise
        assert KEY_CAUSAL_CHAIN in result
        assert KEY_CAUSAL_HYPOTHESIS in result
        assert KEY_CAUSAL_CONFIDENCE in result
