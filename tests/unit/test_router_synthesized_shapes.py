# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``specialist_fanout`` and ``handoff_chain`` end with a synthesis step.

Each agent of these shapes has one tool, so their outputs joined together
are findings, not an answer: the result the caller reads is written by a
last agent from every finding.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from locus.agent.composition import ParallelPipeline, PipelineResult, SequentialPipeline
from locus.router import (
    BuilderContext,
    CapabilityIndex,
    Complexity,
    GoalFrame,
    PipelineRunnable,
    ProtocolExecutionError,
    Risk,
    SynthesizedPipelineRunnable,
    TaskType,
    builtin_protocols,
)
from locus.testing import ScriptedModel
from locus.tools.decorator import tool
from locus.tools.registry import create_registry


@tool
def get_metric(name: str) -> str:
    """Return the latest value of a named metric."""
    return f"{name}=1"


@tool
def list_alerts() -> str:
    """List the alerts that are firing."""
    return "none"


def _frame(goal: TaskType = TaskType.DIAGNOSE) -> GoalFrame:
    return GoalFrame(
        primary_goal=goal, domain="observability", complexity=Complexity.MEDIUM, risk=Risk.LOW
    )


def _index() -> CapabilityIndex:
    index = CapabilityIndex(create_registry(get_metric, list_alerts))
    index.annotate("metric_probe", tool_name="get_metric", description="m", domain="observability")
    index.annotate("alert_list", tool_name="list_alerts", description="a", domain="observability")
    return index


def _build(protocol_id: str, capability_ids: list[str], goal: TaskType) -> Any:
    index = _index()
    protocol = next(p for p in builtin_protocols() if p.id == protocol_id)
    ctx = BuilderContext(model=ScriptedModel([]), capabilities=index)
    return protocol.builder(_frame(goal), index.lookup(capability_ids), ctx)


class _Writer:
    """A synthesizer double: records the prompt it was given and answers."""

    def __init__(self, answer: str = "the answer") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def arun(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        return SimpleNamespace(message=self.answer)


class _Agent:
    def __init__(self, reply: str = "", *, raises: bool = False) -> None:
        self.reply = reply
        self.raises = raises
        self.prompts: list[str] = []

    async def arun(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        if self.raises:
            raise RuntimeError("endpoint refused")
        return SimpleNamespace(message=self.reply)


# ---------------------------------------------------------------- runnable --


async def test_a_fan_out_answers_with_the_synthesis_of_every_finding() -> None:
    metric, alerts, writer = (
        _Agent("latency_p99 is 1840 ms"),
        _Agent("HighLatency firing"),
        _Writer(),
    )
    runnable = SynthesizedPipelineRunnable(
        pipeline=ParallelPipeline(agents=[metric, alerts]),
        synthesizer=writer,
        protocol_id="specialist_fanout",
        frame=_frame(),
        labels=["metric_probe", "alert_list"],
        part_prompt="Request: {task}\n\nYour part: {label}.",
    )

    result = await runnable.execute("Why is checkout slow?")

    assert result.text == "the answer"
    assert result.protocol_id == "specialist_fanout"
    # Every agent's finding stays on the raw result.
    assert result.raw.outputs == ["latency_p99 is 1840 ms", "HighLatency firing"]
    # Each specialist is told its own part, not the bare request.
    assert metric.prompts == ["Request: Why is checkout slow?\n\nYour part: metric_probe."]
    assert alerts.prompts == ["Request: Why is checkout slow?\n\nYour part: alert_list."]
    # The synthesizer reads the request and each finding under its label.
    (prompt,) = writer.prompts
    assert prompt.startswith("Request:\nWhy is checkout slow?")
    assert "[metric_probe]\nlatency_p99 is 1840 ms" in prompt
    assert "[alert_list]\nHighLatency firing" in prompt


async def test_a_chain_answers_from_every_link_not_the_last_one() -> None:
    first, second, writer = _Agent("alerts: QueueBacklog"), _Agent("queue_depth 182000"), _Writer()
    runnable = SynthesizedPipelineRunnable(
        pipeline=SequentialPipeline(agents=[first, second]),
        synthesizer=writer,
        protocol_id="handoff_chain",
        frame=_frame(TaskType.COORDINATE),
        labels=["alert_list", "metric_probe"],
    )

    result = await runnable.execute("Alerts first, then metrics, then summarise.")

    assert result.text == "the answer"
    # Without a part prompt the pipeline chains as it always did.
    assert first.prompts == ["Alerts first, then metrics, then summarise."]
    assert "alerts: QueueBacklog" in second.prompts[0]
    (prompt,) = writer.prompts
    assert "[alert_list]\nalerts: QueueBacklog" in prompt
    assert "[metric_probe]\nqueue_depth 182000" in prompt


async def test_a_partial_failure_is_synthesised_from_the_findings_that_arrived() -> None:
    writer = _Writer()
    runnable = SynthesizedPipelineRunnable(
        pipeline=ParallelPipeline(agents=[_Agent(raises=True), _Agent("HighLatency firing")]),
        synthesizer=writer,
        protocol_id="specialist_fanout",
        frame=_frame(),
        labels=["metric_probe", "alert_list"],
    )

    result = await runnable.execute("Why is checkout slow?")

    assert result.text == "the answer"
    assert result.raw.success is False
    assert "[metric_probe]" not in writer.prompts[0]
    assert "[alert_list]\nHighLatency firing" in writer.prompts[0]


async def test_no_finding_at_all_is_a_failed_dispatch_and_nothing_is_synthesised() -> None:
    writer = _Writer()
    runnable = SynthesizedPipelineRunnable(
        pipeline=ParallelPipeline(agents=[_Agent(raises=True), _Agent(raises=True)]),
        synthesizer=writer,
        protocol_id="specialist_fanout",
        frame=_frame(),
        labels=["metric_probe", "alert_list"],
    )

    with pytest.raises(ProtocolExecutionError, match=r"specialist_fanout.*agent\[0\]"):
        await runnable.execute("Why is checkout slow?")
    assert writer.prompts == []


async def test_an_agent_without_a_label_is_numbered() -> None:
    async def run(_task: str) -> PipelineResult:
        return PipelineResult(outputs=["x", "y"], final_output="y")

    writer = _Writer()
    runnable = SynthesizedPipelineRunnable(
        pipeline=SimpleNamespace(run=run),
        synthesizer=writer,
        protocol_id="handoff_chain",
        frame=_frame(),
        labels=["alert_list"],
    )

    await runnable.execute("task")

    assert "[alert_list]\nx" in writer.prompts[0]
    assert "[agent 2]\ny" in writer.prompts[0]


# ---------------------------------------------------------------- builders --


def test_the_fan_out_builder_adds_a_tool_less_synthesizer() -> None:
    runnable = _build("specialist_fanout", ["metric_probe", "alert_list"], TaskType.DIAGNOSE)

    assert isinstance(runnable, SynthesizedPipelineRunnable)
    assert isinstance(runnable, PipelineRunnable)
    assert isinstance(runnable.pipeline, ParallelPipeline)
    assert len(runnable.pipeline.agents) == 2
    assert runnable.labels == ["metric_probe", "alert_list"]
    assert runnable.part_prompt is not None
    assert "{task}" in runnable.part_prompt
    assert "{label}" in runnable.part_prompt
    assert runnable.synthesizer.config.tools == []
    assert "Use only data the findings contain" in runnable.synthesizer.config.system_prompt
    for agent in runnable.pipeline.agents:
        assert len(agent.config.tools) == 1
        assert "do not comment on the parts" in agent.config.system_prompt


def test_the_chain_builder_adds_a_tool_less_synthesizer() -> None:
    runnable = _build("handoff_chain", ["alert_list", "metric_probe"], TaskType.COORDINATE)

    assert isinstance(runnable, SynthesizedPipelineRunnable)
    assert isinstance(runnable.pipeline, SequentialPipeline)
    assert runnable.labels == ["alert_list", "metric_probe"]
    assert runnable.part_prompt is None
    assert runnable.synthesizer.config.tools == []


def test_a_chain_with_no_capability_stays_a_single_answering_agent() -> None:
    runnable = _build("handoff_chain", [], TaskType.COORDINATE)

    assert isinstance(runnable, PipelineRunnable)
    assert not isinstance(runnable, SynthesizedPipelineRunnable)
    assert len(runnable.pipeline.agents) == 1
