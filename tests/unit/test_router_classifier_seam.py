# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The two seams at the front of the router: who frames the ask, who breaks ties.

A `RequestClassifier` produces the `GoalFrame` — by default `AgentFrameExtractor`
wrapping an `Agent(output_schema=GoalFrame)` — and a `ProtocolPicker` resolves a
pick with more than one survivor. Both are protocols, so a stub of either flows
through the same `Router.dispatch` / `CognitiveCompiler.compile` path as the
shipped implementation, and nothing downstream can tell the difference.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from locus import Agent
from locus.router import (
    AgentFrameExtractor,
    CapabilityIndex,
    CognitiveCompiler,
    Complexity,
    GoalFrame,
    LLMProtocolPicker,
    PolicyGate,
    ProtocolPicker,
    ProtocolRegistry,
    RequestClassifier,
    Risk,
    Router,
    TaskType,
    builtin_protocols,
)
from locus.router.protocol import Protocol
from locus.router.runtime import FrameExtractionError
from locus.testing import ScriptedModel, text
from locus.tools.registry import create_registry


def _frame(
    goal: TaskType = TaskType.ANSWER,
    complexity: Complexity = Complexity.LOW,
    risk: Risk = Risk.LOW,
) -> GoalFrame:
    return GoalFrame(primary_goal=goal, domain="ops", complexity=complexity, risk=risk)


def _compiler(**overrides: Any) -> CognitiveCompiler:
    registry = ProtocolRegistry()
    registry.register_many(builtin_protocols())
    kwargs: dict[str, Any] = {
        "protocols": registry,
        "capabilities": CapabilityIndex(create_registry()),
        "policy": PolicyGate(),
        "model": ScriptedModel([text("done")], repeat_last=True),
    }
    kwargs.update(overrides)
    return CognitiveCompiler(**kwargs)


class _TableClassifier:
    """A closed lookup — the kind of classifier the seam exists for."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def classify(self, text: str) -> GoalFrame:
        self.seen.append(text)
        if text.lower().startswith("why"):
            return _frame(TaskType.DIAGNOSE, Complexity.HIGH, Risk.MEDIUM)
        return _frame()


def _scripted_extractor(payload: str) -> Agent:
    # repeat_last: the agent retries a reply its output schema rejects.
    return Agent(
        model=ScriptedModel([text(payload)], repeat_last=True),
        system_prompt="extract",
        output_schema=GoalFrame,
    )


_FRAME_JSON = json.dumps(
    {
        "primary_goal": "explain",
        "domain": "ops",
        "complexity": "low",
        "risk": "low",
    }
)


# --------------------------------------------------------------- classifier --


@pytest.mark.asyncio
async def test_a_stub_classifier_flows_through_dispatch() -> None:
    classifier = _TableClassifier()
    observed: list[GoalFrame] = []
    router = Router(classifier=classifier, compiler=_compiler(), on_frame=observed.append)

    result = await router.dispatch("What is our refund policy?")

    assert result.protocol_id == "direct_response"
    assert result.text == "done"
    assert classifier.seen == ["What is our refund policy?"]
    assert [f.primary_goal for f in observed] == [TaskType.ANSWER]
    assert router.extractor is None
    assert isinstance(classifier, RequestClassifier)


@pytest.mark.asyncio
async def test_the_classifier_frame_drives_the_shape() -> None:
    router = Router(classifier=_TableClassifier(), compiler=_compiler())

    runnable = await router.compile("Why did checkout latency double?")
    explanation = await router.explain("Why did checkout latency double?")

    assert getattr(runnable, "protocol_id", None) == "specialist_fanout"
    assert explanation.goal_frame.primary_goal is TaskType.DIAGNOSE
    assert explanation.selected_protocol == "specialist_fanout"


@pytest.mark.asyncio
async def test_the_default_classifier_is_the_agent_extractor() -> None:
    extractor = _scripted_extractor(_FRAME_JSON)
    router = Router(extractor=extractor, compiler=_compiler())

    assert router.extractor is extractor
    assert isinstance(router.classifier, AgentFrameExtractor)
    assert router.classifier.extractor is extractor
    frame = await router.extract("explain this")
    assert frame.primary_goal is TaskType.EXPLAIN


@pytest.mark.asyncio
async def test_the_agent_extractor_rejects_a_result_that_is_not_a_frame() -> None:
    wrapped = AgentFrameExtractor(_scripted_extractor("not a frame"))
    with pytest.raises(FrameExtractionError, match="did not produce a GoalFrame"):
        await wrapped.classify("anything")


@pytest.mark.asyncio
async def test_a_classifier_failure_is_reported_on_the_bus_and_reraised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Refusing:
        async def classify(self, text: str) -> GoalFrame:
            msg = f"no frame for {text!r}"
            raise FrameExtractionError(msg)

    failed = AsyncMock()
    monkeypatch.setattr("locus.router.runtime.emit_frame_failed", failed)
    router = Router(classifier=_Refusing(), compiler=_compiler())

    with pytest.raises(FrameExtractionError, match="no frame"):
        await router.extract("hi", run_id="r1")
    failed.assert_awaited_once_with("r1", "no frame for 'hi'")


def test_the_router_needs_exactly_one_front_end() -> None:
    with pytest.raises(ValueError, match="extractor= agent or a classifier="):
        Router(compiler=_compiler())
    with pytest.raises(ValueError, match="not both"):
        Router(extractor=MagicMock(), classifier=_TableClassifier(), compiler=_compiler())


# ------------------------------------------------------------------- picker --


class _LastListedPicker:
    """Picks the last candidate, so the choice is visibly not the ranker's."""

    def __init__(self) -> None:
        self.offered: list[list[str]] = []

    async def pick(self, frame: GoalFrame, candidates: list[Protocol]) -> tuple[Protocol, str]:
        self.offered.append([p.id for p in candidates])
        return candidates[-1], "last one listed"


@pytest.mark.asyncio
async def test_a_stub_picker_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    picker = _LastListedPicker()
    compiler = _compiler(protocol_picker=picker)
    selected = AsyncMock()
    monkeypatch.setattr("locus.router.compiler.emit_protocol_selected", selected)
    # HIGH-complexity RESEARCH leaves several survivors, so the picker is asked.
    frame = _frame(TaskType.RESEARCH, Complexity.HIGH)

    runnable = await compiler.compile(frame, run_id="r2")

    assert len(picker.offered) == 1
    assert len(picker.offered[0]) > 1
    assert getattr(runnable, "protocol_id", None) == picker.offered[0][-1]
    assert selected.await_args.kwargs == {"method": "llm_picked", "rationale": "last one listed"}


def test_the_shipped_picker_and_a_stub_both_satisfy_the_protocol() -> None:
    assert isinstance(_LastListedPicker(), ProtocolPicker)
    assert isinstance(LLMProtocolPicker(model=MagicMock()), ProtocolPicker)
