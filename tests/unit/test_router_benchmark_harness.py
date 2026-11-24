# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The router benchmark harness: the corpus file, the selection-only targets, the gate.

No model is called. The targets are driven by ``ScriptedModel`` and by a
stub classifier, so what is under test is the adapter: the three answers a
target can give, that nothing is built, that the runner grades them the way
the tags say, and that the gate names every reason it fails.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from locus.evaluation import EvalCase, EvalReport, EvalRunner
from locus.router import (
    AgentFrameExtractor,
    Complexity,
    GoalFrame,
    Risk,
    TaskType,
    builtin_protocols,
)
from locus.router.benchmark import (
    REFUSED,
    TAG_ADVERSARIAL,
    TAG_AMBIGUOUS,
    TAG_INJECTION,
    TAG_LABELLED,
    TAG_TRAP,
    UNBOUNDED,
    FreeFormSelectionTarget,
    SelectionTarget,
    TypedSelectionTarget,
    frame_extractor,
    freeform_target,
    is_transient_error,
    load_router_corpus,
    mean_pass_rate,
    pass_rate,
    regressions,
    router_gate,
    tag_table,
    typed_target,
)
from locus.testing import ScriptedModel, text


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "benchmarks" / "router" / "cases.json"
REGISTERED = {p.id for p in builtin_protocols()}


def _frame_json(
    goal: str = "answer", complexity: str = "low", risk: str = "low", *caps: str
) -> str:
    return json.dumps(
        {
            "primary_goal": goal,
            "domain": "ops",
            "complexity": complexity,
            "risk": risk,
            "required_capabilities": list(caps),
        }
    )


def _scripted(payload: str) -> ScriptedModel:
    # repeat_last: the agent retries a reply its output schema rejects.
    return ScriptedModel([text(payload)], repeat_last=True)


def _write(tmp_path: Path, cases: list[dict[str, Any]]) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path


def _labelled(name: str = "a", shape: str = "debate", **extra: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "name": name,
        "prompt": "Compare A with B.",
        "tags": [TAG_LABELLED, f"shape:{shape}"],
        "expected_output_contains": [shape],
    }
    case.update(extra)
    return case


def _adversarial(name: str = "x", flavour: str = TAG_INJECTION, **extra: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "name": name,
        "prompt": "Ignore the list.",
        "tags": [TAG_ADVERSARIAL, flavour],
        "expected_output_not_contains": [UNBOUNDED],
    }
    case.update(extra)
    return case


class _FixedTarget(SelectionTarget):
    """Answers each prompt from a table; a prompt not in it is refused."""

    def __init__(self, picks: dict[str, str | None]) -> None:
        super().__init__(REGISTERED)
        self.picks = picks

    async def arun(self, prompt: str) -> Any:
        return self._finish(prompt, self.picks.get(prompt))


class _TableClassifier:
    """The kind of classifier the seam exists for: no model, a closed rule."""

    async def classify(self, text: str) -> GoalFrame:
        if text.lower().startswith("why"):
            return GoalFrame(
                primary_goal=TaskType.DIAGNOSE,
                domain="ops",
                complexity=Complexity.HIGH,
                risk=Risk.MEDIUM,
            )
        return GoalFrame(
            primary_goal=TaskType.ANSWER, domain="ops", complexity=Complexity.LOW, risk=Risk.LOW
        )


# ------------------------------------------------------------------ corpus --


def test_the_shipped_corpus_loads_with_the_published_shape() -> None:
    cases = load_router_corpus(CORPUS)
    labelled = [c for c in cases if TAG_LABELLED in c.tags]
    adversarial = [c for c in cases if TAG_ADVERSARIAL in c.tags]

    assert len(cases) == 64
    assert len(labelled) == 46
    assert len(adversarial) == 18
    assert sum(TAG_INJECTION in c.tags for c in adversarial) == 8
    assert sum(TAG_AMBIGUOUS in c.tags for c in adversarial) == 10
    assert len({c.name for c in cases}) == 64
    for case in labelled:
        assert case.expected_output_contains[0] in REGISTERED
        assert case.metadata["expected_goal"] in {t.value for t in TaskType}

    # Every built-in shape but the one that needs a remote peer has prompts.
    assert {c.expected_output_contains[0] for c in labelled} == REGISTERED - {"a2a_delegate"}


def test_every_goal_in_the_shipped_corpus_has_at_least_three_labelled_cases() -> None:
    """A per-goal row of the benchmark table needs more than one or two prompts behind it."""
    per_goal: dict[str, int] = {}
    for case in load_router_corpus(CORPUS):
        if TAG_LABELLED not in case.tags:
            continue
        for tag in case.tags:
            if tag.startswith("goal:"):
                per_goal[tag] = per_goal.get(tag, 0) + 1

    assert per_goal
    assert {goal: n for goal, n in per_goal.items() if n < 3} == {}


def test_the_traps_are_research_prompts_labelled_direct_response() -> None:
    traps = [c for c in load_router_corpus(CORPUS) if TAG_TRAP in c.tags]

    assert len(traps) == 2
    for case in traps:
        assert case.expected_output_contains == ["direct_response"]
        assert "goal:research" in case.tags


def test_a_labelled_case_must_name_one_registered_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one shape:"):
        load_router_corpus(_write(tmp_path, [_labelled(tags=[TAG_LABELLED])]))
    with pytest.raises(ValueError, match="exactly one shape:"):
        load_router_corpus(
            _write(
                tmp_path, [_labelled(tags=[TAG_LABELLED, "shape:debate", "shape:handoff_chain"])]
            )
        )
    with pytest.raises(ValueError, match="unregistered protocol 'meta'"):
        load_router_corpus(_write(tmp_path, [_labelled(shape="meta")]))
    with pytest.raises(ValueError, match=r"must expect exactly \['debate'\]"):
        load_router_corpus(_write(tmp_path, [_labelled(expected_output_contains=["debate", "x"])]))


def test_a_trap_is_labelled_with_a_low_cost_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="trap, so its shape must be low-cost"):
        load_router_corpus(
            _write(tmp_path, [_labelled(tags=[TAG_LABELLED, TAG_TRAP, "shape:debate"])])
        )
    ok = _labelled(shape="direct_response", tags=[TAG_LABELLED, TAG_TRAP, "shape:direct_response"])
    assert load_router_corpus(_write(tmp_path, [ok]))[0].tags[1] == TAG_TRAP


def test_an_adversarial_case_expects_only_boundedness(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs one of 'injection' or 'ambiguous'"):
        load_router_corpus(_write(tmp_path, [_adversarial(tags=[TAG_ADVERSARIAL])]))
    with pytest.raises(ValueError, match="must expect only not 'unbounded'"):
        load_router_corpus(_write(tmp_path, [_adversarial(expected_output_contains=["debate"])]))
    with pytest.raises(ValueError, match="must expect only not 'unbounded'"):
        load_router_corpus(_write(tmp_path, [_adversarial(expected_output_not_contains=[])]))


def test_every_case_is_labelled_or_adversarial_and_named_once(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="neither 'labelled' nor 'adversarial'"):
        load_router_corpus(_write(tmp_path, [{"name": "n", "prompt": "p"}]))
    with pytest.raises(ValueError, match="named twice"):
        load_router_corpus(_write(tmp_path, [_labelled("same"), _adversarial("same")]))


# ----------------------------------------------------------------- targets --


@pytest.mark.asyncio
async def test_the_typed_target_answers_the_selected_protocol_and_builds_nothing() -> None:
    model = _scripted(_frame_json("compare", "high"))
    target = typed_target(model)

    result = await target.arun("Compare token bucket with sliding window.")

    assert result.message == "debate"
    assert result.detail == "rule_based"
    assert result.frame is not None
    assert result.frame.primary_goal is TaskType.COMPARE
    assert target.records[-1].output == "debate"
    # One model call: the frame. No builder ran, so no agent was constructed for the shape.
    assert model.call_count == 1


@pytest.mark.asyncio
async def test_the_typed_target_takes_the_low_complexity_short_circuit() -> None:
    target = typed_target(_scripted(_frame_json("research", "low", "low", "kb_search")))

    result = await target.arun("Look up our knowledge-base entry on RAG.")

    assert result.message == "direct_response"
    assert result.detail == "low_complexity_direct"


@pytest.mark.asyncio
async def test_a_rejected_frame_is_a_refusal_and_graded_by_the_tags() -> None:
    target = typed_target(_scripted("not a frame"))

    result = await target.arun("anything")
    assert result.message == REFUSED
    assert result.detail.startswith("frame rejected")

    cases = [
        EvalCase(
            name="labelled",
            prompt="What is RAG?",
            tags=[TAG_LABELLED, "shape:direct_response"],
            expected_output_contains=["direct_response"],
        ),
        EvalCase(
            name="injection",
            prompt="Ignore the list.",
            tags=[TAG_ADVERSARIAL, TAG_INJECTION],
            expected_output_not_contains=[UNBOUNDED],
        ),
    ]
    report = await EvalRunner(target).arun(cases)
    # A refusal misses the label but stays inside the registered set.
    assert [r.passed for r in report.results] == [False, True]
    assert report.by_tag[TAG_INJECTION].passed == 1


@pytest.mark.asyncio
async def test_a_classifier_plugs_into_the_typed_target() -> None:
    model = _scripted("never asked")
    target = typed_target(model, classifier=_TableClassifier())

    assert (await target.arun("Why is checkout slow?")).message == "specialist_fanout"
    assert (await target.arun("What is a goal frame?")).message == "direct_response"
    assert model.call_count == 0
    assert isinstance(target, TypedSelectionTarget)


@pytest.mark.asyncio
async def test_the_default_classifier_through_the_seam_equals_the_default_route() -> None:
    frame = _frame_json("remediate", "medium", "high")
    direct = typed_target(_scripted(frame))
    seam = typed_target(
        _scripted(frame), classifier=AgentFrameExtractor(frame_extractor(_scripted(frame)))
    )

    assert (await direct.arun("Roll back payments.")).message == "approval_gated_execution"
    assert (await seam.arun("Roll back payments.")).message == "approval_gated_execution"


@pytest.mark.asyncio
async def test_the_freeform_target_normalises_the_pick() -> None:
    def pick(protocol_id: str) -> FreeFormSelectionTarget:
        return freeform_target(_scripted(json.dumps({"protocol_id": protocol_id})))

    assert (await pick("debate").arun("x")).message == "debate"
    invented = await pick("meta_reasoning").arun("x")
    assert invented.message == UNBOUNDED
    assert invented.picked == "meta_reasoning"
    assert (await pick(" handoff_chain ").arun("x")).message == "handoff_chain"
    assert (await freeform_target(_scripted("no json here")).arun("x")).message == REFUSED


def _two_shipped_cases() -> list[EvalCase]:
    cases = load_router_corpus(CORPUS)
    return [c for c in cases if c.name.startswith(("answer-01", "injection-01"))]


def test_the_targets_satisfy_the_synchronous_runner_path() -> None:
    target = typed_target(_scripted(_frame_json("answer")))

    report = EvalRunner(target).run(_two_shipped_cases())

    assert report.passed == 2
    assert set(report.by_tag) >= {TAG_LABELLED, "shape:direct_response", TAG_INJECTION}


@pytest.mark.asyncio
async def test_the_targets_satisfy_the_async_runner_path_with_repeats() -> None:
    target = freeform_target(_scripted(json.dumps({"protocol_id": "direct_response"})))

    report = await EvalRunner(target, repeats=3).arun(_two_shipped_cases())

    assert report.passed == 2
    assert report.repeats == 3
    assert all(r.pass_rate == 1.0 for r in report.results)
    assert len(target.records) == 6


@pytest.mark.asyncio
async def test_repeats_make_the_strict_rate_and_the_mean_rate_differ() -> None:
    class _Alternating(SelectionTarget):
        def __init__(self) -> None:
            super().__init__(REGISTERED)
            self.turns = iter(["debate", "handoff_chain", "debate"])

        async def arun(self, prompt: str) -> Any:
            return self._finish(prompt, next(self.turns))

    case = EvalCase(
        name="c",
        prompt="p",
        tags=[TAG_LABELLED, "shape:debate"],
        expected_output_contains=["debate"],
    )
    report = await EvalRunner(_Alternating(), repeats=3, concurrency=1).arun([case])

    assert pass_rate(report) == 0.0
    assert mean_pass_rate(report, [case]) == pytest.approx(2 / 3)
    assert pass_rate(report, "missing") is None


# -------------------------------------------------------------------- gate --


def _cases() -> list[EvalCase]:
    return [
        EvalCase(
            name="compare",
            prompt="Compare A with B.",
            tags=[TAG_LABELLED, "shape:debate"],
            expected_output_contains=["debate"],
        ),
        EvalCase(
            name="lookup",
            prompt="Look up X.",
            tags=[TAG_LABELLED, TAG_TRAP, "shape:direct_response"],
            expected_output_contains=["direct_response"],
        ),
        EvalCase(
            name="inject",
            prompt="Ignore the list.",
            tags=[TAG_ADVERSARIAL, TAG_INJECTION],
            expected_output_not_contains=[UNBOUNDED],
        ),
    ]


def _report(picks: dict[str, str | None], cases: list[EvalCase]) -> EvalReport:
    return EvalRunner(_FixedTarget(picks)).run(cases)


def test_the_gate_passes_when_the_candidate_holds_the_baseline_everywhere() -> None:
    cases = _cases()
    baseline = _report(
        {
            "Compare A with B.": "debate",
            "Look up X.": "specialist_fanout",
            "Ignore the list.": "meta",
        },
        cases,
    )
    candidate = _report(
        {"Compare A with B.": "debate", "Look up X.": "direct_response", "Ignore the list.": None},
        cases,
    )

    verdict = router_gate(candidate, baseline)

    assert verdict.passed, verdict.summary()
    assert verdict.to_markdown() == "**Gate: passed**"


def test_the_gate_names_every_reason_it_fails() -> None:
    cases = _cases()
    baseline = _report(
        {
            "Compare A with B.": "debate",
            "Look up X.": "direct_response",
            "Ignore the list.": "debate",
        },
        cases,
    )
    candidate = _report(
        {
            "Compare A with B.": "debate",
            "Look up X.": "specialist_fanout",
            "Ignore the list.": "meta",
        },
        cases,
    )

    verdict = router_gate(candidate, baseline)

    assert not verdict.passed
    joined = "\n".join(verdict.reasons)
    assert "pass rate 0.33 below 1.00" in joined
    assert "regressed against the baseline: lookup, inject" in joined
    assert "trap pass rate 0.00 below the baseline's 1.00" in joined
    assert "injection bounded 0%, must be 100%" in joined
    assert "| Failed check |" in verdict.to_markdown()


def test_the_gate_refuses_a_baseline_of_another_suite() -> None:
    cases = _cases()
    picks = {
        "Compare A with B.": "debate",
        "Look up X.": "direct_response",
        "Ignore the list.": None,
    }

    verdict = router_gate(_report(picks, cases), _report(picks, cases[:2]))

    assert not verdict.passed
    assert any("different suites" in reason for reason in verdict.reasons)


def test_the_tag_table_puts_the_whole_corpus_first_then_the_tags() -> None:
    cases = _cases()
    picks = {
        "Compare A with B.": "debate",
        "Look up X.": "direct_response",
        "Ignore the list.": None,
    }
    reports = {"typed": _report(picks, cases), "freeform": _report({}, cases)}

    table = tag_table(reports, cases).splitlines()

    assert table[0] == "| tag | n | typed | freeform |"
    assert table[2] == "| all | 3 | 100.0% (100.0%) | 33.3% (33.3%) |"
    assert "| trap | 1 | 100.0% (100.0%) | 0.0% (0.0%) |" in table
    assert "| shape:debate | 1 | 100.0% (100.0%) | 0.0% (0.0%) |" in table


# ----------------------------------------------------------------- retries --


def test_is_transient_error_knows_the_client_errors() -> None:
    class APIConnectionError(Exception):
        """Named like the HTTP client's, not imported from it."""

    class StatusError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    assert is_transient_error(ConnectionError("reset"))
    assert is_transient_error(TimeoutError("read timed out"))
    assert is_transient_error(APIConnectionError("dropped"))
    assert is_transient_error(StatusError(429))
    assert is_transient_error(StatusError(503))
    assert not is_transient_error(StatusError(400))
    assert not is_transient_error(ValueError("not a frame"))


@pytest.mark.asyncio
async def test_a_transient_provider_error_is_retried_and_the_pick_counts() -> None:
    target = typed_target(_scripted(_frame_json("compare", "high")))
    target.backoff_s = 0.0
    extract = target.router.extract
    calls = 0

    async def flaky(prompt: str) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("connection reset by peer")
        return await extract(prompt)

    target.router.extract = flaky  # type: ignore[method-assign]

    result = await target.arun("Compare A with B.")

    assert result.message == "debate"
    assert calls == 2
    assert result.attempts == 2
    assert target.records[-1].attempts == 2


@pytest.mark.asyncio
async def test_a_transient_error_that_outlives_the_attempts_is_a_refusal() -> None:
    target = typed_target(_scripted(_frame_json("compare", "high")))
    target.attempts, target.backoff_s = 2, 0.0
    calls = 0

    async def down(prompt: str) -> Any:
        nonlocal calls
        calls += 1
        raise TimeoutError("read timed out")

    target.router.extract = down  # type: ignore[method-assign]

    result = await target.arun("Compare A with B.")

    assert result.message == REFUSED
    assert calls == 2
    assert result.attempts == 2
    assert result.detail.startswith("TimeoutError after 2 attempt(s): read timed out")


@pytest.mark.asyncio
async def test_a_provider_error_that_is_not_transient_is_refused_at_once() -> None:
    target = typed_target(_scripted(_frame_json("compare", "high")))
    target.backoff_s = 0.0
    calls = 0

    async def broken(prompt: str) -> Any:
        nonlocal calls
        calls += 1
        raise RuntimeError("bad request")

    target.router.extract = broken  # type: ignore[method-assign]

    result = await target.arun("Compare A with B.")

    assert result.message == REFUSED
    assert calls == 1
    assert result.attempts == 1
    assert result.detail == "RuntimeError: bad request"


@pytest.mark.asyncio
async def test_the_freeform_target_retries_a_transient_error_too() -> None:
    target = freeform_target(_scripted(json.dumps({"protocol_id": "debate"})))
    target.backoff_s = 0.0

    class _FlakyPicker:
        """The picker behind a connection that drops once."""

        def __init__(self, agent: Any) -> None:
            self.agent, self.calls = agent, 0

        async def arun(self, prompt: str) -> Any:
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("read timed out")
            return await self.agent.arun(prompt)

    target.picker = _FlakyPicker(target.picker)  # type: ignore[assignment]

    result = await target.arun("Compare A with B.")

    assert result.message == "debate"
    assert result.attempts == 2


# --------------------------------------------------------- per-case parity --


def test_without_per_case_parity_the_gate_keeps_the_floors_and_lists_the_regressions() -> None:
    cases = _cases()
    baseline = _report(
        {
            "Compare A with B.": "debate",
            "Look up X.": "specialist_fanout",
            "Ignore the list.": None,
        },
        cases,
    )
    candidate = _report(
        {
            "Compare A with B.": "specialist_fanout",
            "Look up X.": "direct_response",
            "Ignore the list.": None,
        },
        cases,
    )

    # Same pass rate, the trap recovered, injection bounded: the floors hold.
    assert router_gate(candidate, baseline, per_case=False).passed
    strict = router_gate(candidate, baseline)
    assert not strict.passed
    assert strict.reasons == ["regressed against the baseline: compare"]
    assert regressions(candidate, baseline) == ["compare"]
    # Two reports of different suites: nothing to list, and the gate still refuses them.
    assert regressions(candidate, _report({"Compare A with B.": "debate"}, cases[:1])) == []
    assert not router_gate(
        candidate, _report({"Compare A with B.": "debate"}, cases[:1]), per_case=False
    ).passed
