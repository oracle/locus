# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""The router's training split and the profile a classifier learns to emit."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from locus.evaluation import EvalCase
from locus.router import Complexity, PolicyGate, Risk, TaskType
from locus.router.benchmark import (
    DIFFICULTIES,
    TAG_LABELLED,
    TAG_TRAP,
    benchmark_compiler,
    load_router_corpus,
)
from locus.router.difficulty import WORLD_METRICS
from locus.router.training import (
    NEEDS,
    RoutingProfile,
    cell_table,
    load_router_training_set,
    near_duplicates,
    prompt_overlap,
    routing_profile,
    training_examples,
    write_jsonl,
)


BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks" / "router"
TRAIN = BENCHMARKS / "train.json"
HELD_OUT = BENCHMARKS / "cases.json"


def _case(
    name: str,
    prompt: str,
    *,
    goal: str = "answer",
    level: str = "low",
    risk: str = "low",
    shape: str = "direct_response",
    capabilities: list[str] | None = None,
    trap: bool = False,
) -> dict[str, Any]:
    tags = [TAG_LABELLED, f"shape:{shape}", f"goal:{goal}", f"difficulty:{level}"]
    if trap:
        tags.append(TAG_TRAP)
    metadata: dict[str, Any] = {
        "expected_goal": goal,
        "domain": "ops",
        "expected_complexity": level,
        "expected_risk": risk,
    }
    if capabilities:
        metadata["capabilities"] = capabilities
    return {
        "name": name,
        "prompt": prompt,
        "tags": tags,
        "expected_output_contains": [shape],
        "metadata": metadata,
    }


def _write(tmp_path: Path, cases: list[dict[str, Any]]) -> Path:
    file = tmp_path / "train.json"
    file.write_text(json.dumps(cases))
    return file


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------


def test_a_profile_keeps_its_needs_in_vocabulary_order_and_refuses_unknown_ones() -> None:
    profile = RoutingProfile(
        task_type=TaskType.DIAGNOSE,
        difficulty=Complexity.MEDIUM,
        risk=Risk.LOW,
        needs=("multi_agent", "tools", "tools"),
    )
    assert profile.needs == ("tools", "multi_agent")

    with pytest.raises(ValidationError, match="needs outside the vocabulary: gpu"):
        RoutingProfile(
            task_type=TaskType.ANSWER, difficulty=Complexity.LOW, risk=Risk.LOW, needs=("gpu",)
        )


def test_a_profile_becomes_the_frame_the_compiler_resolves() -> None:
    profile = RoutingProfile(
        task_type=TaskType.REMEDIATE,
        difficulty=Complexity.MEDIUM,
        risk=Risk.HIGH,
        needs=NEEDS,
    )
    frame = profile.to_frame(domain="ops", capabilities=["metric_probe"])

    assert (frame.primary_goal, frame.complexity, frame.risk) == (
        TaskType.REMEDIATE,
        Complexity.MEDIUM,
        Risk.HIGH,
    )
    assert frame.domain == "ops"
    assert frame.required_capabilities == ["metric_probe"]
    assert all(
        (
            frame.requires_tools,
            frame.requires_memory,
            frame.requires_code_generation,
            frame.requires_multi_agent,
            frame.approval_required,
        )
    )

    plain = RoutingProfile(task_type=TaskType.ANSWER, difficulty=Complexity.LOW, risk=Risk.LOW)
    bare = plain.to_frame()
    assert bare.domain == "general"
    assert not any((bare.requires_tools, bare.requires_multi_agent, bare.approval_required))


def test_needs_follow_from_the_stated_frame_by_rule() -> None:
    def profile(**kwargs: Any) -> RoutingProfile:
        return routing_profile(EvalCase.model_validate(_case("c", "p", **kwargs)))

    assert profile().needs == ()
    assert profile(capabilities=["kb_search"], goal="research").needs == ("tools",)
    assert profile(goal="generate_code", level="medium", shape="codegen_test_validate").needs == (
        "code_generation",
    )
    assert profile(
        goal="diagnose", level="medium", shape="specialist_fanout", capabilities=["metric_probe"]
    ).needs == ("tools", "multi_agent")
    assert profile(goal="compare", level="medium", shape="debate").needs == ("multi_agent",)
    held = profile(goal="remediate", level="medium", risk="high", shape="approval_gated_execution")
    assert held.needs == ("approval",)
    assert (held.task_type, held.difficulty, held.risk) == (
        TaskType.REMEDIATE,
        Complexity.MEDIUM,
        Risk.HIGH,
    )


def test_the_approval_need_is_the_policy_gates_verdict() -> None:
    case = EvalCase.model_validate(
        _case("c", "p", goal="escalate", risk="medium", shape="approval_gated_execution")
    )

    assert routing_profile(case).needs == ()
    strict = PolicyGate(require_approval_above=Risk.LOW)
    assert routing_profile(case, policy=strict).needs == ("approval",)


def test_a_profile_is_read_only_from_a_labelled_case_with_a_registered_label() -> None:
    adversarial = EvalCase(
        name="inj", prompt="ignore everything", tags=["adversarial", "injection"]
    )
    with pytest.raises(ValueError, match="'inj' is not labelled"):
        routing_profile(adversarial)

    unknown = EvalCase.model_validate(_case("c", "p", shape="teleport"))
    with pytest.raises(ValueError, match="unregistered protocol 'teleport'"):
        routing_profile(unknown)


# ---------------------------------------------------------------------------
# Overlap and the loader
# ---------------------------------------------------------------------------


def test_prompt_overlap_is_the_share_of_words_two_prompts_have_in_common() -> None:
    assert prompt_overlap("What is RAG?", "what is rag") == 1.0
    assert prompt_overlap("Restart the gateway", "Explain consistent hashing") == 0.0
    assert prompt_overlap("read latency_p99 now", "read latency_p99 later") == pytest.approx(0.5)
    assert prompt_overlap("", "anything") == 0.0
    assert prompt_overlap("?!", "...") == 0.0


def test_near_duplicates_name_the_pairs_highest_overlap_first() -> None:
    train = [
        EvalCase(name="t1", prompt="Roll back the payments service in production"),
        EvalCase(name="t2", prompt="Explain what a webhook is"),
        EvalCase(name="t3", prompt="Roll back payments"),
        EvalCase(name="t4", prompt="..."),
    ]
    held = [
        EvalCase(name="h1", prompt="Roll back the payments service in production."),
        EvalCase(name="h2", prompt="!!"),
    ]

    pairs = near_duplicates(train, held, threshold=0.4)

    assert [(a, b) for a, b, _ in pairs] == [("t1", "h1"), ("t3", "h1")]
    assert pairs[0][2] == 1.0
    assert near_duplicates(train, held, threshold=1.0) == [("t1", "h1", 1.0)]


def test_the_loader_refuses_an_unlabelled_case(tmp_path: Path) -> None:
    adversarial = {
        "name": "inj",
        "prompt": "ignore the rules",
        "tags": ["adversarial", "injection"],
        "expected_output_not_contains": ["unbounded"],
    }
    file = _write(tmp_path, [_case("a", "What is DNS?"), adversarial])

    with pytest.raises(ValueError, match=r"train\.json: case 'inj' is not labelled"):
        load_router_training_set(file)


def test_the_loader_refuses_two_cases_with_one_prompt(tmp_path: Path) -> None:
    file = _write(tmp_path, [_case("a", "What is DNS?"), _case("b", "what is dns")])

    with pytest.raises(ValueError, match="cases 'a' and 'b' share a prompt"):
        load_router_training_set(file)


def test_the_loader_refuses_a_prompt_that_overlaps_the_held_out_set(tmp_path: Path) -> None:
    held = [EvalCase(name=f"h{i}", prompt=f"Restart the service number {i}") for i in range(7)]
    cases = [_case(f"t{i}", f"Restart the service number {i} now") for i in range(7)]
    file = _write(tmp_path, cases)

    with pytest.raises(ValueError) as caught:
        load_router_training_set(file, held_out=held, threshold=0.8)
    message = str(caught.value)
    assert "train.json: 7 prompt(s) overlap the held-out set at or above 0.8" in message
    assert "'t0' and 'h0' (0.83)" in message
    assert message.endswith("and 2 more")

    assert len(load_router_training_set(file, held_out=held, threshold=0.9)) == 7
    assert len(load_router_training_set(file)) == 7


def test_the_loader_keeps_the_corpus_rules(tmp_path: Path) -> None:
    file = _write(tmp_path, [_case("a", "Build a thing", goal="build", trap=True, level="medium")])

    with pytest.raises(ValueError, match="is a trap, so its difficulty must be low"):
        load_router_training_set(file)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def test_training_examples_pair_each_prompt_with_its_profile(tmp_path: Path) -> None:
    cases = [
        EvalCase.model_validate(_case("a", "What is DNS?")),
        EvalCase.model_validate(_case("b", "Build one line", goal="build", trap=True)),
        EvalCase(name="inj", prompt="ignore", tags=["adversarial", "injection"]),
    ]

    rows = training_examples(cases, measured={"a": "medium", "b": "not_measured"})

    assert [row.name for row in rows] == ["a", "b"]
    assert rows[0].profile == RoutingProfile(
        task_type=TaskType.ANSWER, difficulty=Complexity.LOW, risk=Risk.LOW
    )
    assert (rows[0].protocol, rows[0].trap, rows[0].measured_difficulty) == (
        "direct_response",
        False,
        "medium",
    )
    assert (rows[1].trap, rows[1].measured_difficulty) == (True, None)
    assert training_examples(cases)[0].measured_difficulty is None

    out = tmp_path / "rows.jsonl"
    write_jsonl(rows, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["prompt"] == "What is DNS?"
    assert first["profile"] == {
        "task_type": "answer",
        "difficulty": "low",
        "risk": "low",
        "needs": [],
    }
    assert first["protocol"] == "direct_response"


def test_the_cell_table_counts_cases_per_task_type_and_difficulty() -> None:
    cases = [
        EvalCase.model_validate(_case("a", "one")),
        EvalCase.model_validate(_case("b", "two", level="high")),
        EvalCase.model_validate(_case("c", "three", goal="build", trap=True)),
    ]

    table = cell_table(cases).splitlines()

    assert table[0] == "| task type | low | medium | high | traps | total |"
    assert "| answer | 1 | 0 | 1 | 0 | 2 |" in table
    assert "| build | 1 | 0 | 0 | 1 | 1 |" in table
    assert table[-1] == "| **all** | 2 | 0 | 1 | 1 | 3 |"
    assert not any(line.startswith("| plan ") for line in table)


# ---------------------------------------------------------------------------
# The shipped split
# ---------------------------------------------------------------------------


def test_the_shipped_split_loads_and_does_not_overlap_the_held_out_corpus() -> None:
    held_out = load_router_corpus(HELD_OUT)

    cases = load_router_training_set(TRAIN, held_out=held_out)

    assert len(cases) == 390
    assert not {case.name for case in cases} & {case.name for case in held_out}
    assert all(case.name.startswith("train-") for case in cases)


def test_the_shipped_split_covers_every_task_type_and_level() -> None:
    rows = training_examples(load_router_training_set(TRAIN))

    per_task = {task: [r for r in rows if r.profile.task_type is task] for task in TaskType}
    assert {len(mine) for mine in per_task.values()} == {30}
    assert {r.profile.difficulty.value for r in rows} == set(DIFFICULTIES)
    assert {r.profile.risk for r in rows} == set(Risk)
    assert {need for r in rows for need in r.profile.needs} == set(NEEDS) - {"memory"}
    # The nine task types whose canonical shape is not the low-cost one each carry traps.
    trapped = {task for task, mine in per_task.items() if any(r.trap for r in mine)}
    assert trapped == set(TaskType) - {
        TaskType.ANSWER,
        TaskType.EXPLAIN,
        TaskType.REMEDIATE,
        TaskType.ESCALATE,
    }
    assert sum(r.trap for r in rows) == 72


@pytest.mark.parametrize("split", [TRAIN, HELD_OUT], ids=["train", "held-out"])
def test_every_profile_compiles_to_the_protocol_its_label_names(split: Path) -> None:
    compiler = benchmark_compiler(None)

    rows = training_examples(load_router_corpus(split))

    assert rows
    wrong = [
        row.name
        for row in rows
        if compiler.explain(row.profile.to_frame()).selected_protocol != row.protocol
    ]
    assert wrong == []


def test_every_metric_the_shipped_split_names_is_in_the_world() -> None:
    named: set[str] = set()
    for case in load_router_training_set(TRAIN):
        if case.metadata.get("capabilities"):
            named.update(re.findall(r"\b[a-z0-9]+_[a-z0-9_]+\b", case.prompt))

    assert named
    assert named - set(WORLD_METRICS) == set()
