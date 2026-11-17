# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Cases as data: the shapes ``load_cases`` reads and ``dump_cases`` writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from locus.evaluation import Dimension, EvalCase, Rubric, dump_cases, load_cases


def test_a_json_list_loads_with_string_rubrics(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {"name": "a", "prompt": "p", "expected_output_contains": ["x"]},
                {"name": "b", "prompt": "q", "rubric": "Passes if polite."},
            ]
        ),
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert [c.name for c in cases] == ["a", "b"]
    assert cases[0].rubric is None
    assert cases[1].rubric == "Passes if polite."


def test_a_file_level_rubric_applies_to_cases_that_name_none(tmp_path: Path) -> None:
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "rubric": {
                    "dimensions": [{"key": "quality", "guidance": "Is it right?"}],
                    "grade_thresholds": {"A": 90, "B": 75, "C": 60, "D": 45, "F": 0},
                },
                "cases": [
                    {"name": "a", "prompt": "p"},
                    {"name": "b", "prompt": "q", "rubric": "its own criterion"},
                    {"name": "c", "prompt": "r", "rubric": {"dimensions": [{"key": "k"}]}},
                ],
            }
        ),
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert isinstance(cases[0].rubric, Rubric)
    assert cases[0].rubric.keys == ["quality"]
    assert cases[1].rubric == "its own criterion"
    assert isinstance(cases[2].rubric, Rubric)
    assert cases[2].rubric.keys == ["k"]


def test_a_rubric_argument_fills_in_for_a_file_without_one(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"name": "a", "prompt": "p"}]), encoding="utf-8")
    rubric = Rubric.from_text("be right")
    assert load_cases(path, rubric=rubric)[0].rubric == rubric


def test_jsonl_loads_one_case_per_line(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"name": "a", "prompt": "p"}\n\n{"name": "b", "prompt": "q", "tags": ["smoke"]}\n',
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert [c.name for c in cases] == ["a", "b"]
    assert cases[1].tags == ["smoke"]


def test_yaml_loads_when_pyyaml_is_installed(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "cases.yaml"
    path.write_text(
        "rubric: Passes if it names the evidence.\n"
        "cases:\n"
        "  - name: a\n    prompt: p\n    expected_tools: [lookup]\n",
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert cases[0].expected_tools == ["lookup"]
    assert isinstance(cases[0].rubric, Rubric)
    assert cases[0].rubric.dimensions[0].guidance == "Passes if it names the evidence."


def test_bad_shapes_are_refused_with_the_file_named(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": "nope"}), encoding="utf-8")
    with pytest.raises(TypeError, match="expected a list of cases"):
        load_cases(path)
    path.write_text(json.dumps([1]), encoding="utf-8")
    with pytest.raises(TypeError, match="case 0 is int"):
        load_cases(path)
    path.write_text(json.dumps([{"name": "a", "prompt": "p", "rubric": 5}]), encoding="utf-8")
    with pytest.raises(TypeError, match="a case rubric must be"):
        load_cases(path)


def test_dump_and_load_round_trip(tmp_path: Path) -> None:
    rubric = Rubric(dimensions=[Dimension(key="q")], grade_thresholds=Rubric.default_grades())
    cases = [
        EvalCase(name="a", prompt="p", expected_output_contains=["x"], tags=["smoke"]),
        EvalCase(name="b", prompt="q", rubric=rubric, require_receipts=True),
    ]
    path = tmp_path / "out.json"
    dump_cases(cases, path)
    assert load_cases(path) == cases
