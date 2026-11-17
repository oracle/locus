# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``Rubric``: the arithmetic a judge should never be asked to do.

Each test pins a reading of the rubric that the judge used to own — the
composite, the letter, the band a cap moves to, the revision that says which
rubric produced a number — and one way each used to go wrong when a model did
it: a forgotten dimension renormalised away, a half rounded differently per
platform, a reweighted weight leaving two reports looking comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from locus.evaluation import Cap, Dimension, Rubric


def _rubric(**overrides: object) -> Rubric:
    fields: dict[str, object] = {
        "dimensions": [
            Dimension(key="root_cause", weight=0.5, guidance="Names the component."),
            Dimension(key="evidence", weight=0.3),
            Dimension(key="actionable", weight=0.2),
        ],
        "grade_thresholds": Rubric.default_grades(),
    }
    fields.update(overrides)
    return Rubric.model_validate(fields)


# --------------------------------------------------------------------------
# composite
# --------------------------------------------------------------------------


def test_the_composite_is_the_weighted_sum() -> None:
    assert _rubric().composite({"root_cause": 80, "evidence": 100, "actionable": 60}) == 82


def test_weights_are_normalised_so_their_scale_does_not_matter() -> None:
    raw = Rubric(
        dimensions=[Dimension(key="a", weight=3), Dimension(key="b", weight=1)],
    )
    fractional = Rubric(
        dimensions=[Dimension(key="a", weight=0.75), Dimension(key="b", weight=0.25)],
    )
    scores = {"a": 100, "b": 0}
    assert raw.composite(scores) == fractional.composite(scores) == 75


def test_a_missing_dimension_counts_as_zero_not_as_absent() -> None:
    """A judge that forgot a dimension has not graded it; renormalising would hide that."""
    assert _rubric().composite({"root_cause": 100, "evidence": 100}) == 80


def test_unknown_keys_are_ignored() -> None:
    assert (
        _rubric().composite({"root_cause": 100, "evidence": 100, "actionable": 100, "x": 0}) == 100
    )


def test_scores_are_clamped_to_the_scale() -> None:
    assert _rubric().composite({"root_cause": 250, "evidence": -40, "actionable": 100}) == 70


def test_the_composite_rounds_half_up() -> None:
    """74.5 is a B floor question; it must not depend on banker's rounding."""
    rubric = Rubric(dimensions=[Dimension(key="a"), Dimension(key="b")])
    assert rubric.composite({"a": 74, "b": 75}) == 75
    assert rubric.composite({"a": 72, "b": 73}) == 73


# --------------------------------------------------------------------------
# letters
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "letter"),
    [(100, "A"), (90, "A"), (89, "B"), (75, "B"), (60, "C"), (45, "D"), (44, "F"), (0, "F")],
)
def test_the_letter_is_the_highest_band_at_or_below_the_score(score: int, letter: str) -> None:
    assert _rubric().grade_for(score) == letter


def test_a_rubric_without_bands_reports_no_letter() -> None:
    rubric = Rubric(dimensions=[Dimension(key="quality")])
    assert rubric.grade_for(95) is None
    assert rubric.band_top("A") == 100


def test_a_score_below_every_band_takes_the_lowest_letter() -> None:
    rubric = Rubric(dimensions=[Dimension(key="q")], grade_thresholds={"pass": 70, "fail": 40})
    assert rubric.grade_for(10) == "fail"


def test_band_top_is_one_below_the_next_letter() -> None:
    rubric = _rubric()
    assert rubric.band_top("C") == 74
    assert rubric.band_top("A") == 100
    assert rubric.band_top("Z") == 100


def test_rank_orders_letters_best_first() -> None:
    rubric = _rubric()
    assert rubric.rank("A") == 0
    assert rubric.rank("F") == 4
    assert rubric.rank("Z") is None
    assert Rubric(dimensions=[Dimension(key="q")]).rank("A") is None


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_duplicate_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate dimension key"):
        Rubric(dimensions=[Dimension(key="a"), Dimension(key="a")])


def test_a_threshold_outside_the_scale_is_rejected() -> None:
    with pytest.raises(ValidationError, match="within 0-100"):
        Rubric(dimensions=[Dimension(key="a")], grade_thresholds={"A": 120})


def test_a_cap_must_name_a_letter_the_rubric_has() -> None:
    with pytest.raises(ValidationError, match="not one of the rubric's letters"):
        _rubric(caps=[Cap(when="contradicted_claim", grade="Z")])
    with pytest.raises(ValidationError, match="caps need grade_thresholds"):
        Rubric(
            dimensions=[Dimension(key="a")],
            caps=[Cap(when="contradicted_claim", grade="C")],
        )


def test_a_cap_trigger_must_be_one_code_can_decide() -> None:
    with pytest.raises(ValidationError, match="unknown cap trigger"):
        Cap(when="judge_felt_like_it", grade="C")


def test_from_text_is_one_dimension_with_no_letters() -> None:
    rubric = Rubric.from_text("  Passes if it names the evidence.  ")
    assert rubric.keys == ["quality"]
    assert rubric.dimensions[0].guidance == "Passes if it names the evidence."
    assert rubric.grade_thresholds is None


def test_default_grades_is_a_copy() -> None:
    grades = Rubric.default_grades()
    grades["A"] = 1
    assert Rubric.default_grades()["A"] == 90


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------


def test_the_revision_is_stable_for_the_same_rubric() -> None:
    assert _rubric().revision() == _rubric().revision()
    assert len(_rubric().revision()) == 12


def test_the_revision_moves_when_a_weight_or_the_prompt_changes() -> None:
    base = _rubric().revision()
    reweighted = _rubric(
        dimensions=[
            Dimension(key="root_cause", weight=0.6, guidance="Names the component."),
            Dimension(key="evidence", weight=0.2),
            Dimension(key="actionable", weight=0.2),
        ]
    ).revision()
    assert reweighted != base
    assert _rubric().revision("prompt v2") != base


def test_the_prompt_block_names_every_dimension_and_the_bands() -> None:
    block = _rubric().prompt_block()
    assert "root_cause (weight 50%)" in block
    assert "Names the component." in block
    assert "A >= 90" in block
    assert "60 or more passes" in block


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------


def test_a_rubric_round_trips_through_json(tmp_path: Path) -> None:
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps(_rubric().to_dict()), encoding="utf-8")
    loaded = Rubric.load(path)
    assert loaded == _rubric()
    assert loaded.revision() == _rubric().revision()


def test_a_rubric_loads_from_yaml(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "rubric.yaml"
    path.write_text(
        "dimensions:\n  - key: a\n    weight: 2\n  - key: b\n"
        "grade_thresholds: {A: 90, B: 75, C: 60, D: 45, F: 0}\n",
        encoding="utf-8",
    )
    loaded = Rubric.load(path)
    assert loaded.keys == ["a", "b"]
    assert loaded.weights["a"] == pytest.approx(2 / 3)


def test_a_file_that_is_not_one_object_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "rubric.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(TypeError, match="one object"):
        Rubric.load(path)
