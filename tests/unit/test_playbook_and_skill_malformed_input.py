# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""What a playbook or skill file does when it is wrong.

These are the branches that decide how bad input is handled, and 2.6.0 added
several of them without tests — which is what dropped ``playbooks/loader.py``
and ``skills/models.py`` below the coverage ratchet.

They are worth testing on their own merits, because the two files make
opposite calls on purpose and the reasoning is load-bearing.
``skills/models.py`` skips a malformed probe and keeps the skill, on the
grounds that "a dropped probe is one fewer obligation; a failed load is no
procedure at all". ``PlaybookLoader`` refuses a playbook it cannot validate,
because a half-loaded procedure is worse than none. Neither is obviously right
and both are deliberate, so both get pinned rather than left to be re-derived
by whoever next touches them.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tulip.playbooks.loader import PlaybookLoader, PlaybookLoadError
from tulip.skills.models import Skill


VALID = {
    "id": "pb-1",
    "name": "Example",
    "description": "An example playbook.",
    "steps": [{"id": "s1", "description": "Do the thing"}],
}


# --------------------------------------------------------------------------
# The loader refuses what it cannot validate
# --------------------------------------------------------------------------


def test_an_unknown_extension_is_refused_by_name(tmp_path: pathlib.Path) -> None:
    """Silently guessing the format of a .txt would be worse than refusing it."""
    path = tmp_path / "playbook.txt"
    path.write_text(json.dumps(VALID))

    with pytest.raises(PlaybookLoadError, match="Unsupported file format"):
        PlaybookLoader().load_file(path)


def test_a_missing_file_names_the_path(tmp_path: pathlib.Path) -> None:
    with pytest.raises(PlaybookLoadError) as caught:
        PlaybookLoader().load_file(tmp_path / "nope.json")

    assert "nope.json" in str(caught.value)


def test_invalid_yaml_is_reported_as_invalid_yaml(tmp_path: pathlib.Path) -> None:
    """Not as a validation failure — the reader has to know which layer broke."""
    path = tmp_path / "playbook.yaml"
    path.write_text("steps: [unclosed\n")

    with pytest.raises(PlaybookLoadError, match="Invalid YAML"):
        PlaybookLoader().load_file(path)


def test_invalid_yaml_from_a_string_is_reported_too() -> None:
    with pytest.raises(PlaybookLoadError, match="Invalid YAML"):
        PlaybookLoader().load_yaml_string("steps: [unclosed\n")


def test_a_playbook_that_fails_validation_carries_the_reasons() -> None:
    """A count without the errors would send the author back to guessing."""
    with pytest.raises(PlaybookLoadError) as caught:
        PlaybookLoader().load_dict({"id": "pb-1"})

    assert caught.value.errors


def test_a_valid_playbook_still_loads(tmp_path: pathlib.Path) -> None:
    """The guard against a suite that only proves things fail."""
    path = tmp_path / "playbook.json"
    path.write_text(json.dumps(VALID))

    assert PlaybookLoader().load_file(path).id == "pb-1"


# --------------------------------------------------------------------------
# Step groups — the 2.6.0 flattening, and what it does with junk
# --------------------------------------------------------------------------


def test_a_grouped_playbook_is_flattened_into_steps() -> None:
    playbook = PlaybookLoader().load_dict(
        {
            "id": "pb-2",
            "name": "Grouped",
            "description": "Grouped playbook.",
            "step_groups": [
                {
                    "id": "triage",
                    "goal": "Blast radius",
                    "steps": [{"id": "a", "description": "A"}],
                },
                {"id": "fix", "goal": "Fix it", "steps": [{"id": "b", "description": "B"}]},
            ],
        }
    )

    assert [step.id for step in playbook.steps] == ["a", "b"]


def test_junk_among_the_groups_is_skipped_rather_than_fatal() -> None:
    """One malformed group must not cost the reader the whole procedure."""
    playbook = PlaybookLoader().load_dict(
        {
            "id": "pb-3",
            "name": "Mixed",
            "description": "Mixed playbook.",
            "step_groups": [
                "not a group",
                {
                    "id": "real",
                    "goal": "Do it",
                    "steps": [{"id": "a", "description": "A"}, "not a step", None],
                },
            ],
        }
    )

    assert [step.id for step in playbook.steps] == ["a"]


def test_a_duplicate_step_id_is_made_unique_rather_than_colliding() -> None:
    """Two groups naming a step "check" is ordinary; losing one is not."""
    playbook = PlaybookLoader().load_dict(
        {
            "id": "pb-4",
            "name": "Duplicated",
            "description": "Duplicated ids.",
            "step_groups": [
                {"id": "one", "goal": "First", "steps": [{"id": "check", "description": "A"}]},
                {"id": "two", "goal": "Second", "steps": [{"id": "check", "description": "B"}]},
            ],
        }
    )

    ids = [step.id for step in playbook.steps]
    assert len(ids) == 2
    assert len(set(ids)) == 2, f"ids collided: {ids}"


def test_an_explicit_steps_list_wins_over_groups() -> None:
    """Flattening a playbook that already has steps would duplicate them."""
    playbook = PlaybookLoader().load_dict(
        {**VALID, "step_groups": [{"id": "g", "steps": [{"id": "ignored", "description": "X"}]}]}
    )

    assert [step.id for step in playbook.steps] == ["s1"]


# --------------------------------------------------------------------------
# Skills keep going where the loader stops
# --------------------------------------------------------------------------


def _skill(frontmatter: str) -> Skill:
    """``name`` and ``description`` are required, so every fixture carries them."""
    return Skill.from_content(
        f"---\ndescription: An example skill.\n{frontmatter}---\n\nDo the thing.\n"
    )


def test_a_malformed_probe_is_dropped_and_the_skill_survives() -> None:
    """The deliberate asymmetry with the loader, and the reason for it.

    A probe that cannot be read is one fewer obligation; refusing the file
    would leave the agent with no procedure at all.
    """
    skill = _skill(
        "name: example\n"
        "required_probes:\n"
        "  - not a mapping\n"
        "  - name: ''\n"
        "    match: 'errors'\n"
        "  - name: error_count\n"
        "    match: ''\n"
        "  - name: pool_state\n"
        "    match: 'pool'\n"
        "    description: how full the pool is\n"
    )

    assert [probe.name for probe in skill.required_probes] == ["pool_state"]


def test_both_spellings_of_the_probe_key_are_read() -> None:
    """The corpus this was ported from mixes them; accepting one drops the rest."""
    hyphenated = _skill("name: example\nrequired-probes:\n  - name: p\n    match: 'm'\n")
    underscored = _skill("name: example\nrequired_probes:\n  - name: p\n    match: 'm'\n")

    assert [probe.name for probe in hyphenated.required_probes] == ["p"]
    assert [probe.name for probe in underscored.required_probes] == ["p"]


def test_a_skill_with_no_probes_declares_none() -> None:
    assert _skill("name: example\n").required_probes == []
