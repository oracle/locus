"""The grouped playbook shape and probe frontmatter, pinned without the corpus.

`test_grouped_corpus_loads.py` checks the files on disk under
tests/fixtures/grouped_corpus; this pins each rule on its own, with inline
fixtures, so a failure names the rule that broke.
"""

from __future__ import annotations

from tulip.playbooks.loader import _flatten_step_groups, load_playbook
from tulip.skills.models import Skill


GROUPED = {
    "id": "diskfull",
    "title": "Disk-full investigation",
    "summary": "Establish blast radius, then confirm disk pressure.",
    "mode": "diagnosis",
    "completion": {"conclusion_requirements": {"all_applicable_required_steps_resolved": True}},
    "decision_policy": {"id": "remediation", "rules": ["Choose exactly one."]},
    "step_groups": [
        {
            "id": "investigation",
            "title": "Establish blast radius.",
            "goal": "Determine which instances are impacted.",
            "steps": [
                {
                    "id": "errors",
                    "title": "Count the errors.",
                    "goal": "Read the disk-full error counts by instance.",
                    "required": True,
                    "priority": "critical",
                    "min_tool_calls": 2,
                    "max_tool_calls": 6,
                    "guidance": {"skill_refs": ["disk-full-investigation"]},
                }
            ],
        },
        {
            "id": "confirmation",
            "title": "Confirm the disk state.",
            "goal": "Confirm volume pressure.",
            "steps": [
                {"id": "errors", "goal": "Read pool metrics.", "required": False},
            ],
        },
    ],
}


def test_groups_flatten_in_order() -> None:
    flat = _flatten_step_groups(GROUPED)
    assert [step["id"] for step in flat["steps"]] == ["errors", "confirmation.errors"]


def test_a_repeated_step_id_is_qualified_by_its_group() -> None:
    """Both groups call their step 'errors'; the enforcer keys on a flat id."""
    flat = _flatten_step_groups(GROUPED)
    assert flat["steps"][1]["id"] == "confirmation.errors"


def test_skill_refs_become_uses() -> None:
    flat = _flatten_step_groups(GROUPED)
    assert flat["steps"][0]["uses"] == ["disk-full-investigation"]
    assert "guidance" not in flat["steps"][0]


def test_the_goal_becomes_the_description_and_the_group_heading_survives() -> None:
    """A phase is context a reader needs; dropping it loses why the step exists."""
    flat = _flatten_step_groups(GROUPED)
    first = flat["steps"][0]["description"]
    assert "Determine which instances are impacted." in first  # the group's goal
    assert "Read the disk-full error counts by instance." in first  # the step's own


def test_the_playbook_is_named_from_its_title() -> None:
    flat = _flatten_step_groups(GROUPED)
    assert flat["name"] == "Disk-full investigation"
    assert flat["description"] == "Establish blast radius, then confirm disk pressure."


def test_unmodelled_fields_are_carried_not_dropped() -> None:
    """completion / decision_policy / mode are not modelled YET.

    Carried in metadata so the day they are, the data is already there — and so
    an author who wrote them is not silently ignored.
    """
    flat = _flatten_step_groups(GROUPED)
    assert flat["metadata"]["mode"] == "diagnosis"
    assert flat["metadata"]["completion"]["conclusion_requirements"]
    assert flat["metadata"]["decision_policy"]["id"] == "remediation"
    for key in ("completion", "decision_policy", "mode"):
        assert key not in flat


def test_priority_is_kept_on_the_step() -> None:
    flat = _flatten_step_groups(GROUPED)
    assert flat["steps"][0]["metadata"]["priority"] == "critical"


def test_an_already_flat_playbook_is_untouched() -> None:
    """Additive: no existing playbook changes meaning."""
    flat_in = {"id": "p", "name": "P", "steps": [{"id": "a", "description": "b"}]}
    assert _flatten_step_groups(flat_in) is flat_in


def test_the_grouped_shape_loads_end_to_end() -> None:
    playbook = load_playbook(GROUPED)
    assert playbook.name == "Disk-full investigation"
    assert [s.id for s in playbook.steps] == ["errors", "confirmation.errors"]
    assert playbook.steps[0].uses == ["disk-full-investigation"]
    assert playbook.steps[0].min_tool_calls == 2


SKILL_MD = """---
name: disk-full-investigation
description: Establish blast radius for disk-full errors.
allowed-tools: ai_query_prometheus
min-tool-calls: 2
max-tool-calls: 10
required_probes:
  - name: disk_full_count
    match: "node_disk_full_error_count"
    description: The error counts.
  - name: broken_probe
  - not_a_mapping
---

# Body
Query the error counts.
"""


def test_probes_parse_from_frontmatter() -> None:
    skill = Skill.from_content(SKILL_MD)
    assert [p.name for p in skill.required_probes] == ["disk_full_count"]
    assert skill.required_probes[0].match == "node_disk_full_error_count"


def test_a_malformed_probe_is_dropped_not_fatal() -> None:
    """A skill is a document an operations person edits.

    Refusing to load the whole thing because one probe lost its `match` takes
    the instructions away too — one fewer obligation beats no procedure.
    """
    skill = Skill.from_content(SKILL_MD)
    assert len(skill.required_probes) == 1
    assert skill.instructions.startswith("# Body")


def test_both_spellings_of_the_probe_key_are_accepted() -> None:
    """Skill files mix `allowed-tools` with `required_probes`."""
    hyphenated = SKILL_MD.replace("required_probes:", "required-probes:")
    assert [p.name for p in Skill.from_content(hyphenated).required_probes] == ["disk_full_count"]


def test_effort_bounds_parse() -> None:
    skill = Skill.from_content(SKILL_MD)
    assert (skill.min_tool_calls, skill.max_tool_calls) == (2, 10)


def test_a_skill_without_bounds_or_probes_still_loads() -> None:
    plain = "---\nname: n\ndescription: d\n---\n\nbody"
    skill = Skill.from_content(plain)
    assert skill.required_probes == []
    assert skill.min_tool_calls is None


def test_malformed_groups_and_steps_are_skipped_not_fatal() -> None:
    """A hand-edited JSON file with a stray value should not take the run down."""
    messy = {
        "id": "m",
        "title": "Messy",
        "step_groups": [
            "not-a-group",
            {"id": "g", "steps": ["not-a-step", {"id": "ok", "goal": "do it"}]},
        ],
    }
    flat = _flatten_step_groups(messy)
    assert [s["id"] for s in flat["steps"]] == ["ok"]


def test_a_step_with_no_id_gets_a_positional_one() -> None:
    """An id is how the enforcer keys a step; absence cannot mean collision."""
    flat = _flatten_step_groups(
        {"id": "p", "title": "P", "step_groups": [{"steps": [{"goal": "a"}, {"goal": "b"}]}]}
    )
    ids = [s["id"] for s in flat["steps"]]
    assert len(set(ids)) == 2, ids


def test_a_title_only_step_uses_its_title_as_the_description() -> None:
    flat = _flatten_step_groups(
        {
            "id": "p",
            "title": "P",
            "step_groups": [{"steps": [{"id": "s", "title": "Just a title"}]}],
        }
    )
    assert flat["steps"][0]["description"] == "Just a title"


def test_the_name_falls_back_through_playbook_id_then_id() -> None:
    """Grouped files carry `playbook_id`; some carry only `id`."""
    assert (
        _flatten_step_groups({"id": "x", "playbook_id": "pb.x", "step_groups": []})["name"]
        == "pb.x"
    )
    assert _flatten_step_groups({"id": "x", "step_groups": []})["name"] == "x"


def test_a_playbook_carrying_none_of_the_unmodelled_fields_gets_no_metadata() -> None:
    """Don't manufacture an empty metadata bag on every playbook."""
    flat = _flatten_step_groups({"id": "x", "title": "X", "step_groups": []})
    assert "metadata" not in flat


def test_product_and_service_type_ride_along_on_the_step() -> None:
    flat = _flatten_step_groups(
        {
            "id": "p",
            "title": "P",
            "step_groups": [
                {"steps": [{"id": "s", "goal": "g", "product": "billing", "service_type": "db"}]}
            ],
        }
    )
    assert flat["steps"][0]["metadata"] == {"product": "billing", "service_type": "db"}
