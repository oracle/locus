"""A step names a capability; the skill supplies the tools.

`expected_tools` makes a PROCEDURE — a business artefact written by an
operations lead — depend on the names of individual callables. Rename a tool,
swap a vendor, move a capability behind an MCP server, and the procedure breaks.
`uses` lets a step say "establish the blast radius" and let the skill decide
what that means.

This is optic's design (`guidance.skill_refs`), and the reason it had to wait
for `required_probes`: enforcing a skill's `allowed_tools` needs a MOMENT at
which it applies. A skill is prose folded into a system prompt — on its own
there is no such moment. A step supplies one. Outside a step that names it, a
skill still constrains nothing, which is honest rather than convenient.
"""

from __future__ import annotations

from tulip.playbooks.enforcer import PlaybookEnforcer
from tulip.playbooks.models import Playbook, PlaybookStep, RequiredProbe
from tulip.skills.models import Skill


TRIAGE = Skill(
    name="blast-radius",
    description="Establish which instances are affected and whether it is spreading.",
    instructions="Query the error count, then the pool.",
    allowed_tools=["query_metrics", "list_instances"],
    required_probes=[
        RequiredProbe(name="error_count", match="ora_04031_error_count"),
        RequiredProbe(name="pool", match="shared_pool_free"),
    ],
)


def _enforcer(*, with_skill: bool = True, **step_kw: object) -> PlaybookEnforcer:
    base: dict[str, object] = {
        "id": "investigate",
        "description": "Establish the blast radius.",
        "required": True,
        "uses": ["blast-radius"],
    }
    base.update(step_kw)
    playbook = Playbook(id="p", name="P", steps=[PlaybookStep(**base)])  # type: ignore[arg-type]
    return PlaybookEnforcer.from_playbook(
        playbook, skills={"blast-radius": TRIAGE} if with_skill else None
    )


def test_a_step_inherits_the_evidence_its_skill_requires() -> None:
    e = _enforcer()
    step = e.plan.playbook.steps[0]

    assert [p.name for p in e.effective_probes(step)] == ["error_count", "pool"]


def test_the_skills_allow_list_now_binds_inside_the_step() -> None:
    """The whole reason `uses` had to exist: allowed_tools finally has a scope."""
    e = _enforcer()

    assert e.validate_tool_call("query_metrics").allowed
    refused = e.validate_tool_call("delete_database")
    assert not refused.allowed
    assert refused.violation is not None
    assert refused.violation.violation_type == "tool_outside_skill"
    assert "blast-radius" in refused.violation.message


def test_a_step_that_names_no_skill_constrains_nothing_by_this_route() -> None:
    """A skill outside a step still gates nothing — that is the honest answer."""
    e = _enforcer(uses=[])
    step = e.plan.playbook.steps[0]

    assert e.allowed_tools_for(step) is None
    assert e.validate_tool_call("anything_at_all").allowed


def test_an_unresolvable_skill_reference_is_inert_not_fatal() -> None:
    """A playbook may be enforced where the skill bodies are not loaded.

    Failing there would make the enforcer refuse to run rather than enforce
    less, which trades a working guarantee for no guarantee at all.
    """
    e = _enforcer(with_skill=False)
    step = e.plan.playbook.steps[0]

    assert e.allowed_tools_for(step) is None
    assert e.effective_probes(step) == []
    assert e.validate_tool_call("query_metrics").allowed


def test_the_step_may_sharpen_a_skills_probe_and_wins() -> None:
    """Same name, step-declared: the more specific author's intent survives."""
    sharper = RequiredProbe(name="pool", match="shared_pool_free_percent")
    e = _enforcer(required_probes=[sharper])
    step = e.plan.playbook.steps[0]

    pool = next(p for p in e.effective_probes(step) if p.name == "pool")
    assert pool.match == "shared_pool_free_percent"


def test_evidence_from_a_skills_probes_is_matched_and_scored() -> None:
    e = _enforcer()
    e.record_tool_call("query_metrics", arguments={"expr": "sum(ora_04031_error_count)"})
    e.complete_current_step()

    violation = next(v for v in e.violations if v.violation_type == "evidence_incomplete")
    assert "pool" in violation.message
    assert "1/2 matched" in violation.message


def test_a_fully_evidenced_step_scores_one() -> None:
    e = _enforcer()
    e.record_tool_call("query_metrics", arguments={"expr": "ora_04031_error_count"})
    e.record_tool_call("query_metrics", arguments={"expr": "shared_pool_free"})
    e.complete_current_step()

    assert e.adherence_score() == 1.0
    assert not [v for v in e.violations if v.violation_type == "evidence_incomplete"]


def test_adherence_counts_the_evidence_a_skill_required() -> None:
    """The bug this file's scenario found: two answers to one question.

    `adherence_score` used to live on the plan, which cannot resolve skills, so
    it counted a step's OWN probes only. A step whose evidence comes entirely
    from its skill declares none of its own — and the run reported 1.00
    adherence while a violation on the same step said 1 of 3 matched.

    Found by running the Stripe scenario, not by a fixture written to pass.
    """
    e = _enforcer()
    e.record_tool_call("query_metrics", arguments={"expr": "ora_04031_error_count"})
    e.complete_current_step()

    assert e.adherence_score() == 0.5
    violation = next(v for v in e.violations if v.violation_type == "evidence_incomplete")
    assert "1/2 matched" in violation.message
