"""A step is done when the evidence exists, not when a tool was called.

`expected_tools` asks "was the right tool called?". `required_probes` asks "was
the right thing LOOKED AT?" — a different question, and the one an auditor has.
An agent can call the correct tool against the wrong target and satisfy the
first while failing the second.

Two failure modes from the literature are what this closes:

* the **evidence-grounding defect** — treating a claim as sufficient evidence
  for action without resolving it against evidence that was available
  (arXiv 2605.08828);
* **Premature Conclusion** — concluding while necessary steps are still missing
  (arXiv 2606.04874), which `min_tool_calls` puts a floor under.

The design is optic's (`observai/optic`), where a skill declares
`required_probes` and coverage is scored per step; see
REVIEW-what-optic-has-that-tulip-lost.md.
"""

from __future__ import annotations

from tulip.playbooks.enforcer import PlaybookEnforcer
from tulip.playbooks.models import Playbook, PlaybookStep, RequiredProbe


def _playbook(**step_kw: object) -> Playbook:
    base: dict[str, object] = {
        "id": "investigate",
        "description": "Establish the blast radius.",
        "required": True,
        "required_probes": [
            RequiredProbe(name="error_count", match="ora_04031_error_count"),
            RequiredProbe(name="shared_pool", match="shared_pool_free"),
        ],
    }
    base.update(step_kw)
    return Playbook(id="p", name="P", steps=[PlaybookStep(**base)])  # type: ignore[arg-type]


def _enforcer(**step_kw: object) -> PlaybookEnforcer:
    return PlaybookEnforcer.from_playbook(_playbook(**step_kw))


def test_evidence_is_matched_from_what_the_call_looked_at() -> None:
    e = _enforcer()
    e.record_tool_call("query_metrics", arguments={"expr": "sum(ora_04031_error_count)"})

    execution = e.plan.step_executions["investigate"]
    assert execution.matched_probes == ["error_count"]
    assert execution.probe_coverage(e.effective_probes(e.plan.playbook.steps[0])) == 0.5


def test_a_step_that_gathered_everything_is_fully_covered() -> None:
    e = _enforcer()
    e.record_tool_call("query_metrics", arguments={"expr": "sum(ora_04031_error_count)"})
    e.record_tool_call("query_metrics", arguments={"expr": "min(shared_pool_free)"})

    step = e.plan.playbook.steps[0]
    assert e.plan.step_executions["investigate"].probe_coverage(e.effective_probes(step)) == 1.0
    assert e.adherence_score() == 1.0


def test_closing_a_step_with_evidence_missing_is_a_violation() -> None:
    """The whole point: calling the right tool is not the same as doing the work."""
    e = _enforcer()
    e.record_tool_call("query_metrics", arguments={"expr": "sum(ora_04031_error_count)"})
    e.complete_current_step()

    violation = next(v for v in e.violations if v.violation_type == "evidence_incomplete")
    assert "shared_pool" in violation.message
    assert "1/2 matched" in violation.message


def test_the_evidence_is_named_so_it_can_be_gone_and_got() -> None:
    """Unmatched probes are reportable — which is what lets a run be steered back."""
    e = _enforcer()
    e.record_tool_call("query_metrics", arguments={"expr": "min(shared_pool_free)"})

    assert e.plan.step_executions["investigate"].unmatched_probes(
        e.effective_probes(e.plan.playbook.steps[0])
    ) == ["error_count"]


def test_a_probe_is_not_satisfied_by_what_merely_came_back() -> None:
    """Seeking is the agent's responsibility; receiving is partly luck.

    A tool asked something else entirely can mention the string in passing. If
    that counted, "the agent gathered the required evidence" would be satisfied
    by coincidence — which is the claim this whole mechanism exists to make
    honestly. optic matches executed queries for the same reason.
    """
    e = _enforcer()
    e.record_tool_call("run_diagnostic", arguments={"target": "db1"}, result="shared_pool_free=12M")

    assert e.plan.step_executions["investigate"].matched_probes == []

    e.record_tool_call("run_diagnostic", arguments={"metric": "shared_pool_free"})
    assert e.plan.step_executions["investigate"].matched_probes == ["shared_pool"]


def test_a_probe_matches_once_however_often_it_is_seen() -> None:
    e = _enforcer()
    for _ in range(3):
        e.record_tool_call("query_metrics", arguments={"expr": "ora_04031_error_count"})

    assert e.plan.step_executions["investigate"].matched_probes == ["error_count"]


def test_a_step_declaring_no_probes_is_covered_by_definition() -> None:
    """Otherwise every playbook written before today reports zero adherence."""
    e = _enforcer(required_probes=[])
    e.record_tool_call("anything")
    e.complete_current_step()

    assert e.adherence_score() == 1.0
    assert not [v for v in e.violations if v.violation_type == "evidence_incomplete"]


def test_a_floor_catches_the_agent_that_answered_without_looking() -> None:
    e = _enforcer(required_probes=[], min_tool_calls=2)
    e.record_tool_call("query_metrics", arguments={"expr": "x"})
    e.complete_current_step()

    violation = next(v for v in e.violations if v.violation_type == "insufficient_effort")
    assert "floor of 2" in violation.message


def test_meeting_the_floor_says_nothing() -> None:
    e = _enforcer(required_probes=[], min_tool_calls=2)
    e.record_tool_call("query_metrics", arguments={"expr": "x"})
    e.record_tool_call("query_metrics", arguments={"expr": "y"})
    e.complete_current_step()

    assert not [v for v in e.violations if v.violation_type == "insufficient_effort"]


def test_a_required_step_never_reached_is_reported_unresolved() -> None:
    """The conclusion contract: a run must not end honestly with one outstanding."""
    e = _enforcer()
    assert e.plan.unresolved_required_steps() == ["investigate"]

    e.record_tool_call(
        "query_metrics", arguments={"expr": "ora_04031_error_count shared_pool_free"}
    )
    e.complete_current_step()
    assert e.plan.unresolved_required_steps() == []


def test_a_caller_that_passes_no_evidence_gets_an_honest_zero() -> None:
    """Backward compatible, and not by pretending: no evidence, no coverage."""
    e = _enforcer()
    e.record_tool_call("query_metrics")

    assert e.plan.step_executions["investigate"].matched_probes == []
