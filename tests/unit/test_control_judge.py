# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The tool-call judge: a model that reads the call and can only tighten the gate.

Every test drives the judge with :class:`locus.testing.ScriptedModel`, so the
verdicts are fixed and no key is needed. What is under test is everything
around the model: the fold into the policy decision (never looser), the
fail-closed paths (error, unparseable reply, spent budget), the read-only
skip, the per-run cache, the audit entry and the event.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from typing import Any

import pytest

from locus.agent import Agent
from locus.control import (
    Action,
    AdmissionError,
    AuditTrail,
    ControlPolicy,
    JudgeVerdict,
    ToolCallJudge,
    admit,
    gate_tool,
)
from locus.control.judge import canonical_arguments, tighten
from locus.control.policy import ApprovalDecision, ApprovalOutcome, approve
from locus.observability import get_event_bus, run_context
from locus.testing import ScriptedModel, text, tool_call
from locus.tools.decorator import tool


_ORDER = {ApprovalOutcome.ALLOW: 0, ApprovalOutcome.REQUIRE_HUMAN: 1, ApprovalOutcome.DENY: 2}
_WORDS = {
    "allow": ApprovalOutcome.ALLOW,
    "hold": ApprovalOutcome.REQUIRE_HUMAN,
    "deny": ApprovalOutcome.DENY,
}


def _says(verdict: str, reason: str = "because") -> ScriptedModel:
    """A judge model that always returns one verdict, as the schema asks."""
    return ScriptedModel(
        [text(json.dumps({"verdict": verdict, "reason": reason}))], repeat_last=True
    )


def _open() -> ControlPolicy:
    """A policy that allows everything, so only the judge can hold or deny."""
    return ControlPolicy(require_verification_score=0.0, require_human_for=frozenset())


def _tools() -> tuple[list[str], Any, Any]:
    applied: list[str] = []

    @tool
    def terminate_instance(instance_id: str, environment: str = "staging") -> str:
        """Terminate a compute instance."""
        applied.append(instance_id)
        return f"terminated {instance_id}"

    @tool(read_only=True)
    def list_instances(environment: str) -> str:
        """List compute instances."""
        return "i-1, i-2"

    return applied, terminate_instance, list_instances


# ---------------------------------------------------------------------------
# Each verdict, through the gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_allow_verdict_lets_the_call_run() -> None:
    applied, terminate, _ = _tools()
    model = _says("allow", "staging instance, within intent")
    trail = AuditTrail()
    gated = gate_tool(terminate, policy=_open(), trail=trail, judge=ToolCallJudge(model))

    assert await gated.execute(instance_id="i-1") == "terminated i-1"
    assert applied == ["i-1"]
    entry = trail.records()[0].payload
    assert entry["outcome"] == "allow"
    assert entry["judge"]["verdict"] == "allow"
    assert "tool-call judge advised allow; no escalation" in entry["reason"]


@pytest.mark.asyncio
async def test_a_hold_verdict_holds_the_call() -> None:
    applied, terminate, _ = _tools()
    trail = AuditTrail()
    judge = ToolCallJudge(_says("hold", "production instance"), intent="never touch production")
    gated = gate_tool(terminate, policy=_open(), trail=trail, judge=judge)

    refusal = json.loads(await gated.execute(instance_id="i-9", environment="production"))

    assert applied == []
    assert refusal["status"] == "held_for_approval"
    assert refusal["outcome"] == "require_human"
    assert "production instance" in refusal["reason"]
    entry = trail.records()[0].payload
    assert entry["outcome"] == "require_human"
    assert entry["judge"] == {
        "model": "ScriptedModel",
        "verdict": "hold",
        "reason": "production instance",
        "cache_hit": False,
        "source": "model",
    }


@pytest.mark.asyncio
async def test_a_deny_verdict_denies_the_call() -> None:
    applied, terminate, _ = _tools()
    trail = AuditTrail()
    gated = gate_tool(terminate, policy=_open(), trail=trail, judge=ToolCallJudge(_says("deny")))

    refusal = json.loads(await gated.execute(instance_id="i-9"))

    assert applied == []
    assert refusal["status"] == "denied"
    assert refusal["outcome"] == "deny"
    assert trail.records()[0].payload["judge"]["verdict"] == "deny"


@pytest.mark.asyncio
async def test_admit_takes_a_judgement_directly() -> None:
    ran: list[str] = []

    async def perform() -> str:
        ran.append("x")
        return "done"

    verdict = JudgeVerdict(tool="wipe", verdict="deny", reason="out of scope", model="m")
    with pytest.raises(AdmissionError) as info:
        await admit(Action(name="wipe"), perform, policy=_open(), judgement=verdict)
    assert ran == []
    assert info.value.decision.outcome == "deny"
    assert info.value.decision.policy_outcome == "allow"
    assert info.value.decision.escalated_by_model


# ---------------------------------------------------------------------------
# The rule: tighten, never loosen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy_outcome", list(_ORDER))
@pytest.mark.parametrize("word", list(_WORDS))
def test_tighten_takes_the_stricter_of_policy_and_judge(policy_outcome: str, word: str) -> None:
    decision = ApprovalDecision(
        outcome=policy_outcome,
        reason="r",
        action=Action(name="a"),
        checks=["r"] if policy_outcome != "allow" else [],
        policy_outcome=policy_outcome,
    )
    folded = tighten(decision, JudgeVerdict(tool="a", verdict=word, reason="why", model="m"))

    expected = max(policy_outcome, _WORDS[word], key=lambda o: _ORDER[o])
    assert folded.outcome == expected
    assert _ORDER[folded.outcome] >= _ORDER[policy_outcome], "a judge widened a decision"
    assert folded.policy_outcome == policy_outcome, "the policy's own outcome is the ground truth"
    assert folded.checks[-1].startswith("tool-call judge ")
    assert folded.reason.endswith(folded.checks[-1])


@pytest.mark.asyncio
async def test_a_policy_hold_stands_when_the_judge_says_allow() -> None:
    applied, terminate, _ = _tools()
    trail = AuditTrail()
    policy = ControlPolicy(
        require_verification_score=0.0, require_human_for=frozenset({"production"})
    )
    gated = gate_tool(
        terminate,
        policy=policy,
        action=lambda name, args: Action(name=name, environment=str(args.get("environment"))),
        trail=trail,
        judge=ToolCallJudge(_says("allow", "looks fine to me")),
    )

    refusal = json.loads(await gated.execute(instance_id="i-9", environment="production"))

    assert applied == []
    assert refusal["outcome"] == "require_human"
    entry = trail.records()[0].payload
    assert entry["judge"]["verdict"] == "allow"
    assert "require human approval" in entry["reason"]
    assert "no escalation" in entry["reason"]


def test_a_policy_deny_stands_when_the_judge_says_allow() -> None:
    policy = ControlPolicy(require_verification_score=0.0, deny_for=frozenset({"irreversible"}))
    decision = approve(Action(name="wipe", tags=frozenset({"irreversible"})), policy=policy)
    folded = tighten(decision, JudgeVerdict(tool="wipe", verdict="allow", reason="ok", model="m"))
    assert folded.outcome == "deny"


def test_an_unknown_verdict_word_reads_as_a_hold() -> None:
    assert (
        JudgeVerdict(tool="t", verdict="sure", reason="").outcome == ApprovalOutcome.REQUIRE_HUMAN
    )


def test_tighten_keeps_an_advisors_opinion_and_fills_it_in_otherwise() -> None:
    base = ApprovalDecision(outcome="allow", reason="ok", action=Action(name="a"))
    judged = JudgeVerdict(tool="a", verdict="hold", reason="why", model="m")
    assert tighten(base, judged).model_outcome == "require_human"
    with_advisor = ApprovalDecision(
        outcome="allow", reason="ok", action=Action(name="a"), model_outcome="allow"
    )
    assert tighten(with_advisor, judged).model_outcome == "allow"
    assert tighten(with_advisor, judged).outcome == "require_human"


# ---------------------------------------------------------------------------
# Budget and latency: read-only skip, cache, budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_read_only_tool_is_not_judged() -> None:
    _, _, list_instances = _tools()
    model = _says("deny", "would deny if asked")
    trail = AuditTrail()
    gated = gate_tool(list_instances, policy=_open(), trail=trail, judge=ToolCallJudge(model))

    assert await gated.execute(environment="production") == "i-1, i-2"
    assert model.call_count == 0
    entry = trail.records()[0].payload
    assert entry["outcome"] == "allow"
    assert entry["judge"]["source"] == "read_only"
    assert entry["judge"]["model"] == ""
    assert "read-only tool" in entry["reason"]


def test_read_only_is_off_by_default_and_survives_gating() -> None:
    _, terminate, list_instances = _tools()
    assert terminate.read_only is False
    assert list_instances.read_only is True
    assert gate_tool(list_instances, policy=_open()).read_only is True


@pytest.mark.asyncio
async def test_a_repeated_identical_call_hits_the_cache() -> None:
    applied, terminate, _ = _tools()
    model = _says("allow")
    trail = AuditTrail()
    gated = gate_tool(terminate, policy=_open(), trail=trail, judge=ToolCallJudge(model))

    await gated.execute(instance_id="i-1", environment="staging")
    await gated.execute(environment="staging", instance_id="i-1")  # same call, other key order
    await gated.execute(instance_id="i-2", environment="staging")  # a different call

    assert model.call_count == 2
    first, second, third = (rec.payload["judge"] for rec in trail.records())
    assert first["cache_hit"] is False
    assert first["source"] == "model"
    assert second["cache_hit"] is True
    assert second["source"] == "cache"
    assert third["cache_hit"] is False
    assert "(cached)" in trail.records()[1].payload["reason"]
    assert applied == ["i-1", "i-1", "i-2"]


def test_canonical_arguments_ignore_key_order() -> None:
    assert canonical_arguments({"b": 1, "a": 2}) == canonical_arguments({"a": 2, "b": 1})
    assert canonical_arguments({"n": 1}) != canonical_arguments({"n": 2})
    assert canonical_arguments({"when": object}) == canonical_arguments({"when": object})


@pytest.mark.asyncio
async def test_a_spent_budget_holds_rather_than_allows() -> None:
    applied, terminate, _ = _tools()
    model = _says("allow")
    trail = AuditTrail()
    judge = ToolCallJudge(model, budget=1)
    gated = gate_tool(terminate, policy=_open(), trail=trail, judge=judge)

    assert await gated.execute(instance_id="i-1") == "terminated i-1"
    refusal = json.loads(await gated.execute(instance_id="i-2"))
    assert await gated.execute(instance_id="i-1") == "terminated i-1"  # cached: free

    assert model.call_count == 1
    assert applied == ["i-1", "i-1"]
    assert refusal["outcome"] == "require_human"
    assert "budget of 1 spent" in refusal["reason"]
    held = trail.records()[1].payload["judge"]
    assert held["source"] == "budget"
    assert held["verdict"] == "hold"
    assert judge.calls() == 1


def test_a_negative_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="budget"):
        ToolCallJudge(_says("allow"), budget=-1)


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


class _Broken:
    async def complete(self, **kwargs: Any) -> Any:
        raise ConnectionError("judge endpoint down")


@pytest.mark.asyncio
async def test_a_judge_model_error_holds_and_is_on_the_entry() -> None:
    applied, terminate, _ = _tools()
    trail = AuditTrail()
    gated = gate_tool(terminate, policy=_open(), trail=trail, judge=ToolCallJudge(_Broken()))

    refusal = json.loads(await gated.execute(instance_id="i-1"))

    assert applied == []
    assert refusal["outcome"] == "require_human"
    judge = trail.records()[0].payload["judge"]
    assert judge["source"] == "error"
    assert judge["verdict"] == "hold"
    assert judge["error"] == "ConnectionError: judge endpoint down"


@pytest.mark.asyncio
async def test_a_reply_that_is_not_a_verdict_holds() -> None:
    applied, terminate, _ = _tools()
    trail = AuditTrail()
    chatty = ScriptedModel([text("PROCEED, this looks fine.")], repeat_last=True)
    gated = gate_tool(terminate, policy=_open(), trail=trail, judge=ToolCallJudge(chatty))

    refusal = json.loads(await gated.execute(instance_id="i-1"))

    assert applied == []
    assert refusal["outcome"] == "require_human"
    judge = trail.records()[0].payload["judge"]
    assert judge["source"] == "error"
    assert judge["error"].startswith("unparseable verdict")


@pytest.mark.asyncio
async def test_a_verdict_outside_the_three_words_is_not_a_verdict() -> None:
    applied, terminate, _ = _tools()
    model = ScriptedModel([text('{"verdict": "maybe", "reason": "?"}')], repeat_last=True)
    gated = gate_tool(terminate, policy=_open(), judge=ToolCallJudge(model))
    assert json.loads(await gated.execute(instance_id="i-1"))["outcome"] == "require_human"
    assert applied == []


@pytest.mark.asyncio
async def test_no_model_at_all_holds() -> None:
    verdict = await ToolCallJudge().judge("terminate_instance", {"instance_id": "i-1"})
    assert verdict.verdict == "hold"
    assert verdict.source == "error"


# ---------------------------------------------------------------------------
# The event, and the catalogue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_verdict_is_emitted_on_the_bus() -> None:
    _, terminate, _ = _tools()
    gated = gate_tool(terminate, policy=_open(), judge=ToolCallJudge(_says("hold", "nope")))
    bus = get_event_bus()
    seen: list[Any] = []

    async def consumer() -> None:
        async for ev in bus.subscribe("judge-run"):
            seen.append(ev)
            if ev.event_type == "stop":
                return

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    async with run_context("judge-run"):
        await gated.execute(instance_id="i-1")
        await gated.execute(instance_id="i-1")
        from locus.observability import emit

        await emit("stop")
    await asyncio.wait_for(task, timeout=2.0)

    judged = [ev for ev in seen if ev.event_type == "control.judge.decided"]
    assert len(judged) == 2
    assert judged[0].data == {
        "tool_name": "terminate_instance",
        "verdict": "hold",
        "reason": "nope",
        "judge_model": "ScriptedModel",
        "source": "model",
        "cache_hit": False,
    }
    assert judged[1].data["source"] == "cache"
    assert judged[1].data["cache_hit"] is True


def test_the_judge_event_is_in_the_catalogue() -> None:
    import locus.observability.emit  # noqa: F401 - make sure the module is loaded

    emit_mod = sys.modules["locus.observability.emit"]
    assert emit_mod.EV_CONTROL_JUDGE_DECIDED == "control.judge.decided"

    page = pathlib.Path(__file__).resolve().parents[2] / "docs" / "concepts" / "sse-events.md"
    if not page.is_file():
        pytest.skip("docs are not part of this checkout")
    catalogue = page.read_text()
    missing = [
        getattr(emit_mod, name)
        for name in dir(emit_mod)
        if name.startswith("EV_") and f"`{getattr(emit_mod, name)}`" not in catalogue
    ]
    assert missing == [], f"events missing from docs/concepts/sse-events.md: {missing}"


# ---------------------------------------------------------------------------
# As a hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_as_a_hook_a_deny_cancels_the_call_before_it_runs() -> None:
    applied, terminate, _ = _tools()
    trail = AuditTrail()
    judge = ToolCallJudge(_says("deny", "not in scope"), trail=trail)
    agent_model = ScriptedModel([tool_call("terminate_instance", instance_id="i-7"), text("ok")])
    agent = Agent(model=agent_model, tools=[terminate], hooks=[judge])

    result = await agent.arun("terminate i-7")

    assert applied == []
    assert result.text == "ok"
    refusal = json.loads(result.tool_executions[0].result)
    assert refusal["status"] == "denied"
    assert refusal["action"] == "terminate_instance"
    assert "not in scope" in refusal["reason"]
    rec = trail.records()[0]
    assert rec.event_type == "tool-call-judged"
    assert rec.payload["outcome"] == "deny"
    assert rec.payload["judge"]["verdict"] == "deny"


@pytest.mark.asyncio
async def test_as_a_hook_an_allow_lets_the_call_run_and_is_recorded() -> None:
    applied, terminate, _ = _tools()
    trail = AuditTrail()
    judge = ToolCallJudge(_says("allow"), trail=trail)
    agent_model = ScriptedModel([tool_call("terminate_instance", instance_id="i-7"), text("ok")])
    agent = Agent(model=agent_model, tools=[terminate], hooks=[judge])

    await agent.arun("terminate i-7")

    assert applied == ["i-7"]
    assert trail.records()[0].payload["outcome"] == "allow"


@pytest.mark.asyncio
async def test_a_judge_without_a_model_borrows_the_agents() -> None:
    applied, terminate, list_instances = _tools()
    judge = ToolCallJudge()
    # One script serves both roles: the agent's turn, then the judge's reply.
    shared = ScriptedModel(
        [
            tool_call("terminate_instance", instance_id="i-7"),
            text(json.dumps({"verdict": "hold", "reason": "ask first"})),
            text("held"),
        ]
    )
    agent = Agent(model=shared, tools=[terminate, list_instances], hooks=[judge])

    assert judge.model is shared
    result = await agent.arun("terminate i-7")

    assert applied == []
    assert result.text == "held"
    assert json.loads(result.tool_executions[0].result)["status"] == "held_for_approval"


@pytest.mark.asyncio
async def test_as_a_hook_read_only_tools_learned_from_the_agent_are_skipped() -> None:
    _, terminate, list_instances = _tools()
    judge_model = _says("deny")
    judge = ToolCallJudge(judge_model)
    agent_model = ScriptedModel([tool_call("list_instances", environment="prod"), text("ok")])
    agent = Agent(model=agent_model, tools=[terminate, list_instances], hooks=[judge])

    result = await agent.arun("list")

    assert judge_model.call_count == 0
    assert result.tool_executions[0].result == "i-1, i-2"


@pytest.mark.asyncio
async def test_hook_and_gate_together_ask_the_model_once() -> None:
    applied, terminate, _ = _tools()
    judge_model = _says("allow")
    trail = AuditTrail()
    judge = ToolCallJudge(judge_model, trail=trail)
    gated = gate_tool(terminate, policy=_open(), trail=trail, judge=judge)
    agent_model = ScriptedModel([tool_call("terminate_instance", instance_id="i-7"), text("ok")])
    agent = Agent(model=agent_model, tools=[gated], hooks=[judge])

    await agent.arun("terminate i-7")

    assert applied == ["i-7"]
    assert judge_model.call_count == 1
    kinds = [rec.event_type for rec in trail.records()]
    assert kinds == ["tool-call-judged", "action-admission"]
    assert trail.records()[0].payload["judge"]["cache_hit"] is False
    assert trail.records()[1].payload["judge"]["cache_hit"] is True


@pytest.mark.asyncio
async def test_a_new_run_starts_with_a_fresh_budget_and_cache() -> None:
    applied, terminate, _ = _tools()
    judge_model = _says("allow")
    judge = ToolCallJudge(judge_model, budget=1)
    agent_model = ScriptedModel(
        [tool_call("terminate_instance", instance_id="i-7"), text("ok")], repeat_last=False
    )
    agent = Agent(model=agent_model, tools=[terminate], hooks=[judge])
    await agent.arun("one")
    assert judge.calls() == 0, "the run's state is dropped when it ends"

    again = ScriptedModel([tool_call("terminate_instance", instance_id="i-7"), text("ok")])
    agent = Agent(model=again, tools=[terminate], hooks=[judge])
    await agent.arun("two")

    assert applied == ["i-7", "i-7"], "the second run was judged afresh, not held on a spent budget"
    assert judge_model.call_count == 2
