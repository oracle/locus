# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Gating a Tulip tool, which until now only non-Tulip agents could do.

`tulip-frameworks` has shipped `gate_langchain_tool` and its siblings for a
while: wrap one tool and the model's decision goes through `admit()` before
anything happens. Building on Tulip itself, you hand-wrote the try/except —
so a LangChain user got better ergonomics from Tulip than a Tulip user did, on
the one feature the project is built around.

The tests that matter are the ones a governance feature is worthless without:
the side effect really does not happen, the model cannot tell it is gated, and
a policy that admits still lets work through. A gate that blocks everything is
as useless as one that blocks nothing, and easier to ship by accident.
"""

from __future__ import annotations

import json

import pytest

from tulip.agent import Agent
from tulip.control import Action, AdmissionError, AuditTrail, ControlPolicy, gate_tool
from tulip.testing import ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


PAID: list[tuple[str, float]] = []


@tool
def issue_refund(order_id: str, amount_usd: float) -> str:
    """Issue a customer refund."""
    PAID.append((order_id, amount_usd))
    return f"refunded ${amount_usd:,.2f}"


@tool
async def async_refund(order_id: str, amount_usd: float) -> str:
    """Issue a customer refund, asynchronously."""
    PAID.append((order_id, amount_usd))
    return "refunded"


def _by_amount(name: str, kwargs: dict) -> Action:
    """Risk scales with the payout, not with the tool's name."""
    amount = float(kwargs.get("amount_usd", 0))
    return Action(
        name=name,
        asset=str(kwargs.get("order_id", "")),
        kind="payment",
        environment="production",
        blast_radius=max(1, int(amount // 1_000)),
    )


def _policy() -> ControlPolicy:
    # Gate on amount alone: verification is not the question for a refund, and
    # holding every production action would hold the $12 credit too.
    return ControlPolicy(
        require_verification_score=0.0, max_blast_radius=1, require_human_for=frozenset()
    )


@pytest.fixture(autouse=True)
def _clear() -> None:
    PAID.clear()


def _run(tools: list, amount: float) -> object:
    agent = Agent(
        model=ScriptedModel(
            [tool_call("issue_refund", order_id="ord-4821", amount_usd=amount), text("Done.")]
        ),
        tools=tools,
    )
    return agent.run_sync("refund it")


# --------------------------------------------------------------------------
# The side effect
# --------------------------------------------------------------------------


def test_an_ungated_tool_pays_out() -> None:
    """The baseline. Without it, the gated case proves nothing."""
    _run([issue_refund], 4_000_000.0)

    assert PAID == [("ord-4821", 4_000_000.0)]


def test_the_gate_stops_the_side_effect() -> None:
    """Same agent, same model, same prompt — one wrapper."""
    gated = gate_tool(issue_refund, policy=_policy(), action=_by_amount)

    _run([gated], 4_000_000.0)

    assert PAID == [], "the gate let a $4,000,000 refund through"


def test_a_policy_that_admits_still_lets_work_through() -> None:
    """A gate that blocks everything is as useless as one that blocks nothing."""
    gated = gate_tool(issue_refund, policy=_policy(), action=_by_amount)

    _run([gated], 12.0)

    assert PAID == [("ord-4821", 12.0)]


@pytest.mark.asyncio
async def test_an_async_tool_is_gated_too() -> None:
    """Half the tools in a real agent are coroutines."""
    gated = gate_tool(async_refund, policy=_policy(), action=_by_amount)

    await gated.execute(order_id="ord-1", amount_usd=4_000_000.0)

    assert PAID == []


# --------------------------------------------------------------------------
# What the model sees
# --------------------------------------------------------------------------


def test_the_model_cannot_tell_the_tool_is_gated() -> None:
    """The point of a structural control: nothing to notice, nothing to route
    around. A changed name or schema would also break every prompt that
    mentions the tool."""
    gated = gate_tool(issue_refund, policy=_policy())

    assert gated.name == issue_refund.name
    assert gated.description == issue_refund.description
    assert gated.parameters == issue_refund.parameters
    assert gated.labels == issue_refund.labels


def test_a_refusal_reaches_the_model_as_a_readable_result() -> None:
    """So the agent can explain itself instead of the run exploding."""
    gated = gate_tool(issue_refund, policy=_policy(), action=_by_amount)

    result = _run([gated], 4_000_000.0)
    payload = json.loads(result.tool_executions[0].result)

    assert payload["outcome"] == "require_human"
    assert payload["action"] == "issue_refund"
    assert payload["asset"] == "ord-4821"
    assert payload["reason"]


def test_the_refusal_shape_matches_the_framework_bridges() -> None:
    """A policy should read the same whether the agent is Tulip-native or
    wrapped from LangChain; two shapes would make that a per-framework detail."""
    gated = gate_tool(issue_refund, policy=_policy(), action=_by_amount)

    payload = json.loads(_run([gated], 4_000_000.0).tool_executions[0].result)

    assert set(payload) == {"status", "outcome", "action", "asset", "reason"}
    assert payload["status"] == "held_for_approval"


@pytest.mark.asyncio
async def test_raise_is_available_for_a_caller_that_wants_to_stop() -> None:
    gated = gate_tool(issue_refund, policy=_policy(), action=_by_amount, on_refusal="raise")

    with pytest.raises(AdmissionError):
        await gated.execute(order_id="ord-1", amount_usd=4_000_000.0)


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


def test_every_decision_lands_on_the_trail() -> None:
    """Both the hold and the payment — a trail with only refusals cannot show
    that the gate was ever consulted for the work that went through."""
    trail = AuditTrail()
    gated = gate_tool(issue_refund, policy=_policy(), action=_by_amount, trail=trail)

    _run([gated], 4_000_000.0)
    _run([gated], 12.0)

    outcomes = [record.payload["outcome"] for record in trail.records()]
    assert outcomes == ["require_human", "allow"]
    assert trail.verify()


def test_it_works_without_a_trail() -> None:
    """Recording is opt-in; the gate still decides."""
    gated = gate_tool(issue_refund, policy=_policy(), action=_by_amount)

    _run([gated], 4_000_000.0)

    assert PAID == []


# --------------------------------------------------------------------------
# Deriving the action
# --------------------------------------------------------------------------


def test_without_an_action_the_tool_name_is_still_matchable() -> None:
    """``default_action`` tags with the tool's own name, so a policy can gate
    one specific tool without the caller writing a derivation."""
    trail = AuditTrail()
    gated = gate_tool(
        issue_refund,
        policy=ControlPolicy(
            require_verification_score=0.0, require_human_for=frozenset({"issue_refund"})
        ),
        trail=trail,
    )

    _run([gated], 12.0)

    assert PAID == []
    assert trail.records()[0].payload["outcome"] == "require_human"


def test_the_original_tool_is_left_ungated() -> None:
    """Trusted code calling the same function directly must not be affected."""
    gate_tool(issue_refund, policy=_policy(), action=_by_amount)

    _run([issue_refund], 4_000_000.0)

    assert PAID == [("ord-4821", 4_000_000.0)]
