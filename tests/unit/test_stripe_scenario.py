"""A real vendor's API, governed — Stripe, whose whole surface is two tools.

Stripe's MCP server does not expose one tool per operation. It exposes:

    stripe_api_read    → ANY GET method
    stripe_api_write   → ANY POST/PATCH/PUT/DELETE method

which means per-tool scoping, the usual answer to "what may this agent touch?",
is nearly useless against it. Allowing `stripe_api_write` allows refunds,
voiding invoices, cancelling subscriptions, deleting coupons and updating tax
settings — the tool name carries no information about the consequence.

That makes it the right scenario to test this machinery against, because the
procedure has to do the work the tool name cannot: a step admits the reader and
not the writer, and its probes match on the API PATH inside the arguments.

The pack under `examples/stripe/` is the artefact these cases exercise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tulip.playbooks.enforcer import PlaybookEnforcer
from tulip.playbooks.loader import load_playbook
from tulip.playbooks.models import Playbook
from tulip.skills.models import Skill


PACK = Path(__file__).resolve().parents[2] / "examples" / "stripe"


def _playbook() -> Playbook:
    return load_playbook(json.loads((PACK / "playbook.stripe-triage.json").read_text()))


def _skills() -> dict[str, Skill]:
    return {
        skill.name: skill
        for skill in (
            Skill.from_content(path.read_text(), path=path)
            for path in sorted((PACK / "skills").glob("*.md"))
        )
    }


@pytest.fixture
def enforcer() -> PlaybookEnforcer:
    return PlaybookEnforcer.from_playbook(_playbook(), skills=_skills())


def test_the_pack_loads_and_every_step_resolves_a_skill() -> None:
    """A `uses` that resolves to nothing is inert — the pack must not ship that."""
    playbook, skills = _playbook(), _skills()
    unresolved = [
        (step.id, ref) for step in playbook.steps for ref in step.uses if ref not in skills
    ]
    assert not unresolved, f"steps naming skills that do not exist: {unresolved}"
    assert len(playbook.steps) == 6


def test_the_read_steps_admit_the_reader_and_refuse_the_writer(
    enforcer: PlaybookEnforcer,
) -> None:
    """The whole point: same vendor, two tools, one of which can do anything."""
    enforcer.record_tool_call("stripe_api_search", arguments={"query": "refund a charge"})
    enforcer.complete_current_step()  # past discovery, into 'payments'

    assert enforcer.current_step is not None
    assert enforcer.current_step.id == "payments"
    assert enforcer.validate_tool_call("stripe_api_read").allowed

    refused = enforcer.validate_tool_call("stripe_api_write")
    assert not refused.allowed
    assert refused.violation is not None
    assert refused.violation.violation_type == "tool_outside_skill"


def test_evidence_is_matched_on_the_api_path_not_the_tool_name(
    enforcer: PlaybookEnforcer,
) -> None:
    """Every call here is `stripe_api_read`. The path is what distinguishes them.

    This is the argument-level governance a generic tool forces: the name is
    constant, so the obligation has to be expressed against the arguments.
    """
    enforcer.record_tool_call("stripe_api_search", arguments={"query": "refund"})
    enforcer.complete_current_step()

    enforcer.record_tool_call(
        "stripe_api_read", arguments={"method": "GET", "path": "/v1/charges/ch_3PxyzABC"}
    )
    assert enforcer.plan.step_executions["payments"].matched_probes == ["charge_read"]

    enforcer.record_tool_call(
        "stripe_api_read", arguments={"method": "GET", "path": "/v1/refunds?charge=ch_3PxyzABC"}
    )
    assert enforcer.plan.step_executions["payments"].matched_probes == [
        "charge_read",
        "refunds_checked",
    ]


def test_reading_the_wrong_resource_does_not_satisfy_the_step(
    enforcer: PlaybookEnforcer,
) -> None:
    """`/v1/customers` is a legitimate call that is not the evidence required."""
    enforcer.record_tool_call("stripe_api_search", arguments={"query": "refund"})
    enforcer.complete_current_step()

    enforcer.record_tool_call(
        "stripe_api_read", arguments={"method": "GET", "path": "/v1/customers/cus_abc"}
    )
    enforcer.complete_current_step()

    violation = next(v for v in enforcer.violations if v.violation_type == "evidence_incomplete")
    assert "charge_read" in violation.message
    assert "refunds_checked" in violation.message


def test_money_movement_names_the_specific_tool_not_the_generic_writer() -> None:
    """`create_refund`, never `stripe_api_write`.

    A procedure that reaches for the generic writer to issue one refund has
    handed itself Stripe's entire write surface to do it.
    """
    skills = _skills()
    movement = skills["stripe-money-movement"]
    assert movement.allowed_tools == ["create_refund"]
    assert "stripe_api_write" not in (movement.allowed_tools or [])


def test_no_read_skill_can_write_anything() -> None:
    """The read half of the pack must not carry a write tool by accident."""
    offenders = {
        name: skill.allowed_tools
        for name, skill in _skills().items()
        if name.endswith("-read")
        and any("write" in t or t == "create_refund" for t in (skill.allowed_tools or []))
    }
    assert not offenders, f"read skills carrying write capability: {offenders}"


#: The resource families Stripe documents as reachable through its MCP server
#: (docs.stripe.com/mcp, "Supported API methods"). Kept here rather than derived
#: because the point is to notice when Stripe's surface moves and ours does not.
DOCUMENTED_FAMILIES = (
    "/v1/customers",
    "/v1/charges",
    "/v1/refunds",
    "/v1/payment_intents",
    "/v1/checkout/sessions",
    "/v1/invoices",
    "/v1/invoiceitems",
    "/v1/subscriptions",
    "/v1/subscription_schedules",
    "/v1/coupons",
    "/v1/promotion_codes",
    "/v1/products",
    "/v1/prices",
    "/v1/payment_links",
    "/v1/disputes",
    "/v1/webhook_endpoints",
    "/v1/billing_portal",
    "/v1/balance",
    "/v1/balance_transactions",
    "/v1/tax",
    "/v1/payouts",
    "/v1/issuing",
    "/v2/core/accounts",
    "/v2/money_management",
)


def test_the_pack_covers_stripes_documented_surface() -> None:
    """Every family Stripe exposes has a skill that knows about it.

    Not a claim that every method is exercised — a claim that no documented
    resource family is somewhere an agent could wander with no procedure
    written for it. `stripe_api_read` will happily fetch any of these whether
    or not anyone thought about it.
    """
    corpus = "\n".join(
        f"{skill.name} {skill.description} {skill.instructions} "
        + " ".join(probe.match for probe in skill.required_probes)
        for skill in _skills().values()
    ).lower()

    missing = [family for family in DOCUMENTED_FAMILIES if family.lower() not in corpus]
    assert not missing, (
        f"Stripe resource families no skill mentions: {missing}. Either add the "
        f"skill or drop the family from DOCUMENTED_FAMILIES with a reason."
    )


def test_write_capability_stays_rare_and_named() -> None:
    """Three skills may write. That number should move only on purpose.

    Stripe's `stripe_api_write` reaches its entire API, so every skill holding
    it is a place the whole write surface is one path away.
    """
    writers = sorted(
        name
        for name, skill in _skills().items()
        if any(
            tool in ("stripe_api_write", "create_refund") for tool in (skill.allowed_tools or [])
        )
    )
    assert writers == [
        "stripe-invoices-write",
        "stripe-money-movement",
        "stripe-subscriptions-write",
    ], f"the set of write-capable skills changed: {writers}"
