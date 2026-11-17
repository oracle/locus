# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 73: A support agent changes a customer account — admit() holds the big ones.

A customer-support agent resolves tickets by changing the customer's
account: applying a goodwill credit, bumping a plan, waiving a fee. Most
of these are routine and should proceed without review. A few move real
money or change what the customer is entitled to, and those should pause
for a human before anything is written.

The agent does not decide that for itself. Every account change runs
through ``admit()`` first. ``admit()`` weighs the change against a
:class:`ControlPolicy` you wrote, runs the side-effecting function only
if the policy allows it, and records the decision either way::

    Support agent proposes an account change (an Action)
       │
       ▼
    admit(action, perform, policy=…, trail=…)
       │
       ├─ small credit, single account, routine ─────────> ALLOW   → perform() runs
       │
       └─ plan upgrade / large credit (tagged high_value) > HOLD    → AdmissionError
                                                                       perform() never runs
       │
       ▼
    AuditTrail records BOTH decisions (allowed and held)

The gate is code that runs *before* the write, not a line in the agent's
prompt. A model that is persuaded to issue a $500 credit still cannot
issue it: the high-value change holds for a human, and the attempt lands
on a tamper-evident audit trail. Fooling the model does not move money.

The policy here gates on the action's scope and labels, not on a human
reading the ticket: a change tagged ``high_value`` always holds, and any
change that touches more than ``max_blast_radius`` entitlements holds too.
Everything else auto-allows.

Part 2 follows the held upgrade to the person who decides it. The same
policy sits behind ``gate_tool(..., on_refusal="interrupt")`` on an agent
with a checkpointer and a file-backed approval store, so the hold pauses
the run and outlives the process: the thread's checkpoint carries the hold
(``pending_holds``), a support lead approves it from another process with
``locus approvals``, and a fresh agent resumes the thread and performs the
upgrade exactly once::

    Agent run  ──  gate holds upgrade  ──>  InterruptEvent, run parked
                                            checkpoint carries the hold
                                            FileApprovals keeps the record
    Another process:  locus approvals --store file://… approve <id> --by lead
    New agent, same checkpointer + store:  resume(perform_dangling=True)
                                            ──> the upgrade runs once

Run it:
    # Offline — no network, no credentials. The "perform" is a local stub
    # that mutates an in-memory account dict, and with no provider
    # configured the agent's model is a scripted test double.
    python examples/notebook_73_support_account_gate.py

    # The series convention (examples/config.py): with a provider set, the
    # agent in Part 2 runs on the configured model and the run names it.
    export LOCUS_MODEL_PROVIDER=oci LOCUS_MODEL_ID=openai.gpt-4.1
    python examples/notebook_73_support_account_gate.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Any

from config import get_model, resolved_provider

from locus.agent import Agent
from locus.control import (
    Action,
    AdmissionError,
    AuditTrail,
    ControlPolicy,
    FileApprovals,
    admit,
    gate_tool,
    pending_holds,
)
from locus.core.events import InterruptEvent, TerminateEvent
from locus.memory.backends.file import FileCheckpointer
from locus.models.metadata import model_id_of
from locus.testing import ScriptedModel, text, tool_call
from locus.tools.decorator import tool


# The "side effect" — a local stub standing in for the billing/account API.
#
# In production this would call the real account service. Here it just
# mutates an in-memory dict so the example runs offline with no creds.
# The point is that this function only ever runs once admit() has allowed
# it: there is no path to a write that the policy didn't clear.

ACCOUNTS: dict[str, dict[str, object]] = {
    "cust:4821": {"plan": "starter", "credit_balance_usd": 0.0},
}


async def apply_account_change(account: str, field: str, value: object) -> str:
    """Mutate the customer account. Stands in for a real billing API call."""
    ACCOUNTS.setdefault(account, {})[field] = value
    return f"{account}: set {field} = {value!r}"


# Run the two account changes through the gate and show the trail.


async def main() -> None:
    print("Notebook 73: A support agent changes a customer account — admit() holds the big ones")
    print("=" * 60)

    # The support-ops knobs. Routine changes auto-allow; anything tagged
    # high_value, or touching more than one entitlement, holds for a human.
    #
    # require_verification_score=0.0 turns off the security-grounding bar:
    # that check belongs to the security domain (a finding that survived
    # verification), not to a support agent crediting an account. Here the
    # gate is purely blast radius + labels.
    policy = ControlPolicy(
        require_verification_score=0.0,
        max_blast_radius=1,
        require_human_for=frozenset({"high_value"}),
    )
    trail = AuditTrail()

    # --- Change 1: a small goodwill credit. Routine — should auto-allow. ---
    print("\n--- Ticket SUP-7781: $5 goodwill credit for a late reply ---")
    small_credit = Action(
        name="apply_credit",
        asset="cust:4821",
        blast_radius=1,
        environment="live",
        kind="account_credit",
        tags=frozenset(),  # not high_value
    )
    print(f"  Agent proposes: {small_credit.name} (+$5) on {small_credit.asset}")
    try:
        result = await admit(
            small_credit,
            lambda: apply_account_change("cust:4821", "credit_balance_usd", 5.0),
            policy=policy,
            trail=trail,
        )
        print(f"  ✓ ALLOWED — the change ran: {result}")
    except AdmissionError as exc:
        print(f"  ✗ HELD — {exc.decision.outcome}: {exc.decision.reason}")

    # --- Change 2: a plan upgrade + $500 credit. High value — should hold. ---
    print("\n--- Ticket SUP-7782: upgrade to enterprise + $500 retention credit ---")
    big_change = Action(
        name="upgrade_plan_and_credit",
        asset="cust:4821",
        blast_radius=2,  # touches the plan entitlement *and* the billing balance
        environment="live",
        kind="account_credit",
        tags=frozenset({"high_value"}),
    )
    print(f"  Agent proposes: {big_change.name} (enterprise + $500) on {big_change.asset}")
    held = False
    try:
        await admit(
            big_change,
            lambda: apply_account_change("cust:4821", "plan", "enterprise"),
            policy=policy,
            trail=trail,
        )
        print("  ✓ ALLOWED — the change ran (unexpected)")
    except AdmissionError as exc:
        held = True
        print(f"  ⏸  HELD for a human — {exc.decision.outcome}: {exc.decision.reason}")
        print("     The upgrade did NOT run. It waits for a support lead to approve.")

    # The high-value change must not have touched the account.
    assert held, "high-value change should have been held for a human"
    assert ACCOUNTS["cust:4821"]["plan"] == "starter", "held change must not mutate the account"
    assert ACCOUNTS["cust:4821"]["credit_balance_usd"] == 5.0, "only the allowed credit applied"

    # --- The audit trail recorded BOTH decisions. ---
    print("\nAudit trail")
    print("-" * 60)
    records = [r for r in trail.records() if r.event_type == "action-admission"]
    for rec in records:
        p = rec.payload
        print(f"  {p['outcome']:<14} {p['action']:<26} {p['asset']}  —  {p['reason']}")

    outcomes = [r.payload["outcome"] for r in records]
    assert len(records) == 2, f"expected 2 admission records, got {len(records)}"
    assert "allow" in outcomes, "the small credit should be recorded as allowed"
    assert "require_human" in outcomes, "the plan upgrade should be recorded as held"
    assert trail.verify(), "the audit trail must verify (tamper-evident)"

    print("\nBoth decisions are on the trail: one allowed, one held. No write went unrecorded.")

    await approve_from_another_process(policy)


# Part 2: the hold survives a restart and is decided from another process.
#
# The upgrade is now a tool the agent calls. Wrapped with gate_tool() in
# interrupt mode, a hold pauses the run instead of refusing it: the
# checkpointer keeps the conversation, the FileApprovals store keeps the
# pending approval, and the checkpoint carries the hold itself. Nothing
# below needs a service: the store is a JSON file, the checkpointer a
# directory, and the "other process" is the locus command in a subprocess.


@tool
async def upgrade_plan_and_credit(account: str, plan: str, credit_usd: float) -> str:
    """Move the customer to a new plan and add a retention credit."""
    await apply_account_change(account, "plan", plan)
    balance = float(ACCOUNTS[account].get("credit_balance_usd", 0.0))  # type: ignore[arg-type]
    await apply_account_change(account, "credit_balance_usd", balance + credit_usd)
    return f"{account}: plan={plan}, credit_balance_usd={balance + credit_usd}"


def _as_action(name: str, arguments: dict[str, object]) -> Action:
    """The same shape as Change 2: two entitlements, tagged high_value."""
    return Action(
        name=name,
        asset=str(arguments["account"]),
        blast_radius=2,
        environment="live",
        kind="account_credit",
        tags=frozenset({"high_value"}),
    )


OFFLINE = resolved_provider() == "mock"


def _agent_model(scripted: ScriptedModel) -> Any:
    """The series convention: the scripted double offline, else the configured model."""
    return scripted if OFFLINE else get_model()


def _support_agent(model: Any, root: Path, policy: ControlPolicy) -> Agent:
    """An agent process: rebuilt from the same policy, store file and checkpoint dir."""
    gated = gate_tool(
        upgrade_plan_and_credit,
        policy=policy,
        action=_as_action,
        approval=FileApprovals(root / "approvals.json"),
        on_refusal="interrupt",
        principal="support-agent",
    )
    return Agent(
        model=model,
        tools=[gated],
        system_prompt=(
            "You are a customer-support agent working tickets. Resolve a ticket by "
            "calling upgrade_plan_and_credit with the account, plan and credit it asks "
            "for; when the tool has run, confirm in one sentence."
        ),
        checkpointer=FileCheckpointer(root / "checkpoints"),
        reflexion=False,
        grounding=False,
    )


async def approve_from_another_process(policy: ControlPolicy) -> None:
    print("\nPart 2: the held upgrade waits for a support lead, across a restart")
    print("-" * 60)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store_spec = f"file://{root / 'approvals.json'}"

        # --- Process 1: the agent proposes the upgrade and the gate holds it. ---
        first = _support_agent(
            _agent_model(
                ScriptedModel(
                    [
                        tool_call(
                            "upgrade_plan_and_credit",
                            account="cust:4821",
                            plan="enterprise",
                            credit_usd=500.0,
                        )
                    ]
                )
            ),
            root,
            policy,
        )
        if not OFFLINE:
            print(f"  model: {resolved_provider()} {model_id_of(first.model)}")
        approval_id = ""
        ticket = (
            "Ticket SUP-7782: move account cust:4821 to the enterprise plan "
            "and add a 500 USD retention credit."
        )
        async for event in first.run(ticket, thread_id="SUP-7782"):
            if isinstance(event, InterruptEvent):
                approval_id = event.metadata["approval_id"]
                print(
                    f"  ⏸  run parked on {event.metadata['action']} for {event.metadata['asset']}"
                )
        assert approval_id, "the gate should have held the upgrade"
        assert ACCOUNTS["cust:4821"]["plan"] == "starter", "nothing ran yet"

        # The process that paused is gone. What is left on disk is enough: the
        # thread's checkpoint names the approval and the store it is in.
        [hold] = await pending_holds(FileCheckpointer(root / "checkpoints"), "SUP-7782")
        print(f"  checkpoint carries the hold: {hold.action} on {hold.asset}")
        print(
            f"  approval {hold.approval_id} is {hold.state(FileApprovals(root / 'approvals.json'))}"
        )
        print(f"  store: {'file://…/approvals.json' if hold.store == store_spec else hold.store}")

        # --- Another process: a support lead approves from the command line. ---
        lead = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "locus.cli",
            "approvals",
            "--store",
            store_spec,
            "approve",
            hold.approval_id,
            "--by",
            "lead@example.com",
            "--reason",
            "retention offer within budget",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await lead.communicate()
        assert lead.returncode == 0, err.decode()
        print(f"  locus approvals: {out.decode().strip()}")

        # --- Process 2: a fresh agent resumes the thread; the upgrade runs once. ---
        second = _support_agent(
            _agent_model(ScriptedModel([text("Upgrade applied and credit added.")])), root, policy
        )
        async for event in second.resume("approved", thread_id="SUP-7782", perform_dangling=True):
            if isinstance(event, TerminateEvent):
                print(f"  agent: {event.final_message}")

        record = FileApprovals(root / "approvals.json").get(hold.approval_id)
        assert record is not None
        print(
            f"  approval {hold.approval_id} is now {record.status} (decided by {record.decided_by})"
        )
        print(
            f"  holds left on the thread: {await pending_holds(FileCheckpointer(root / 'checkpoints'), 'SUP-7782')}"
        )
        print(f"  account: {ACCOUNTS['cust:4821']}")

    assert ACCOUNTS["cust:4821"] == {"plan": "enterprise", "credit_balance_usd": 505.0}
    assert record.status == "consumed", "the approval was used exactly once"
    print(
        "\nThe upgrade ran once, after a named person approved it, in a process that never saw the hold placed."
    )


if __name__ == "__main__":
    asyncio.run(main())
