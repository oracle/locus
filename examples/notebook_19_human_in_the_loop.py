# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 19: Human approval gates for containment actions.

Isolating a production host is not something an agent should do on its
own authority — the approval gate is the control against excessive
agency (OWASP LLM06). `interrupt(payload)` pauses the running node and
returns control to the caller with `result.is_interrupted = True`. The caller
inspects the payload, gets a response (SOC console, CLI prompt, Slack
reply — whatever makes sense), then calls
`graph.execute(Command(update=..., resume=...))` to continue. The same
node restarts; its `interrupt()` call now returns the supplied response.

- `interrupt(payload)` — pause and surface a payload to the caller.
- `Command(update=..., resume=...)` — resume execution with a response.
- Multiple interrupts in one workflow.
- Conditional interrupts (only ask for higher-risk containment).
- `graph.config.interrupt_before = [...]` — pause before specific nodes.

Run it:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_19_human_in_the_loop.py

This notebook doesn't call any LLM, so the model provider doesn't
matter. The default is the bundled mock model unless TULIP_MODEL_PROVIDER is set.
"""

import asyncio

from tulip.core import Command, interrupt
from tulip.multiagent import END, START, StateGraph


# =============================================================================
# Part 1: One interrupt
# =============================================================================


async def example_basic_interrupt():
    """Pause before a disruptive containment action and wait for a yes/no response."""
    print("=== Part 1: One interrupt ===\n")

    graph = StateGraph()

    async def prepare(inputs):
        return {"action": "isolate", "target": inputs.get("host", "web-01")}

    async def request_approval(inputs):
        # interrupt() pauses the node, surfaces `payload` to the caller via
        # result.interrupt, and returns the value passed to resume= when the
        # graph is re-executed.
        response = interrupt(
            {
                "question": f"Approve {inputs['action']} on {inputs['target']}?",
                "options": ["yes", "no"],
            }
        )
        return {"approved": response == "yes", "response": response}

    async def execute_action(inputs):
        if inputs.get("approved"):
            return {"result": f"Executed {inputs['action']} on {inputs['target']}"}
        return {"result": "Containment cancelled"}

    graph.add_node("prepare", prepare)
    graph.add_node("approval", request_approval)
    graph.add_node("execute", execute_action)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "approval")
    graph.add_edge("approval", "execute")
    graph.add_edge("execute", END)

    print("Starting containment workflow...")
    result = await graph.execute({"host": "prod-web-01"})

    if result.is_interrupted:
        print(f"PAUSED at: {result.interrupt.node_id}")
        print(f"Question: {result.interrupt.interrupt.payload['question']}")

        print("Incident commander responds: 'yes'")
        result = await graph.execute(Command(update=result.final_state, resume="yes"))

        print(f"Result: {result.final_state.get('result')}")
    print()


# =============================================================================
# Part 2: Several interrupts in a row
# =============================================================================


async def example_multi_step():
    """A containment intake form modelled as one interrupt per question."""
    print("=== Part 2: Several interrupts in a row ===\n")

    graph = StateGraph()

    async def ask_host(inputs):
        host = interrupt({"question": "Which host is compromised?", "type": "text"})
        return {"host": host}

    async def ask_action(inputs):
        action = interrupt({"question": f"What containment action for {inputs['host']}?"})
        return {"action": action}

    async def confirm(inputs):
        confirmed = interrupt(
            {
                "question": f"Confirm: {inputs['action']} on {inputs['host']}?",
                "options": ["confirm", "cancel"],
            }
        )
        return {"confirmed": confirmed == "confirm"}

    async def complete(inputs):
        if inputs.get("confirmed"):
            return {"status": "Containment ticket created", "target": inputs["host"]}
        return {"status": "Cancelled"}

    graph.add_node("host", ask_host)
    graph.add_node("action", ask_action)
    graph.add_node("confirm", confirm)
    graph.add_node("complete", complete)

    graph.add_edge(START, "host")
    graph.add_edge("host", "action")
    graph.add_edge("action", "confirm")
    graph.add_edge("confirm", "complete")
    graph.add_edge("complete", END)

    responses = ["prod-web-01", "isolate", "confirm"]

    print("Containment intake flow:")
    result = await graph.execute({})

    for response in responses:
        if result.is_interrupted:
            print(f"  Q: {result.interrupt.interrupt.payload['question']}")
            print(f"  A: {response}")
            result = await graph.execute(Command(update=result.final_state, resume=response))
        else:
            break

    print(f"\nFinal: {result.final_state.get('status')}")
    print()


# =============================================================================
# Part 3: Interrupt only when it matters
# =============================================================================


async def example_conditional_interrupt():
    """Auto-approve small isolations; pause only for medium and large blast radius."""
    print("=== Part 3: Interrupt only when it matters ===\n")

    graph = StateGraph()

    async def assess_blast_radius(inputs):
        hosts = inputs.get("hosts", 0)
        if hosts < 5:
            radius = "small"
        elif hosts < 50:
            radius = "medium"
        else:
            radius = "large"
        return {"hosts": hosts, "radius": radius}

    async def maybe_approve(inputs):
        radius = inputs.get("radius")
        if radius == "small":
            return {"approved": True, "approver": "auto"}

        required = "SOC lead" if radius == "medium" else "incident commander"
        response = interrupt(
            {
                "message": f"Isolating {inputs['hosts']} hosts requires {required} approval",
                "radius": radius,
            }
        )
        return {"approved": response == "approve", "approver": required}

    async def contain(inputs):
        if inputs.get("approved"):
            return {"result": f"Isolation approved by {inputs['approver']}"}
        return {"result": "Isolation rejected"}

    graph.add_node("assess", assess_blast_radius)
    graph.add_node("approve", maybe_approve)
    graph.add_node("contain", contain)

    graph.add_edge(START, "assess")
    graph.add_edge("assess", "approve")
    graph.add_edge("approve", "contain")
    graph.add_edge("contain", END)

    test_cases = [
        (2, None),
        (20, "approve"),
        (200, "approve"),
    ]

    for hosts, user_response in test_cases:
        print(f"Isolating {hosts} hosts...")
        result = await graph.execute({"hosts": hosts})

        if result.is_interrupted:
            print(f"  Needs approval: {result.interrupt.interrupt.payload['radius']} blast radius")
            result = await graph.execute(Command(update=result.final_state, resume=user_response))

        print(f"  -> {result.final_state.get('result')}")
    print()


# =============================================================================
# Part 4: interrupt_before — pause without modifying the node
# =============================================================================


async def example_interrupt_before():
    """Pause before listed nodes without putting interrupt() inside them."""
    print("=== Part 4: interrupt_before — pause without modifying the node ===\n")

    graph = StateGraph()

    async def snapshot(inputs):
        return {"evidence": inputs.get("evidence", "memory dump"), "snapshotted": True}

    async def isolate(inputs):
        return {"isolated": True, "target": inputs.get("host")}

    async def verify(inputs):
        return {"verified": True}

    graph.add_node("snapshot", snapshot)
    graph.add_node("isolate", isolate)
    graph.add_node("verify", verify)

    graph.add_edge(START, "snapshot")
    graph.add_edge("snapshot", "isolate")
    graph.add_edge("isolate", "verify")
    graph.add_edge("verify", END)

    # Pause before any node in this list. Useful when the sensitive step
    # is third-party code you can't edit to call interrupt() directly.
    graph.config.interrupt_before = ["isolate"]

    print("Isolating a production host...")
    result = await graph.execute({"host": "prod-db-01", "evidence": "memory dump"})

    if result.is_interrupted:
        print(f"PAUSED before: {result.interrupt.node_id}")
        print(f"Current state: snapshotted={result.final_state.get('snapshotted')}")
        print("\nResume with graph.execute(Command(update=..., resume=...)).")
    print()


# =============================================================================
# Part 5: A two-stage approval workflow
# =============================================================================


async def example_complete_workflow():
    """SOC lead review then incident commander sign-off, each its own interrupt."""
    print("=== Part 5: A two-stage approval workflow ===\n")

    graph = StateGraph()

    async def create_request(inputs):
        return {
            "request_id": "CONT-001",
            "type": inputs.get("type", "containment"),
            "description": inputs.get("description", ""),
            "status": "pending",
        }

    async def soc_lead_review(inputs):
        approval = interrupt(
            {
                "step": "SOC Lead Review",
                "request": inputs["request_id"],
                "description": inputs["description"],
                "question": "Is isolation the right containment here?",
            }
        )
        return {
            "lead_approved": approval == "approve",
            "lead_comments": "Reviewed by SOC lead",
        }

    async def commander_approval(inputs):
        if not inputs.get("lead_approved"):
            return {"status": "rejected", "reason": "SOC lead review failed"}

        approval = interrupt(
            {
                "step": "Incident Commander Approval",
                "request": inputs["request_id"],
                "question": "Approve isolating this production host?",
            }
        )
        return {
            "commander_approved": approval == "approve",
            "status": "approved" if approval == "approve" else "rejected",
        }

    async def finalize(inputs):
        status = inputs.get("status")
        return {
            "final_status": status,
            "message": f"Request {inputs['request_id']}: {status}",
        }

    graph.add_node("create", create_request)
    graph.add_node("lead", soc_lead_review)
    graph.add_node("commander", commander_approval)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "create")
    graph.add_edge("create", "lead")
    graph.add_edge("lead", "commander")
    graph.add_edge("commander", "finalize")
    graph.add_edge("finalize", END)

    print("Containment Request Workflow")
    print("-" * 30)

    result = await graph.execute(
        {
            "type": "containment",
            "description": "Isolate prod-db-01 (suspected credential theft)",
        }
    )

    approvals = ["approve", "approve"]
    approval_idx = 0

    while result.is_interrupted and approval_idx < len(approvals):
        step = result.interrupt.interrupt.payload.get("step", "Unknown")
        question = result.interrupt.interrupt.payload.get("question", "")
        print(f"\n{step}: {question}")
        print(f"  -> {approvals[approval_idx]}")

        result = await graph.execute(
            Command(update=result.final_state, resume=approvals[approval_idx])
        )
        approval_idx += 1

    print(f"\nResult: {result.final_state.get('message')}")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    print("=" * 60)
    print("Notebook 19: Human approval gates for containment")
    print("=" * 60)
    print()

    await example_basic_interrupt()
    await example_multi_step()
    await example_conditional_interrupt()
    await example_interrupt_before()
    await example_complete_workflow()

    print("=" * 60)
    print("Next: Notebook 20 — Purple-team advanced patterns")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
