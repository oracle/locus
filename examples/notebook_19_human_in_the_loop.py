# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""
Pause a graph mid-execution, ask a human, then resume with their answer.

`interrupt(payload)` pauses the running node and returns control to the
caller with `result.is_interrupted = True`. The caller inspects the
payload, gets a response (web form, CLI prompt, Slack reply — whatever
makes sense), then calls `graph.execute(Command(update=..., resume=...))`
to continue. The same node restarts; its `interrupt()` call now
returns the supplied response.

- `interrupt(payload)` — pause and surface a payload to the caller.
- `Command(update=..., resume=...)` — resume execution with a response.
- Multiple interrupts in one workflow.
- Conditional interrupts (only ask for higher-risk cases).
- `graph.config.interrupt_before = [...]` — pause before specific nodes.

Run it:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_25_human_in_the_loop.py

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
    """Pause before a destructive action and wait for a yes/no response."""
    print("=== Part 1: One interrupt ===\n")

    graph = StateGraph()

    async def prepare(inputs):
        return {"action": "delete", "target": inputs.get("file", "data.txt")}

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
        return {"result": "Action cancelled"}

    graph.add_node("prepare", prepare)
    graph.add_node("approval", request_approval)
    graph.add_node("execute", execute_action)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "approval")
    graph.add_edge("approval", "execute")
    graph.add_edge("execute", END)

    print("Starting workflow...")
    result = await graph.execute({"file": "important.txt"})

    if result.is_interrupted:
        print(f"PAUSED at: {result.interrupt.node_id}")
        print(f"Question: {result.interrupt.interrupt.payload['question']}")

        print("User responds: 'yes'")
        result = await graph.execute(Command(update=result.final_state, resume="yes"))

        print(f"Result: {result.final_state.get('result')}")
    print()


# =============================================================================
# Part 2: Several interrupts in a row
# =============================================================================


async def example_multi_step():
    """A multi-question form modelled as one interrupt per question."""
    print("=== Part 2: Several interrupts in a row ===\n")

    graph = StateGraph()

    async def ask_name(inputs):
        name = interrupt({"question": "What is your name?", "type": "text"})
        return {"name": name}

    async def ask_email(inputs):
        email = interrupt({"question": f"Hi {inputs['name']}, what's your email?"})
        return {"email": email}

    async def confirm(inputs):
        confirmed = interrupt(
            {
                "question": f"Confirm: {inputs['name']} <{inputs['email']}>?",
                "options": ["confirm", "cancel"],
            }
        )
        return {"confirmed": confirmed == "confirm"}

    async def complete(inputs):
        if inputs.get("confirmed"):
            return {"status": "Account created", "user": inputs["name"]}
        return {"status": "Cancelled"}

    graph.add_node("name", ask_name)
    graph.add_node("email", ask_email)
    graph.add_node("confirm", confirm)
    graph.add_node("complete", complete)

    graph.add_edge(START, "name")
    graph.add_edge("name", "email")
    graph.add_edge("email", "confirm")
    graph.add_edge("confirm", "complete")
    graph.add_edge("complete", END)

    responses = ["Alice", "alice@example.com", "confirm"]

    print("Registration flow:")
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
    """Auto-approve low-risk cases; pause only for medium and high risk."""
    print("=== Part 3: Interrupt only when it matters ===\n")

    graph = StateGraph()

    async def assess_risk(inputs):
        amount = inputs.get("amount", 0)
        if amount < 100:
            risk = "low"
        elif amount < 1000:
            risk = "medium"
        else:
            risk = "high"
        return {"amount": amount, "risk": risk}

    async def maybe_approve(inputs):
        risk = inputs.get("risk")
        if risk == "low":
            return {"approved": True, "approver": "auto"}

        required = "manager" if risk == "medium" else "executive"
        response = interrupt(
            {
                "message": f"${inputs['amount']} requires {required} approval",
                "risk": risk,
            }
        )
        return {"approved": response == "approve", "approver": required}

    async def process(inputs):
        if inputs.get("approved"):
            return {"result": f"Transaction approved by {inputs['approver']}"}
        return {"result": "Transaction rejected"}

    graph.add_node("assess", assess_risk)
    graph.add_node("approve", maybe_approve)
    graph.add_node("process", process)

    graph.add_edge(START, "assess")
    graph.add_edge("assess", "approve")
    graph.add_edge("approve", "process")
    graph.add_edge("process", END)

    test_cases = [
        (50, None),
        (500, "approve"),
        (5000, "approve"),
    ]

    for amount, user_response in test_cases:
        print(f"Processing ${amount}...")
        result = await graph.execute({"amount": amount})

        if result.is_interrupted:
            print(f"  Needs approval: {result.interrupt.interrupt.payload['risk']} risk")
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

    async def prepare(inputs):
        return {"data": inputs.get("data", "sample"), "prepared": True}

    async def deploy(inputs):
        return {"deployed": True, "target": inputs.get("environment")}

    async def verify(inputs):
        return {"verified": True}

    graph.add_node("prepare", prepare)
    graph.add_node("deploy", deploy)
    graph.add_node("verify", verify)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "deploy")
    graph.add_edge("deploy", "verify")
    graph.add_edge("verify", END)

    # Pause before any node in this list. Useful when the sensitive step
    # is third-party code you can't edit to call interrupt() directly.
    graph.config.interrupt_before = ["deploy"]

    print("Deploying to production...")
    result = await graph.execute({"environment": "production", "data": "v2.0"})

    if result.is_interrupted:
        print(f"PAUSED before: {result.interrupt.node_id}")
        print(f"Current state: prepared={result.final_state.get('prepared')}")
        print("\nResume with graph.execute(Command(update=..., resume=...)).")
    print()


# =============================================================================
# Part 5: A two-stage approval workflow
# =============================================================================


async def example_complete_workflow():
    """Technical review then manager sign-off, each its own interrupt."""
    print("=== Part 5: A two-stage approval workflow ===\n")

    graph = StateGraph()

    async def create_request(inputs):
        return {
            "request_id": "REQ-001",
            "type": inputs.get("type", "change"),
            "description": inputs.get("description", ""),
            "status": "pending",
        }

    async def technical_review(inputs):
        approval = interrupt(
            {
                "step": "Technical Review",
                "request": inputs["request_id"],
                "description": inputs["description"],
                "question": "Is this technically feasible?",
            }
        )
        return {
            "tech_approved": approval == "approve",
            "tech_comments": "Reviewed by engineering",
        }

    async def manager_approval(inputs):
        if not inputs.get("tech_approved"):
            return {"status": "rejected", "reason": "Technical review failed"}

        approval = interrupt(
            {
                "step": "Manager Approval",
                "request": inputs["request_id"],
                "question": "Approve this change request?",
            }
        )
        return {
            "manager_approved": approval == "approve",
            "status": "approved" if approval == "approve" else "rejected",
        }

    async def finalize(inputs):
        status = inputs.get("status")
        return {
            "final_status": status,
            "message": f"Request {inputs['request_id']}: {status}",
        }

    graph.add_node("create", create_request)
    graph.add_node("tech", technical_review)
    graph.add_node("manager", manager_approval)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "create")
    graph.add_edge("create", "tech")
    graph.add_edge("tech", "manager")
    graph.add_edge("manager", "finalize")
    graph.add_edge("finalize", END)

    print("Change Request Workflow")
    print("-" * 30)

    result = await graph.execute(
        {
            "type": "change",
            "description": "Update database schema",
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
    print("Notebook 20: Human-in-the-loop")
    print("=" * 60)
    print()

    await example_basic_interrupt()
    await example_multi_step()
    await example_conditional_interrupt()
    await example_interrupt_before()
    await example_complete_workflow()

    print("=" * 60)
    print("Next: Notebook 21 — Advanced patterns")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
