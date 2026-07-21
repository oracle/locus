# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 20: A customer-support ticket pipeline as a graph.

HELPDESK is the support automation: incoming tickets flow through a
graph that classifies them, fans work out across many tickets and many
classifiers, validates payloads with a reusable sub-pipeline, and
remembers customers across runs. The pipeline uses the graph building
blocks you reach for once basic graphs stop being enough: dynamic
routing from inside a node, fan-out to many handlers, reusable
subgraphs, and cross-conversation key/value storage.

The triage branch closes the loop with a confidence check: a ticket
that several independent classifiers agree on is auto-resolved, while a
ticket only one weak signal flagged is escalated to a human agent — so
an uncertain automation decision never ships to the customer on its own.

- Command(update=..., goto=...) — write state and pick the next node in one return value.
- goto() / end() — short helpers for common Command shapes.
- scatter() — fan a list of tickets out to copies of a worker node.
- broadcast() — fan one ticket out to several classifier nodes.
- Subgraph-as-node — call one StateGraph from inside another.
- InMemoryStore — durable key/value space that outlives a single run.

Run it:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_20_advanced_patterns.py

The default provider is the bundled mock model; set TULIP_MODEL_PROVIDER for a live provider.
Set TULIP_MODEL_PROVIDER=mock for offline runs. Pick a live provider with
TULIP_MODEL_ID=openai.gpt-4.1 (or meta.llama-3.3-70b-instruct, etc.).
"""

import asyncio
import time

from config import get_model

from tulip.agent import Agent
from tulip.core import Command, broadcast, end, goto, scatter
from tulip.memory import InMemoryStore
from tulip.multiagent import END, START, StateGraph


async def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 80
) -> str:
    """Run a one-shot Agent and print a timing/token banner. Used by every part."""
    agent = Agent(model=get_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = await agent.arun(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


# =============================================================================
# Part 1: Command — state and routing in one return
# =============================================================================


async def example_command_routing():
    """A node that returns Command picks its own queue for the ticket."""
    print("=== Part 1: Command — state and routing in one return ===\n")
    _ai_note = await _llm_call("In one sentence, why is Tulip Command better than separate edges + state writes?")
    print(f"AI rationale: {_ai_note}")

    graph = StateGraph()

    async def classify_ticket(inputs):
        ticket_type = inputs.get("type", "unknown")

        # Returning a Command both writes state and selects the next node,
        # so this single node replaces a conditional edge + a state writer.
        if ticket_type == "billing":
            return Command(
                update={"queue": "billing", "classified": True},
                goto="billing_branch",
            )
        elif ticket_type == "technical":
            return Command(
                update={"queue": "technical", "classified": True},
                goto="technical_branch",
            )
        else:
            return Command(
                update={"queue": "general", "classified": True},
                goto="triage",
            )

    async def billing_branch(inputs):
        return {"branch": "billing_resolution", "owner": "billing team"}

    async def technical_branch(inputs):
        return {"branch": "technical_support", "owner": "engineering support"}

    async def triage(inputs):
        return {"branch": "general_triage", "owner": "front-line agent"}

    graph.add_node("classify", classify_ticket)
    graph.add_node("billing_branch", billing_branch)
    graph.add_node("technical_branch", technical_branch)
    graph.add_node("triage", triage)

    graph.add_edge(START, "classify")
    # No outgoing edges from classify — Command(goto=...) handles routing.
    graph.add_edge("billing_branch", END)
    graph.add_edge("technical_branch", END)
    graph.add_edge("triage", END)

    for ticket_type in ["billing", "technical", "unknown"]:
        result = await graph.execute({"type": ticket_type})
        print(
            f"{ticket_type}: branch={result.final_state.get('branch')}, "
            f"owner={result.final_state.get('owner')}"
        )
    print()


async def example_goto_helpers():
    """goto() and end() are shorthand for the most common Command shapes."""
    print("=== Part 1b: goto() and end() ===\n")
    _ai_note = await _llm_call("In one sentence, when is goto() preferable to a Command literal?")
    print(f"AI rationale: {_ai_note}")

    graph = StateGraph()

    async def check_hours(inputs):
        token = inputs.get("hours_token", "")
        if token == "hours-open":  # noqa: S105 — notebook literal, not a real secret
            # goto("name", k=v) == Command(goto="name", update={"k": v})
            return goto("handle_ticket", in_hours=True)
        return goto("after_hours", in_hours=False)

    async def handle_ticket(inputs):
        # end(k=v) == Command(goto=END, update={"k": v})
        return end(message="Live agent picked up the ticket", status="handled")

    async def after_hours(inputs):
        return end(message="Outside support hours — queued for the morning", status="queued")

    graph.add_node("hours", check_hours)
    graph.add_node("handle_ticket", handle_ticket)
    graph.add_node("after_hours", after_hours)

    graph.add_edge(START, "hours")
    graph.add_edge("handle_ticket", END)
    graph.add_edge("after_hours", END)

    for token in ["hours-open", "hours-closed"]:
        result = await graph.execute({"hours_token": token})
        print(f"Hours token '{token}': {result.final_state.get('message')}")
    print()


# =============================================================================
# Part 2: scatter — fan one list out to many worker copies
# =============================================================================


async def example_scatter():
    """scatter("worker", items, key="x") runs `worker` once per ticket, in parallel."""
    print("=== Part 2: scatter() ===\n")
    _ai_note = await _llm_call("In one sentence, give a customer-support use-case for the scatter() fan-out helper.")
    print(f"AI rationale: {_ai_note}")

    graph = StateGraph()

    async def split_tickets(inputs):
        tickets = inputs.get("tickets", [])
        return scatter("auto_ack", tickets, key="ticket")

    async def auto_ack(inputs):
        ticket = inputs.get("ticket", "")
        # Mock auto-acknowledgement — invented ticket data, clearly fake.
        return {"acked": f"auto-reply sent for {ticket}"}

    async def collect(inputs):
        # Each scattered invocation lands its result under a send_* key.
        results = []
        for key, value in inputs.items():
            if key.startswith("send_") and isinstance(value, dict):
                results.append(value.get("acked"))
        return {"results": results, "count": len(results)}

    graph.add_node("split", split_tickets)
    graph.add_node("auto_ack", auto_ack)
    graph.add_node("collect", collect)

    graph.add_edge(START, "split")
    graph.add_edge("split", "collect")
    graph.add_edge("collect", END)

    result = await graph.execute(
        {"tickets": ["password reset", "refund request", "shipping delay"]}
    )
    print(f"Acknowledged {result.final_state.get('count')} tickets")
    print(f"Results: {result.final_state.get('results')}")
    print()


async def example_broadcast():
    """broadcast(nodes, payload) sends one ticket to several classifier nodes.

    The fan-in node closes the triage loop with a confidence check: a
    ticket that several independent classifiers agree on is auto-resolved,
    while one that only a weak heuristic flagged is escalated to a human
    agent, so an uncertain automated decision never ships on its own.
    """
    print("=== Part 2b: broadcast() + confidence-gated triage ===\n")
    _ai_note = await _llm_call("In one sentence, when is broadcast() better than scatter() in a graph?")
    print(f"AI rationale: {_ai_note}")

    graph = StateGraph()

    async def route_ticket(inputs):
        # HELPDESK sends one ticket to every classifier at once.
        ticket = inputs.get("ticket", "")
        return broadcast(["intent", "knowledge_base", "heuristic"], {"ticket": ticket})

    async def intent(inputs):
        # Intent classifier over the known-issue corpus.
        ticket = inputs.get("ticket", "").lower()
        matched = "reset my password" in ticket
        return {"intent_hit": matched}

    async def knowledge_base(inputs):
        # Look for an exact knowledge-base article tag in the ticket. A
        # tagged ticket maps cleanly to a canned, vetted resolution.
        ticket = inputs.get("ticket", "")
        marker = "[kb:password-reset]" in ticket
        return {"kb_marker": marker}

    async def heuristic(inputs):
        # A soft heuristic — useful as a hint, never a decision on its own.
        ticket = inputs.get("ticket", "")
        return {"heuristic_flag": len(ticket) > 60}

    async def adjudicate(inputs):
        # Each broadcast classifier lands its dict under a send_* key; merge them.
        signals: dict = {}
        for key, value in inputs.items():
            if key.startswith("send_") and isinstance(value, dict):
                signals.update(value)

        # Strong signals — intent match and a knowledge-base tag — are the
        # ones we trust enough to auto-resolve. The length heuristic is a
        # hint only and never auto-resolves a ticket by itself.
        strong = []
        if signals.get("intent_hit"):
            strong.append("intent matched a known self-service issue")
        if signals.get("kb_marker"):
            strong.append("ticket carries a vetted knowledge-base tag")

        weak = []
        if signals.get("heuristic_flag") and not strong:
            weak.append("ticket length exceeded the heuristic threshold")

        if len(strong) >= 2:
            decision = {
                "action": "auto_resolve",
                "reply": "Sent the self-service password-reset guide automatically.",
                "confidence": "high",
                "reasons": strong,
            }
        elif strong:
            decision = {
                "action": "suggest",
                "reply": "Drafted a suggested reply for an agent to review.",
                "confidence": "medium",
                "reasons": strong,
            }
        else:
            decision = {
                "action": "escalate",
                "reply": "Routed to a human agent — no confident match.",
                "confidence": "low",
                "reasons": weak,
            }
        return {"triage": decision}

    graph.add_node("route", route_ticket)
    graph.add_node("intent", intent)
    graph.add_node("knowledge_base", knowledge_base)
    graph.add_node("heuristic", heuristic)
    graph.add_node("adjudicate", adjudicate)

    graph.add_edge(START, "route")
    graph.add_edge("route", "adjudicate")
    graph.add_edge("adjudicate", END)

    # A confident ticket: both the intent and the knowledge-base tag match.
    confident = await graph.execute(
        {
            "ticket": "Hi, I need to reset my password please [kb:password-reset] — locked out since today"
        }
    )
    # A merely-suspected ticket: long, but no classifier matched.
    suspected = await graph.execute(
        {
            "ticket": "I have a long winded question about something vaguely related to my account here"
        }
    )

    for label, run in [("confident", confident), ("suspected", suspected)]:
        decision = run.final_state.get("triage")
        print(
            f"{label}: {decision['action'].upper()} "
            f"(confidence={decision['confidence']}, reasons={decision['reasons']})"
        )
    print()


# =============================================================================
# Part 3: Subgraph as a node
# =============================================================================


async def example_subgraph():
    """A complete StateGraph can be added as a node in another graph."""
    print("=== Part 3: Subgraph as a node ===\n")
    _ai_note = await _llm_call("In one sentence, when should you factor a piece of graph logic out as a subgraph?")
    print(f"AI rationale: {_ai_note}")

    validation_graph = StateGraph()

    async def check_required(inputs):
        ticket = inputs.get("ticket", {})
        missing = [f for f in ["ticket_id", "customer"] if f not in ticket]
        return {"missing_fields": missing, "has_required": len(missing) == 0}

    async def check_format(inputs):
        ticket = inputs.get("ticket", {})
        ticket_id = ticket.get("ticket_id", "")
        return {"valid_id": ticket_id.startswith("TKT-")}

    validation_graph.add_node("required", check_required)
    validation_graph.add_node("format", check_format)
    validation_graph.add_edge(START, "required")
    validation_graph.add_edge("required", "format")
    validation_graph.add_edge("format", END)

    main_graph = StateGraph()

    async def prepare_ticket(inputs):
        return {"ticket": inputs}

    main_graph.add_node("prepare", prepare_ticket)
    # The subgraph plugs in like any other node — its START/END become
    # entry/exit hooks inside the parent.
    main_graph.add_node("validate", validation_graph)

    async def process_result(inputs):
        is_valid = inputs.get("has_required") and inputs.get("valid_id")
        return {"ticket_state": "accepted" if is_valid else "rejected"}

    main_graph.add_node("result", process_result)

    main_graph.add_edge(START, "prepare")
    main_graph.add_edge("prepare", "validate")
    main_graph.add_edge("validate", "result")
    main_graph.add_edge("result", END)

    result = await main_graph.execute({"ticket_id": "TKT-0042", "customer": "acme-co"})
    print(f"Well-formed ticket: state = {result.final_state.get('ticket_state')}")

    result = await main_graph.execute({"ticket_id": "TKT-0042"})
    print(f"Missing customer field: state = {result.final_state.get('ticket_state')}")
    print()


# =============================================================================
# Part 4: Store — memory that outlives one graph run
# =============================================================================


async def example_store():
    """Graph state is per-run; Store persists across ticket runs (or threads)."""
    print("=== Part 4: Store — memory that outlives one graph run ===\n")
    _ai_note = await _llm_call("In one sentence, what kind of state belongs in InMemoryStore vs in graph state?")
    print(f"AI rationale: {_ai_note}")

    store = InMemoryStore()
    graph = StateGraph()

    async def check_seen(inputs):
        issue = inputs.get("issue")
        outcome = await store.get(("issues", issue), "outcome")

        if outcome:
            return {"briefing": f"Seen before — last outcome: {outcome}", "known": True}
        return {"briefing": "New issue for this team — handle with care", "known": False}

    async def record_outcome(inputs):
        if not inputs.get("known"):
            issue = inputs.get("issue")
            outcome = inputs.get("observed_outcome", "unresolved")
            await store.put(("issues", issue), "outcome", outcome)
            return {"recorded": True, "stored_outcome": outcome}
        return {"recorded": False}

    graph.add_node("check", check_seen)
    graph.add_node("record", record_outcome)

    graph.add_edge(START, "check")
    graph.add_edge("check", "record")
    graph.add_edge("record", END)

    print("Ticket 1:")
    result = await graph.execute({"issue": "password-reset", "observed_outcome": "resolved in 40s"})
    print(f"  {result.final_state.get('briefing')}")

    print("\nTicket 2:")
    result = await graph.execute({"issue": "password-reset"})
    print(f"  {result.final_state.get('briefing')}")
    print()


# =============================================================================
# Part 5: All five primitives in one workflow
# =============================================================================


async def example_combined():
    """A ticket pipeline that uses Command, scatter, and Store together."""
    print("=== Part 5: All five primitives in one workflow ===\n")
    _ai_note = await _llm_call("In one sentence, why is combining Command + scatter + Store typical for a recurring support pipeline?")
    print(f"AI rationale: {_ai_note}")

    store = InMemoryStore()
    graph = StateGraph()

    async def classify_ticket(inputs):
        urgency = inputs.get("urgency", 0)
        customer_id = inputs.get("customer_id")
        is_vip = await store.get(("customers", customer_id), "vip") or False

        if urgency > 80 or is_vip:
            return Command(
                update={"handling": "priority", "vip_customer": is_vip},
                goto="full_handling",
            )
        return Command(
            update={"handling": "standard", "vip_customer": is_vip},
            goto="light_handling",
        )

    async def full_handling(inputs):
        return scatter("handler", ["send_ack", "notify_agent", "create_case"], key="action")

    async def light_handling(inputs):
        return {"processed": True, "path": "light"}

    async def handler(inputs):
        action = inputs.get("action", "")
        return {f"{action}_done": True}

    async def finalize(inputs):
        customer_id = inputs.get("customer_id")
        await store.put(
            ("customers", customer_id, "tickets"),
            f"ticket_{inputs.get('urgency')}",
            {"urgency": inputs.get("urgency"), "handling": inputs.get("handling")},
        )
        return {"status": "complete", "handling": inputs.get("handling")}

    graph.add_node("classify", classify_ticket)
    graph.add_node("full_handling", full_handling)
    graph.add_node("light_handling", light_handling)
    graph.add_node("handler", handler)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "classify")
    graph.add_edge("full_handling", "finalize")
    graph.add_edge("light_handling", "finalize")
    graph.add_edge("finalize", END)

    await store.put(("customers", "vip_acme"), "vip", True)  # noqa: FBT003 — store.put signature is (namespace, key, value)

    result = await graph.execute({"customer_id": "std_user", "urgency": 20})
    print(f"Standard customer, urgency 20: {result.final_state.get('handling')} handling")

    result = await graph.execute({"customer_id": "std_user", "urgency": 95})
    print(f"Standard customer, urgency 95: {result.final_state.get('handling')} handling")

    result = await graph.execute({"customer_id": "vip_acme", "urgency": 10})
    print(f"VIP customer, urgency 10: {result.final_state.get('handling')} handling")
    print()


# =============================================================================
# Part 6: LLM-decided Command target
# =============================================================================


async def example_command_with_llm():
    """An LLM reads a ticket update; the node returns Command(goto=label)."""
    print("=== Part 6: LLM-decided Command target ===\n")

    graph = StateGraph()

    async def triage(inputs):
        import time as _t

        update = inputs.get("update", "")
        agent = Agent(
            model=get_model(max_tokens=10),
            system_prompt=(
                "You are a support triage assistant. Output one of: resolved, pending, escalate. "
                "Reply with just that single word."
            ),
        )
        t0 = _t.perf_counter()
        result = await agent.arun(update)
        dt = _t.perf_counter() - t0
        print(
            f"  [model call: {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
        )
        label = result.message.strip().lower()
        # Clamp anything unexpected so goto= always lands on a real node.
        if label not in {"resolved", "pending", "escalate"}:
            label = "escalate"
        return Command(update={"label": label}, goto=label)

    async def resolved(_inputs):
        return {"resolution": "closed as resolved"}

    async def pending(_inputs):
        return {"resolution": "awaiting customer reply — kept open"}

    async def escalate(_inputs):
        return {"resolution": "escalated to a senior agent"}

    graph.add_node("triage", triage)
    graph.add_node("resolved", resolved)
    graph.add_node("pending", pending)
    graph.add_node("escalate", escalate)
    graph.add_edge(START, "triage")
    graph.add_edge("resolved", END)
    graph.add_edge("pending", END)
    graph.add_edge("escalate", END)

    samples = [
        "The customer confirmed the password reset worked and thanked us.",
        "Sent the customer troubleshooting steps and are waiting on their reply.",
        "Customer is furious about a double charge and demands a manager now.",
    ]
    for upd in samples:
        result = await graph.execute({"update": upd})
        print(f"  '{upd[:40]}…' → {result.final_state.get('resolution')}")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    print("=" * 60)
    print("Notebook 20: Customer-support advanced patterns")
    print("=" * 60)
    print()

    await example_command_routing()
    await example_goto_helpers()
    await example_scatter()
    await example_broadcast()
    await example_subgraph()
    await example_store()
    await example_combined()
    await example_command_with_llm()

    print("=" * 60)
    print("Next: Notebook 21 — Composing support agents into pipelines")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
