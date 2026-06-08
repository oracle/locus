# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""
Pick the next node at runtime based on graph state.

A conditional edge is a function attached to a node. It runs after the
node returns, looks at the current state, and picks the next node by
name. That's all you need to express branching workflows, fallback
paths, and LLM-decided routing.

- Binary and multi-way branching with `add_conditional_edges`.
- Router function — receives state, returns a node name.
- Optional `targets` mapping to translate router output to node ids.
- `default` to handle unexpected router output.
- An LLM acting as the router for one node.

Run it:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_23_conditional_routing.py

The default provider is the bundled mock model; set TULIP_MODEL_PROVIDER for a live provider.
Set TULIP_MODEL_PROVIDER=mock for offline runs. Pick a live provider with
TULIP_MODEL_ID=openai.gpt-4.1 (or meta.llama-3.3-70b-instruct, etc.).
"""

import asyncio
import time

from config import get_model

from tulip.agent import Agent
from tulip.multiagent import END, START, StateGraph


def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 80
) -> str:
    """Run a one-shot Agent and print a timing/token banner. Used by every part."""
    agent = Agent(model=get_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = agent.run_sync(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


# =============================================================================
# Part 1: Binary branch
# =============================================================================


async def example_binary_routing():
    """Pick one of two downstream nodes based on a boolean in state."""
    print("=== Part 1: Binary branch ===\n")

    graph = StateGraph()

    async def check_age(inputs):
        age = inputs.get("age", 0)
        return {"age": age, "is_adult": age >= 18}

    async def adult_path(inputs):
        msg = _llm_call(
            f"Write a one-line welcome message for an adult user (age "
            f"{inputs.get('age')}) with full system access.",
        )
        return {"message": msg}

    async def minor_path(inputs):
        msg = _llm_call(
            f"Write a one-line welcome message for a minor user (age "
            f"{inputs.get('age')}) that mentions parental guidance.",
        )
        return {"message": msg}

    graph.add_node("check", check_age)
    graph.add_node("adult", adult_path)
    graph.add_node("minor", minor_path)

    graph.add_edge(START, "check")

    # add_conditional_edges(source, router, targets=None)
    # router — function that receives state and returns the next node name
    # targets — optional mapping from router output to actual node ids
    graph.add_conditional_edges(
        "check",
        lambda state: "adult" if state.get("is_adult") else "minor",
        {"adult": "adult", "minor": "minor"},
    )

    graph.add_edge("adult", END)
    graph.add_edge("minor", END)

    for age in [25, 15]:
        result = await graph.execute({"age": age})
        print(f"Age {age}: {result.final_state.get('message')}")
    print()


# =============================================================================
# Part 2: Multi-way branch
# =============================================================================


async def example_multiway_routing():
    """One router function picks among four outcomes."""
    print("=== Part 2: Multi-way branch ===\n")

    graph = StateGraph()

    async def classify_ticket(inputs):
        priority = inputs.get("priority", "low")
        return {"priority": priority}

    async def handle_critical(inputs):
        line = _llm_call("In one short line, escalate a CRITICAL ticket. SLA 1 hour.")
        return {"response": line, "sla": "1 hour"}

    async def handle_high(inputs):
        line = _llm_call("In one short line, route a HIGH priority ticket. SLA 4 hours.")
        return {"response": line, "sla": "4 hours"}

    async def handle_normal(inputs):
        line = _llm_call("In one short line, queue a NORMAL ticket. SLA 24 hours.")
        return {"response": line, "sla": "24 hours"}

    async def handle_low(inputs):
        line = _llm_call("In one short line, backlog a LOW ticket. SLA 1 week.")
        return {"response": line, "sla": "1 week"}

    graph.add_node("classify", classify_ticket)
    graph.add_node("critical", handle_critical)
    graph.add_node("high", handle_high)
    graph.add_node("normal", handle_normal)
    graph.add_node("low", handle_low)

    graph.add_edge(START, "classify")

    def priority_router(state):
        priority = state.get("priority", "low")
        if priority == "critical":  # noqa: SIM116 — explicit if/elif reads cleaner here
            return "critical"
        elif priority == "high":
            return "high"
        elif priority == "medium":
            return "normal"
        else:
            return "low"

    graph.add_conditional_edges("classify", priority_router)

    graph.add_edge("critical", END)
    graph.add_edge("high", END)
    graph.add_edge("normal", END)
    graph.add_edge("low", END)

    for priority in ["critical", "high", "medium", "low"]:
        result = await graph.execute({"priority": priority})
        print(f"{priority.upper()}: {result.final_state.get('response')}")
    print()


# =============================================================================
# Part 3: Two routers in sequence
# =============================================================================


async def example_chained_conditions():
    """First check auth, then — if authenticated — check the user's role."""
    print("=== Part 3: Two routers in sequence ===\n")

    graph = StateGraph()

    async def authenticate(inputs):
        token = inputs.get("token", "")
        is_valid = token == "secret123"  # noqa: S105 — notebook literal, not a real secret
        return {"authenticated": is_valid}

    async def check_permissions(inputs):
        role = inputs.get("role", "guest")
        return {"is_admin": role == "admin"}

    async def admin_action(inputs):
        line = _llm_call("In one line, log a successful admin operation.")
        return {"result": line}

    async def user_action(inputs):
        line = _llm_call("In one line, log a successful regular-user operation.")
        return {"result": line}

    async def access_denied(inputs):
        line = _llm_call("In one line, politely decline an unauthenticated user.")
        return {"result": line}

    graph.add_node("auth", authenticate)
    graph.add_node("permissions", check_permissions)
    graph.add_node("admin", admin_action)
    graph.add_node("user", user_action)
    graph.add_node("denied", access_denied)

    graph.add_edge(START, "auth")

    graph.add_conditional_edges(
        "auth", lambda s: "permissions" if s.get("authenticated") else "denied"
    )
    graph.add_conditional_edges("permissions", lambda s: "admin" if s.get("is_admin") else "user")

    graph.add_edge("admin", END)
    graph.add_edge("user", END)
    graph.add_edge("denied", END)

    test_cases = [
        {"token": "wrong", "role": "admin"},
        {"token": "secret123", "role": "user"},
        {"token": "secret123", "role": "admin"},
    ]

    for case in test_cases:
        result = await graph.execute(case)
        print(f"Token: {case['token'][:6]}..., Role: {case['role']}")
        print(f"  -> {result.final_state.get('result')}")
    print()


# =============================================================================
# Part 4: Default fallback
# =============================================================================


async def example_default_route():
    """Use `default` to catch router outputs not in the targets mapping."""
    print("=== Part 4: Default fallback ===\n")

    graph = StateGraph()

    async def categorize(inputs):
        category = inputs.get("category", "unknown")
        return {"category": category}

    async def handle_tech(inputs):
        line = _llm_call("In one short line, name the team that handles technical issues.")
        return {"handler": line}

    async def handle_billing(inputs):
        line = _llm_call("In one short line, name the team that handles billing issues.")
        return {"handler": line}

    async def handle_sales(inputs):
        line = _llm_call("In one short line, name the team that handles sales inquiries.")
        return {"handler": line}

    async def handle_other(inputs):
        line = _llm_call("In one short line, name a generic support fallback team.")
        return {"handler": line}

    graph.add_node("categorize", categorize)
    graph.add_node("tech", handle_tech)
    graph.add_node("billing", handle_billing)
    graph.add_node("sales", handle_sales)
    graph.add_node("other", handle_other)

    graph.add_edge(START, "categorize")

    graph.add_conditional_edges(
        "categorize",
        lambda s: s.get("category", "other"),
        targets={
            "tech": "tech",
            "billing": "billing",
            "sales": "sales",
        },
        default="other",
    )

    graph.add_edge("tech", END)
    graph.add_edge("billing", END)
    graph.add_edge("sales", END)
    graph.add_edge("other", END)

    for category in ["tech", "billing", "returns", "xyz"]:
        result = await graph.execute({"category": category})
        print(f"Category '{category}': {result.final_state.get('handler')}")
    print()


# =============================================================================
# Part 5: Routing on multiple fields
# =============================================================================


async def example_complex_routing():
    """A router can read several state fields and combine them however it likes."""
    print("=== Part 5: Routing on multiple fields ===\n")

    graph = StateGraph()

    async def evaluate_order(inputs):
        amount = inputs.get("amount", 0)
        customer_type = inputs.get("customer_type", "regular")
        items = inputs.get("items", 1)

        return {
            "amount": amount,
            "customer_type": customer_type,
            "items": items,
            "is_bulk": items > 10,
            "is_vip": customer_type == "vip",
            "is_large": amount > 1000,
        }

    async def express_processing(inputs):
        line = _llm_call("In one short line, confirm same-day express order processing.")
        return {"processing": line, "eta": "Same day"}

    async def priority_processing(inputs):
        line = _llm_call("In one short line, confirm 1-2 day priority order processing.")
        return {"processing": line, "eta": "1-2 days"}

    async def standard_processing(inputs):
        line = _llm_call("In one short line, confirm 3-5 day standard order processing.")
        return {"processing": line, "eta": "3-5 days"}

    graph.add_node("evaluate", evaluate_order)
    graph.add_node("express", express_processing)
    graph.add_node("priority", priority_processing)
    graph.add_node("standard", standard_processing)

    graph.add_edge(START, "evaluate")

    def order_router(state):
        is_vip = state.get("is_vip", False)
        is_large = state.get("is_large", False)
        is_bulk = state.get("is_bulk", False)

        if is_vip and is_large:
            return "express"
        elif is_vip or is_large or is_bulk:
            return "priority"
        else:
            return "standard"

    graph.add_conditional_edges("evaluate", order_router)

    graph.add_edge("express", END)
    graph.add_edge("priority", END)
    graph.add_edge("standard", END)

    test_cases = [
        {"amount": 500, "customer_type": "regular", "items": 2},
        {"amount": 500, "customer_type": "vip", "items": 2},
        {"amount": 2000, "customer_type": "regular", "items": 2},
        {"amount": 2000, "customer_type": "vip", "items": 20},
    ]

    for case in test_cases:
        result = await graph.execute(case)
        print(f"Order: ${case['amount']}, {case['customer_type']}, {case['items']} items")
        print(f"  -> {result.final_state.get('processing')}: {result.final_state.get('eta')}")
    print()


# =============================================================================
# Part 6: LLM as the router
# =============================================================================


async def example_llm_router():
    """The classifier node asks an LLM for a single-word label, then routes on it."""
    print("=== Part 6: LLM as the router ===\n")

    graph = StateGraph()

    async def classify_with_llm(inputs):
        import time as _t

        text = inputs.get("text", "")
        agent = Agent(
            model=get_model(max_tokens=10),
            system_prompt=(
                "Classify the user's message into exactly one of: "
                "billing, tech, sales. Reply with just the single word."
            ),
        )
        t0 = _t.perf_counter()
        result = agent.run_sync(text)
        dt = _t.perf_counter() - t0
        print(
            f"  [model call: {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
        )
        label = result.message.strip().lower()
        # Defensive: clamp anything unexpected back onto a known label
        # so the conditional edge always finds a target.
        if label not in {"billing", "tech", "sales"}:
            label = "tech"
        return {"category": label}

    async def billing(_inputs):
        return {"handler": "Billing Department"}

    async def tech(_inputs):
        return {"handler": "Tech Support Team"}

    async def sales(_inputs):
        return {"handler": "Sales Team"}

    graph.add_node("classify", classify_with_llm)
    graph.add_node("billing", billing)
    graph.add_node("tech", tech)
    graph.add_node("sales", sales)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", lambda s: s["category"])
    graph.add_edge("billing", END)
    graph.add_edge("tech", END)
    graph.add_edge("sales", END)

    samples = [
        "My invoice last month has a duplicate charge.",
        "I want to compare your enterprise plans.",
        "The dashboard keeps throwing a 500 error.",
    ]
    for text in samples:
        result = await graph.execute({"text": text})
        print(f"  '{text[:40]}…' → {result.final_state.get('handler')}")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    print("=" * 60)
    print("Notebook 18: Conditional routing")
    print("=" * 60)
    print()

    await example_binary_routing()
    await example_multiway_routing()
    await example_chained_conditions()
    await example_default_route()
    await example_complex_routing()
    await example_llm_router()

    print("=" * 60)
    print("Next: Notebook 19 — State reducers")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
