# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""
Notebook 13: Severity-based cloud incident escalation routing.

A conditional edge is a function attached to a node. It runs after the
node returns, looks at the current state, and picks the next node by
name. That's all a cloud ops team needs to express escalation policy:
low-severity monitoring alerts auto-resolve, high-severity ones page an
on-call SRE, and an LLM can sort raw alerts into families (compute,
storage, network).

Key ideas:
- Binary and multi-way branching with `add_conditional_edges`.
- Router function — receives state, returns a node name.
- Optional `targets` mapping to translate router output to node ids.
- `default` to handle unexpected router output.
- An LLM acting as the router for one node.

Run it:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_13_conditional_routing.py

The default provider is OCI GenAI when ``~/.oci/config`` holds a usable profile, otherwise the bundled mock model; set ``LOCUS_MODEL_PROVIDER`` (``oci``, ``openai``, ``anthropic``, ``mock``) to choose.
Set LOCUS_MODEL_PROVIDER=mock for offline runs. Pick a live provider with
LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o
(or LOCUS_MODEL_PROVIDER=anthropic, or any OpenAI-compatible
provider: ollama, vllm, groq, together, openrouter, ...).
"""

import asyncio

from config import get_model, print_config

from locus.agent import Agent
from locus.multiagent import END, START, StateGraph


# Parts 1-5 are deterministic on purpose: the lesson is the router, and a
# handler that spends a model round-trip to say "paged on-call" would bury
# it. The one model call in this file is Part 6, where the model *is* the
# router.


def _banner(result) -> None:
    m = result.metrics
    print(
        f"  [model call: {m.duration_ms / 1000.0:.2f}s · "
        f"{m.prompt_tokens}→{m.completion_tokens} tokens]"
    )


# =============================================================================
# Part 1: Binary branch — page or auto-resolve
# =============================================================================


async def example_binary_routing():
    """Pick one of two downstream nodes based on a boolean in state."""
    print("=== Part 1: Binary branch — page or auto-resolve ===\n")

    graph = StateGraph()

    async def check_confidence(inputs):
        confidence = inputs.get("confidence", 0)
        return {"confidence": confidence, "needs_human": confidence >= 80}

    async def escalate_path(inputs):
        return {
            "disposition": (
                f"Escalated: {inputs.get('confidence')}% anomaly confidence — paging on-call SRE."
            )
        }

    async def auto_close_path(inputs):
        return {
            "disposition": (
                f"Auto-resolved: {inputs.get('confidence')}% anomaly confidence is below the "
                "paging threshold; alert kept on record."
            )
        }

    graph.add_node("check", check_confidence)
    graph.add_node("escalate", escalate_path)
    graph.add_node("auto_close", auto_close_path)

    graph.add_edge(START, "check")

    # add_conditional_edges(source, router, targets=None)
    # router — function that receives state and returns the next node name
    # targets — optional mapping from router output to actual node ids
    graph.add_conditional_edges(
        "check",
        lambda state: "escalate" if state.get("needs_human") else "auto_close",
        {"escalate": "escalate", "auto_close": "auto_close"},
    )

    graph.add_edge("escalate", END)
    graph.add_edge("auto_close", END)

    for confidence in [95, 40]:
        result = await graph.execute({"confidence": confidence})
        print(f"Confidence {confidence}%: {result.final_state.get('disposition')}")
    print()


# =============================================================================
# Part 2: Multi-way branch — severity ladders
# =============================================================================


async def example_multiway_routing():
    """One router function picks among four severity handlers."""
    print("=== Part 2: Multi-way branch — severity ladders ===\n")

    graph = StateGraph()

    async def classify_alert(inputs):
        severity = inputs.get("severity", "low")
        return {"severity": severity}

    async def handle_critical(inputs):
        return {"response": "Paging on-call SRE — CRITICAL outage", "sla": "15 minutes"}

    async def handle_high(inputs):
        return {"response": "Assigned to the platform team", "sla": "1 hour"}

    async def handle_medium(inputs):
        return {"response": "Queued for triage", "sla": "24 hours"}

    async def handle_low(inputs):
        return {"response": "Batched for weekly review", "sla": "1 week"}

    graph.add_node("classify", classify_alert)
    graph.add_node("critical", handle_critical)
    graph.add_node("high", handle_high)
    graph.add_node("medium", handle_medium)
    graph.add_node("low", handle_low)

    graph.add_edge(START, "classify")

    def severity_router(state):
        severity = state.get("severity", "low")
        if severity == "critical":  # noqa: SIM116 — explicit if/elif reads cleaner here
            return "critical"
        elif severity == "high":
            return "high"
        elif severity == "medium":
            return "medium"
        else:
            return "low"

    graph.add_conditional_edges("classify", severity_router)

    graph.add_edge("critical", END)
    graph.add_edge("high", END)
    graph.add_edge("medium", END)
    graph.add_edge("low", END)

    for severity in ["critical", "high", "medium", "low"]:
        result = await graph.execute({"severity": severity})
        state = result.final_state
        print(f"{severity.upper()}: {state.get('response')} (SLA {state.get('sla')})")
    print()


# =============================================================================
# Part 3: Two routers in sequence
# =============================================================================


async def example_chained_conditions():
    """First check the alert source, then — if trusted — check the operator's role."""
    print("=== Part 3: Two routers in sequence ===\n")

    graph = StateGraph()

    async def validate_source(inputs):
        webhook_key = inputs.get("webhook_key", "")
        is_trusted = webhook_key == "webhook-key-123"  # noqa: S105 — notebook literal, not a secret
        return {"trusted_source": is_trusted}

    async def check_role(inputs):
        role = inputs.get("role", "readonly")
        return {"is_operator": role == "operator"}

    async def rollback_action(inputs):
        return {"result": "Operator rolled back the affected service"}

    async def ticket_action(inputs):
        return {"result": "Read-only viewer opened a remediation ticket"}

    async def discard_alert(inputs):
        return {"result": "Alert from untrusted webhook discarded"}

    graph.add_node("source", validate_source)
    graph.add_node("role", check_role)
    graph.add_node("rollback", rollback_action)
    graph.add_node("ticket", ticket_action)
    graph.add_node("discard", discard_alert)

    graph.add_edge(START, "source")

    graph.add_conditional_edges(
        "source", lambda s: "role" if s.get("trusted_source") else "discard"
    )
    graph.add_conditional_edges("role", lambda s: "rollback" if s.get("is_operator") else "ticket")

    graph.add_edge("rollback", END)
    graph.add_edge("ticket", END)
    graph.add_edge("discard", END)

    test_cases = [
        {"webhook_key": "forged", "role": "operator"},
        {"webhook_key": "webhook-key-123", "role": "readonly"},
        {"webhook_key": "webhook-key-123", "role": "operator"},
    ]

    for case in test_cases:
        result = await graph.execute(case)
        print(f"Webhook key: {case['webhook_key'][:6]}..., Role: {case['role']}")
        print(f"  -> {result.final_state.get('result')}")
    print()


# =============================================================================
# Part 4: Default fallback
# =============================================================================


async def example_default_route():
    """Use `default` to catch alert families not in the targets mapping."""
    print("=== Part 4: Default fallback ===\n")

    graph = StateGraph()

    async def categorize(inputs):
        family = inputs.get("family", "unknown")
        return {"family": family}

    async def handle_compute(inputs):
        return {"handler": "Compute Platform Team"}

    async def handle_storage(inputs):
        return {"handler": "Storage Platform Team"}

    async def handle_network(inputs):
        return {"handler": "Network Platform Team"}

    async def handle_other(inputs):
        return {"handler": "General on-call queue (unmatched family)"}

    graph.add_node("categorize", categorize)
    graph.add_node("compute", handle_compute)
    graph.add_node("storage", handle_storage)
    graph.add_node("network", handle_network)
    graph.add_node("other", handle_other)

    graph.add_edge(START, "categorize")

    graph.add_conditional_edges(
        "categorize",
        lambda s: s.get("family", "other"),
        targets={
            "compute": "compute",
            "storage": "storage",
            "network": "network",
        },
        default="other",
    )

    graph.add_edge("compute", END)
    graph.add_edge("storage", END)
    graph.add_edge("network", END)
    graph.add_edge("other", END)

    for family in ["compute", "storage", "database", "xyz"]:
        result = await graph.execute({"family": family})
        print(f"Alert family '{family}': {result.final_state.get('handler')}")
    print()


# =============================================================================
# Part 5: Routing on multiple fields
# =============================================================================


async def example_complex_routing():
    """A router can read several state fields and combine them however it likes."""
    print("=== Part 5: Routing on multiple fields ===\n")

    graph = StateGraph()

    async def evaluate_incident(inputs):
        error_rate = inputs.get("error_rate", 0.0)
        service_tier = inputs.get("service_tier", "standard")
        instances = inputs.get("instances", 1)

        return {
            "error_rate": error_rate,
            "service_tier": service_tier,
            "instances": instances,
            "is_widespread": instances > 10,
            "is_tier_zero": service_tier == "tier_zero",
            "is_high_error": error_rate > 10.0,
        }

    async def page_oncall(inputs):
        return {"routing": "Paged the incident commander", "response_window": "Immediate"}

    async def priority_queue(inputs):
        return {"routing": "Priority queue", "response_window": "4 hours"}

    async def standard_queue(inputs):
        return {"routing": "Standard queue", "response_window": "24 hours"}

    graph.add_node("evaluate", evaluate_incident)
    graph.add_node("page", page_oncall)
    graph.add_node("priority", priority_queue)
    graph.add_node("standard", standard_queue)

    graph.add_edge(START, "evaluate")

    def incident_router(state):
        is_tier_zero = state.get("is_tier_zero", False)
        is_high_error = state.get("is_high_error", False)
        is_widespread = state.get("is_widespread", False)

        if is_tier_zero and is_high_error:
            return "page"
        elif is_tier_zero or is_high_error or is_widespread:
            return "priority"
        else:
            return "standard"

    graph.add_conditional_edges("evaluate", incident_router)

    graph.add_edge("page", END)
    graph.add_edge("priority", END)
    graph.add_edge("standard", END)

    test_cases = [
        {"error_rate": 2.5, "service_tier": "standard", "instances": 2},
        {"error_rate": 2.5, "service_tier": "tier_zero", "instances": 2},
        {"error_rate": 35.0, "service_tier": "standard", "instances": 2},
        {"error_rate": 35.0, "service_tier": "tier_zero", "instances": 20},
    ]

    for case in test_cases:
        result = await graph.execute(case)
        print(
            f"Incident: error rate {case['error_rate']}%, "
            f"{case['service_tier']}, {case['instances']} instances"
        )
        print(
            f"  -> {result.final_state.get('routing')}: {result.final_state.get('response_window')}"
        )
    print()


# =============================================================================
# Part 6: LLM as the router
# =============================================================================


async def example_llm_router():
    """The classifier node asks an LLM for a single-word label, then routes on it."""
    print("=== Part 6: LLM as the router ===\n")

    graph = StateGraph()

    # One classifier agent for every alert the graph sees. Built outside the
    # node so the three executions below share a single model client.
    classifier = Agent(
        model=get_model(max_tokens=10),
        system_prompt=(
            "Classify the cloud alert into exactly one of: "
            "compute, storage, network. Reply with just the single word."
        ),
    )

    async def classify_with_llm(inputs):
        result = await classifier.arun(inputs.get("alert", ""))
        _banner(result)
        label = result.message.strip().lower()
        # Defensive: clamp anything unexpected back onto a known label
        # so the conditional edge always finds a target.
        if label not in {"compute", "storage", "network"}:
            label = "network"
        return {"family": label}

    async def compute(_inputs):
        return {"handler": "Compute Platform Team"}

    async def storage(_inputs):
        return {"handler": "Storage Platform Team"}

    async def network(_inputs):
        return {"handler": "Network Platform Team"}

    graph.add_node("classify", classify_with_llm)
    graph.add_node("compute", compute)
    graph.add_node("storage", storage)
    graph.add_node("network", network)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", lambda s: s["family"])
    graph.add_edge("compute", END)
    graph.add_edge("storage", END)
    graph.add_edge("network", END)

    samples = [
        "CPU pinned at 100% on autoscaling group web-asg for ten minutes.",
        "Object store bucket prod-assets returning 503 SlowDown on PUT requests.",
        "Cross-region VPC peering link dropping 4% of packets to eu-west-1.",
    ]
    for alert in samples:
        result = await graph.execute({"alert": alert})
        print(f"  '{alert[:40]}…' → {result.final_state.get('handler')}")
    print()


async def main():
    print("=" * 60)
    print("Notebook 13: Severity-based cloud incident escalation routing")
    print("=" * 60)
    print()

    print_config()
    print()

    await example_binary_routing()
    await example_multiway_routing()
    await example_chained_conditions()
    await example_default_route()
    await example_complex_routing()
    await example_llm_router()

    print("=" * 60)
    print("Next: Notebook 14 — Merging multi-check payment signals with state reducers")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
