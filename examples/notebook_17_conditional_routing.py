# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 17: Severity-based alert escalation routing.

A conditional edge is a function attached to a node. It runs after the
node returns, looks at the current state, and picks the next node by
name. That's all a SOC needs to express escalation policy: low-severity
alerts auto-close, high-severity ones page a human, and an LLM can sort
raw alerts into families (phishing T1566, malware execution T1204,
suspicious access T1078 — all MITRE ATT&CK).

- Binary and multi-way branching with `add_conditional_edges`.
- Router function — receives state, returns a node name.
- Optional `targets` mapping to translate router output to node ids.
- `default` to handle unexpected router output.
- An LLM acting as the router for one node.

Run it:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_17_conditional_routing.py

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
# Part 1: Binary branch — escalate or auto-close
# =============================================================================


async def example_binary_routing():
    """Pick one of two downstream nodes based on a boolean in state."""
    print("=== Part 1: Binary branch — escalate or auto-close ===\n")

    graph = StateGraph()

    async def check_confidence(inputs):
        confidence = inputs.get("confidence", 0)
        return {"confidence": confidence, "needs_human": confidence >= 80}

    async def escalate_path(inputs):
        msg = _llm_call(
            f"Write a one-line escalation note for a security alert with "
            f"{inputs.get('confidence')}% detection confidence that needs analyst review.",
        )
        return {"disposition": msg}

    async def auto_close_path(inputs):
        msg = _llm_call(
            f"Write a one-line auto-close note for a security alert with only "
            f"{inputs.get('confidence')}% detection confidence, mentioning it stays on record.",
        )
        return {"disposition": msg}

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
        line = _llm_call("In one short line, page on-call IR for a CRITICAL alert. SLA 15 min.")
        return {"response": line, "sla": "15 minutes"}

    async def handle_high(inputs):
        line = _llm_call("In one short line, assign a HIGH alert to Tier 2. SLA 1 hour.")
        return {"response": line, "sla": "1 hour"}

    async def handle_medium(inputs):
        line = _llm_call("In one short line, queue a MEDIUM alert for triage. SLA 24 hours.")
        return {"response": line, "sla": "24 hours"}

    async def handle_low(inputs):
        line = _llm_call("In one short line, batch a LOW alert for weekly review. SLA 1 week.")
        return {"response": line, "sla": "1 week"}

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
        print(f"{severity.upper()}: {result.final_state.get('response')}")
    print()


# =============================================================================
# Part 3: Two routers in sequence
# =============================================================================


async def example_chained_conditions():
    """First check the alert source, then — if trusted — check the analyst's role."""
    print("=== Part 3: Two routers in sequence ===\n")

    graph = StateGraph()

    async def validate_source(inputs):
        sensor_key = inputs.get("sensor_key", "")
        is_trusted = sensor_key == "sensor-key-123"  # noqa: S105 — notebook literal, not a secret
        return {"trusted_source": is_trusted}

    async def check_role(inputs):
        role = inputs.get("role", "readonly")
        return {"is_responder": role == "responder"}

    async def contain_action(inputs):
        line = _llm_call("In one line, log that a responder isolated the affected host.")
        return {"result": line}

    async def ticket_action(inputs):
        line = _llm_call("In one line, log that a read-only analyst opened a containment ticket.")
        return {"result": line}

    async def discard_alert(inputs):
        line = _llm_call("In one line, log that an alert from an untrusted sensor was discarded.")
        return {"result": line}

    graph.add_node("source", validate_source)
    graph.add_node("role", check_role)
    graph.add_node("contain", contain_action)
    graph.add_node("ticket", ticket_action)
    graph.add_node("discard", discard_alert)

    graph.add_edge(START, "source")

    graph.add_conditional_edges(
        "source", lambda s: "role" if s.get("trusted_source") else "discard"
    )
    graph.add_conditional_edges("role", lambda s: "contain" if s.get("is_responder") else "ticket")

    graph.add_edge("contain", END)
    graph.add_edge("ticket", END)
    graph.add_edge("discard", END)

    test_cases = [
        {"sensor_key": "forged", "role": "responder"},
        {"sensor_key": "sensor-key-123", "role": "readonly"},
        {"sensor_key": "sensor-key-123", "role": "responder"},
    ]

    for case in test_cases:
        result = await graph.execute(case)
        print(f"Sensor key: {case['sensor_key'][:6]}..., Role: {case['role']}")
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

    async def handle_phishing(inputs):
        line = _llm_call("In one short line, name the team that handles phishing alerts.")
        return {"handler": line}

    async def handle_malware(inputs):
        line = _llm_call("In one short line, name the team that handles malware alerts.")
        return {"handler": line}

    async def handle_network(inputs):
        line = _llm_call("In one short line, name the team that handles network anomaly alerts.")
        return {"handler": line}

    async def handle_other(inputs):
        line = _llm_call("In one short line, name a generic SOC queue for unmatched alerts.")
        return {"handler": line}

    graph.add_node("categorize", categorize)
    graph.add_node("phishing", handle_phishing)
    graph.add_node("malware", handle_malware)
    graph.add_node("network", handle_network)
    graph.add_node("other", handle_other)

    graph.add_edge(START, "categorize")

    graph.add_conditional_edges(
        "categorize",
        lambda s: s.get("family", "other"),
        targets={
            "phishing": "phishing",
            "malware": "malware",
            "network": "network",
        },
        default="other",
    )

    graph.add_edge("phishing", END)
    graph.add_edge("malware", END)
    graph.add_edge("network", END)
    graph.add_edge("other", END)

    for family in ["phishing", "malware", "insider", "xyz"]:
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
        cvss = inputs.get("cvss", 0.0)
        asset_tier = inputs.get("asset_tier", "standard")
        hosts = inputs.get("hosts", 1)

        return {
            "cvss": cvss,
            "asset_tier": asset_tier,
            "hosts": hosts,
            "is_widespread": hosts > 10,
            "is_crown_jewel": asset_tier == "crown_jewel",
            "is_high_cvss": cvss > 7.0,
        }

    async def page_oncall(inputs):
        line = _llm_call("In one short line, confirm the on-call incident commander was paged.")
        return {"routing": line, "response_window": "Immediate"}

    async def priority_queue(inputs):
        line = _llm_call("In one short line, confirm the incident entered the priority queue.")
        return {"routing": line, "response_window": "4 hours"}

    async def standard_queue(inputs):
        line = _llm_call("In one short line, confirm the incident entered the standard queue.")
        return {"routing": line, "response_window": "24 hours"}

    graph.add_node("evaluate", evaluate_incident)
    graph.add_node("page", page_oncall)
    graph.add_node("priority", priority_queue)
    graph.add_node("standard", standard_queue)

    graph.add_edge(START, "evaluate")

    def incident_router(state):
        is_crown_jewel = state.get("is_crown_jewel", False)
        is_high_cvss = state.get("is_high_cvss", False)
        is_widespread = state.get("is_widespread", False)

        if is_crown_jewel and is_high_cvss:
            return "page"
        elif is_crown_jewel or is_high_cvss or is_widespread:
            return "priority"
        else:
            return "standard"

    graph.add_conditional_edges("evaluate", incident_router)

    graph.add_edge("page", END)
    graph.add_edge("priority", END)
    graph.add_edge("standard", END)

    test_cases = [
        {"cvss": 4.3, "asset_tier": "standard", "hosts": 2},
        {"cvss": 4.3, "asset_tier": "crown_jewel", "hosts": 2},
        {"cvss": 9.8, "asset_tier": "standard", "hosts": 2},
        {"cvss": 9.8, "asset_tier": "crown_jewel", "hosts": 20},
    ]

    for case in test_cases:
        result = await graph.execute(case)
        print(f"Incident: CVSS {case['cvss']}, {case['asset_tier']}, {case['hosts']} hosts")
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

    async def classify_with_llm(inputs):
        import time as _t

        alert = inputs.get("alert", "")
        agent = Agent(
            model=get_model(max_tokens=10),
            system_prompt=(
                "Classify the security alert into exactly one of: "
                "phishing, malware, access. Reply with just the single word."
            ),
        )
        t0 = _t.perf_counter()
        result = agent.run_sync(alert)
        dt = _t.perf_counter() - t0
        print(
            f"  [model call: {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
        )
        label = result.message.strip().lower()
        # Defensive: clamp anything unexpected back onto a known label
        # so the conditional edge always finds a target.
        if label not in {"phishing", "malware", "access"}:
            label = "access"
        return {"family": label}

    async def phishing(_inputs):
        return {"handler": "Phishing Response Team"}

    async def malware(_inputs):
        return {"handler": "Malware Analysis Team"}

    async def access(_inputs):
        return {"handler": "Identity & Access Team"}

    graph.add_node("classify", classify_with_llm)
    graph.add_node("phishing", phishing)
    graph.add_node("malware", malware)
    graph.add_node("access", access)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", lambda s: s["family"])
    graph.add_edge("phishing", END)
    graph.add_edge("malware", END)
    graph.add_edge("access", END)

    samples = [
        "User reports an email asking them to verify a password at phish.example.net.",
        "EDR flagged a binary with test hash aa11bb22 on workstation WS-204.",
        "Login for j.doe from 198.51.100.7 minutes after a login from another country.",
    ]
    for alert in samples:
        result = await graph.execute({"alert": alert})
        print(f"  '{alert[:40]}…' → {result.final_state.get('handler')}")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    print("=" * 60)
    print("Notebook 17: Severity-based escalation routing")
    print("=" * 60)
    print()

    await example_binary_routing()
    await example_multiway_routing()
    await example_chained_conditions()
    await example_default_route()
    await example_complex_routing()
    await example_llm_router()

    print("=" * 60)
    print("Next: Notebook 18 — Merging scanner findings with state reducers")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
