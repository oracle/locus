# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 16: Phishing triage pipeline as a graph.

A StateGraph is a directed graph where each node is an async function
that takes the current state in and returns updates to merge back. Use
it when one Agent isn't enough — here we build a phishing triage
pipeline (parse → enrich → verdict) plus the fan-out and streaming
shapes a SOC workflow needs. Scenario: phishing intake
(MITRE ATT&CK T1566, Phishing).

- Nodes and edges: nodes do triage work, edges describe order.
- START and END: the two sentinel node ids that frame execution.
- Sequential, parallel, and conditional flow on the same primitives.
- GraphResult: final state plus per-node status, timing, and ordering.
- Streaming: emit progress events from inside a node with emit_custom.

Run it:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_16_basic_graph.py

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
# Part 1: A one-node graph
# =============================================================================


async def example_first_graph():
    """The smallest possible graph: START -> assess -> END."""
    print("=== Part 1: A one-node graph ===\n")

    graph = StateGraph()

    async def assess(inputs):
        subject = inputs.get("subject", "(no subject)")
        ai_line = _llm_call(
            f"Give a one-sentence first-pass phishing assessment of an email "
            f"with the subject '{subject}'.",
            system="You are a SOC phishing-triage assistant.",
        )
        return {"assessment": ai_line}

    graph.add_node("assess", assess)
    graph.add_edge(START, "assess")
    graph.add_edge("assess", END)

    result = await graph.execute({"subject": "URGENT: verify your account now"})

    print("Input:   subject = 'URGENT: verify your account now'")
    print(f"Output:  assessment = '{result.final_state.get('assessment')}'")
    print(f"Success: {result.success}")
    print()


# =============================================================================
# Part 2: A sequence of nodes — parse → enrich → verdict
# =============================================================================


async def example_sequence():
    """Chain three triage nodes; each node's return value merges into shared state."""
    print("=== Part 2: A sequence of nodes — parse → enrich → verdict ===\n")

    graph = StateGraph()

    async def parse(inputs):
        report = inputs.get("report", "")
        return {
            "report": report.strip(),
            "has_content": len(report.strip()) > 0,
        }

    async def enrich(inputs):
        report = inputs.get("report", "")
        return {
            "mentions_lure_domain": "phish.example.net" in report,
            "token_count": len(report.split()),
        }

    async def verdict(inputs):
        tc = inputs.get("token_count")
        ai = _llm_call(
            f"In one short sentence, give a triage verdict for a {tc}-token user "
            f"report that {'does' if inputs.get('mentions_lure_domain') else 'does not'} "
            "mention a known lure domain.",
        )
        return {
            "verdict": f"{tc} tokens, lure_domain={inputs.get('mentions_lure_domain')} — {ai}",
        }

    graph.add_node("parse", parse)
    graph.add_node("enrich", enrich)
    graph.add_node("verdict", verdict)

    graph.add_edge(START, "parse")
    graph.add_edge("parse", "enrich")
    graph.add_edge("enrich", "verdict")
    graph.add_edge("verdict", END)

    result = await graph.execute({"report": "  user clicked link to phish.example.net  "})

    print("Input:    report = '  user clicked link to phish.example.net  '")
    print(f"Parsed:   has_content = {result.final_state.get('has_content')}")
    print(f"Enriched: mentions_lure_domain = {result.final_state.get('mentions_lure_domain')}")
    print(f"Verdict:  {result.final_state.get('verdict')}")
    print()


# =============================================================================
# Part 3: How state accumulates
# =============================================================================


async def example_state_flow():
    """Print what each node receives so you can watch investigation state grow."""
    print("=== Part 3: How state accumulates ===\n")

    graph = StateGraph()

    async def intake(inputs):
        print(f"  Intake receives:    {list(inputs.keys())}")
        return {"intake_note": "alert ingested", "risk_score": 10}

    async def correlate(inputs):
        print(f"  Correlate receives: {list(inputs.keys())}")
        risk = inputs.get("risk_score", 0)
        # Two related alerts on the same host double the working risk score.
        return {"correlate_note": "matched a sibling alert", "combined_risk": risk * 2}

    async def score(inputs):
        print(f"  Score receives:     {list(inputs.keys())}")
        combined = inputs.get("combined_risk", 0)
        ai = _llm_call(
            f"Comment on a triage pipeline that raised an alert's risk score to {combined}.",
        )
        return {"score_note": "final score issued", "final_risk": combined + 5, "ai_comment": ai}

    graph.add_node("intake", intake)
    graph.add_node("correlate", correlate)
    graph.add_node("score", score)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "correlate")
    graph.add_edge("correlate", "score")
    graph.add_edge("score", END)

    print("Executing graph...")
    result = await graph.execute({"alert_id": "ALR-1042"})

    print("\nFinal state:")
    for key, value in result.final_state.items():
        if not key.startswith("_"):
            print(f"  {key}: {value}")
    print()


# =============================================================================
# Part 4: Fan-out and fan-in
# =============================================================================


async def example_parallel():
    """Three independent enrichment nodes run concurrently, then converge in one."""
    print("=== Part 4: Fan-out and fan-in ===\n")

    graph = StateGraph()
    graph.config.parallel = True

    async def classify_tone(inputs):
        email = inputs.get("email", "")
        label = _llm_call(
            f"Classify the tone of the email '{email}' as urgent, neutral, or "
            "friendly. Reply with one word.",
            system="Output one of: urgent | neutral | friendly. Nothing else.",
            max_tokens=10,
        )
        return {"tone": label.lower()}

    async def count_links(inputs):
        email = inputs.get("email", "")
        await asyncio.sleep(0.1)
        return {"link_count": email.count("http")}

    async def check_sender(inputs):
        await asyncio.sleep(0.1)
        # Mock reputation lookup — invented data, clearly fake.
        return {"sender_reputation": "unknown-newly-registered"}

    async def combine_results(inputs):
        return {
            "enrichment": {
                "tone": inputs.get("tone"),
                "links": inputs.get("link_count"),
                "sender": inputs.get("sender_reputation"),
            }
        }

    graph.add_node("tone", classify_tone)
    graph.add_node("links", count_links)
    graph.add_node("sender", check_sender)
    graph.add_node("combine", combine_results)

    graph.add_edge(START, "tone")
    graph.add_edge(START, "links")
    graph.add_edge(START, "sender")

    graph.add_edge("tone", "combine")
    graph.add_edge("links", "combine")
    graph.add_edge("sender", "combine")

    graph.add_edge("combine", END)

    import time

    start = time.time()
    result = await graph.execute(
        {"email": "Your account is suspended! Verify at http://phish.example.net/login"}
    )
    elapsed = (time.time() - start) * 1000

    print("Input: 'Your account is suspended! Verify at http://phish.example.net/login'")
    print(f"Enrichment: {result.final_state.get('enrichment')}")
    print(f"Time: {elapsed:.0f}ms (parallel nodes run concurrently)")
    print()


# =============================================================================
# Part 5: Reading GraphResult
# =============================================================================


async def example_results():
    """GraphResult gives you final state plus per-node status and timing."""
    print("=== Part 5: Reading GraphResult ===\n")

    graph = StateGraph()

    async def normalize(inputs):
        v = inputs.get("raw_score", 0)
        comment = _llm_call(f"In one sentence, comment on normalizing a raw alert score of {v}.")
        return {"normalized": True, "risk_score": v * 2, "comment": comment}

    graph.add_node("normalize", normalize)
    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", END)

    result = await graph.execute({"raw_score": 21})

    print("GraphResult fields:")
    print(f"  .success         = {result.success}")
    print(f"  .graph_id        = {result.graph_id}")
    print(f"  .duration_ms     = {result.duration_ms:.1f}")
    print(f"  .iterations      = {result.iterations}")
    print(f"  .execution_order = {result.execution_order}")

    print("\n  .final_state:")
    for k, v in result.final_state.items():
        if not k.startswith("_"):
            print(f"    {k}: {v}")

    print("\n  .node_results:")
    for node_id, node_result in result.node_results.items():
        print(f"    {node_id}: status={node_result.status.value}")
    print()


async def example_streaming():
    """Stream node updates and push custom progress events from inside a node."""
    print("=== Part 6: Streaming with emit_custom ===\n")
    from tulip.multiagent import StreamMode, emit_custom

    graph = StateGraph()

    async def enrich(inputs):
        # emit_custom — push an arbitrary payload onto the event stream
        # while the node is still running. Useful for long-running lookups.
        await emit_custom({"phase": "starting", "ioc_count": inputs.get("iocs", 0)})
        ai = _llm_call(
            f"In one sentence, narrate an enrichment pass that grew the IOC list "
            f"to {inputs.get('iocs', 0) * 2} entries.",
        )
        await emit_custom({"phase": "halfway"})
        return {"enriched_iocs": inputs.get("iocs", 0) * 2, "ai": ai}

    async def report(inputs):
        ai = _llm_call(
            f"In one short sentence, narrate adding 10 context entries to "
            f"{inputs.get('enriched_iocs', 0)} IOCs in a triage report.",
        )
        return {"report_entries": inputs.get("enriched_iocs", 0) + 10, "ai": ai}

    graph.add_node("enrich", enrich)
    graph.add_node("report", report)
    graph.add_edge(START, "enrich")
    graph.add_edge("enrich", "report")
    graph.add_edge("report", END)

    print("Streaming UPDATES + CUSTOM events as they arrive:")
    async for event in graph.stream({"iocs": 21}, mode=StreamMode.UPDATES):
        if event.mode == StreamMode.CUSTOM:
            print(f"  [CUSTOM]  {event.node_id}: {event.data}")
        else:
            print(f"  [UPDATE]  {event.node_id}: {event.data}")
    print()


# =============================================================================
# Part 7: An Agent inside a node
# =============================================================================


async def example_graph_with_llm():
    """A node can hold a full Agent — graphs and agents compose freely."""
    print("=== Part 7: An Agent inside a node ===\n")

    graph = StateGraph()

    async def ai_summarize(inputs):
        import time as _t

        campaign = inputs.get("campaign", "")
        agent = Agent(
            model=get_model(max_tokens=80),
            system_prompt="You write one-sentence factual summaries for SOC handover notes.",
        )
        t0 = _t.perf_counter()
        result = agent.run_sync(f"Summarize the threat campaign '{campaign}' in one sentence.")
        dt = _t.perf_counter() - t0
        print(
            f"  [model call: {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
        )
        return {"summary": result.message}

    graph.add_node("summarize", ai_summarize)
    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", END)

    result = await graph.execute({"campaign": "credential-phishing wave spoofing IT helpdesk"})
    print(f"AI summary: {result.final_state.get('summary')}")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    print("=" * 60)
    print("Notebook 16: Phishing triage pipeline as a graph")
    print("=" * 60)
    print()

    await example_first_graph()
    await example_sequence()
    await example_state_flow()
    await example_parallel()
    await example_results()
    await example_streaming()
    await example_graph_with_llm()

    print("=" * 60)
    print("Next: Notebook 17 — Severity-based escalation routing")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
