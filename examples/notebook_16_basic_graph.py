# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 16: Data-subject-request triage pipeline as a graph.

A StateGraph is a directed graph where each node is an async function
that takes the current state in and returns updates to merge back. Use
it when one Agent isn't enough — here we build a privacy-request triage
pipeline (parse → enrich → verdict) plus the fan-out and streaming
shapes a data-protection workflow needs. Scenario: data subject access
request (DSAR) intake under GDPR Article 15.

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
# Part 1: A one-node graph
# =============================================================================


async def example_first_graph():
    """The smallest possible graph: START -> assess -> END."""
    print("=== Part 1: A one-node graph ===\n")

    graph = StateGraph()

    async def assess(inputs):
        subject = inputs.get("subject", "(no subject)")
        ai_line = await _llm_call(
            f"Give a one-sentence first-pass privacy-request assessment of a message "
            f"with the subject '{subject}'.",
            system="You are a data-protection request-triage assistant.",
        )
        return {"assessment": ai_line}

    graph.add_node("assess", assess)
    graph.add_edge(START, "assess")
    graph.add_edge("assess", END)

    result = await graph.execute({"subject": "Please delete all data you hold about me"})

    print("Input:   subject = 'Please delete all data you hold about me'")
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
        request = inputs.get("request", "")
        return {
            "request": request.strip(),
            "has_content": len(request.strip()) > 0,
        }

    async def enrich(inputs):
        request = inputs.get("request", "")
        return {
            "mentions_special_category": "health record" in request,
            "token_count": len(request.split()),
        }

    async def verdict(inputs):
        tc = inputs.get("token_count")
        ai = await _llm_call(
            f"In one short sentence, give a triage verdict for a {tc}-token DSAR "
            f"that {'does' if inputs.get('mentions_special_category') else 'does not'} "
            "mention special-category personal data.",
        )
        return {
            "verdict": f"{tc} tokens, special_category={inputs.get('mentions_special_category')} — {ai}",
        }

    graph.add_node("parse", parse)
    graph.add_node("enrich", enrich)
    graph.add_node("verdict", verdict)

    graph.add_edge(START, "parse")
    graph.add_edge("parse", "enrich")
    graph.add_edge("enrich", "verdict")
    graph.add_edge("verdict", END)

    result = await graph.execute(
        {"request": "  please export my health record and contact details  "}
    )

    print("Input:    request = '  please export my health record and contact details  '")
    print(f"Parsed:   has_content = {result.final_state.get('has_content')}")
    print(
        f"Enriched: mentions_special_category = {result.final_state.get('mentions_special_category')}"
    )
    print(f"VerificationResult:  {result.final_state.get('verdict')}")
    print()


# =============================================================================
# Part 3: How state accumulates
# =============================================================================


async def example_state_flow():
    """Print what each node receives so you can watch the request state grow."""
    print("=== Part 3: How state accumulates ===\n")

    graph = StateGraph()

    async def intake(inputs):
        print(f"  Intake receives:    {list(inputs.keys())}")
        return {"intake_note": "request ingested", "exposure_score": 10}

    async def correlate(inputs):
        print(f"  Correlate receives: {list(inputs.keys())}")
        exposure = inputs.get("exposure_score", 0)
        # Two requests from the same data subject double the working exposure score.
        return {"correlate_note": "matched a sibling request", "combined_exposure": exposure * 2}

    async def score(inputs):
        print(f"  Score receives:     {list(inputs.keys())}")
        combined = inputs.get("combined_exposure", 0)
        ai = await _llm_call(
            f"Comment on a triage pipeline that raised a request's exposure score to {combined}.",
        )
        return {
            "score_note": "final score issued",
            "final_exposure": combined + 5,
            "ai_comment": ai,
        }

    graph.add_node("intake", intake)
    graph.add_node("correlate", correlate)
    graph.add_node("score", score)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "correlate")
    graph.add_edge("correlate", "score")
    graph.add_edge("score", END)

    print("Executing graph...")
    result = await graph.execute({"request_id": "DSAR-1042"})

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

    async def classify_sensitivity(inputs):
        record = inputs.get("record", "")
        label = await _llm_call(
            f"Classify the sensitivity of the data in '{record}' as high, medium, or "
            "low. Reply with one word.",
            system="Output one of: high | medium | low. Nothing else.",
            max_tokens=10,
        )
        return {"sensitivity": label.lower()}

    async def count_identifiers(inputs):
        record = inputs.get("record", "")
        await asyncio.sleep(0.1)
        return {"identifier_count": record.count("@") + record.count("id:")}

    async def check_consent(inputs):
        await asyncio.sleep(0.1)
        # Mock consent-ledger lookup — invented data, clearly fake.
        return {"consent_status": "withdrawn-no-active-basis"}

    async def combine_results(inputs):
        return {
            "enrichment": {
                "sensitivity": inputs.get("sensitivity"),
                "identifiers": inputs.get("identifier_count"),
                "consent": inputs.get("consent_status"),
            }
        }

    graph.add_node("sensitivity", classify_sensitivity)
    graph.add_node("identifiers", count_identifiers)
    graph.add_node("consent", check_consent)
    graph.add_node("combine", combine_results)

    graph.add_edge(START, "sensitivity")
    graph.add_edge(START, "identifiers")
    graph.add_edge(START, "consent")

    graph.add_edge("sensitivity", "combine")
    graph.add_edge("identifiers", "combine")
    graph.add_edge("consent", "combine")

    graph.add_edge("combine", END)

    import time

    start = time.time()
    result = await graph.execute(
        {"record": "subject id:8842 email jordan@example.com phone +1-555-0100"}
    )
    elapsed = (time.time() - start) * 1000

    print("Input: 'subject id:8842 email jordan@example.com phone +1-555-0100'")
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
        comment = await _llm_call(f"In one sentence, comment on normalizing a raw exposure score of {v}.")
        return {"normalized": True, "exposure_score": v * 2, "comment": comment}

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
        await emit_custom({"phase": "starting", "record_count": inputs.get("records", 0)})
        ai = await _llm_call(
            f"In one sentence, narrate a discovery pass that grew the matched-record list "
            f"to {inputs.get('records', 0) * 2} entries.",
        )
        await emit_custom({"phase": "halfway"})
        return {"matched_records": inputs.get("records", 0) * 2, "ai": ai}

    async def report(inputs):
        ai = await _llm_call(
            f"In one short sentence, narrate adding 10 redaction notes to "
            f"{inputs.get('matched_records', 0)} records in a DSAR response pack.",
        )
        return {"report_entries": inputs.get("matched_records", 0) + 10, "ai": ai}

    graph.add_node("enrich", enrich)
    graph.add_node("report", report)
    graph.add_edge(START, "enrich")
    graph.add_edge("enrich", "report")
    graph.add_edge("report", END)

    print("Streaming UPDATES + CUSTOM events as they arrive:")
    async for event in graph.stream({"records": 21}, mode=StreamMode.UPDATES):
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

        activity = inputs.get("activity", "")
        agent = Agent(
            model=get_model(max_tokens=80),
            system_prompt="You write one-sentence factual summaries for DPO handover notes.",
        )
        t0 = _t.perf_counter()
        result = await agent.arun(f"Summarize the processing activity '{activity}' in one sentence.")
        dt = _t.perf_counter() - t0
        print(
            f"  [model call: {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
        )
        return {"summary": result.message}

    graph.add_node("summarize", ai_summarize)
    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", END)

    result = await graph.execute(
        {"activity": "marketing analytics joining email opens to CRM profiles"}
    )
    print(f"AI summary: {result.final_state.get('summary')}")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    print("=" * 60)
    print("Notebook 16: Data-subject-request triage pipeline as a graph")
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
    print("Next: Notebook 17 — Routing requests by privacy risk")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
