# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 54: reasoning patterns — a missed-redaction postmortem.

A privacy-team exercise seeded a synthetic PII canary (a fake SSN) into a
data export and the redaction pipeline never flagged it. The postmortem
question is why the redaction pipeline stayed silent: a stale PII
classifier model and a saturated scan queue amount to an unintended
privacy gap (the kind that ships personal data in the clear — a
data-protection-by-design failure under GDPR Art. 25 / Art. 32). Each
part exercises one piece of the Locus reasoning toolkit against that
postmortem; the parts that run an agent print a
``[model call: X.XXs · prompt→completion tokens]`` banner.

The thesis the reasoning layer enforces here: an ungrounded claim in a
postmortem is how a false root cause gets written into a runbook, so each
claim is scored against tool evidence before it is allowed to stand.

- ``@tool`` + ``Agent(tools=...)`` — let the agent call real Python
  functions over the redaction pipeline's logs and stats.
- ``Agent(reflexion=True)`` and ``Reflector`` — Reflexion is a
  self-critique loop; the agent looks at its own trajectory and
  decides whether it is making progress or stuck.
- ``Agent(output_schema=...)`` — typed JSON for postmortem claims and
  event timelines.
- ``GroundingEvaluator`` — score each claim against tool evidence and
  decide whether to replan. Ungrounded claims in a postmortem are how
  false root causes get written into runbooks.
- ``CausalChain`` / ``build_causal_chain`` — build and walk a graph of
  cause/effect relationships from canary seed to shipped-in-the-clear PII.

Read end to end, the parts form one pipeline — typed claims → grounding
score → replan guidance → causal chain → reflexion over the run — which
``docs/concepts/reasoning.md`` describes piece by piece.

Run it:
    # Default: OCI GenAI if ~/.oci/config has a usable profile, else the bundled mock model
    LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_54_reasoning_patterns.py

    # Offline:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_54_reasoning_patterns.py

Prerequisites:
- An OpenAI or Anthropic API key, or set ``LOCUS_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
- A model that supports constrained JSON decoding for the
  ``output_schema=`` parts. The ``check_structured_output_capable()``
  helper exits cleanly under mock.
"""

import asyncio

from config import check_structured_output_capable, get_model
from pydantic import BaseModel, Field

from locus.agent import Agent
from locus.core.state import AgentState
from locus.reasoning import (
    CausalChain,
    GroundingEvaluator,
    Reflector,
    RelationshipType,
    build_causal_chain,
    evaluate_progress,
)
from locus.tools import tool


def _banner(result, label: str = "") -> None:
    m = result.metrics
    tag = f" {label}" if label else ""
    print(
        f"  [model call{tag}: {m.duration_ms / 1000.0:.2f}s · "
        f"{m.prompt_tokens}→{m.completion_tokens} tokens · iters={m.iterations}]"
    )


# ---------------------------------------------------------------------------
# Pydantic schemas passed to Agent(output_schema=...).
# ---------------------------------------------------------------------------


class ClaimList(BaseModel):
    """Three factual claims about the missed redaction."""

    claims: list[str] = Field(..., description="Three short factual claims.")


class EventList(BaseModel):
    """Causal-ordered list of events leading to the unredacted export."""

    events: list[str] = Field(..., description="Events in causal order.")


# ---------------------------------------------------------------------------
# Tools the agent can call. Real Python — deterministic mock telemetry.
# ---------------------------------------------------------------------------


@tool
def read_redaction_logs(dataset: str) -> str:
    """Pull the last few lines of the redaction pipeline's log."""
    return (
        "[14:02:01] INFO privacy-team synthetic PII canary seeded into export ds-0231\n"
        "[14:02:14] WARN classifier heartbeat stale on ds-0231 (last seen 46 min ago)\n"
        "[14:02:18] ERROR redaction pipeline dropped 3 records (scan queue full)"
    )


@tool
def query_dlp(dataset: str) -> str:
    """Query the DLP console for the dataset's redaction-pipeline vital signs."""
    return (
        f"dataset={dataset} classifier_version=4.2.1 latest_version=5.0.3 records_skipped=3 "
        "redactions_applied=0 scan_latency_min=45 queue_depth_pct=98"
    )


# =============================================================================
# Part 1: Reflexion — self-critique on a real agent trajectory.
# =============================================================================


async def part1_reflexion() -> tuple[Reflector, AgentState]:
    print("\n=== Part 1: Reflexion on a real Agent run (Agent + tool) ===\n")
    postmortem_agent = Agent(
        model=get_model(max_tokens=300),
        tools=[read_redaction_logs],
        system_prompt=(
            "You are a privacy engineer running a missed-redaction postmortem. "
            "Use the read_redaction_logs tool to investigate, then summarise "
            "what went wrong in one short sentence."
        ),
    )
    postmortem_result = await postmortem_agent.arun(
        "Investigate why the synthetic PII canary in export ds-0231 was never redacted."
    )
    _banner(postmortem_result, "Part 1")
    print(f"Agent verdict: {postmortem_result.message[:200]}")

    # These are the library defaults, spelled out so the knobs are visible:
    # three identical tool calls count as a loop, each successful call adds
    # 0.15 confidence, each failed one subtracts 0.2.
    reflector = Reflector(loop_threshold=3, success_weight=0.15, error_penalty=0.2)
    reflection = reflector.reflect(postmortem_result.state)
    print(f"Reflexion assessment: {reflection.assessment.value}")
    print(f"Confidence delta: {reflection.confidence_delta:+.2f}")
    if reflection.guidance:
        print(f"Guidance: {reflection.guidance}")
    return reflector, postmortem_result.state


# =============================================================================
# Part 2: Loop detection — Reflector flags the same tool called repeatedly.
# =============================================================================


def part2_loop_detection(reflector: Reflector) -> None:
    print("\n=== Part 2: Loop detection ===\n")
    loop_state = AgentState(
        agent_id="looping_agent",
        tool_history=("query_dlp",) * 4,
    )
    loop_reflection = reflector.reflect(loop_state)
    print(f"Assessment: {loop_reflection.assessment.value}")
    if loop_reflection.loop_pattern:
        print(f"Loop pattern: {loop_reflection.loop_pattern}")


# =============================================================================
# Part 3: evaluate_progress — one-shot progress check, no Reflector instance.
# =============================================================================


def part3_quick_progress(state: AgentState) -> None:
    print("\n=== Part 3: Quick progress evaluation ===\n")
    quick = evaluate_progress(state=state, loop_threshold=3, success_weight=0.2)
    print(f"Quick assessment: {quick.assessment.value}")


# =============================================================================
# Part 4: Typed claims plus tool-gathered evidence. Two agents:
#   (4a) one fetches evidence, (4b) one produces claims as JSON,
#   (4c) GroundingEvaluator scores claims against evidence.
# =============================================================================


async def part4_grounding() -> tuple[GroundingEvaluator, object]:
    print("\n=== Part 4: Structured claims (output_schema) + tool-fetched evidence ===\n")

    evidence_agent = Agent(
        model=get_model(max_tokens=200),
        tools=[query_dlp],
        system_prompt=(
            "You are a privacy analyst. Call query_dlp for dataset ds-0231 and report "
            "back what it returned, verbatim, on a single line."
        ),
    )
    evidence_result = await evidence_agent.arun("Pull the redaction stats for ds-0231 right now.")
    _banner(evidence_result, "Part 4a")

    # Ground against what the tool actually returned, not the model's
    # paraphrase of it — a paraphrase can drop or round the very field a
    # claim hangs on. A model that answered from memory gives us nothing to
    # ground against, so in that case read the probe directly.
    tool_runs = evidence_result.tool_executions
    evidence_line = (
        tool_runs[0].result if tool_runs and tool_runs[0].result else query_dlp("ds-0231")
    )
    evidence_pieces = [chunk for chunk in evidence_line.split() if "=" in chunk]
    print(f"Tool-gathered evidence ({len(tool_runs)} tool call(s)):")
    for e in evidence_pieces:
        print(f"  - {e}")

    claim_agent = Agent(
        model=get_model(max_tokens=200),
        output_schema=ClaimList,
        system_prompt=(
            "You are a privacy analyst writing a missed-redaction postmortem. Make "
            "exactly three factual claims about the redaction pipeline based on "
            "the stats provided."
        ),
    )
    claim_result = await claim_agent.arun(
        f"Stats from query_dlp: {evidence_line}\n"
        "Produce three factual claims about the redaction-pipeline state."
    )
    _banner(claim_result, "Part 4b")

    parsed_claims: ClaimList | None = claim_result.parsed
    if not isinstance(parsed_claims, ClaimList) or not parsed_claims.claims:
        raise RuntimeError(
            "Claim agent returned no parsed ClaimList. The configured model "
            "could not honor the JSON schema. Use a stronger model "
            "(e.g. openai.gpt-4o, openai.gpt-5, anthropic.claude-3-5-sonnet) "
            f"for notebook 54. Raw output: {claim_result.message!r}"
        )
    claims = parsed_claims.claims[:3]
    print("Model-produced typed claims:")
    for c in claims:
        print(f"  - {c}")

    # Library defaults again: a run whose overall grounding falls under 0.65
    # is sent back to replan, a single claim needs 0.5 to count as grounded,
    # and a claim with no evidence behind it is penalised rather than ignored.
    evaluator = GroundingEvaluator(
        replan_threshold=0.65, claim_threshold=0.5, require_evidence=True
    )
    grounding = evaluator.evaluate(claims, evidence_pieces)
    print(f"\nOverall grounding score: {grounding.score:.2f}")
    print(f"Requires replan: {grounding.requires_replan}")
    for ce in grounding.claims:
        status = "grounded" if ce.is_grounded else "UNGROUNDED"
        print(f"  [{status}] {ce.claim}  (score={ce.score:.2f})")
    return evaluator, grounding


# =============================================================================
# Part 5: Replan guidance — when grounding is low, the evaluator says what
#         to go and fetch.
# =============================================================================


def part5_replan_guidance(evaluator: GroundingEvaluator, grounding) -> None:
    print("\n=== Part 5: Replan guidance ===\n")
    if evaluator.should_replan(grounding):
        print(evaluator.get_replan_guidance(grounding))
    else:
        print("All postmortem claims are sufficiently grounded — no replan needed.")


# =============================================================================
# Part 6: CausalChain — wire typed events into a cause/effect graph.
# =============================================================================


async def part6_causal_chain() -> CausalChain:
    print("\n=== Part 6: Causal chain from typed events (output_schema) ===\n")
    event_agent = Agent(
        model=get_model(max_tokens=300),
        output_schema=EventList,
        system_prompt=(
            "You are a privacy engineer describing a missed-redaction "
            "timeline. Output exactly five events in causal order, no numbering."
        ),
    )
    event_result = await event_agent.arun(
        "Walk through how an outdated PII classifier plus a saturated scan queue "
        "leads to a missed redaction (PII shipped in the clear). Output exactly "
        "five events in causal order."
    )
    _banner(event_result, "Part 6")
    parsed_events: EventList | None = event_result.parsed
    if not isinstance(parsed_events, EventList) or not parsed_events.events:
        raise RuntimeError(
            "Event agent returned no parsed EventList. The configured model "
            "could not honor the JSON schema. Use a stronger model "
            "(e.g. openai.gpt-4o, openai.gpt-5, anthropic.claude-3-5-sonnet) "
            f"for notebook 54. Raw output: {event_result.message!r}"
        )
    event_phrases = parsed_events.events[:5]
    print("Model-generated events:")
    for e in event_phrases:
        print(f"  - {e}")

    events_list: list[dict] = []
    prev: str | None = None
    for phrase in event_phrases:
        entry: dict = {"label": phrase}
        if prev is not None:
            entry["causes"] = [prev]
        events_list.append(entry)
        prev = phrase
    chain = build_causal_chain(events_list, auto_classify=True)
    print("\nAuto-classified chain:")
    for node_id, node_type in chain.classify_nodes().items():
        node = chain.get_node(node_id)
        print(f"  [{node_type.value:12}] {node.label}")
    return chain


# =============================================================================
# Part 7: Walk the chain — get the path from a root cause to a symptom.
# =============================================================================


def part7_causal_path(chain: CausalChain) -> None:
    print("\n=== Part 7: Causal path analysis ===\n")
    roots = chain.identify_root_causes()
    symptoms = chain.identify_symptoms()
    if not (roots and symptoms):
        print("No root→symptom path: the chain has no distinct root and symptom nodes.")
        return
    path = chain.get_causal_path(roots[0].id, symptoms[0].id) or []
    print("Causal path from root cause to symptom:")
    for i, n in enumerate(path):
        prefix = "  " * i + ("-> " if i > 0 else "")
        print(f"{prefix}{n.label}")


# =============================================================================
# Part 8: Conflict detection — flag cycles in the causal graph.
# =============================================================================


def part8_conflicts() -> None:
    print("\n=== Part 8: Conflict detection ===\n")
    conflict_chain = CausalChain()
    a = conflict_chain.create_node(label="Event A")
    b = conflict_chain.create_node(label="Event B")
    conflict_chain.link(a.id, b.id, relationship=RelationshipType.CAUSES)
    conflict_chain.link(b.id, a.id, relationship=RelationshipType.CAUSES)
    for c in conflict_chain.detect_conflicts():
        print(f"  Type: {c.conflict_type}")
        print(f"  Description: {c.description}")
        if c.resolution_hint:
            print(f"  Built-in hint: {c.resolution_hint}")


# =============================================================================
# Part 9: Agent(reflexion=True) — the reflexion loop wired into a live run.
# =============================================================================


async def part9_live_reflexion() -> None:
    print("\n=== Part 9: Live Agent with Reflexion ===\n")
    reflexive_agent = Agent(
        model=get_model(max_tokens=300),
        system_prompt=(
            "You are a privacy-engineering root-cause analyst. Reason step by "
            "step before giving a final one-paragraph conclusion."
        ),
        reflexion=True,
    )
    live = await reflexive_agent.arun(
        "A synthetic PII canary in export ds-0231 was shipped without redaction. "
        "The PII classifier is two major versions behind and the scan queue sat "
        "at 98% capacity. What's the most likely root cause?"
    )
    _banner(live, "Part 9")
    print(f"Conclusion: {live.message[:400]}")


async def main():
    check_structured_output_capable()
    print("=" * 60)
    print("Notebook 54: reasoning patterns — missed-redaction postmortem")
    print("=" * 60)

    reflector, state = await part1_reflexion()
    part2_loop_detection(reflector)
    part3_quick_progress(state)
    evaluator, grounding = await part4_grounding()
    part5_replan_guidance(evaluator, grounding)
    chain = await part6_causal_chain()
    part7_causal_path(chain)
    part8_conflicts()
    await part9_live_reflexion()

    print("\n" + "=" * 60)
    print("Done. Next: notebook 55 — GSAR typed grounding.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
