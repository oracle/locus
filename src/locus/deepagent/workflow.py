# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Research workflow primitives — composable nodes for long-horizon research.

Each public function returns an **async node function** compatible with
``locus.StateGraph``.  Nodes read from and write to a plain ``dict[str, Any]``
state; no fixed schema is imposed.  The well-known key names are exported as
constants so callers and nodes stay in sync without a shared TypedDict.

Typical composition::

    from locus.multiagent.graph import END, START, StateGraph
    from locus.deepagent.workflow import (
        make_execute_node,
        make_causal_inference_node,
        make_summarize_node,
        make_grounding_eval_node,
        make_regenerate_summary_node,
        make_replan_node,
        route_after_grounding,
        KEY_PROMPT,
    )

    graph = StateGraph()
    graph.add_node("execute", make_execute_node(model, tools))
    graph.add_node("causal_inference", make_causal_inference_node(model))
    graph.add_node("summarize", make_summarize_node(model))
    graph.add_node("grounding_eval", make_grounding_eval_node(model))
    graph.add_node("regenerate", make_regenerate_summary_node(model))
    graph.add_node("replan", make_replan_node())

    router = route_after_grounding(threshold=0.65, max_replans=2)

    graph.add_edge(START, "execute")
    graph.add_edge("execute", "causal_inference")
    graph.add_edge("causal_inference", "summarize")
    graph.add_edge("summarize", "grounding_eval")
    graph.add_conditional_edges(
        "grounding_eval",
        router,
        {"regenerate": "regenerate", "replan": "replan", END: END},
    )
    graph.add_edge("regenerate", "grounding_eval")
    graph.add_edge("replan", "execute")

    workflow = graph.compile()
    result = await workflow.execute({KEY_PROMPT: "Investigate ..."})

Or use the convenience factory :func:`create_research_workflow` which wires
the same graph with sensible defaults.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Well-known state keys
# ---------------------------------------------------------------------------

KEY_PROMPT = "prompt"
"""The original research prompt (str)."""

KEY_EXECUTE_PROMPT = "execute_prompt"
"""Override prompt for the current execute phase; falls back to KEY_PROMPT."""

KEY_EVIDENCE = "evidence"
"""List[str] — raw tool output strings collected during execute."""

KEY_GROUNDING_FACTS = "grounding_facts"
"""List[dict] — structured facts with stable IDs: {id, text, source}."""

KEY_CAUSAL_CHAIN = "causal_chain"
"""locus.reasoning.causal.CausalChain built from the evidence."""

KEY_CAUSAL_HYPOTHESIS = "causal_hypothesis"
"""str — primary root-cause hypothesis extracted from the causal chain."""

KEY_CAUSAL_CONFIDENCE = "causal_confidence"
"""float — confidence in the causal hypothesis (0.0 – 1.0)."""

KEY_SUMMARY = "summary"
"""str — distilled summary produced by the summarize node."""

KEY_STRUCTURED_OUTPUT = "structured_output"
"""Any — parsed output_schema instance from the summary (if configured)."""

KEY_GROUNDING_SCORE = "grounding_score"
"""float — GroundingEvaluator score for the current summary (0.0 – 1.0)."""

KEY_UNGROUNDED_CLAIMS = "ungrounded_claims"
"""List[str] — claims that fell below the grounding threshold."""

KEY_REPLAN_COUNT = "replan_count"
"""int — number of full execute-phase replans consumed so far."""

KEY_REGENERATION_COUNT = "regeneration_count"
"""int — number of lightweight summary-regeneration attempts so far."""

KEY_STOP_REASON = "stop_reason"
"""str — terminal reason: 'grounded' | 'max_replans' | 'max_regenerations'."""


# ---------------------------------------------------------------------------
# Node: execute
# ---------------------------------------------------------------------------


def make_execute_node(
    model: Any,
    tools: list[Any],
    system_prompt: str = "",
    reflexion: bool = True,
    max_iterations: int = 20,
) -> Any:
    """Return an async node that runs the ReAct agent loop.

    Reads:  ``KEY_EXECUTE_PROMPT`` (falls back to ``KEY_PROMPT``)
    Writes: ``KEY_EVIDENCE``, ``KEY_GROUNDING_FACTS``

    Each tool result is appended to ``evidence`` as a raw string and also
    stored in ``grounding_facts`` with a stable ``fact_id`` so downstream
    nodes (grounding_eval, regenerate) can cite specific pieces of evidence.
    """
    from locus.agent.agent import Agent  # noqa: PLC0415
    from locus.core.events import TerminateEvent, ToolCompleteEvent  # noqa: PLC0415
    from locus.observability.emit import (  # noqa: PLC0415
        EV_RESEARCH_EXECUTE_COMPLETED,
        EV_RESEARCH_EXECUTE_STARTED,
        emit,
    )

    base_prompt = system_prompt or (
        "You are a research agent. Use tools to investigate the given topic. "
        "Gather as much evidence as possible before concluding."
    )

    async def _execute(state: dict[str, Any]) -> dict[str, Any]:
        prompt = state.get(KEY_EXECUTE_PROMPT) or state.get(KEY_PROMPT, "")
        replan_count = state.get(KEY_REPLAN_COUNT, 0)

        await emit(EV_RESEARCH_EXECUTE_STARTED, prompt_preview=prompt[:120], replan=replan_count)

        agent = Agent(
            model=model,
            tools=tools,
            system_prompt=base_prompt,
            reflexion=reflexion,
            max_iterations=max_iterations,
        )

        evidence: list[str] = []
        grounding_facts: list[dict[str, Any]] = []
        fact_idx = 0

        async for event in agent.run(prompt):
            if isinstance(event, ToolCompleteEvent) and event.result:
                raw = str(event.result)[:2000]
                evidence.append(raw)
                grounding_facts.append(
                    {
                        "id": f"fact_{fact_idx:03d}",
                        "text": raw,
                        "source": event.tool_name,
                    }
                )
                fact_idx += 1
            elif isinstance(event, TerminateEvent) and event.final_message:
                conclusion = f"[conclusion] {event.final_message[:2000]}"
                evidence.append(conclusion)
                grounding_facts.append(
                    {
                        "id": f"fact_{fact_idx:03d}",
                        "text": conclusion,
                        "source": "agent_conclusion",
                    }
                )

        await emit(EV_RESEARCH_EXECUTE_COMPLETED, fact_count=len(grounding_facts))
        return {KEY_EVIDENCE: evidence, KEY_GROUNDING_FACTS: grounding_facts}

    return _execute


# ---------------------------------------------------------------------------
# Node: causal_inference
# ---------------------------------------------------------------------------


def make_causal_inference_node(
    model: Any,
    max_nodes: int = 10,
) -> Any:
    """Return an async node that builds a causal chain from evidence.

    Uses an LLM call to extract causal events from the evidence, then
    constructs a ``locus.reasoning.causal.CausalChain`` and identifies
    the primary root-cause hypothesis.

    Reads:  ``KEY_EVIDENCE``, ``KEY_PROMPT``
    Writes: ``KEY_CAUSAL_CHAIN``, ``KEY_CAUSAL_HYPOTHESIS``, ``KEY_CAUSAL_CONFIDENCE``
    """
    from locus.agent.agent import Agent  # noqa: PLC0415
    from locus.observability.emit import EV_RESEARCH_CAUSAL_BUILT, emit  # noqa: PLC0415
    from locus.reasoning.causal import build_causal_chain  # noqa: PLC0415

    async def _causal_inference(state: dict[str, Any]) -> dict[str, Any]:
        evidence = state.get(KEY_EVIDENCE, [])
        prompt = state.get(KEY_PROMPT, "")

        if not evidence:
            return {
                KEY_CAUSAL_CHAIN: None,
                KEY_CAUSAL_HYPOTHESIS: "",
                KEY_CAUSAL_CONFIDENCE: 0.0,
            }

        evidence_block = "\n".join(f"- {e[:500]}" for e in evidence[:20])
        causal_prompt = (
            f"Research goal: {prompt}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            f"Extract up to {max_nodes} causal events as a JSON array. "
            f'Each item: {{"label": "event description", "causes": ["prior event"], '
            f'"type": "root_cause|intermediate|symptom|unknown", "confidence": 0.0-1.0}}.\n'
            f"Order from root cause to final symptom. Return ONLY the JSON array."
        )

        extractor = Agent(
            model=model,
            system_prompt="You extract causal chains from evidence. Return only valid JSON.",
        )
        result = extractor.run_sync(causal_prompt)

        events: list[dict[str, Any]] = []
        try:
            raw = result.message or ""
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                events = json.loads(raw[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        if not events:
            return {
                KEY_CAUSAL_CHAIN: None,
                KEY_CAUSAL_HYPOTHESIS: "",
                KEY_CAUSAL_CONFIDENCE: 0.0,
            }

        chain = build_causal_chain(events)

        # Extract primary hypothesis from root-cause nodes
        from locus.reasoning.causal import NodeType  # noqa: PLC0415

        root_causes = [n for n in chain.nodes.values() if n.node_type == NodeType.ROOT_CAUSE]
        hypothesis = (
            root_causes[0].label
            if root_causes
            else (next(iter(chain.nodes.values())).label if chain.nodes else "")
        )
        confidence = root_causes[0].confidence if root_causes else 0.5

        await emit(
            EV_RESEARCH_CAUSAL_BUILT,
            node_count=len(chain.nodes),
            hypothesis_preview=hypothesis[:120],
            confidence=confidence,
        )
        return {
            KEY_CAUSAL_CHAIN: chain,
            KEY_CAUSAL_HYPOTHESIS: hypothesis,
            KEY_CAUSAL_CONFIDENCE: confidence,
        }

    return _causal_inference


# ---------------------------------------------------------------------------
# Node: summarize
# ---------------------------------------------------------------------------


def make_summarize_node(
    model: Any,
    output_schema: type[BaseModel] | None = None,
) -> Any:
    """Return an async node that distills evidence + causal context into a summary.

    Incorporates the causal hypothesis (if present) so the summary narrative
    is causally grounded before claims are evaluated.

    Reads:  ``KEY_EVIDENCE``, ``KEY_PROMPT``, ``KEY_CAUSAL_HYPOTHESIS``
    Writes: ``KEY_SUMMARY``, ``KEY_STRUCTURED_OUTPUT`` (if output_schema set)
    """
    from locus.agent.agent import Agent  # noqa: PLC0415
    from locus.observability.emit import EV_RESEARCH_SUMMARIZE_COMPLETED, emit  # noqa: PLC0415

    async def _summarize(state: dict[str, Any]) -> dict[str, Any]:
        evidence = state.get(KEY_EVIDENCE, [])
        prompt = state.get(KEY_PROMPT, "")
        causal_hypothesis = state.get(KEY_CAUSAL_HYPOTHESIS, "")

        evidence_block = "\n\n".join(f"[{i + 1}] {e}" for i, e in enumerate(evidence))

        causal_context = f"\n\nCausal hypothesis: {causal_hypothesis}" if causal_hypothesis else ""

        schema_hint = ""
        if output_schema:
            schema_hint = (
                f"\n\nReturn your answer as a JSON object matching this schema:\n"
                f"{json.dumps(output_schema.model_json_schema(), indent=2)}"
            )

        summarize_prompt = (
            f"Research goal: {prompt}{causal_context}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            f"Write a concise, factually grounded summary of your findings. "
            f"Only assert what the evidence supports.{schema_hint}"
        )

        summarizer = Agent(
            model=model,
            system_prompt="You are a precise summarizer. Cite only what the evidence supports.",
            output_schema=output_schema,
        )
        result = summarizer.run_sync(summarize_prompt)

        update: dict[str, Any] = {KEY_SUMMARY: result.message or ""}
        if output_schema and result.parsed:
            update[KEY_STRUCTURED_OUTPUT] = result.parsed

        await emit(
            EV_RESEARCH_SUMMARIZE_COMPLETED,
            summary_length=len(result.message or ""),
            has_structured_output=bool(output_schema and result.parsed),
        )
        return update

    return _summarize


# ---------------------------------------------------------------------------
# Node: grounding_eval
# ---------------------------------------------------------------------------


def make_grounding_eval_node(model: Any) -> Any:
    """Return an async node that scores summary claims against evidence (LLM-as-judge).

    Reads:  ``KEY_SUMMARY``, ``KEY_EVIDENCE``
    Writes: ``KEY_GROUNDING_SCORE``, ``KEY_UNGROUNDED_CLAIMS``
    """
    from locus.observability.emit import EV_RESEARCH_GROUNDING_EVALUATED, emit  # noqa: PLC0415
    from locus.reasoning.grounding import GroundingEvaluator  # noqa: PLC0415

    evaluator = GroundingEvaluator()

    async def _grounding_eval(state: dict[str, Any]) -> dict[str, Any]:
        summary = state.get(KEY_SUMMARY, "")
        evidence = state.get(KEY_EVIDENCE, [])

        if not summary or not evidence:
            return {KEY_GROUNDING_SCORE: 0.0, KEY_UNGROUNDED_CLAIMS: []}

        claims = [s.strip() for s in summary.replace("\n", " ").split(".") if len(s.strip()) > 10]

        grounding_result = await evaluator.evaluate_with_llm(
            claims=claims,
            evidence=evidence,
            model=model,
        )

        await emit(
            EV_RESEARCH_GROUNDING_EVALUATED,
            score=grounding_result.score,
            claims_evaluated=len(claims),
            ungrounded_count=len(grounding_result.ungrounded_claims),
            requires_replan=grounding_result.requires_replan,
        )
        return {
            KEY_GROUNDING_SCORE: grounding_result.score,
            KEY_UNGROUNDED_CLAIMS: grounding_result.ungrounded_claims,
        }

    return _grounding_eval


# ---------------------------------------------------------------------------
# Node: regenerate_summary (lightweight recovery — no tool re-run)
# ---------------------------------------------------------------------------


def make_regenerate_summary_node(
    model: Any,
    output_schema: type[BaseModel] | None = None,
) -> Any:
    """Return an async node that rewrites the summary using grounding feedback.

    Cheaper than a full replan: preserves all tool outputs and only
    re-synthesizes the narrative, targeting ungrounded claims specifically.

    Reads:  ``KEY_SUMMARY``, ``KEY_EVIDENCE``, ``KEY_UNGROUNDED_CLAIMS``,
            ``KEY_GROUNDING_FACTS``, ``KEY_REGENERATION_COUNT``
    Writes: ``KEY_SUMMARY``, ``KEY_STRUCTURED_OUTPUT``, ``KEY_REGENERATION_COUNT``
    """
    from locus.agent.agent import Agent  # noqa: PLC0415
    from locus.observability.emit import (  # noqa: PLC0415
        EV_RESEARCH_REGENERATE_COMPLETED,
        EV_RESEARCH_REGENERATE_STARTED,
        emit,
    )

    async def _regenerate(state: dict[str, Any]) -> dict[str, Any]:
        ungrounded = state.get(KEY_UNGROUNDED_CLAIMS, [])
        evidence = state.get(KEY_EVIDENCE, [])
        old_summary = state.get(KEY_SUMMARY, "")
        regen_count = state.get(KEY_REGENERATION_COUNT, 0)

        await emit(EV_RESEARCH_REGENERATE_STARTED, ungrounded_count=len(ungrounded))

        ungrounded_block = "\n".join(f"- {c}" for c in ungrounded[:8])
        evidence_block = "\n\n".join(f"[{i + 1}] {e}" for i, e in enumerate(evidence))

        schema_hint = ""
        if output_schema:
            schema_hint = (
                f"\n\nReturn as JSON matching:\n"
                f"{json.dumps(output_schema.model_json_schema(), indent=2)}"
            )

        regen_prompt = (
            f"Previous summary:\n{old_summary}\n\n"
            f"The following claims were NOT grounded by the evidence:\n"
            f"{ungrounded_block}\n\n"
            f"Evidence available:\n{evidence_block}\n\n"
            f"Rewrite the summary. Remove or qualify any ungrounded claims. "
            f"Only assert what the evidence explicitly supports.{schema_hint}"
        )

        agent = Agent(
            model=model,
            system_prompt="You are a precise editor. Remove ungrounded claims.",
            output_schema=output_schema,
        )
        result = agent.run_sync(regen_prompt)

        update: dict[str, Any] = {
            KEY_SUMMARY: result.message or old_summary,
            KEY_REGENERATION_COUNT: regen_count + 1,
        }
        if output_schema and result.parsed:
            update[KEY_STRUCTURED_OUTPUT] = result.parsed

        await emit(EV_RESEARCH_REGENERATE_COMPLETED, regeneration=regen_count + 1)
        return update

    return _regenerate


# ---------------------------------------------------------------------------
# Node: replan (full retry — returns focused execute prompt)
# ---------------------------------------------------------------------------


def make_replan_node() -> Any:
    """Return an async node that generates a focused re-plan prompt.

    Unlike regenerate_summary, this triggers a full execute phase with a
    narrower scope targeting the ungrounded claims.

    Reads:  ``KEY_PROMPT``, ``KEY_UNGROUNDED_CLAIMS``, ``KEY_REPLAN_COUNT``
    Writes: ``KEY_EXECUTE_PROMPT``, ``KEY_REPLAN_COUNT``
    """

    from locus.observability.emit import EV_RESEARCH_REPLAN, emit  # noqa: PLC0415

    async def _replan(state: dict[str, Any]) -> dict[str, Any]:
        ungrounded = state.get(KEY_UNGROUNDED_CLAIMS, [])
        prompt = state.get(KEY_PROMPT, "")
        replan_count = state.get(KEY_REPLAN_COUNT, 0)

        if ungrounded:
            focused = "\n".join(f"- {c}" for c in ungrounded[:6])
            execute_prompt = (
                f"Previous investigation of '{prompt}' left these claims unverified:\n"
                f"{focused}\n\n"
                f"Gather specific evidence to verify or refute each claim. "
                f"Focus only on the gaps above."
            )
        else:
            execute_prompt = (
                f"Re-investigate '{prompt}' with a focus on gathering "
                f"stronger, more specific evidence."
            )

        await emit(
            EV_RESEARCH_REPLAN,
            replan=replan_count + 1,
            ungrounded_count=len(ungrounded),
            prompt_preview=execute_prompt[:120],
        )
        return {
            KEY_EXECUTE_PROMPT: execute_prompt,
            KEY_REPLAN_COUNT: replan_count + 1,
        }

    return _replan


# ---------------------------------------------------------------------------
# Routing helper
# ---------------------------------------------------------------------------


def route_after_grounding(
    threshold: float = 0.65,
    max_replans: int = 2,
    max_regenerations: int = 1,
) -> Any:
    """Return a routing function for the grounding_eval conditional edge.

    Recovery strategy (mirrors Optic's two-level approach):
    1. First failure  → ``"regenerate"``  (cheap: rewrite without re-running tools)
    2. Subsequent     → ``"replan"``      (expensive: full execute retry)
    3. Limits reached → ``END``

    Compatible with ``StateGraph.add_conditional_edges``.
    """
    from locus.multiagent.graph import END  # noqa: PLC0415

    def _route(state: dict[str, Any]) -> str:
        score = state.get(KEY_GROUNDING_SCORE, 0.0)
        replans = state.get(KEY_REPLAN_COUNT, 0)
        regens = state.get(KEY_REGENERATION_COUNT, 0)

        if score >= threshold:
            return END
        if replans >= max_replans and regens >= max_regenerations:
            return END
        if regens < max_regenerations:
            return "regenerate"
        return "replan"

    return _route


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_research_workflow(
    *,
    model: Any,
    tools: list[Any],
    system_prompt: str = "",
    output_schema: type[BaseModel] | None = None,
    grounding_threshold: float = 0.65,
    max_replans: int = 2,
    max_regenerations: int = 1,
    max_iterations: int = 20,
    summarization_model: Any | None = None,
    grounding_model: Any | None = None,
    reflexion: bool = True,
    causal_inference: bool = True,
    checkpointer: Any | None = None,
    datastores: dict[str, Any] | None = None,
    datastore_top_k: int = 5,
) -> Any:
    """Compose the standard research workflow from individual node primitives.

    Graph topology::

        START → execute → [causal_inference →] summarize → grounding_eval
                                                                ↓
                                            score ≥ threshold → END
                                            regens < max      → regenerate → grounding_eval
                                            else              → replan → execute

    Args:
        model: Primary model for execute + causal inference phases.
        tools: Tools available to the execute agent.
        system_prompt: Domain identity for the execute agent.
        output_schema: Optional Pydantic model for structured output.
        grounding_threshold: Minimum grounding score to accept (default 0.65).
        max_replans: Maximum full-execute retries (default 2).
        max_regenerations: Maximum lightweight summary rewrites (default 1).
        max_iterations: Max ReAct iterations per execute phase (default 20).
        summarization_model: Model for summarize + regenerate nodes.
        grounding_model: Model for grounding_eval node.
        reflexion: Enable per-turn self-evaluation in execute agent.
        causal_inference: Insert causal_inference node before summarize.
        checkpointer: Optional locus checkpointer for the StateGraph.
        datastores: Optional mapping of ``{name: RAGRetriever}`` (or
            ``{name: {"retriever": ..., "description": ..., "top_k": ...,
            "threshold": ...}}``). For each entry, a ``search_{name}`` tool
            is auto-wired via ``locus.rag.tools.create_rag_tool`` and
            appended to the execute agent's tool list, and a per-store
            routing hint block is prepended to ``system_prompt`` so the
            model picks the right store per query. Same shape as
            ``create_deepagent(datastores=...)`` and the langchain-oci
            ``create_deep_research_agent(datastores=...)`` contract.
        datastore_top_k: Default top-k for auto-wired datastore tools when
            an entry doesn't set its own ``top_k``. Default 5.

    Returns:
        A compiled ``locus.StateGraph``.
    """
    from locus.deepagent.factory import wire_datastores  # noqa: PLC0415
    from locus.multiagent.graph import END, START, StateGraph  # noqa: PLC0415

    _sum_model = summarization_model or model
    _grd_model = grounding_model or model

    # Datastore auto-wiring: identical surface to create_deepagent — the
    # execute node gets the same search_<name> tools plus a routing block
    # prepended to its system_prompt.
    ds_tools, ds_routing_block = wire_datastores(datastores, datastore_top_k)
    final_tools = [*tools, *ds_tools]
    final_system_prompt = (
        f"{ds_routing_block}\n\n---\n\n{system_prompt}" if ds_routing_block else system_prompt
    )

    graph = StateGraph()

    graph.add_node(
        "execute",
        make_execute_node(model, final_tools, final_system_prompt, reflexion, max_iterations),
    )
    if causal_inference:
        graph.add_node("causal_inference", make_causal_inference_node(model))
    graph.add_node("summarize", make_summarize_node(_sum_model, output_schema))
    graph.add_node("grounding_eval", make_grounding_eval_node(_grd_model))
    graph.add_node("regenerate", make_regenerate_summary_node(_sum_model, output_schema))
    graph.add_node("replan", make_replan_node())

    router = route_after_grounding(grounding_threshold, max_replans, max_regenerations)

    graph.add_edge(START, "execute")
    if causal_inference:
        graph.add_edge("execute", "causal_inference")
        graph.add_edge("causal_inference", "summarize")
    else:
        graph.add_edge("execute", "summarize")
    graph.add_edge("summarize", "grounding_eval")
    graph.add_conditional_edges(
        "grounding_eval",
        router,
        {"regenerate": "regenerate", "replan": "replan", END: END},
    )
    graph.add_edge("regenerate", "grounding_eval")
    graph.add_edge("replan", "execute")

    kwargs: dict[str, Any] = {}
    if checkpointer:
        kwargs["checkpointer"] = checkpointer

    return graph.compile(**kwargs)
