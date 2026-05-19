# Notebooks

63 runnable `examples/notebook_NN_*.py` files. Every one runs end-to-end
against the bundled `MockModel` (no credentials required) and upgrades
to live **Oracle Cloud Infrastructure (OCI) Generative AI** —
or OpenAI / Anthropic / Ollama — by setting one environment variable.

<div class="notebook-filter">
  <input
    type="search"
    id="notebook-filter-input"
    class="notebook-filter__input"
    placeholder="Filter notebooks (e.g. oracle, rag, hooks)…  press ⌘K"
    autocomplete="off"
    autocorrect="off"
    spellcheck="false" />
</div>

Run any notebook directly:

```bash
git clone https://github.com/oracle-samples/locus.git
cd locus && pip install -e .
python examples/notebook_01_oci_transports.py
```

The notebooks are numbered in **suggested reading order**. Start at
01 and walk forward; each one builds on the last. If you're shipping
on OCI, the first seven are the path you came for.

## 01–05 · OCI Generative AI

The OCI inference platform end-to-end: pick a transport, point at a
cluster, on-demand reranking. **Start here if you're shipping on OCI.**

| # | Notebook | What you get |
|---|---|---|
| 01 | [OCI transports — start here][t01] | The three OCI transports side by side (V1, Responses, generic chat) |
| 02 | [OCI v1 (`OCIOpenAIModel`)][t02] | The default transport — every OCI model family in one class |
| 03 | [OCI Responses (`OCIResponsesModel`)][t03] | Opt-in stateful path with built-in tools |
| 04 | [OCI Dedicated AI Cluster (DAC)][t04] | Wiring a private endpoint OCID into Locus |
| 05 | [Cohere Reranker V4 on OCI][t05] | Retrieve-then-rerank with OCI on-demand `rerank-v4` |

## 06–07 · Oracle Database 26ai

The Oracle data layer: native `VECTOR(N, FLOAT32)` data type and the
`VECTOR_DISTANCE` SQL function let your agent ground answers and
durably checkpoint conversations directly in Oracle Autonomous Database.

| # | Notebook | What you get |
|---|---|---|
| 06 | [Oracle 26ai RAG][t06] | `OracleVectorStore` against an Autonomous Database wallet — native `VECTOR(N, FLOAT32)`, `VECTOR_DISTANCE` SQL |
| 07 | [Oracle 26ai checkpointer][t07] | `oracle_checkpointer` — resume agent conversations from ADB |

## 13–20 · Agent Foundations

The agent loop, tools, memory, streaming, hooks. Where to send a
brand-new developer.

| # | Notebook |
|---|---|
| 08 | [Basic agent][t08] |
| 09 | [Agent with tools][t09] |
| 10 | [Conversation memory][t10] |
| 11 | [Streaming events][t11] |
| 12 | [Lifecycle hooks][t12] |
| 13 | [SSE streaming][t13] |
| 14 | [Hooks (advanced)][t14] |
| 15 | [Termination conditions][t15] |

## 16–23 · Graphs & composition

`StateGraph`, conditional edges, reducers, retries, the functional API.

| # | Notebook |
|---|---|
| 16 | [Basic graph][t16] |
| 17 | [Conditional routing][t17] |
| 18 | [State reducers][t18] |
| 19 | [Human-in-the-loop][t19] |
| 20 | [Command + advanced patterns][t20] |
| 21 | [Composition (Sequential / Parallel / Loop)][t21] |
| 22 | [Graph (advanced) — retries, subgraphs][t22] |
| 23 | [Functional API (`@task`, `@entrypoint`)][t23] |

## 24–34 · Multi-agent

In-process patterns plus A2A, DeepAgent, and real-world crew workflows.

| # | Notebook | Shape |
|---|---|---|
| 24 | [Swarm][t24] | Peer-to-peer shared context |
| 25 | [Agent handoff][t25] | Sequential escalation |
| 26 | [Orchestrator pattern][t26] | Coordinator + parallel specialists |
| 27 | [Specialist agents][t27] | Named domain experts |
| 28 | [A2A protocol (cross-process)][t28] | HTTP + SSE mesh |
| 29 | [DeepAgent — research factory][t29] | Reflexion + grounding + subagents |
| 30 | [Map-reduce code review][t30] | `Send` fan-out / reduce |
| 31 | [Supervisor + critic loop][t31] | Refinement loop with cycles |
| 32 | [Adversarial debate + judge][t32] | Typed `Verdict` via `output_schema` |
| 33 | [Multi-agent + human-in-the-loop][t33] | Three HITL patterns in one file |
| 34 | [Emergent routing][t34] | Opt-in LLM-as-picker |

## 35–37 · Reasoning & structured output

Pydantic schemas, Reflexion, Grounding, Causal, GSAR.

| # | Notebook |
|---|---|
| 35 | [Structured output (Pydantic)][t35] |
| 36 | [Reasoning patterns][t36] |
| 37 | [GSAR — typed grounding][t37] |

## 38–40 · RAG

| # | Notebook |
|---|---|
| 38 | [RAG basics][t38] |
| 39 | [RAG providers (vector stores, embeddings)][t39] |
| 40 | [RAG agents][t40] |

## 41–45 · Skills, playbooks & plugins

| # | Notebook |
|---|---|
| 41 | [MCP integration][t41] |
| 42 | [Playbooks][t42] |
| 43 | [Plugins][t43] |
| 44 | [Skills][t44] |
| 45 | [Steering (LLM-as-policy hook)][t45] |

## 46–51 · Production

Guardrails, checkpointers, evaluation, provider matrix, multi-modal.

| # | Notebook |
|---|---|
| 46 | [Guardrails & security (basics)][t46] |
| 47 | [Guardrails (advanced)][t47] |
| 48 | [Checkpoint backends][t48] |
| 49 | [Evaluation][t49] |
| 50 | [Model providers][t50] |
| 51 | [Multi-modal providers (web, images, audio)][t51] |

## 52–56 · Cognitive router & observability

Cognitive router + opt-in EventBus telemetry.

| # | Notebook |
|---|---|
| 52 | [Cognitive router (PRISM)][t52] |
| 53 | [Observability basics — opt-in SSE telemetry][t53] |
| 54 | [Agent yield bridge + token usage][t54] |
| 55 | [EventBus subscriber patterns][t55] |
| 56 | [Full event catalogue tour][t56] |

## 57–61 · Real-world workflows

End-to-end use cases — incident response, contract review, audio chat.

| # | Notebook |
|---|---|
| 57 | [On-call incident response][t57] |
| 58 | [Tiered procurement approval][t58] |
| 59 | [Contract review + negotiation][t59] |
| 60 | [Voice output (TTS)][t60] |
| 61 | [Voice in → voice out (gpt-audio)][t61] |

## 62–63 · Server & full pipelines

| # | Notebook |
|---|---|
| 62 | [Agent server (FastAPI)][t62] |
| 63 | [Research workflow (full pipeline)][t63] |

[t01]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_01_oci_transports.py
[t02]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_02_oci_openai_chat.py
[t03]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_03_oci_responses.py
[t04]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_04_oci_dac.py
[t05]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_05_cohere_reranker.py
[t06]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_06_oracle_26ai_rag.py
[t07]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_07_oracle_26ai_checkpointer.py
[t08]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_14_basic_agent.py
[t09]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_15_agent_with_tools.py
[t10]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_16_agent_memory.py
[t11]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_17_agent_streaming.py
[t12]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_18_agent_hooks.py
[t13]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_19_sse_streaming.py
[t14]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_20_hooks_advanced.py
[t15]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_21_termination.py
[t16]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_22_basic_graph.py
[t17]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_23_conditional_routing.py
[t18]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_24_state_reducers.py
[t19]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_25_human_in_the_loop.py
[t20]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_26_advanced_patterns.py
[t21]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_27_composition.py
[t22]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_28_graph_advanced.py
[t23]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_29_functional_api.py
[t24]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_30_swarm_multiagent.py
[t25]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_31_agent_handoff.py
[t26]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_32_orchestrator_pattern.py
[t27]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_33_specialist_agents.py
[t28]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_34_a2a_protocol.py
[t29]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_35_deepagent.py
[t30]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_36_map_reduce_code_review.py
[t31]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_37_supervisor_critic_loop.py
[t32]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_38_debate_with_judge.py
[t33]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_39_multiagent_human_in_loop.py
[t34]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_40_emergent_routing.py
[t35]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_41_structured_output.py
[t36]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_42_reasoning_patterns.py
[t37]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_43_gsar_typed_grounding.py
[t38]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_44_rag_basics.py
[t39]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_45_rag_providers.py
[t40]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_46_rag_agents.py
[t41]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_47_mcp_integration.py
[t42]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_48_playbooks.py
[t43]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_49_plugins.py
[t44]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_50_skills.py
[t45]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_51_steering.py
[t46]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_52_guardrails_security.py
[t47]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_53_guardrails_advanced.py
[t48]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_54_checkpoint_backends.py
[t49]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_55_evaluation.py
[t50]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_56_model_providers.py
[t51]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_57_multimodal_providers.py
[t52]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_58_cognitive_router.py
[t53]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_59_observability_basics.py
[t54]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_60_agent_yield_bridge.py
[t55]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_61_eventbus_subscribers.py
[t56]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_62_event_catalogue.py
[t57]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_63_incident_response.py
[t58]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_64_procurement_approval.py
[t59]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_65_contract_review.py
[t60]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_66_audio_response.py
[t61]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_67_audio_chat.py
[t62]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_68_agent_server.py
[t63]: https://github.com/oracle-samples/locus/blob/main/examples/notebook_69_research_workflow.py
