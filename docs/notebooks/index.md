---
title: Examples catalog
---

# Examples

Every example is a runnable `.py` file. Most use the bundled `MockModel` or a
local stub, need no credentials, and print simulated results. Examples that
contact a live provider say so on their page. A mock run demonstrates control
flow; it is not evidence of live-model or vendor behavior.

| Catalog status | Value |
|---|---|
| Tested SDK | `locus-sdk` {{ locus_sdk_version }} |
| Python | 3.11–3.14 |
| Default execution mode | Offline simulation unless the page says **Live** |
| Credentials | None for offline examples; named on live-example pages |
| Source format | Self-contained Python files under `examples/` |

These examples build agents with typed tools, memory, streaming, graphs, RAG,
multi-agent shapes, and the provider and gateway matrix, on everyday work in
payments, customer support, infrastructure, and privacy. Many examples put a
gate you wrote in front of a consequential action: a hook that cancels the tool
call, a graph interrupt that waits for a person, or `admit()` checking a
`ControlPolicy`. The [control-layer examples](#control-layer-and-security) also
record every decision, allowed or held, on a tamper-evident audit trail.
The tracks below work through those domains; one dedicated track applies that
gate to security operations.

<div class="notebook-filter">
  <input
    type="search"
    id="notebook-filter-input"
    class="notebook-filter__input"
    placeholder="Filter examples (e.g. oracle, oci, rag, hooks)…  press ⌘K"
    autocomplete="off"
    autocorrect="off"
    spellcheck="false" />
</div>

Run any example — each link below opens its file:

```bash
git clone https://github.com/oracle/locus.git
cd locus-sdk
python -m pip install -e .
python examples/<file>.py
```

!!! tip "New to Locus?"
    Start with [Start here](#start-here) — the agent loop, tools, memory,
    streaming, hooks, and termination — then [Multi-agent shapes](#multi-agent-shapes)
    and [Oracle Cloud and Database 26ai](#oracle-cloud-and-database-26ai). The
    five gate notebooks are under [Control layer and security](#control-layer-and-security).

The series runs 01 to 89 (81–89 live in `locus-sdk-security`), ordered by topic: start here, the multi-agent shapes, Oracle Cloud and Database 26ai, the harness, observability and serving, the control layer, real-world workflows. Every notebook runs against OCI Generative AI with an OCI profile; a few need a database or service, stated at the top of each page.

## Start here

- **01 · [OCI transports — start here](notebook_01_oci_transports.md)** — pick the right Oracle Cloud Infrastructure (OCI) Generative AI client for your model
- **02 · [Basic agent](notebook_02_basic_agent.md)** — your first agent — build LEDGER, send it a transaction, read the result
- **03 · [Agent with tools](notebook_03_agent_with_tools.md)** — giving an agent tools
- **04 · [A chat loop](notebook_04_chat_loop.md)** — a chat loop — actually talking to your agent
- **05 · [Conversation memory](notebook_05_agent_memory.md)** — conversation memory — remembering what you told it
- **06 · [Streaming events](notebook_06_agent_streaming.md)** — streaming a data-subject-request (DSAR) analysis
- **07 · [Structured output](notebook_07_structured_output.md)** — structured output — typed support schemas a help desk can consume
- **08 · [Termination conditions](notebook_08_termination.md)** — bounded resolution — deciding when the support loop stops
- **09 · [Lifecycle hooks](notebook_09_agent_hooks.md)** — audit hooks — every agent action on the record
- **10 · [Advanced hooks and guardrails](notebook_10_hooks_advanced.md)** — RELEASE GUARD — guardrails around a deploy-ops agent
- **11 · [Model providers](notebook_11_model_providers.md)** — Model providers — one cloud-ops agent, swap models with a string

## Multi-agent shapes

- **12 · [Basic graph](notebook_12_basic_graph.md)** — Data-subject-request triage pipeline as a graph
- **13 · [Conditional routing](notebook_13_conditional_routing.md)** — Severity-based cloud incident escalation routing
- **14 · [State reducers](notebook_14_state_reducers.md)** — Merging multi-check payment signals with state reducers
- **15 · [Human in the loop](notebook_15_human_in_the_loop.md)** — Human approval gates for production changes
- **16 · [Advanced graph patterns](notebook_16_advanced_patterns.md)** — A customer-support ticket pipeline as a graph
- **17 · [Resilient graphs](notebook_17_graph_advanced.md)** — Resilient cloud-provisioning workflows
- **18 · [Functional API](notebook_18_functional_api.md)** — Payment dispute triage with the functional API
- **19 · [Composition — pipelines](notebook_19_composition.md)** — Composing data-privacy agents into pipelines
- **20 · [Scatter-gather with Send](notebook_20_map_reduce_support_triage.md)** — customer-support triage at scale — scatter-gather with Send
- **21 · [Supervisor and critic loop](notebook_21_supervisor_critic_loop.md)** — PII-exposure report vs skeptical reviewer — kill unproven claims
- **22 · [Orchestrator](notebook_22_orchestrator_pattern.md)** — STEWARD — one privacy officer, many data-privacy specialists
- **23 · [Specialist agents](notebook_23_specialist_agents.md)** — RIGHTSIZER — the cloud-rightsizing specialist
- **24 · [Swarm](notebook_24_swarm_multiagent.md)** — Outage war room — mitigation, diagnostics, and comms agents in a swarm
- **25 · [Handoff](notebook_25_agent_handoff.md)** — Support tier escalation — L1 → L2 → L3 handoff with typed context
- **26 · [Debate with a judge](notebook_26_debate_with_judge.md)** — incident adjudication debate — real incident vs noise
- **27 · [Multi-agent human in the loop](notebook_27_multiagent_human_in_loop.md)** — human-approved disbursement — multi-agent HITL
- **28 · [Emergent routing](notebook_28_emergent_routing.md)** — emergent incident routing — the model picks the specialist protocol
- **29 · [Cognitive router](notebook_29_cognitive_router.md)** — Cognitive router — risk-gated routing for infra/devops requests
- **30 · [A2A protocol](notebook_30_a2a_protocol.md)** — A2A — cross-org payment-intel sharing between spec-compliant peers
- **31 · [DeepAgent](notebook_31_deepagent.md)** — DeepAgent — scoped service-fleet reliability review grounded in prior notes
- **32 · [Research workflow](notebook_32_research_workflow.md)** — Customer-support research workflow — recon, hypothesize, validate, report

## Oracle Cloud and Database 26ai

- **33 · [OCI v1 chat](notebook_33_oci_openai_chat.md)** — call any non-R-series OCI GenAI model with ``OCIChatCompletionsModel``
- **34 · [OCI Responses](notebook_34_oci_responses.md)** — reach Responses-only OCI GenAI models with ``OCIResponsesModel``
- **35 · [OCI Dedicated AI Cluster](notebook_35_oci_dac.md)** — target an Oracle Cloud Infrastructure (OCI) Dedicated AI Cluster endpoint
- **36 · [OCI tools — drive the control plane](notebook_36_oci_tools.md)** — Drive the entire OCI control plane from natural language — `use_oci` + `describe_oci`
- **37 · [Oracle 26ai store](notebook_37_oracle_store.md)** — a Locus agent with cross-thread memory on Oracle 26ai
- **38 · [Oracle 26ai RAG](notebook_38_oracle_26ai_rag.md)** — a Locus agent that retrieves over Oracle Database 26ai
- **39 · [Oracle ADB loader](notebook_39_oracle_adb_loader.md)** — a Locus agent that pulls rows from Autonomous Database
- **40 · [Oracle in-DB chunker](notebook_40_oracle_indb_chunker.md)** — a Locus agent driving an in-database chunker
- **41 · [Oracle in-DB embeddings](notebook_41_oracle_indb_embeddings.md)** — a Locus agent doing data-residency-aware RAG in-database
- **42 · [Oracle 26ai checkpointer](notebook_42_oracle_26ai_checkpointer.md)** — durable agent threads on Oracle Database 26ai
- **43 · [Oracle versioned saver](notebook_43_oracle_versioned_saver.md)** — versioned agent history on Oracle Database 26ai
- **44 · [Oracle agent memory](notebook_44_oracle_agent_memory.md)** — a Locus agent with Oracle's official agent-memory client
- **45 · [Cohere reranker on Oracle 26ai](notebook_45_cohere_reranker.md)** — retrieve-then-rerank on Oracle 26ai with Cohere Reranker V4

## The harness — tools, skills, memory, RAG, reasoning

- **46 · [MCP integration](notebook_46_mcp_integration.md)** — MCP integration — wire support tooling into an agent
- **47 · [Plugins](notebook_47_plugins.md)** — plugins — package cloud-cost triage as a reusable extension
- **48 · [Skills](notebook_48_skills.md)** — skills — a payments-ops skill library with progressive disclosure
- **49 · [Playbooks](notebook_49_playbooks.md)** — playbooks — a GDPR data-subject-access runbook the agent must follow
- **50 · [Checkpoint backends](notebook_50_checkpoint_backends.md)** — Durable support cases — checkpoint backends on object storage
- **51 · [RAG basics](notebook_51_rag_basics.md)** — RAG basics — a cloud Well-Architected knowledge base
- **52 · [RAG providers](notebook_52_rag_providers.md)** — RAG providers — a payments runbook KB on any vector store
- **53 · [RAG agents](notebook_53_rag_agents.md)** — RAG agents — ATLAS, an on-call SRE copilot over the Index
- **54 · [Reasoning patterns](notebook_54_reasoning_patterns.md)** — reasoning patterns — a missed-redaction postmortem
- **55 · [GSAR typed grounding](notebook_55_gsar_typed_grounding.md)** — GSAR grounded findings — a finding ships, or the agent abstains
- **56 · [Evaluation](notebook_56_evaluation.md)** — Evaluation — regression-test a privacy-review agent on known requests
- **57 · [Multi-modal providers](notebook_57_multimodal_providers.md)** — Multi-modal dispute triage — order pages, ledger, image, speech
- **58 · [Audio response](notebook_58_audio_response.md)** — Spoken cloud status advisory — voice output
- **59 · [Audio chat](notebook_59_audio_chat.md)** — Payments-support voice assistant — voice in, voice out

## Observability, streaming, serving

- **60 · [Observability basics](notebook_60_observability_basics.md)** — Support timeline — every step a ticket bot takes, on the record
- **61 · [Agent yield bridge](notebook_61_agent_yield_bridge.md)** — Privacy-review telemetry — the agent yield bridge
- **62 · [EventBus subscribers](notebook_62_eventbus_subscribers.md)** — Shipping agent events to a cloud telemetry platform
- **63 · [Event catalogue](notebook_63_event_catalogue.md)** — The event taxonomy as a compliance artifact
- **64 · [SSE streaming](notebook_64_sse_streaming.md)** — stream a payment review to an ops dashboard with SSE
- **65 · [Agent server](notebook_65_agent_server.md)** — Agent server — deploy an on-call incident triage copilot as an HTTP API
- **66 · [LiteLLM gateway](notebook_66_litellm_gateway.md)** — Routing cloud-ops agent traffic through the LiteLLM AI Gateway
- **67 · [LiteLLM cost guardrails](notebook_67_litellm_gateway_cost.md)** — Cost guardrails for always-on payments automation

## Control layer and security

- **68 · [Steering](notebook_68_steering.md)** — steering — a policy LLM gates a live production incident
- **69 · [Guardrails — prompt injection](notebook_69_guardrails_security.md)** — Guardrails — indirect prompt injection via untrusted tool output
- **70 · [Trace hygiene](notebook_70_guardrails_advanced.md)** — Trace hygiene — redact PII and secrets before they persist
- **71 · [Payment refund gate](notebook_71_payment_refund_gate.md)** — A refund gate that pays out small refunds and holds big ones
- **72 · [Infra deploy gate](notebook_72_infra_deploy_gate.md)** — An ops agent that ships to staging on its own and waits for a human in production
- **73 · [Support account gate](notebook_73_support_account_gate.md)** — A support agent changes a customer account — admit() holds the big ones
- **74 · [Data deletion gate](notebook_74_data_deletion_gate.md)** — A human signs off before an agent erases personal data
- **75 · [Cloud resource gate](notebook_75_cloud_resource_gate.md)** — A cloud-ops agent that can only touch production with a human's say-so
- **76 · [Incident response with an audit chain](notebook_76_ir_audit_trail.md)** — Incident response with a tamper-evident audit chain

## Real-world workflows

- **77 · [Incident response runbook](notebook_77_incident_response.md)** — SRE on-call incident runbook
- **78 · [Concession approval](notebook_78_concession_approval.md)** — Customer-support concession approval with risk-tiered gates
- **79 · [Contract review](notebook_79_contract_review.md)** — Vendor DPA review — data-subject rights, retention, transfer gaps
- **80 · [Vendor integrations](notebook_80_vendor_integrations.md)** — Live vendor integrations — PII discovery, data map, scan dispatch

## Security operations (locus-sdk-security)

- **81 · [Grounded cloud-posture agent](notebook_81_cloud_posture_agent.md)** — Grounded AWS cloud-posture agent
- **82 · [SOC playbooks](notebook_82_security_playbooks.md)** — SOC playbooks over the first-class security toolset
- **83 · [Red-team an AI agent](notebook_83_agent_red_team.md)** — agentic AI red-teaming — grounded findings or abstentions
- **84 · [Red-team a support bot](notebook_84_redteam_support_bot.md)** — red-team a customer-support chatbot (a real scenario)
- **85 · [CI security gate](notebook_85_ci_security_gate.md)** — a CI security gate for AI agents
- **86 · [Verify findings](notebook_86_verify_findings.md)** — verify findings — the SDK that prevents security hallucinations
- **87 · [SOC alert triage](notebook_87_soc_alert_triage.md)** — SOC alert triage with SIEM-grounded verdicts
- **88 · [Model and hardware fingerprinting](notebook_88_model_fingerprint.md)** — Model & hardware fingerprinting via timing side-channels
- **89 · [Investigate with SecurityContext](notebook_89_investigate_with_ctx.md)** — investigate an incident with SecurityContext — no vendor names
