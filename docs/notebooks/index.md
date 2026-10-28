---
title: Examples
---

# Examples

Every example is a runnable `.py` file. Examples that call a model pick it
through `examples/config.py`: OCI Generative AI when `~/.oci/config` has a
usable profile, otherwise the bundled mock model, so every file also runs
offline with no credentials. Examples that need a service beyond the model, such
as a database or a vendor API, say so at the top of their page. A mock run
demonstrates control flow; it is not evidence of live-model or vendor behavior.

<dl class="locus-facts" markdown="block">
<div markdown="block"><dt>Tested SDK</dt><dd markdown="span">`locus-sdk` {{ locus_sdk_version }}</dd></div>
<div markdown="block"><dt>Python</dt><dd>3.11–3.14</dd></div>
<div markdown="block"><dt>Model</dt><dd>OCI Generative AI with an OCI profile; the bundled mock model without one</dd></div>
<div markdown="block"><dt>Credentials</dt><dd>An OCI profile for live model calls; none for the offline mock; service credentials named on the page</dd></div>
<div markdown="block"><dt>Source format</dt><dd markdown="span">Self-contained Python files under `examples/`</dd></div>
</dl>

These examples build agents with typed tools, memory, streaming, graphs, RAG,
multi-agent shapes, and the provider and gateway matrix, on everyday work in
payments, customer support, infrastructure, and privacy. Many examples put a
gate you wrote in front of a consequential action: a hook that cancels the tool
call, a graph interrupt that waits for a person, or `admit()` checking a
`ControlPolicy`. The [control-layer examples](#control-layer-and-security) also
record every decision, allowed or held, on a tamper-evident audit trail.
The tracks below work through those domains; one dedicated track applies that
gate to security operations.

Run any example:

```bash
git clone https://github.com/oracle/locus.git
cd locus
python -m pip install -e ".[oci,oracledb]"
python examples/<file>.py
```

!!! tip "New to Locus?"
    Begin with [Start here](#start-here) (the agent loop, tools, memory,
    streaming, hooks, and termination), then [Multi-agent shapes](#multi-agent-shapes)
    and [Oracle Cloud and Database 26ai](#oracle-cloud-and-database-26ai). The
    five gate examples are under [Control layer and security](#control-layer-and-security).

The series runs 01 to 89, ordered by topic: start here, the multi-agent shapes, Oracle Cloud and Database 26ai, the harness, observability and serving, the control layer, real-world workflows, and security operations. Examples 81–89 ship in the `locus-sdk-security` package: install it from the clone with `python -m pip install -e packages/locus-sdk-security` and run its files from `packages/locus-sdk-security/examples/`. Each entry below links to the example's page, which carries the source and names any database or service the example needs beyond the model.

<div class="notebook-filter">
  <input
    type="search"
    id="notebook-filter-input"
    class="notebook-filter__input"
    aria-label="Filter examples"
    placeholder="Filter examples (e.g. oracle, oci, rag, hooks)…"
    autocomplete="off"
    autocorrect="off"
    spellcheck="false" />
</div>

## Start here

- **01 · [OCI transports — start here](notebook_01_oci_transports.md)** — pick the right Oracle Cloud Infrastructure (OCI) Generative AI client for your model
- **02 · [Basic agent](notebook_02_basic_agent.md)** — your first agent — build LEDGER, send it a transaction, read the result
- **03 · [Agent with tools](notebook_03_agent_with_tools.md)** — giving an agent tools
- **04 · [A chat loop](notebook_04_chat_loop.md)** — an interactive session with the agent
- **05 · [Conversation memory](notebook_05_agent_memory.md)** — remembering what you told the agent
- **06 · [Streaming events](notebook_06_agent_streaming.md)** — streaming a data-subject-request (DSAR) analysis
- **07 · [Structured output](notebook_07_structured_output.md)** — typed support schemas a help desk can consume
- **08 · [Termination conditions](notebook_08_termination.md)** — bounded resolution — deciding when the support loop stops
- **09 · [Lifecycle hooks](notebook_09_agent_hooks.md)** — audit hooks — every agent action on the record
- **10 · [Advanced hooks and guardrails](notebook_10_hooks_advanced.md)** — RELEASE GUARD — guardrails around a deploy-ops agent
- **11 · [Model providers](notebook_11_model_providers.md)** — one cloud-ops agent, swap models with a string

## Multi-agent shapes

- **12 · [Basic graph](notebook_12_basic_graph.md)** — data-subject-request triage pipeline as a graph
- **13 · [Conditional routing](notebook_13_conditional_routing.md)** — severity-based cloud incident escalation routing
- **14 · [State reducers](notebook_14_state_reducers.md)** — merging multi-check payment signals with state reducers
- **15 · [Human in the loop](notebook_15_human_in_the_loop.md)** — human approval gates for production changes
- **16 · [Advanced graph patterns](notebook_16_advanced_patterns.md)** — a customer-support ticket pipeline as a graph
- **17 · [Resilient graphs](notebook_17_graph_advanced.md)** — resilient cloud-provisioning workflows
- **18 · [Functional API](notebook_18_functional_api.md)** — payment dispute triage with the functional API
- **19 · [Composition — pipelines](notebook_19_composition.md)** — composing data-privacy agents into pipelines
- **20 · [Scatter-gather with Send](notebook_20_map_reduce_support_triage.md)** — customer-support triage at scale
- **21 · [Supervisor and critic loop](notebook_21_supervisor_critic_loop.md)** — a PII-exposure report against a skeptical reviewer that rejects unsupported claims
- **22 · [Orchestrator](notebook_22_orchestrator_pattern.md)** — STEWARD — one privacy officer, many data-privacy specialists
- **23 · [Specialist agents](notebook_23_specialist_agents.md)** — RIGHTSIZER — the cloud-rightsizing specialist
- **24 · [Swarm](notebook_24_swarm_multiagent.md)** — outage war room — mitigation, diagnostics, and comms agents in a swarm
- **25 · [Handoff](notebook_25_agent_handoff.md)** — support tier escalation — L1 → L2 → L3 handoff with typed context
- **26 · [Debate with a judge](notebook_26_debate_with_judge.md)** — incident adjudication debate — real incident vs noise
- **27 · [Multi-agent human in the loop](notebook_27_multiagent_human_in_loop.md)** — human-approved disbursement in a multi-agent payments fleet
- **28 · [Emergent routing](notebook_28_emergent_routing.md)** — emergent incident routing — the model picks the specialist protocol
- **29 · [Cognitive router](notebook_29_cognitive_router.md)** — risk-gated routing for infra/devops requests
- **30 · [A2A protocol](notebook_30_a2a_protocol.md)** — cross-org payment-intel sharing between spec-compliant peers
- **31 · [DeepAgent](notebook_31_deepagent.md)** — scoped service-fleet reliability review grounded in prior notes
- **32 · [Research workflow](notebook_32_research_workflow.md)** — customer-support research workflow — recon, hypothesize, validate, report

## Oracle Cloud and Database 26ai

- **33 · [OCI v1 chat](notebook_33_oci_openai_chat.md)** — call any non-R-series OCI GenAI model with ``OCIChatCompletionsModel``
- **34 · [OCI Responses](notebook_34_oci_responses.md)** — reach Responses-only OCI GenAI models with ``OCIResponsesModel``
- **35 · [OCI Dedicated AI Cluster](notebook_35_oci_dac.md)** — target an Oracle Cloud Infrastructure (OCI) Dedicated AI Cluster endpoint
- **36 · [OCI tools — drive the control plane](notebook_36_oci_tools.md)** — drive the OCI control plane from natural language with `use_oci` and `describe_oci`
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

- **46 · [MCP integration](notebook_46_mcp_integration.md)** — wire support tooling into an agent
- **47 · [Plugins](notebook_47_plugins.md)** — package cloud-cost triage as a reusable extension
- **48 · [Skills](notebook_48_skills.md)** — a payments-ops skill library with progressive disclosure
- **49 · [Playbooks](notebook_49_playbooks.md)** — a GDPR data-subject-access runbook the agent must follow
- **50 · [Checkpoint backends](notebook_50_checkpoint_backends.md)** — durable support cases on object storage
- **51 · [RAG basics](notebook_51_rag_basics.md)** — a cloud Well-Architected knowledge base
- **52 · [RAG providers](notebook_52_rag_providers.md)** — a payments runbook KB on any vector store
- **53 · [RAG agents](notebook_53_rag_agents.md)** — ATLAS, an on-call SRE copilot over the Index
- **54 · [Reasoning patterns](notebook_54_reasoning_patterns.md)** — a missed-redaction postmortem
- **55 · [GSAR typed grounding](notebook_55_gsar_typed_grounding.md)** — a finding ships only when grounded, or the agent abstains
- **56 · [Evaluation](notebook_56_evaluation.md)** — regression-test a privacy-review agent on known requests
- **57 · [Multi-modal providers](notebook_57_multimodal_providers.md)** — multi-modal dispute triage — order pages, ledger, image, speech
- **58 · [Audio response](notebook_58_audio_response.md)** — spoken cloud status advisory — voice output
- **59 · [Audio chat](notebook_59_audio_chat.md)** — payments-support voice assistant — voice in, voice out

## Observability, streaming, serving

- **60 · [Observability basics](notebook_60_observability_basics.md)** — support timeline — every step a ticket bot takes, on the record
- **61 · [Agent yield bridge](notebook_61_agent_yield_bridge.md)** — privacy-review telemetry — yielded agent events republished on the bus
- **62 · [EventBus subscribers](notebook_62_eventbus_subscribers.md)** — shipping agent events to a cloud telemetry platform
- **63 · [Event catalogue](notebook_63_event_catalogue.md)** — the event taxonomy as a compliance artifact
- **64 · [SSE streaming](notebook_64_sse_streaming.md)** — stream a payment review to an ops dashboard with SSE
- **65 · [Agent server](notebook_65_agent_server.md)** — deploy an on-call incident triage copilot as an HTTP API
- **66 · [LiteLLM gateway](notebook_66_litellm_gateway.md)** — routing cloud-ops agent traffic through the LiteLLM AI Gateway
- **67 · [LiteLLM cost guardrails](notebook_67_litellm_gateway_cost.md)** — cost guardrails for always-on payments automation

## Control layer and security

- **68 · [Steering](notebook_68_steering.md)** — a policy LLM gates a live production incident
- **69 · [Guardrails — prompt injection](notebook_69_guardrails_security.md)** — indirect prompt injection via untrusted tool output
- **70 · [Trace hygiene](notebook_70_guardrails_advanced.md)** — redact PII and secrets before they persist
- **71 · [Payment refund gate](notebook_71_payment_refund_gate.md)** — a refund gate that pays small refunds automatically and holds large ones for a human approver
- **72 · [Infra deploy gate](notebook_72_infra_deploy_gate.md)** — an ops agent that ships to staging on its own and waits for a human in production
- **73 · [Support account gate](notebook_73_support_account_gate.md)** — a support agent changes customer accounts; `admit()` holds high-impact changes for a human
- **74 · [Data deletion gate](notebook_74_data_deletion_gate.md)** — a human signs off before an agent erases personal data
- **75 · [Cloud resource gate](notebook_75_cloud_resource_gate.md)** — a cloud-ops agent that touches production only with human approval
- **76 · [Incident response with an audit chain](notebook_76_ir_audit_trail.md)** — an SRE runbook whose every tool call lands on a tamper-evident `AuditTrail`

## Real-world workflows

- **77 · [Incident response runbook](notebook_77_incident_response.md)** — SRE on-call incident runbook
- **78 · [Concession approval](notebook_78_concession_approval.md)** — customer-support concession approval with risk-tiered gates
- **79 · [Contract review](notebook_79_contract_review.md)** — vendor DPA review — data-subject rights, retention, transfer gaps
- **80 · [Vendor integrations](notebook_80_vendor_integrations.md)** — live vendor integrations — PII discovery, data map, scan dispatch

## Security operations (locus-sdk-security)

- **81 · [Grounded cloud-posture agent](notebook_81_cloud_posture_agent.md)** — a read-only AWS posture audit that grounds every finding in observed API facts
- **82 · [SOC playbooks](notebook_82_security_playbooks.md)** — bundled incident-response playbooks run over the security toolset, step-gated by `PlaybookEnforcer`
- **83 · [Red-team an AI agent](notebook_83_agent_red_team.md)** — the OWASP-ASI / MITRE-ATLAS suite against a vulnerable and a hardened bot — grounded findings or abstentions
- **84 · [Red-team a support bot](notebook_84_redteam_support_bot.md)** — red-team a customer-support chatbot for prompt injection and data leakage
- **85 · [CI security gate](notebook_85_ci_security_gate.md)** — a regression gate that fails the build when an agent's guardrail coverage drops
- **86 · [Verify findings](notebook_86_verify_findings.md)** — an independent skeptic challenges each finding before anyone acts on it
- **87 · [SOC alert triage](notebook_87_soc_alert_triage.md)** — verdicts on SIEM alerts that ship only with corroborating evidence, otherwise an abstention
- **88 · [Model and hardware fingerprinting](notebook_88_model_fingerprint.md)** — verify a vendor's claimed model and hardware through timing side-channels
- **89 · [Investigate with SecurityContext](notebook_89_investigate_with_ctx.md)** — a vendor-neutral investigation that reasons in domains (logs, identity, threat intel, endpoint) rather than products
