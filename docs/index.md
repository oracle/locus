---
hide:
  - navigation
  - toc
description: "Oracle Generative AI Agentic Harness. Locus is a Python agent harness with eight multi-agent shapes, native OCI Generative AI (including Dedicated AI Clusters) and Oracle Database 26ai, and the runtime around the model: tools, memory, RAG, observability, guardrails, admission gate."
---

<div class="locus-hero" markdown>
<div class="locus-hero__copy" markdown>

# Multi-agent workflows built for <span class="accent">production.</span>

One `Agent` class, eight ways to combine agents, Oracle Cloud underneath. Around the model, the agentic harness; around the code an agent writes, an OCI GenAI Sandbox in your tenancy.

- **The agentic harness.** Tools, memory, RAG, observability, guardrails. Durable threads, typed events, claims checked against evidence. `admit()` in front of every consequential action: allow, hold for a person, deny, on a hash-chained audit trail.
- **The coding harness, sandboxed.** Read, write, edit, search, a shell, on the host or inside an OCI GenAI Sandbox: isolated Linux, created in seconds, expiring on its own. Every call gated, every command recorded.
- **Eight multi-agent shapes.** Pipelines, fan-out, orchestrator, swarm, handoff, graphs, debate, A2A, each running the same `Agent`. A cognitive router picks one from the task.
- **Oracle Cloud, native.** OCI Generative AI on day 0, Dedicated AI Clusters, Oracle Database 26ai for vectors, threads, memory.

[Quickstart](how-to/quickstart.md){ .md-button .md-button--primary }
[Multi-agent shapes](concepts/multi-agent.md){ .md-button }
[GitHub](https://github.com/oracle/locus){ .md-button }

```bash
pip install "locus-sdk[oci,oracledb]"
```

Built inside Oracle · Open source (<span class="locus-nowrap">UPL-1.0</span>)

</div>

<div class="locus-hero__code" markdown>

```python
from locus.agent import Agent
from locus.agent.composition import (
    SequentialPipeline,
)

model = "oci:openai.gpt-5.5"  # OCI, day 0

diagnose = Agent(
    model=model,
    tools=[query_metrics, search_logs],
    system_prompt="Find what changed.",
)
mitigate = Agent(
    model=model,
    tools=[scale_up, rollback],  # admit()
    system_prompt="Fix it, smallest step.",
)
brief = Agent(
    model=model,
    system_prompt="Write the update.",
)

war_room = SequentialPipeline(
    agents=[diagnose, mitigate, brief],
)
result = await war_room.run(
    "checkout p99 hit 4.2s: fix and brief"
)
print(result.final_output)
```

</div>
</div>

## Your first agent

{{ locus_model_tabs }}

```python
from locus.agent import Agent

agent = Agent(model="{{ locus_model }}")
print(agent.run_sync("What is the capital of France?").text)
# → Paris
```

Pick a provider above; the call does not change. With an OCI profile in
`~/.oci/config`, every notebook runs against OCI Generative AI with no further
setup.

[The Agent →](concepts/agent.md) ·
[Notebook 02 — Basic agent →](notebooks/notebook_02_basic_agent.md) ·
[Notebook 03 — Agent with tools →](notebooks/notebook_03_agent_with_tools.md)
{ .locus-links }

## What are you building?

Each tab shows one workload in a few lines and links to the notebook that
runs the full example.

=== "Support bot"

    A support agent credits customer accounts to resolve tickets; `admit()`
    runs before each write, so a credit over the policy's limit is held for a
    human and every decision, allowed or held, lands on the audit trail,
    whatever the prompt asked for.

    ```python
    from locus.agent import Agent
    from locus.tools import tool
    from locus.control import Action, AuditTrail, ControlPolicy, gate_tool

    @tool
    async def apply_credit(account: str, usd: float) -> str:
        """Credit a customer account."""
        return await billing.credit(account, usd)

    policy = ControlPolicy(require_human_over_usd=50, require_verification_score=0)
    gated = gate_tool(  # no finding to verify here; the amount decides
        apply_credit, policy=policy, trail=AuditTrail(),
        action=lambda name, a: Action(name, a["account"], cost_usd=a["usd"]))
    agent = Agent(model="oci:openai.gpt-5.5", tools=[gated])
    print(agent.run_sync("SUP-7782: apply a $500 retention credit.").text)
    ```

    [Notebook 73 — Support account gate →](notebooks/notebook_73_support_account_gate.md)

=== "Incident response"

    An on-call runbook as a `StateGraph`: triage grades the page, three
    investigators fan out in parallel, a synthesizer names the likely cause,
    and on a critical grade the run halts at `interrupt()`, so the mitigation
    agent is not reached until a human resumes it with `Command(resume=...)`.

    ```python
    from locus.agent import Agent
    from locus.core import Command, interrupt
    from locus.multiagent.graph import StateGraph

    agent = Agent(model="oci:openai.gpt-5.5", system_prompt="Propose one mitigation.")

    async def page_human(state):  # critical only; a human answers yes or no
        return {"approved": interrupt(state["hypothesis"]) == "yes"}

    async def mitigate(state):
        return {"mitigation": (await agent.arun(state["hypothesis"])).message}

    def severity_gate(state):  # info and warn mitigate on their own
        return "page_human" if state["severity"] == "critical" else "auto_mitigate"

    g = StateGraph(name="incident-runbook")  # earlier nodes as in the notebook
    g.add_node("page_human", page_human).add_node("auto_mitigate", mitigate)
    g.add_edge("page_human", "auto_mitigate")
    g.add_conditional_edges("synthesize", severity_gate)
    run = await g.execute({"symptom": "checkout p99 latency 8x, 12% 503s"})
    if run.interrupt:  # paused at page_human until a human answers
        run = await g.execute(Command(resume="yes", update=run.final_state))
    ```

    [Notebook 77 — Incident response runbook →](notebooks/notebook_77_incident_response.md)

=== "Contract review"

    Reviewers read a vendor data-processing agreement, a gate rule pauses for
    human counsel while any blocker is open and re-reviews the redline, and
    the sign-off is a `ContractDecision`: Locus validates it against the
    schema, feeds validation errors back to the model for repair, and
    otherwise returns `parsed=None` with the parse error, never an
    unvalidated record.

    ```python
    from pydantic import BaseModel, Field
    from locus.agent import Agent

    class ContractDecision(BaseModel):
        open_blockers: list[str]
        decision: str = Field(description="signed | rejected | abandoned")

    def negotiation_gate(state: dict) -> str:  # "negotiate" pauses for counsel
        return "negotiate" if state["open_blockers"] else "sign_off"

    legal = Agent(model="oci:openai.gpt-5.5", system_prompt="List blockers.")
    dpo = Agent(model="oci:openai.gpt-5.5", output_schema=ContractDecision)
    findings = (await legal.arun(f"DPA clauses:\n{clauses}")).message
    decision = (await dpo.arun(f"Findings:\n{findings}")).parsed
    ```

    [Notebook 79 — Contract review →](notebooks/notebook_79_contract_review.md)

=== "SOC alert triage"

    A `locus-sdk-security` analyst works each alert with the security
    information and event management (SIEM), endpoint detection and response
    (EDR) and threat-intelligence tools and proposes a verdict; `ground_report`
    returns it as a typed `Evidence`, carrying the references it cited, only
    when that evidence clears the Grounding-Stratified Adaptive Replanning
    (GSAR) threshold, and as an `Abstention` otherwise, so an alert with
    nothing behind it is not filed as a finding.

    ```python
    from locus_security import (
        PostureReport, SecurityControls, create_soc_analyst, ground_report,
        is_finding,
    )

    controls = SecurityControls(siem=True, edr=True, threat_intel=True)
    analyst = create_soc_analyst(model="oci:openai.gpt-5.5", controls=controls)
    result = await analyst.arun("Triage the open alerts from the last hour.")
    for verdict in ground_report(result.parsed_as(PostureReport), controls):
        if is_finding(verdict):  # cited evidence cleared the GSAR threshold
            print(verdict.severity, verdict.title, verdict.evidence_refs)
        else:  # below the threshold: withheld, nothing is filed
            print("withheld:", verdict.candidate_title, verdict.reason)
    ```

    [Notebook 87 — SOC alert triage →](notebooks/notebook_87_soc_alert_triage.md)

## One `Agent`, every shape

Every shape runs the same `Agent`, emits the same events, and checkpoints to
the same backends. Pick the shape; the agents do not change.

<div class="grid cards locus-feature-cards" markdown>

- {{ locus_glyph pipelines }} **[Pipelines](concepts/multi-agent/composition.md)**

    ---
    `SequentialPipeline`, `ParallelPipeline`, `LoopAgent`: chain, fan out,
    or revise until a reviewer clears. Three lines each.

- {{ locus_glyph orchestrator }} **[Orchestrator and specialists](concepts/multi-agent/orchestrator.md)**

    ---
    One coordinator delegates to typed specialists with their own tools,
    and merges what comes back.

- {{ locus_glyph swarm }} **[Swarm](concepts/multi-agent/swarm.md)**

    ---
    Peers that hand work to each other on their own judgement, with a
    shared scratchpad and a hard stop.

- {{ locus_glyph handoff }} **[Handoff](concepts/multi-agent/handoff.md)**

    ---
    Tiered escalation with typed context: L1 to L2 to L3, nothing lost
    in between.

- {{ locus_glyph graph }} **[StateGraph and functional API](concepts/multi-agent/graph.md)**

    ---
    Nodes, edges, reducers, `Send` fan-out, interrupts for a human, and
    checkpointed state.

- {{ locus_glyph debate }} **[Debate and critic loops](notebooks/notebook_26_debate_with_judge.md)**

    ---
    Two debaters and a judge, or a skeptical reviewer that rejects unproven
    claims before they ship.

- {{ locus_glyph router }} **[Cognitive router](concepts/multi-agent.md#the-eight-shapes)**

    ---
    Describe the task; the router extracts a typed `GoalFrame`, picks one
    of eight protocols and compiles it onto real agents.

- {{ locus_glyph a2a }} **[Agent-to-agent (A2A) protocol](concepts/multi-agent/a2a.md)**

    ---
    Cross-process agents as spec-compliant peers: expose yours, call
    theirs, same tool contract.

- {{ locus_glyph deepagent }} **[DeepAgent and research workflows](concepts/deepagent.md)**

    ---
    Plan, explore, validate, report, with per-turn grounding and a
    post-hoc grounding evaluation.

</div>

```python
result = await router.dispatch("Diagnose the checkout API slowdown.")
print(result.protocol_id)   # "specialist_fanout"
print(result.text)          # findings from 3 parallel probes
```

[Multi-agent overview →](concepts/multi-agent.md) ·
[Notebook 19 — Composition →](notebooks/notebook_19_composition.md) ·
[Notebook 22 — Orchestrator →](notebooks/notebook_22_orchestrator_pattern.md) ·
[Notebook 29 — Cognitive router →](notebooks/notebook_29_cognitive_router.md)
{ .locus-links }

## Oracle Cloud, native

The OCI and Oracle Database primitives are first-class classes, not adapters:
same `Agent`, same events, same admission gate.

<div class="grid cards locus-feature-cards" markdown>

- {{ locus_glyph oci-genai }} **[OCI Generative AI](concepts/providers/oci.md)**

    ---
    `model="oci:…"` picks the transport from the model id: V1 for OpenAI,
    Meta, xAI, Google; SDK for Cohere; the Responses transport is opt-in
    through `OCIResponsesModel`.

- {{ locus_glyph dac }} **[Dedicated AI Clusters](how-to/oci-dac.md)**

    ---
    `oci:ocid1.generativeaiendpoint…` as the model string targets your own
    Dedicated AI Cluster endpoint, with the compartment and auth resolved
    from your profile.

- {{ locus_glyph oci-tools }} **[OCI tools](notebooks/notebook_36_oci_tools.md)**

    ---
    `use_oci()` and `describe_oci()` put the OCI control plane behind
    tools, so an agent can list, inspect and act on your tenancy.

- {{ locus_glyph oracle-rag }} **[Oracle Database 26ai RAG](notebooks/notebook_38_oracle_26ai_rag.md)**

    ---
    `OracleVectorStore` on native `VECTOR` columns, hybrid search, plus
    in-database chunking and embeddings so text never leaves the database.

- {{ locus_glyph oracle-checkpoints }} **[Oracle Database 26ai checkpoints and history](notebooks/notebook_42_oracle_26ai_checkpointer.md)**

    ---
    `oracle_checkpointer` for durable threads, `OracleCheckpointSaver` for
    versioned history and time travel, OCI Object Storage as the alternative.

- {{ locus_glyph oracle-memory }} **[Oracle Database 26ai long-term memory](notebooks/notebook_44_oracle_agent_memory.md)**

    ---
    `OracleStore` for cross-thread memory and `OracleAgentMemoryManager`
    over Oracle's official agent-memory client.

- {{ locus_glyph oci-sandbox }} **[OCI GenAI Sandboxes](harness/oci-sandbox.md)**

    ---
    `OCISandboxBackend` runs the coding harness inside an isolated, expiring
    Linux workspace in your tenancy: the model's shell commands and file
    edits never touch the host, and every call still clears the gate.

</div>

OCI Generative AI first: `oci:` reaches the whole catalog (OpenAI, Google, Meta,
xAI, Cohere) and your Dedicated AI Clusters, so a new model is a new string, not
new code. Everything else that speaks the OpenAI API plugs in behind a prefix,
hosted or self-hosted.

<p class="locus-chips__vendor">One model string, any provider</p>
<div class="locus-chips locus-chips--prefixes"><span>oci:</span><span>openai:</span><span>anthropic:</span><span>gemini:</span><span>openrouter:</span><span>together:</span><span>runpod:</span><span>groq:</span><span>mistral:</span><span>xai:</span><span>deepseek:</span><span>fireworks:</span><span>cerebras:</span><span>perplexity:</span><span>nvidia:</span><span>litellm:</span><span>ollama:</span><span>vllm:</span><span>llamacpp:</span><span>lmstudio:</span><span>openai-compatible:</span></div>

```python
from locus.agent import Agent
from locus.rag import OCIEmbeddings, OracleVectorStore, RAGRetriever

Agent(model="oci:openai.gpt-5.5")                    # V1 transport
Agent(model="oci:cohere.command-a-03-2025")          # SDK transport, auto-routed
Agent(model="oci:ocid1.generativeaiendpoint.oc1...") # your Dedicated AI Cluster

retriever = RAGRetriever(                            # Oracle Database 26ai
    embedder=OCIEmbeddings(model_id="cohere.embed-v4.0"),
    store=OracleVectorStore(dsn="mydb_low", user="locus_app", password="…",
                            wallet_location="~/.oci/wallets/mydb", dimension=1536),
)
agent = Agent(model="oci:openai.gpt-5.5", tools=[retriever.as_tool()])
```

[Oracle Database 26ai, every primitive →](concepts/oracle-26ai.md) ·
[Which OCI models work →](how-to/oci-models.md#model-support-on-oci) ·
[Configure any provider →](how-to/configure-providers.md) ·
[Dedicated AI Clusters →](how-to/oci-dac.md) ·
[Oracle Database 26ai notebooks →](notebooks/index.md#oracle-cloud-and-database-26ai)
{ .locus-links }

## The harness

<div class="grid cards locus-feature-cards" markdown>

- {{ locus_glyph coding-harness }} **[The coding harness](harness.md)**

    ---
    Read, write, edit, grep and a shell over one workspace protocol, run on
    the host, in an [OCI GenAI Sandbox](harness/oci-sandbox.md) or in
    [NVIDIA OpenShell](harness/openshell.md); one `wrap` gates every call.

- {{ locus_glyph tools-mcp }} **[Tools and MCP](concepts/tools.md)**

    ---
    `@tool` on any function, typed arguments coerced from whatever the
    model sends, idempotent execution, Model Context Protocol (MCP) servers
    in or out.

- {{ locus_glyph memory-rag }} **[Memory and RAG](concepts/memory.md)**

    ---
    Four layers: working memory, the thread (checkpoint backends from local
    files to Oracle Database 26ai),
    long-term memory across threads, and knowledge through RAG (six vector
    stores, rerankers, `retriever.as_tool()`). All four on Oracle Database 26ai.

- {{ locus_glyph control }} **[Grounding, guardrails, admission gate](concepts/control-layer.md)**

    ---
    Grounding-Stratified Adaptive Replanning (GSAR) checks every claim
    against evidence; guardrails and steering catch the rest; `admit()`
    holds or denies consequential actions.

- {{ locus_glyph observability }} **[Observability and streaming](concepts/observability.md)**

    ---
    One `run_context()` streams typed events from every layer; Server-Sent
    Events (SSE) and token streaming for the UI.

- {{ locus_glyph server }} **[Server and deploy](concepts/server.md)**

    ---
    `locus serve` puts an agent behind FastAPI with bearer auth and thread
    persistence; a runner image ships for OCI Kubernetes Engine (OKE).

- {{ locus_glyph skills-eval }} **[Skills, playbooks, evaluation](concepts/skills.md)**

    ---
    Progressive-disclosure skills, enforced playbooks, and regression
    suites with `EvalRunner` so behaviour does not drift.

</div>

A coding agent in an OCI GenAI Sandbox is one backend away from the same
harness that runs on your laptop:

```python
from locus.agent import Agent
from locus.control import ControlPolicy, gate_tool
from locus.harness import OCISandboxBackend, build_harness

backend = OCISandboxBackend.create("ocid1.generativeaiproject.oc1...", expiration="PT1H")
policy = ControlPolicy(require_human_for=frozenset({"workspace.exec"}))
harness = build_harness(
    backend, wrap=lambda tool, spec: gate_tool(tool, policy=policy, action=spec)
)
agent = Agent(
    model="oci:openai.gpt-5.5", tools=harness.tools, system_prompt=harness.prompt_fragment
)
print(agent.run_sync("Add pytest tests for calc/ and run them.").message)
backend.close()   # the sandbox is stopped and deleted
```

[The coding harness →](harness.md) ·
[OCI GenAI Sandboxes →](harness/oci-sandbox.md) ·
[NVIDIA OpenShell →](harness/openshell.md)

## Control, when you need it

The admission gate is one call around the agent you already have, or built in:

```python
from locus.control import governed_agent

agent = governed_agent(model="oci:openai.gpt-5.5", tools=[...])
print(agent.run_sync("Refund order 8812 if the return was received.").text)
assert agent.audit_trail.verify()   # the chain is intact
```

```bash
pip install locus-sdk && python -m locus.rogue
# 🛡️  wipe_database → deny: labels ['irreversible'] are denied by policy
# FINAL SCORE · dangerous attempts: 1 · BREACHES: 0 · audit chain intact: ✓
```

[How the admission gate works →](concepts/control-layer.md) ·
[First controlled action →](how-to/first-controlled-action.md) ·
[Notebook 71 — Payment refund gate →](notebooks/notebook_71_payment_refund_gate.md)
{ .locus-links }

## Where next

[All notebooks →](notebooks/index.md) ·
[Capabilities →](capabilities.md) ·
[API reference →](api/index.md) ·
[Deploy →](how-to/deploy.md)
{ .locus-links }
