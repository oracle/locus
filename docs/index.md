---
hide:
  - navigation
  - toc
description: "Oracle Generative AI Agentic Harness. Locus is a Python agent harness with eight multi-agent shapes, native OCI Generative AI (including Dedicated AI Clusters) and Oracle Database 26ai, and the runtime around the model: tools, memory, RAG, observability, guardrails, admission gate."
---

<div class="locus-hero" markdown>
<div class="locus-hero__copy" markdown>

# Multi-agent workflows built for <span class="accent">production.</span>

One `Agent` class, eight ways to combine agents, Oracle Cloud underneath, and the runtime around the model: tools, memory, RAG, observability, guardrails, an admission gate.

- **Eight multi-agent shapes.** Pipelines, fan-out, orchestrator, swarm, handoff, graphs, debate, A2A. A cognitive router picks one from the task.
- **Oracle Cloud, native.** OCI Generative AI on day 0, Dedicated AI Clusters, and Oracle Database 26ai for vectors, threads and memory.
- **Production from the first line.** Durable threads, typed events, grounded answers, and `admit()` in front of every consequential action.

[Quickstart](how-to/quickstart.md){ .md-button .md-button--primary }
[Multi-agent shapes](concepts/multi-agent.md){ .md-button }
[GitHub](https://github.com/oracle/locus){ .md-button }

```bash
pip install "locus-sdk[oci,oracledb]"
```

Built inside Oracle · Used in production · Open source (UPL-1.0)

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

```python
from locus.agent import Agent

agent = Agent(model="oci:openai.gpt-5.5")
print(agent.run_sync("What is the capital of France?").text)
# → Paris
```

Swap `oci:openai.gpt-5.5` for `openai:gpt-4o` or `anthropic:claude-sonnet-5-5`;
the call stays the same. With an OCI profile in `~/.oci/config`, every notebook
runs against OCI Generative AI with no further setup.

[The Agent →](concepts/agent.md) ·
[Notebook 02 — Basic agent →](notebooks/notebook_02_basic_agent.md) ·
[Notebook 03 — Agent with tools →](notebooks/notebook_03_agent_with_tools.md)

## Eight shapes, one `Agent`

Every shape runs the same `Agent`, emits the same events, and checkpoints to
the same backends. Pick the shape; the agents don't change.

<div class="grid cards locus-feature-cards" markdown>

- :material-arrow-right-bold-box:{ .lg .middle } **[Pipelines](concepts/multi-agent/composition.md)**

    ---
    `SequentialPipeline`, `ParallelPipeline`, `LoopAgent`: chain, fan out,
    or revise until a reviewer clears. Three lines each.

- :material-sitemap:{ .lg .middle } **[Orchestrator and specialists](concepts/multi-agent/orchestrator.md)**

    ---
    One coordinator delegates to typed specialists with their own tools,
    and merges what comes back.

- :material-account-group:{ .lg .middle } **[Swarm](concepts/multi-agent/swarm.md)**

    ---
    Peers that hand work to each other on their own judgement, with a
    shared scratchpad and a hard stop.

- :material-swap-horizontal:{ .lg .middle } **[Handoff](concepts/multi-agent/handoff.md)**

    ---
    Tiered escalation with typed context: L1 to L2 to L3, nothing lost
    in between.

- :material-graph:{ .lg .middle } **[StateGraph and functional API](concepts/multi-agent/graph.md)**

    ---
    Nodes, edges, reducers, `Send` fan-out, interrupts for a human, and
    checkpointed state.

- :material-forum:{ .lg .middle } **[Debate and critic loops](notebooks/notebook_26_debate_with_judge.md)**

    ---
    Two debaters and a judge, or a skeptical reviewer that kills unproven
    claims before they ship.

- :material-routes:{ .lg .middle } **[Cognitive router](concepts/multi-agent.md#the-eight-shapes)**

    ---
    Describe the task; the router extracts a typed `GoalFrame`, picks one
    of eight protocols and compiles it onto real agents.

- :material-lan-connect:{ .lg .middle } **[A2A protocol](concepts/multi-agent/a2a.md)**

    ---
    Cross-process agents as spec-compliant peers: expose yours, call
    theirs, same tool contract.

- :material-telescope:{ .lg .middle } **[DeepAgent and research workflows](concepts/deepagent.md)**

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

## Oracle Cloud, native

The OCI and Oracle Database primitives are first-class classes, not adapters:
same `Agent`, same events, same admission gate.

<div class="grid cards locus-feature-cards" markdown>

- :material-cloud-check:{ .lg .middle } **[OCI Generative AI](concepts/providers/oci.md)**

    ---
    `model="oci:…"` picks the transport from the model id: V1 for OpenAI,
    Meta, xAI, Gemini; SDK for Cohere; Responses where the region serves it.

- :material-server-network:{ .lg .middle } **[Dedicated AI Clusters](how-to/oci-dac.md)**

    ---
    `model="oci:ocid1.generativeaiendpoint…"` targets your own DAC
    endpoint, with the compartment and auth resolved from your profile.

- :material-console:{ .lg .middle } **[OCI tools](notebooks/notebook_36_oci_tools.md)**

    ---
    `use_oci()` and `describe_oci()` put the OCI control plane behind
    tools, so an agent can list, inspect and act on your tenancy.

- :material-database-search:{ .lg .middle } **[Oracle Database 26ai RAG](notebooks/notebook_38_oracle_26ai_rag.md)**

    ---
    `OracleVectorStore` on native `VECTOR` columns, hybrid search, plus
    in-database chunking and embeddings so text never leaves the database.

- :material-content-save:{ .lg .middle } **[26ai checkpoints and history](notebooks/notebook_42_oracle_26ai_checkpointer.md)**

    ---
    `oracle_checkpointer` for durable threads, `OracleCheckpointSaver` for
    versioned history and time travel, OCI Object Storage as the alternative.

- :material-brain:{ .lg .middle } **[26ai long-term memory](notebooks/notebook_44_oracle_agent_memory.md)**

    ---
    `OracleStore` for cross-thread memory and `OracleAgentMemoryManager`
    over Oracle's official agent-memory client.

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

## The harness

<div class="grid cards locus-feature-cards" markdown>

- :material-tools:{ .lg .middle } **[Tools and MCP](concepts/tools.md)**

    ---
    `@tool` on any function, typed arguments coerced from whatever the
    model sends, idempotent execution, MCP servers in or out.

- :material-database:{ .lg .middle } **[Memory and RAG](concepts/memory.md)**

    ---
    Four layers: working memory, the thread (checkpoint backends from local
    files to Oracle Database 26ai),
    long-term memory across threads, and knowledge through RAG (six vector
    stores, rerankers, `retriever.as_tool()`). All four on Oracle Database 26ai.

- :material-shield-lock:{ .lg .middle } **[Grounding, guardrails, admission gate](concepts/control-layer.md)**

    ---
    GSAR checks every claim against evidence; guardrails and steering
    catch the rest; `admit()` holds or denies consequential actions.

- :material-eye:{ .lg .middle } **[Observability and streaming](concepts/observability.md)**

    ---
    One `run_context()` streams typed events from every layer; SSE and
    token streaming for the UI.

- :material-server:{ .lg .middle } **[Server and deploy](concepts/server.md)**

    ---
    `locus serve` puts an agent behind FastAPI with bearer auth and thread
    persistence; a runner image ships for OKE.

- :material-school:{ .lg .middle } **[Skills, playbooks, evaluation](concepts/skills.md)**

    ---
    Progressive-disclosure skills, enforced playbooks, and regression
    suites with `EvalRunner` so behaviour stays put.

</div>

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

## Where next

[All notebooks →](notebooks/index.md) ·
[Capabilities →](capabilities.md) ·
[API reference →](api/index.md) ·
[Deploy →](how-to/deploy.md)

---

**Built inside Oracle. Used in production. Open to everyone.**
