---
hide:
  - navigation
  - toc
---

<div class="tulip-hero" markdown>
<div class="tulip-hero__copy" markdown>

<p class="tulip-product-name">Tulip · Multi-Agent Reasoning Orchestrator SDK</p>

# Multi-agent workflows built for <span class="accent">production.</span>

Describe the task. Tulip selects the protocol and coordinates the agents.

<div class="tulip-stat-strip" markdown><span style="white-space:nowrap">[direct&nbsp;answer](concepts/router.md)</span> · <span style="white-space:nowrap">[pipeline](concepts/multi-agent/composition.md)</span> · <span style="white-space:nowrap">[fan&#8209;out](concepts/multi-agent/composition.md)</span> · <span style="white-space:nowrap">[debate](concepts/multi-agent/composition.md)</span> · <span style="white-space:nowrap">[code&nbsp;+&nbsp;test](concepts/multi-agent/composition.md)</span> · <span style="white-space:nowrap">[approval&nbsp;gate](concepts/interrupts.md)</span> · <span style="white-space:nowrap">[A2A](concepts/multi-agent/a2a.md)</span> · <span style="white-space:nowrap">[handoff](concepts/multi-agent/handoff.md)</span></div>

- **From idea to production agent in minutes, not weeks.** Describe the task; Tulip picks the pattern and assembles the network from eight production-tested protocols.
- **Self-critiquing agents with grounded outputs.** Every turn is scored; every claim is verified against the tool result that produced it.
- **Full causal traceability.** Every decision, tool call, and reasoning step is a typed event you can replay, audit, and debug.

[Workbench guide](workbench.md){ .md-button .md-button--primary }
[GitHub](https://github.com/tuliplabs-ai/sdk-python){ .md-button }

```bash
pip install "tulip-agents[openai]"   # OpenAI · Anthropic
```

Vendor-neutral · Production-tested · Open source

</div>

<div class="tulip-hero__code" markdown>

```python
from tulip.agent import Agent
from tulip.tools import tool
from tulip.observability import run_context, get_event_bus

@tool
def get_metric(name: str) -> float:
    """Current value of a named SRE metric."""
    return monitoring.read(name)

@tool
def fetch_runbook(topic: str) -> str:
    """Pull the runbook section for a topic."""
    return wiki.fetch(topic)

@tool(idempotent=True)
def page_oncall(reason: str) -> str:
    """Page the on-call engineer. Fires exactly once per reason."""
    return pager.send(reason)

agent = Agent(
    model="openai:gpt-4o",
    tools=[get_metric, fetch_runbook, page_oncall],
    reflexion=True,        # self-evaluates every turn
    grounding=True,        # claims verified against tool output
)

async with run_context() as rid:
    result = await agent.run(
        "p99 on checkout-api spiked to 4.2s — investigate and page if critical."
    )
    async for ev in get_event_bus().subscribe(rid):
        match ev.event_type:
            case "agent.tool.started":   print("🔧", ev.data["tool_name"])
            case "agent.tokens.used":    print("🪙", ev.data["total_tokens"])
            case "agent.terminate":      print("✓", ev.data["final_message_preview"])
```

</div>
</div>

## Your first agent in five lines

```python
from tulip.agent import Agent

agent = Agent(model="openai:gpt-4o")
print(agent.run_sync("What is the capital of France?").text)
# → Paris
```

That's the entire interface. `Agent` handles the model call, the
response, and any retries. Swap `openai:gpt-4o` for
`anthropic:claude-sonnet-4-6` or `anthropic:claude-sonnet-4-6` — the call stays the
same.

Add a tool, and the agent loops Think → call tool → Think → answer
until it's done:

```python
from tulip.tools import tool

@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return weather_api.fetch(city)

agent = Agent(
    model="openai:gpt-4o",
    tools=[get_weather],
    system_prompt="You are a helpful travel assistant.",
)
print(agent.run_sync("Should I bring an umbrella to Tokyo tomorrow?").text)
```

[Notebook 14 — basic agent →](notebooks/notebook_06_basic_agent.md)

## What Tulip gives you

<div class="grid cards tulip-feature-cards" markdown>

- :material-graph:{ .lg .middle } **[Multi-agent coordination](concepts/multi-agent.md)**

    ---
    Seven in-process patterns plus A2A: Sequential, Parallel, Loop,
    Orchestrator, Swarm, Handoff, StateGraph, plus DeepAgent. One
    `Agent` class, one event stream.

- :material-routes:{ .lg .middle } **[Cognitive router](concepts/router.md)**

    ---
    Describe a task in plain language. The router extracts a typed
    `GoalFrame`, picks one of eight built-in protocols, and compiles
    it onto a real `Agent` / `Pipeline` / `Orchestrator`.

- :material-chart-timeline-variant:{ .lg .middle } **[Grounded reasoning](concepts/reasoning.md)**

    ---
    Reflexion, Grounding, and Causal are first-class `Think → Execute → Reflect`
    nodes. Claims that don't hold up against tool output get dropped or
    re-researched before the user ever sees them.

- :material-shield-check:{ .lg .middle } **[Idempotent tools](concepts/idempotency.md)**

    ---
    `@tool(idempotent=True)` deduplicates on `(name, args)` inside the
    Execute node. No double-charge, double-book, or double-page — even
    on model retry or checkpoint resume.

- :material-eye:{ .lg .middle } **[In-process observability](concepts/observability.md)**

    ---
    Opt-in `EventBus` with an agent yield bridge. One `run_context()`
    streams 60+ canonical events from every layer — agent, multi-agent,
    router, RAG, memory. Zero allocations when unused.

- :material-code-braces:{ .lg .middle } **[Termination algebra](concepts/termination.md)**

    ---
    `MaxIterations(10) | TextMention("DONE") & ConfidenceMet(0.9)` is
    real Python — `__or__` / `__and__` overloads on typed classes.
    Greppable, unit-testable, serialisable.

</div>

## Eight protocols, one dispatch call

Once you have an agent, the next question is *which shape* to use.
The cognitive router picks for you:

| Protocol | Compiled shape | Best for |
|---|---|---|
| `direct_response` | Single `Agent` | `ANSWER`, `EXPLAIN` |
| `plan_execute_validate` | `SequentialPipeline` (planner → executor → validator) | `PLAN`, `BUILD`, `MODIFY` |
| `specialist_fanout` | `ParallelPipeline` of N tool-bound Agents | `DIAGNOSE`, `MONITOR` |
| `debate` | Two debaters + judge `Agent` | `COMPARE` |
| `codegen_test_validate` | `LoopAgent` (stops on `PASS`) | `GENERATE_CODE` |
| `approval_gated_execution` | `Agent` wrapped in approval interrupt | `ESCALATE`, `REMEDIATE` |
| `handoff_chain` | `SequentialPipeline` of one-tool Agents | `COORDINATE` |
| `a2a_delegate` | Cross-process A2A call (opt-in) | distributed meshes |

```python
result = await router.dispatch("Diagnose the checkout API slowdown.")
print(result.protocol_id)   # "specialist_fanout"
print(result.text)          # findings from 3 parallel probes
```

[Cognitive router →](concepts/router.md)

## Vendor-neutral backends

RAG, memory, and persistence are small contracts — pick any backend that
implements them, with no lock-in and a free/local test path for most.

```python
from tulip.rag import OpenAIEmbeddings, QdrantVectorStore, RAGRetriever
from tulip.rag.reranker import CrossEncoderReranker

retriever = RAGRetriever(
    embedder=OpenAIEmbeddings(model="text-embedding-3-small"),
    store=QdrantVectorStore(location=":memory:", dimension=1536),  # or a server URL
    reranker=CrossEncoderReranker(top_n=5),                        # local, offline
)
await retriever.add_documents(corpus)
hits = await retriever.retrieve("…", limit=5)
```

Vector stores: **pgvector · Qdrant · Chroma · OpenSearch · in-memory**.
Checkpointers: **Redis · Postgres · MySQL · OpenSearch · S3 / MinIO ·
file · in-memory · HTTP**. Long-term memory: **Mem0** or `LLMMemoryManager`
over any `BaseStore`.

[RAG concept page →](concepts/rag.md)

## Walk the notebooks

The fastest way to learn Tulip is to run the notebooks. Each one is a
single self-contained file under [`examples/`][gh-examples] with a
matching docs page — start at the topic you want, click through to
the source from there.

| What | Notebooks (click any one) |
|---|---|
| Agent + ReAct loop | [13 — Basic agent](notebooks/notebook_06_basic_agent.md) · [14 — Agent with tools](notebooks/notebook_07_agent_with_tools.md) · [17 — Lifecycle hooks](notebooks/notebook_12_agent_hooks.md) |
| Cognitive router (PRISM) | [57 — Cognitive router](notebooks/notebook_58_cognitive_router.md) · [39 — Emergent routing](notebooks/notebook_34_emergent_routing.md) |
| Multi-agent shapes | [21 — Basic graph](notebooks/notebook_16_basic_graph.md) · [26 — Composition](notebooks/notebook_21_composition.md) · [29 — Swarm](notebooks/notebook_24_swarm_multiagent.md) · [30 — Handoff](notebooks/notebook_25_agent_handoff.md) · [31 — Orchestrator](notebooks/notebook_26_orchestrator_pattern.md) · [33 — A2A](notebooks/notebook_28_a2a_protocol.md) |
| Observability | [16 — Streaming events](notebooks/notebook_11_agent_streaming.md) · [58 — Observability basics](notebooks/notebook_59_observability_basics.md) · [61 — Event catalogue](notebooks/notebook_62_event_catalogue.md) |
| Idempotent tools · termination | [14 — Agent with tools](notebooks/notebook_07_agent_with_tools.md) · [20 — Termination](notebooks/notebook_15_termination.md) |
| RAG · vector stores | [38 — RAG basics](notebooks/notebook_38_rag_basics.md) · [39 — RAG providers](notebooks/notebook_39_rag_providers.md) · [40 — RAG agents](notebooks/notebook_40_rag_agents.md) |
| Memory · checkpointers | [08 — Conversation memory](notebooks/notebook_08_agent_memory.md) · [52 — Checkpoint backends](notebooks/notebook_52_checkpoint_backends.md) |

Full catalog → [Notebooks index](notebooks/index.md) · [Capabilities matrix](capabilities.md) · [API reference](api/agent.md)

[gh-examples]: https://github.com/tuliplabs-ai/sdk-python/tree/main/examples

---

**Vendor-neutral. Production-tested. Open to everyone.**
