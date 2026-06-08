# Capabilities

Everything Tulip ships, what it
does, and where to find it.

!!! sdk-distinctive "Distinctive to the SDK"
    These are architectural choices no other Python agent framework ships
    together in one coherent stack:

    - **Multi-agent reasoning orchestrator** — describe a task; a
      typed registry picks one of eight protocols and instantiates the
      matching SDK primitive. The LLM fills a typed `GoalFrame`; routing is
      rule-based. Eight protocols: `direct_response` (single Agent),
      `plan_execute_validate` (SequentialPipeline), `specialist_fanout`
      (ParallelPipeline), `debate` (two debaters + judge),
      `codegen_test_validate` (LoopAgent), `approval_gated_execution`
      (Agent + interrupt), `a2a_delegate`, `handoff_chain`.
    - **Seven native multi-agent patterns plus A2A** — Composition
      (Sequential / Parallel / Loop), Orchestrator + Specialists, Swarm,
      Handoff, StateGraph, Functional API (`@task` / `@entrypoint`), DeepAgent,
      cross-process A2A. Use them directly, or let the reasoning orchestrator
      dispatch to them. Every pattern shares the same `Agent` class and event
      stream.
    - **In-process observability** — opt-in `EventBus` with agent yield
      bridge. One `run_context()` streams 60+ canonical events from every
      layer (agent, multi-agent, router, RAG, memory, A2A). Zero allocations
      when unused.
    - **Reasoning loop nodes** — Reflexion, Grounding, Causal as first-class
      Think → Execute → **Reflect** → Think nodes, not bolted-on libraries.
    - **GSAR** — typed-grounding safety layer from
      [Federico A. Tulip (2026), arXiv:2604.23366](https://arxiv.org/abs/2604.23366):
      four-way claim partition (grounded / ungrounded / contradicted /
      complementary) + tiered replanning decisions.
    - **Termination algebra** — `MaxIterations(10) | TextMention("DONE") & ConfidenceMet(0.9)` is real Python (`__or__` / `__and__` overloads). Greppable, unit-testable, serialisable.
    - **Idempotent tools** — `@tool(idempotent=True)` dedupes on `(name, args)` inside the Execute node. No double-charge, double-book, double-page — even on model retry or checkpoint resume.
    - **OpenAI, Anthropic, and OpenAI-compatible providers** — OpenAI
      day-zero (two transports, 90+ models including OpenAI commercial and xAI
      Grok, auto-routed by model id); OpenAI and Anthropic through their
      official SDKs. One `get_model()` call, any
      provider.

## Agent core

| Feature | What it does | Surface |
|---|---|---|
| **Agent** + `AgentConfig` + `AgentResult` | The Think → Execute → Reflect → Terminate loop | `tulip.agent` · [Agent loop](concepts/agent-loop.md) |
| **Termination algebra** | Compose stop conditions with `&` and `\|` operator overloads | `tulip.core.termination` · [Termination](concepts/termination.md) |
| **Idempotent tools** | `@tool(idempotent=True)` dedupes repeat calls inside the loop — exactly-once side effects | `tulip.tools.decorator` · [Idempotency](concepts/idempotency.md) |
| **Reflexion** | Self-evaluation node in the ReAct cycle; rewrites the next turn when the last one was wrong | `Agent(reflexion=True)` · [Reasoning](concepts/reasoning.md) |
| **Grounding** | LLM-as-judge claim verification against tool results; below-threshold triggers replanning | `Agent(grounding=True)` · [Reasoning](concepts/reasoning.md) |
| **Causal chains** | Cause-effect graph builder with cycle/contradiction detection | `tulip.reasoning.causal.CausalChain` · [Reasoning](concepts/reasoning.md) |
| **GSAR** | Typed-grounding safety layer ([Federico A. Tulip (2026), arXiv:2604.23366](https://arxiv.org/abs/2604.23366)) — four-way claim partition + tiered replanning | `Agent(gsar=GSARConfig(...))` · [GSAR](concepts/gsar.md) |
| **Cancel** | Thread-safe abort during a run; emits `TerminateEvent` with reason | `agent.cancel()` · [Agent loop](concepts/agent-loop.md) |
| **Interrupts (HITL)** | Pause via `InterruptEvent`; resume with `agent.resume(...)` | `tulip.core.interrupt` · [Interrupts](concepts/interrupts.md) |
| **Structured output** | Pass `output_schema=` (Pydantic), final answer is parsed into a typed instance | `tulip.agent.config`, `tulip.core.structured` · [Structured output](concepts/structured-output.md) |
| **Hooks** | before/after × invocation × tool × model lifecycle observation + steering | `tulip.hooks.provider` · [Hooks](concepts/hooks.md) |
| **Plugins** | Bundle hooks + tools as one drop-in unit | `tulip.hooks.plugin` · [Hooks](concepts/hooks.md) |

## Multi-agent

| Shape | What it does | Surface |
|---|---|---|
| **Composition** | Linear chain · fan-out + merge — the simplest multi-agent shape | `tulip.multiagent.composition` · [Composition](concepts/multi-agent/composition.md) |
| **Orchestrator** | One coordinator dispatches specialists in parallel | `tulip.multiagent.orchestrator` · [Orchestrator](concepts/multi-agent/orchestrator.md) |
| **Swarm** | Open-ended peer-to-peer collaboration | `tulip.multiagent.swarm` · [Swarm](concepts/multi-agent/swarm.md) |
| **Handoff** | Specialist-to-specialist context transfer with chain-of-custody | `tulip.multiagent.handoff` · [Handoff](concepts/multi-agent/handoff.md) |
| **StateGraph** | Cycles, conditional edges, subgraphs — when DAG isn't enough | `tulip.multiagent.graph` · [StateGraph](concepts/multi-agent/graph.md) |
| **Functional API** | Map / reduce over agents with `@task` and `@entrypoint` | `tulip.multiagent.functional` · [Functional](concepts/multi-agent/functional.md) |
| **A2A** | Cross-process agent meshes — `AgentCard` discovery + HTTP/SSE transport | `tulip.a2a` · [A2A](concepts/multi-agent/a2a.md) |

## Cognitive Router

Most agent frameworks force a choice: hand-code the topology (predictable
but brittle) or let the LLM pick it (flexible but unpredictable). The
cognitive router takes a third path — **bounded graph generation**. The LLM fills exactly
one typed `GoalFrame`; a typed registry selects from eight
named protocols; a compiler instantiates real SDK primitives. The
output is always one of the eight proven shapes — never an ad-hoc topology
the model invented.

| Feature | What it does | Surface |
|---|---|---|
| **`Router`** | `dispatch(NL)` → extract GoalFrame → select protocol → compile → execute | `tulip.router.Router` · [Router](concepts/router.md) |
| **`GoalFrame`** | Typed schema the LLM extractor fills — 13 `TaskType`s, `Risk`, `Complexity`, domain, capabilities | `tulip.router.GoalFrame` |
| **`ProtocolRegistry`** | Typed filter (`handles ∋ goal`, `risk_max ≥ frame.risk`) + four-tier ranking (distance · canonical · cost · specificity) | `tulip.router.ProtocolRegistry` |
| **`PolicyGate`** | Two thresholds: `max_risk` (hard deny) and `require_approval_above` (human-in-the-loop gate) | `tulip.router.PolicyGate` |
| **`CognitiveCompiler`** | Instantiates real SDK primitives from frame + protocol; emits a `Runnable` adapter | `tulip.router.CognitiveCompiler` |
| **`builtin_protocols()`** | 8 v1 protocols: `direct_response` · `plan_execute_validate` · `specialist_fanout` · `debate` · `codegen_test_validate` · `approval_gated_execution` · `a2a_delegate` · `handoff_chain` | `tulip.router.builtin_protocols` |
| **`CapabilityIndex`** | Domain + risk overlay on `ToolRegistry` — no parallel storage | `tulip.router.CapabilityIndex` |
| **`SkillIndex`** | Domain-tagged view of installed `Skill` packs; scoped catalog attached to every emitted Agent | `tulip.router.SkillIndex` |
| Custom protocols | `Protocol(id=…, handles=[…], builder=fn)` registered via `ProtocolRegistry.register()` | `tulip.router.Protocol` |
| Error types | `FrameExtractionError` · `NoMatchingProtocolError` · `PolicyDeniedError` | `tulip.router.runtime/protocol/policy` |

## Observability

| Feature | What it does | Surface |
|---|---|---|
| **`EventBus`** | Singleton in-process pub/sub — per-run + global subscribers, bounded queues, history replay, drop accounting | `tulip.observability.EventBus` · [Observability](concepts/observability.md) |
| **`run_context()`** | ContextVar-based opt-in gate — zero allocations when inactive | `tulip.observability.run_context` |
| **Agent yield bridge** | `@_bus_bridge` on `Agent.run` transparently republishes 9 `TulipEvent` types as `agent.*` SSE events | `tulip.agent.runtime_loop` |
| **`EventBusHook`** | `HookProvider` that bridges all agent lifecycle hooks onto the bus (for non-async / pre-built agents) | `tulip.observability.EventBusHook` |
| **Canonical event catalogue** | 60+ `EV_*` constants across 10 prefixes (`agent.*`, `multiagent.*`, `composition.*`, `router.*`, `research.*`, `rag.*`, `memory.*`, `a2a.*`, `skills.*`, `deepagent.*`) | `tulip.observability.emit` · [SSE event catalogue](concepts/sse-events.md) |

## Reasoning

| Feature | What it does | Surface |
|---|---|---|
| **Reflexion** | After each turn, the agent self-evaluates and re-plans on wrong premises | `Agent(reflexion=True)` · [Reasoning](concepts/reasoning.md) |
| **Grounding** | LLM-as-judge over claims vs the tool results that produced them | `Agent(grounding=True)` · [Reasoning](concepts/reasoning.md) |
| **Causal** | Build a cause-effect graph from the trace; surface contradictions | `build_causal_chain()` · [Reasoning](concepts/reasoning.md) |
| **GSAR** | Typed claim partition (grounded / ungrounded / contradicted / complementary) + `proceed`/`regenerate`/`replan`/`abstain` decision | `Agent(gsar=GSARConfig(...))` · [GSAR](concepts/gsar.md) |

## Tools

| Feature | What it does | Surface |
|---|---|---|
| `@tool` decorator | Function → JSON-Schema-typed tool the model can call | `tulip.tools.decorator` · [Tools](concepts/tools.md) |
| Idempotent dedup | `@tool(idempotent=True)` skips repeat calls (same args) in the loop | `tulip.tools.decorator` · [Idempotency](concepts/idempotency.md) |
| **Sequential executor** | Run tool calls one at a time | `tulip.tools.executor` · [Executors](concepts/executors.md) |
| **Concurrent executor** | Run tool calls in parallel | `tulip.tools.executor` · [Executors](concepts/executors.md) |
| **CircuitBreaker executor** | Auto-disable a tool after N failures | `tulip.tools.executor` · [Executors](concepts/executors.md) |
| Result-store offload | Move large tool results to object storage; agent sees a pointer | `tulip.tools.result_storage` |
| Path / URL safety | Validate filesystem and network access from tool args | `tulip.tools.path_safety`, `tulip.tools.url_safety` · [Safety](concepts/safety.md) |
| **MCP — client + server** | Talk to / be talked to by Anthropic-spec MCP servers | `tulip.integrations.fastmcp` · [MCP](concepts/mcp.md) |

## Memory — checkpointer backends

| Backend | Best for | Surface |
|---|---|---|
| `MemoryCheckpointer` | Tests, REPL — in-process dict | `tulip.memory.backends.memory` · [Checkpointers](concepts/checkpointers.md) |
| `FileCheckpointer` | Local dev — JSON files on disk | `tulip.memory.backends.file` |
| `HTTPCheckpointer` | A remote checkpoint service you already run | `tulip.memory.backends.http` |
| **`S3Backend`** | vendor-neutral, lifecycle policies, region replication | `tulip.memory.backends.oci_bucket` |
| `RedisBackend` | Multi-replica, fast, TTLs (a managed Redis) | `tulip.memory.backends.redis` |
| `PostgreSQLBackend` | Production DB with metadata queries | `tulip.memory.backends.postgresql` |
| `MySQLBackend` | Production MySQL with official async Connector/Python | `tulip.memory.backends.mysql` |
| `OpenSearchBackend` | Full-text search across past runs | `tulip.memory.backends.opensearch` |
| `RedisBackend` | Redis key/value store | `tulip.memory.backends.redis` |

## Memory — context management

| Feature | What it does | Surface |
|---|---|---|
| `SlidingWindowManager` | Keeps the last N messages; drops the rest | `tulip.memory.compactor` · [Conversation management](concepts/conversation-management.md) |
| `SummarizingManager` | LLM rollup of older turns | `tulip.memory.compactor` |
| **`LLMCompactor`** | Budget-aware compaction with head + tail protection | `tulip.memory.compactor` |
| Long-term key-value store | Cross-run user prefs / results with optimistic-locking `version` counter | `tulip.memory.store` |

## Hooks (built-in)

| Hook | What it does | Import |
|---|---|---|
| `LoggingHook` / `StructuredLoggingHook` | Stdlib / structured-JSON logs of every event | `tulip.hooks.builtin` · [Observability](concepts/observability.md) |
| **`TelemetryHook`** | OpenTelemetry traces + metrics (counters, histograms) | `tulip.hooks.builtin` |
| `NoOpTelemetryHook` | Opt-out variant for tests | `tulip.hooks.builtin` |
| `ModelRetryHook` | Auto-retry model calls on throttle/empty with exponential back-off | `tulip.hooks.builtin` · [Retry](concepts/retry.md) |
| **`GuardrailsHook`** | Block dangerous tools, redact PII, enforce content/topic policies | `tulip.hooks.builtin` · [Safety](concepts/safety.md) |
| `ContentFilterHook` | Standalone content moderation | `tulip.hooks.builtin` |
| **`SteeringHook`** | LLM-as-judge approval gate on every tool call | `tulip.hooks.builtin` · [Safety](concepts/safety.md) |

## Streaming + Server

| Feature | What it does | Surface |
|---|---|---|
| **Typed events** | Frozen Pydantic events for `match`-statement consumers | `tulip.core.events` · [Events](concepts/events.md) |
| `StructuredStream` | Incremental Pydantic-partial parsing during streaming | `tulip.core.structured` |
| Console + SSE handlers | Render to terminal or stream over Server-Sent Events | `tulip.core.events` · [Streaming](concepts/streaming.md) |
| **`AgentServer`** | Drop-in FastAPI app: `/invoke`, `/stream`, `/threads/{id}`, `/health` | `tulip.server` · [Agent Server](concepts/server.md) |
| Per-principal threads | Bearer-token auth + thread-id namespacing prevents cross-tenant leaks | `AgentServer(api_key=...)` · [Agent Server](concepts/server.md) |
| Graph streaming | Multi-agent state-graph event streams | `tulip.multiagent.graph` · [Graph streaming](concepts/graph-streaming.md) |

## RAG

| Component | Options | Surface |
|---|---|---|
| Vector stores | pgvector · OpenSearch · pgvector · in-memory | `tulip.rag.stores` · [RAG](concepts/rag.md) |
| Embeddings | `OpenAIEmbeddings` (Cohere) · `OpenAIEmbeddings` | `tulip.rag.embeddings` |
| Multimodal processors | Text · PDF (text + OCR) · Image (OCR) · Audio (transcription) | `tulip.rag.multimodal` |
| Tool wiring | `create_rag_tool(retriever)` exposes the retriever as a `@tool` | `tulip.rag.tools` |

## Models

| Provider | Models | Surface |
|---|---|---|
| OpenAI | All commercial models (gpt-5.5, o-series, etc) | `tulip.models.providers.openai` · [OpenAI](concepts/providers/openai.md) |
| Anthropic | Claude 4 / 4.5 / 4.7 / 4.8 — direct API | `tulip.models.providers.anthropic` · [Anthropic](concepts/providers/anthropic.md) |
| Auto-routing | `get_model("anthropic:claude-sonnet-4-6")` picks transport from id | `tulip.models.registry.get_model` |
| Decorators | Failover · pooled · cached · rate-limited wrappers over any provider | `tulip.models.decorators` |

## Skills + Playbooks

| Feature | What it does | Surface |
|---|---|---|
| **Skills** | AgentSkills.io progressive disclosure (catalog → instructions → resources) | `tulip.skills.SkillsPlugin` · [Skills](concepts/skills.md) |
| `Skill.from_directory()` | Load a folder of `SKILL.md` bundles | `tulip.skills.models.Skill` |
| **Playbooks** | Numbered execution plans with per-step `PlaybookEnforcer` | `tulip.playbooks` · [Playbooks](concepts/playbooks.md) |
| YAML / JSON / Python loaders | Author playbooks in any of three formats | `tulip.playbooks.loader` |

## Evaluation

| Class | What it does | Surface |
|---|---|---|
| `EvalCase` | A single test case — expected tools / output / iteration / duration budgets | `tulip.evaluation` · [Evaluation](concepts/evaluation.md) |
| `EvalRunner` | Runs a list of cases against an agent, returns `EvalReport` | `tulip.evaluation` |
| `EvalResult` | Per-case pass / score / duration + diagnostic checks | `tulip.evaluation` |
| `EvalReport` | Aggregate stats with `summary()` + JSON serialisation | `tulip.evaluation` |

## Where to next

- **For first-time visitors**: [Quickstart](how-to/quickstart.md) ships a working agent in five minutes.
- **For architecture**: [Agent loop](concepts/agent-loop.md) is the canonical reference.
- **For depth on any feature**: every row in this matrix links to its concept page. Source lives at [`src/tulip/`](https://github.com/tuliplabs-ai/sdk-python/tree/main/src/tulip); canonical entry is [`src/tulip/__init__.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/__init__.py).
