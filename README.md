<p align="center">
  <img src="https://raw.githubusercontent.com/tuliplabs-ai/sdk-python/main/docs/img/tuliplabs-logo.png" alt="tuliplabs" width="320">
</p>

<p align="center">
  <strong>Tulip — the SDK for agents you can let <em>act</em>.</strong><br>
  <em>Agents that take real action — isolate a host, block an IP, disable an account — where every risky step is policy-gated and human-approvable, and every decision lands in a tamper-evident audit trail. You can fool the model; you can't talk past the runtime. Built for cybersecurity, where a wrong action is a breach.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/tulip-agents/"><img src="https://img.shields.io/pypi/v/tulip-agents.svg?label=PyPI&color=ED5A8B" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/Python-3.11%E2%80%933.14-blue.svg" alt="Python 3.11–3.14">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/mypy-strict-brightgreen.svg" alt="mypy strict">
  <img src="https://img.shields.io/badge/ruff-clean-brightgreen.svg" alt="ruff clean">
</p>

<p align="center">
  <strong>OpenAI · Anthropic · bring your own</strong><br>
  <em>Providers are a one-string swap; the loop, the tools, and the event stream stay put.</em>
</p>

<p align="center">
  <a href="https://tulipagents.ai/">Documentation</a> ·
  <a href="https://tulipagents.ai/concepts/gsar/">GSAR grounding</a> ·
  <a href="https://tulipagents.ai/concepts/router/">Cognitive Router</a> ·
  <a href="https://tulipagents.ai/concepts/multi-agent/">Multi-agent</a> ·
  <a href="https://tulipagents.ai/concepts/deepagent/">DeepAgent</a> ·
  <a href="https://tulipagents.ai/notebooks/">Notebooks</a> ·
  <a href="https://tulipagents.ai/workbench/">Workbench</a>
</p>

<p align="center">
  <strong>Try every Tulip pattern in your browser →</strong>
  <a href="https://tulipagents.ai/workbench/"><strong>Workbench guide</strong></a><br>
  <em>Step-by-step setup for the browser playground — run it on localhost in three terminals, or in a single Docker container. Bring your own OpenAI / Anthropic key.</em>
</p>

<p align="center">
  <em>Vendor-neutral backends are first-class — pgvector · Qdrant · Chroma · OpenSearch RAG, durable agent threads on Redis / Postgres / S3, and pluggable embeddings + rerankers.</em>
</p>

---

## What is Tulip?

Frontier models are smart. The one thing they **can't** do — no matter how smart —
is *prove they won't take a catastrophic action.* That's the gap Tulip closes:
agents that **act** in the real world, where every consequential step is gated by
policy, held for a human when it matters, and recorded so you can prove what
happened. The moat is **control**, not raw intelligence — the model supplies the
brains; Tulip makes its actions safe and provable.

- **Agents that act, not just advise.** Most agents are stuck at read-only
  suggestions because nobody trusts them to pull the trigger. Tulip's admission
  gate (`admit` / `approve`) lets an agent take real response actions safely — and
  a bare LLM reaching for something dangerous gets **stopped cold**.
- **Tamper-evident audit.** Every action and decision is a hash-chained record you
  can replay and **cannot forge** — SOC/forensics-ready.
- **Evidence-grounded findings.** Security findings carry the tool-backed evidence
  behind them (GSAR); abstentions are explicit, not silent guesses.
- **Red-team & assure other AI** *(the fun front door)* — point it at a chatbot,
  agent, or endpoint and run OWASP-ASI / MITRE-ATLAS attacks.

> **Can you make the agent go rogue?** Give an agent live prod tools, then try to
> jailbreak it into wiping the database. You'll fool the model — and the runtime
> won't care. See [`examples/can_you_make_it_go_rogue.py`](examples/can_you_make_it_go_rogue.py).

## See it in 60 seconds

| Run | What it shows |
|-----|---------------|
| [`examples/can_you_make_it_go_rogue.py`](examples/can_you_make_it_go_rogue.py) | Jailbreak an agent with live prod tools — the admission gate blocks the action anyway. 🏆 breaches: 0. |
| [`examples/governed_soc_action.py`](examples/governed_soc_action.py) | Grounded finding → policy gate → execute **or hold-for-human** → tamper-evident audit you can replay. |
| [`examples/grounding_ablation.py`](examples/grounding_ablation.py) | Honest ablation: same model, with vs without grounding. |

## Drop it into the agent you already have

Already have an agent (any framework) that takes actions? Add Tulip's gate +
tamper-evident audit around the dangerous one — **~8 lines, no rebuild**:

```python
from tulip.security import Action, AuditTrail, SecurityPolicy, admit, AdmissionError

trail = AuditTrail()

async def safe_isolate(host: str):
    try:
        return await admit(
            Action(name="isolate_host", asset=host, blast_radius=50, environment="production"),
            lambda: my_edr.isolate(host),          # your code, any SDK / framework
            policy=SecurityPolicy(), trail=trail,
        )
    except AdmissionError as e:
        page_oncall(e.decision)                    # the gate held it; the trail has the record
```

Production-blast isolation now requires a human, the attempt is recorded in a
hash-chained trail you can't forge, and your agent keeps working unchanged.
**You can fool the model; you can't talk past the gate.**
Try it: [`examples/can_you_make_it_go_rogue.py`](examples/can_you_make_it_go_rogue.py).

## Build a full agent — 5 lines

```python
from tulip.agent import Agent
agent = Agent(model="anthropic:claude-sonnet-4-6")
print(agent.run_sync("Triage: outbound beaconing from 192.0.2.14 to a domain registered yesterday.").message)
```

Construction, the model call, retries, and the reply all live behind that one
class. Point `model=` at `"openai:gpt-4o"` instead and nothing else moves.

## Add a tool

A tool is an ordinary Python function — `@tool` publishes its signature and docstring so the model knows when to reach for it.

```python
from tulip.agent import Agent
from tulip.tools import tool
@tool
def domain_reputation(domain: str) -> str:
    """Return registrar age, category, and reputation for a domain."""
    return intel_db.lookup(domain)

agent = Agent(
    model="openai:gpt-4o",
    tools=[domain_reputation],
    system_prompt="You are a SOC triage analyst. Cite the evidence behind every verdict.",
)

print(agent.run_sync("Users got mail linking to login.phish.example.net — phishing or legit?").text)
```

Behind the scenes the agent alternates reasoning with tool calls until it can answer.
For tools where a duplicate call would hurt — isolating a host, paging an on-call, filing a ticket — declare `@tool(idempotent=True)`:
the loop keys every invocation on `(name, args)` and refuses to fire the same one twice, even across retries.

## Install

```bash
pip install "tulip-agents[openai]"        # OpenAI
pip install "tulip-agents[anthropic]"     # Anthropic
pip install "tulip-agents[rag]"           # vector stores + embeddings + rerankers
pip install "tulip-agents[sdk]"           # everything
```

No mandatory cloud account to start — `MockModel` lets every notebook run offline.

→ [Quickstart guide](https://tulipagents.ai/how-to/quickstart/)

---

## Red-team and assure another AI

Point a `Target` at an AI system — a remote endpoint, an in-process
`tulip.Agent`, or an A2A peer — and run the OWASP-ASI / MITRE-ATLAS suite.
Every result is a grounded `Finding` or an explicit `Abstention`: a vulnerable
target yields findings, a hardened one abstains across the board.

```python
from tulip.security import Target, red_team, assure, is_finding

target = Target.endpoint("https://support-bot.example/chat")

report = await red_team(target, suite="owasp-asi")   # attack → grounded findings
for r in report:
    print(r.title, r.taxonomy) if is_finding(r) else print("abstained:", r.reason)

posture = await assure(target)                        # assess → grounded guardrail coverage
```

The agent doing the work is itself **secure by default** — grounded, guarded,
risk-gated, and recorded in a tamper-evident audit trail:

```python
from tulip.security import secure_agent

secured = secure_agent(model="openai:gpt-4o", tools=[...])
assert secured.audit_trail.verify()   # every action is replayable evidence
```

## Why security teams: grounded or it doesn't ship

Security is the one domain where a hallucinated claim isn't an
embarrassment — it's a false positive that burns an analyst's night, or
a false negative that ships a breach. Tulip is built around that fact:

- **GSAR typed grounding** ([paper](https://arxiv.org/abs/2604.23366)) —
  every claim an agent makes is partitioned **grounded / ungrounded /
  contradicted / unknown** against typed evidence, with scanner and tool
  output outranking inference and inference outranking domain priors.
  Below threshold the run regenerates, replans, or **abstains**. An
  ungrounded finding is a false positive *by construction* — it doesn't
  reach your queue.
- **Typed, replayable event streams** — every model call, tool call,
  guardrail verdict, and approval is an immutable event you can ship to
  a SIEM and replay in a postmortem. The audit trail is the default, not
  an add-on.
- **Risk-gated actions** — the router's PolicyGate ranks operations by
  risk; HIGH-risk steps (isolate a host, block a domain) require a
  durable human approval that survives restarts (`interrupt()` +
  checkpointers).
- **Enforced runbooks** — `PlaybookEnforcer` pins an investigation to
  its IR playbook: steps in order, tools per step, violations recorded.
- **Hardened tool layer** — SSRF-safe fetchers (cloud metadata and
  private ranges blocked, DNS fail-closed), prompt-injection and
  secret-leak guardrails on both input and output, idempotent
  side-effecting tools.

That last point is a type-level guarantee, not a convention. The
`tulip.security` layer turns a GSAR evidence partition into a finding
only when it clears the grounding threshold — otherwise you get an
auditable abstention, never a finding:

```python
from tulip.security import ground_finding, Severity, is_finding
from tulip.reasoning.gsar import Claim, EvidenceType, Partition

result = ground_finding(
    title="Expired TLS certificate on 192.0.2.10:443",
    description="Serving endpoint presents an expired certificate.",
    severity=Severity.HIGH,
    asset="192.0.2.10:443",
    remediation="Rotate the certificate; enforce automated renewal.",
    partition=Partition(grounded=[
        Claim(text="cert expired 2026-05-30", type=EvidenceType.TOOL_MATCH,
              evidence_refs=["tool:tls_scan:not_after=2026-05-30"]),
    ]),
)
# A grounded partition → a typed Finding. An ungrounded one → an
# Abstention with the reason it was withheld. There is no third path,
# and no public constructor that makes a Finding without a score.
print(result.title if is_finding(result) else f"withheld: {result.reason}")
```

`Finding` tags carry the standard catalogues — **MITRE ATLAS**
(`AML.Txxxx`), **OWASP Top 10 for LLM Applications (2025)**, and the
**OWASP Top 10 for Agentic Applications (2026)** — so findings drop into
a SIEM or a **NIST AI RMF** report without a translation layer.

Every example in [`examples/`](examples/) is a security workflow.
**AI-security is the primary track** — prompt injection, jailbreaks,
inference fingerprinting, RAG and memory poisoning, model extraction,
excessive agency — with **classic SOC/IR** (triage, IOC enrichment,
phishing, secure code review, incident response with approval gates) as
the second track. Start with
[`notebook_37_gsar_typed_grounding.py`](examples/notebook_37_gsar_typed_grounding.py).

---

## Talk to any provider

A model is a string. The prefix before the colon (`openai:`,
`anthropic:`) tells the SDK which provider to use; the rest is
the model id that provider expects. `get_model()` parses the string and
returns a ready client.

```python
# tools, system_prompt, and every other kwarg are identical across providers
Agent(model="openai:gpt-4o")                     # OpenAI direct
Agent(model="anthropic:claude-sonnet-4-6")       # Anthropic direct
```

The same `Agent` works against any provider — only the model id and the
credentials change.

| Provider | Class | What it covers |
|---|---|---|
| **OpenAI** | `OpenAIModel` | Chat completions, reasoning models (o-series), `base_url` override for Azure · Portkey · LiteLLM · vLLM · together.ai · fireworks · groq |
| **Anthropic** | `AnthropicModel` | Claude family with prompt caching + extended thinking |
| **Custom** | `register_provider("myco", MyModel)` | Implement `BaseModel` — `complete` · `stream` · `count_tokens` (~50 lines) |

Because OpenAI-compatible endpoints accept a `base_url`, `OpenAIModel`
also fronts gateways and self-hosted servers (LiteLLM, vLLM, Azure
OpenAI, together.ai, groq, …) without a dedicated provider.

→ [Model providers concept page](https://tulipagents.ai/concepts/models/)

---

## The cognitive router (PRISM) — describe what you need, get the right shape

Once you know agents, the next step is knowing *which* shape to use.
The cognitive router takes a natural-language task, runs an LLM
classifier that fills a typed `GoalFrame` (intent · domain · complexity ·
risk), matches it to one of eight built-in coordination protocols, and
the `CognitiveCompiler` emits the matching runtime primitive (`Agent`,
`SequentialPipeline`, `ParallelPipeline`, `LoopAgent`, an `A2AClient`
call, or an approval-gated agent) — without you hand-coding the topology.

```python
from tulip.agent import Agent
from tulip.router import (
    CapabilityIndex, CognitiveCompiler, GoalFrame, PolicyGate,
    ProtocolRegistry, Router, SkillIndex, builtin_protocols,
)
from tulip.tools.registry import create_registry

# 1. Capabilities the router can bind to specialists.
registry = create_registry([kb_search, get_metric, list_alerts])

# 2. All 8 built-in protocols (answer / plan / specialist-fanout / debate
#    / codegen-loop / approval / a2a-delegate / handoff-chain).
protocols = ProtocolRegistry()
for p in builtin_protocols():
    protocols.register(p)

# 3. The Router wires an Agent(output_schema=GoalFrame) extractor + the
#    deterministic protocol picker + a CognitiveCompiler over the registry.
router = Router(
    frame_extractor=Agent(model=get_model(), output_schema=GoalFrame),
    protocols=protocols,
    capabilities=CapabilityIndex.from_registry(registry),
    skills=SkillIndex(),
    gate=PolicyGate(),
    compiler=CognitiveCompiler(),
)

# 4. Dispatch — the router picks the protocol + compiles the shape.
result = await router.dispatch(
    "We just got a sev-1 latency alert on the checkout service. "
    "Investigate and recommend remediation."
)
print(f"protocol={result.protocol_id}")
print(result.text)
```

The same `router.dispatch(...)` call resolves a one-shot lookup to a
single `Agent`, a multi-step incident triage to a `SequentialPipeline`
of planner→executor→validator, and a write-affecting action to an
approval-gated agent — chosen by protocol selection, not by the model.

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

→ [Cognitive router concept](https://tulipagents.ai/concepts/router/) ·
[`examples/notebook_58_cognitive_router.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_58_cognitive_router.py)

---

## Eight orchestration shapes

When one agent isn't enough, the SDK gives you seven in-process shapes plus cross-process A2A.
Every pattern uses the same `Agent` class and the same event stream.

| Pattern | When to use |
|---|---|
| **SequentialPipeline** | A → B → C in order; each output feeds the next |
| **ParallelPipeline** | Fan out to N agents simultaneously, merge results |
| **LoopAgent** | Refine until a condition fires (PASS/FAIL, confidence, iteration cap) |
| **Orchestrator + Specialists** | One coordinator routes to domain experts in parallel |
| **Swarm** | Open-ended research; peers share a task queue and context |
| **Handoff** | Escalation desk; conversation moves with full history to the next specialist |
| **StateGraph** | Explicit DAG with conditional edges, cycles, and human-in-the-loop gates |
| **A2A** | Cross-process meshes over HTTP; agents advertise capabilities via AgentCard |

```python
from tulip.agent import Agent, SequentialPipeline
recon    = Agent(model=model, system_prompt="Summarise the exposed services on the host.")
validate = Agent(model=model, system_prompt="Flag which exposures are exploitable; cite evidence.")
report   = Agent(model=model, system_prompt="Write the finding: severity, asset, remediation.")

result = await SequentialPipeline(agents=[recon, validate, report]).run(
    "Triage the attack surface on 192.0.2.10."
)
print(result.text)
```

→ [All patterns](https://tulipagents.ai/concepts/multi-agent/)

---

## What you get

| | |
|---|---|
| **[🔒 Grounded findings](https://tulipagents.ai/concepts/security/)** | `tulip.security` — `ground_finding()` emits a typed `Finding` only above the GSAR threshold, else an auditable `Abstention`. Tagged to MITRE ATLAS · OWASP LLM · OWASP ASI. Plus `FingerprintFinding` for timing side-channel inference fingerprinting. |
| **[🧭 Cognitive router](https://tulipagents.ai/concepts/router/)** | Describe a task → eight named protocols → right primitive compiled automatically. LLM fills a typed schema; routing is deterministic. |
| **[🤝 Multi-agent](https://tulipagents.ai/concepts/multi-agent/)** | Seven native patterns + cross-process A2A. One `Agent` class. One event stream. |
| **[🔬 DeepAgent](https://tulipagents.ai/concepts/deepagent/)** | `create_deepagent` (single agent, per-turn grounding) and `create_research_workflow` (StateGraph with post-hoc grounding eval + two-level recovery). |
| **[📡 Observability](https://tulipagents.ai/concepts/observability/)** | Opt-in `EventBus` — one `run_context()` streams 40+ canonical events from every layer, no external broker. `TelemetryHook` for OpenTelemetry/OTLP. |
| **[🧠 Reasoning](https://tulipagents.ai/concepts/reasoning/)** | `reflexion=True` · `grounding=True` · `CausalChain` · **GSAR** typed grounding layer (`arXiv:2604.23366`). |
| **[🛡 Idempotent tools](https://tulipagents.ai/concepts/idempotency/)** | `@tool(idempotent=True)` — dedupes on `(name, args)`. The model can't double-charge, double-book, or double-page. |
| **[💾 Durable memory](https://tulipagents.ai/concepts/checkpointers/)** | 8 checkpoint backends — PostgreSQL · MySQL · Redis · OpenSearch · S3 / MinIO / R2 · in-memory · file · HTTP. |
| **[🧠 Long-term memory](https://tulipagents.ai/concepts/memory-manager/)** | `Mem0MemoryManager` over [`mem0`](https://github.com/mem0ai/mem0) — fact extraction, scoped retrieval, self-hostable. Portable path: `LLMMemoryManager` over any `BaseStore` (InMemory / Redis / Postgres / OpenSearch). |
| **[🔎 RAG](https://tulipagents.ai/concepts/rag/)** | 5 vector stores — pgvector · Qdrant · Chroma · OpenSearch · in-memory. OpenAI + Cohere embeddings · local + Cohere rerankers · multimodal (PDF, image OCR, audio). |
| **[📡 Streaming + Server](https://tulipagents.ai/concepts/server/)** | Typed events · SSE · `AgentServer` (FastAPI, per-principal thread isolation). |
| **[🪝 Hooks](https://tulipagents.ai/concepts/hooks/)** | Logging · OpenTelemetry · ModelRetry · Guardrails · Steering (LLM-as-judge). |
| **[🪙 MCP](https://tulipagents.ai/concepts/mcp/)** | `MCPClient` consumes MCP servers. `TulipMCPServer` exposes the SDK's tools as MCP. |
| **[🌐 Multi-modal](https://tulipagents.ai/concepts/multi-modal-providers/)** | `Agent(web_search=…, web_fetch=…, image_generator=…, speech_provider=…)` auto-registers tools. |
| **[📊 Evaluation](https://tulipagents.ai/concepts/evaluation/)** | `EvalCase` / `EvalRunner` / `EvalReport` regression suites. |
| **[🧰 Models](https://tulipagents.ai/concepts/models/)** | OpenAI · Anthropic — plus any OpenAI-compatible gateway via `base_url`. |

---

## Inside the SDK — the stack, not just the loop

A Tulip agent isn't a one-shot ReAct loop. The same `Agent` class runs
inside eight orchestration shapes, chosen automatically by the **PRISM
cognitive router** from a natural-language task, with typed reasoning
around every Execute and a pluggable, vendor-neutral backend stack. The
agent loop is the inner engine — the SDK is the whole stack around it.

<p align="center">
  <img src="docs/img/tulip-stack.svg" alt="The SDK stack — PRISM cognitive router compiles natural-language tasks into one of 8 orchestration shapes (SequentialPipeline, ParallelPipeline, LoopAgent, StateGraph, Orchestrator + Specialists, Swarm, Handoff Chain, A2A Mesh), each of which runs the agent loop (Think → Execute → Reflect → Terminate), powered by foundations (Models, Memory, RAG, Observability, Tools/MCP/Skills)" width="100%">
</p>

- **PRISM Cognitive Router** — an LLM classifier reads the task and fills a typed `GoalFrame` (intent · domain · complexity · risk); the `CognitiveCompiler` emits the matching runtime shape. The model classifies, never authors graph topology.
- **Eight orchestration shapes** — `SequentialPipeline`, `ParallelPipeline`, `LoopAgent`, `StateGraph`, `Orchestrator + Specialists`, `Swarm`, `Handoff Chain`, and cross-process `A2A Mesh`. One `Agent` class composes them all; one event stream covers them all.
- **The agent loop** — **Think → Execute → Reflect → Terminate** with one immutable state flowing through. `@tool(idempotent=True)` dedupes Execute on `(name, args)`; Reflect runs Reflexion + Grounding + Causal on cadence or on error; Terminate is composable algebra (`MaxIterations(10) | ToolCalled("submit") & ConfidenceMet(0.9)`).
- **Foundations** — Models (OpenAI · Anthropic · any OpenAI-compatible gateway), Memory (8 checkpoint backends, long-term store), RAG (5 vector stores, multimodal, rerank), Observability (40+ typed events, EventBus, OpenTelemetry), Tools / MCP / Skills (idempotent, both-ways MCP, guardrails, steering, evaluation).
- **Vendor-neutral storage** — pgvector / Qdrant / Chroma / OpenSearch vector search, durable agent threads on Redis / Postgres / MySQL / S3, and a cross-thread long-term store — all behind one set of contracts.

Every node at every layer emits a write-protected typed event — the same stream powers SSE, telemetry hooks, and your own `async for event in agent.run(…)` consumer.

---

## Vendor-neutral backends

RAG, memory, and persistence are defined by small contracts in
`tulip.rag` and `tulip.memory` — pick any backend that implements them.
Nothing is wired to a single vendor, and most have a free/local test
path (in-memory Qdrant, embedded Chroma, MinIO via `moto`, an offline
cross-encoder reranker).

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

| Capability | Backends |
|---|---|
| **Vector stores** (`tulip.rag.stores`) | `PgVectorStore` · `QdrantVectorStore` · `ChromaVectorStore` · `OpenSearchVectorStore` · `InMemoryVectorStore` |
| **Embeddings** (`tulip.rag.embeddings`) | `OpenAIEmbeddings` · `CohereEmbeddings` |
| **Rerankers** (`tulip.rag.reranker`) | `CrossEncoderReranker` (local sentence-transformers) · `CohereReranker` (Cohere API) |
| **Checkpointers** (`tulip.memory.backends`) | `RedisBackend` · `PostgreSQLBackend` · `MySQLBackend` · `OpenSearchBackend` · `S3Backend` (AWS S3 / MinIO / R2) · `FileCheckpointer` · `MemoryCheckpointer` · `HTTPCheckpointer` |
| **Long-term memory** (`tulip.memory.managers`) | `Mem0MemoryManager` (mem0) · `LLMMemoryManager` over any `BaseStore` |

Every backend is an optional extra — install only what you use
(`pip install "tulip-agents[qdrant,s3,rerank-local]"`).

---

## Vendor security integrations

Core ships **offline reference adapters** for every security domain, so the SDK
runs standalone with no credentials. The maintained, vendor-specific integrations
live in a separate community package —
**[`tulip-integrations`](https://github.com/tuliplabs-ai/tulip-integrations)**
(import `tulip_integrations`) — which depends one-way on core (the LangChain
`core` + `community` split).

```bash
pip install "tulip-integrations[edr-crowdstrike]"   # + any per-vendor extra
```

| Domain | Vendors |
|---|---|
| **SIEM** | Splunk / Elastic |
| **EDR** | CrowdStrike Falcon |
| **Identity** | Okta · Auth0 |
| **Threat intel** | VirusTotal |
| **Vuln / AI-SPM** | Wiz |
| **Compute** | RunPod · Lambda (GPU fingerprint probe) |

Wire one in via `security_toolset(extra=[…])` or a `SecurityContext` provider —
see the [integrations docs](https://tulipagents.ai/integrations/) and the
[`tulip-integrations` repo](https://github.com/tuliplabs-ai/tulip-integrations).

> **Not to be confused with `tulip.integrations`** (this repo) — that's the
> built-in **MCP** client/server (`MCPClient`, `TulipMCPServer`). The community
> **`tulip-integrations`** package is the vendor security adapters above.

---

## Notebooks

[`examples/`](examples/) has a set of progressive notebooks, numbered in
suggested reading order. Each one defaults to a bundled **mock model**
when no API key is present, so every example runs offline with no
credentials needed; set `OPENAI_API_KEY`
to run them against a real provider.

```bash
git clone https://github.com/tuliplabs-ai/sdk-python.git
cd tulip-agents && pip install -e .

python examples/notebook_06_basic_agent.py           # your first agent
python examples/notebook_29_deepagent.py             # deep-research factory
python examples/notebook_69_research_workflow.py     # full research pipeline
```

| Track | What you learn |
|---|---|
| **Agent Foundations** | Agent, tools, memory, streaming, hooks, termination |
| **Graphs & composition** | StateGraph, conditional routing, reducers, HITL, composition, functional API |
| **Multi-agent** | Swarm, handoff, orchestrator, A2A, DeepAgent, debate, emergent routing |
| **Reasoning & structured** | Pydantic schemas, reasoning patterns, GSAR typed grounding |
| **RAG** | Basics, vector stores, embeddings, rerankers, RAG agents |
| **Skills, playbooks, plugins** | MCP, playbooks, plugins, skills, steering |
| **Production** | Guardrails, checkpoints, evaluation, providers, multi-modal |
| **Cognitive router + observability** | Routing, EventBus, yield bridge, event catalogue |
| **Real-world workflows** | Incident response, vendor security review, DPA review, spoken advisories |
| **Server & full pipelines** | Agent server (FastAPI), full research workflow |

→ [Full notebooks index](https://tulipagents.ai/notebooks/)

---

## Workbench

A browser-based playground for every SDK pattern. Two clicks to a
running agent — no CLI install, no editor setup. Three model slots
(A / B / C) so multi-agent notebooks can mix a fast triage model
with a deeper specialist. The **Notebooks** sidebar lists the runnable
`notebook_*.py` files, grouped by track, with live client-side
filtering. A per-tab **Provider Settings** panel collects OpenAI /
Anthropic credentials.

It lives in its own repo —
[tuliplabs-ai/workbench](https://github.com/tuliplabs-ai/workbench).
Two ways to run it. Pick whichever fits.

### Run locally (from source)

```bash
git clone https://github.com/tuliplabs-ai/workbench.git && cd workbench

# Three terminals, one per tier (the python tier is hatch-managed):
cd backend && hatch run sdk && hatch run serve   # FastAPI runner on :8100
cd bff     && npm install && npm run dev         # BFF on :3101
cd web     && npm install && npm run dev         # Vite on :5173
```

Open <http://localhost:5173>, click **Provider settings**, pick a
provider, paste the key, and save.

### Run in Docker

```bash
git clone https://github.com/tuliplabs-ai/workbench.git && cd workbench
docker build -t tulip-workbench .
docker run --rm -p 5173:5173 -p 3101:3101 -p 8100:8100 tulip-workbench
# open http://localhost:5173
```

OpenAI and Anthropic work as-is — paste the key into *Provider settings*.

→ Full walkthrough: [Workbench guide](https://tulipagents.ai/workbench/)

---

## Deploy

```bash
pip install "tulip-agents[server,openai]"
```

`AgentServer` is a drop-in FastAPI app: `POST /invoke`, `POST /stream`, `GET/DELETE /threads/{id}`, `GET /health`.

```python
from tulip.server import AgentServer

server = AgentServer(agent=my_agent, api_key=os.environ["API_KEY"])
server.run(host="0.0.0.0", port=8080)
```

The repo ships a multi-stage `Dockerfile` ready to drop into your own image
pipeline. Deploy anywhere FastAPI runs — Kubernetes, ECS / Fargate, Cloud
Run, Fly.io, a plain VM, or any cloud equivalent.

→ [Deploy guide](https://tulipagents.ai/how-to/deploy/)

---

## Repo layout

```text
src/tulip/
├── agent/          Agent runtime, config, SequentialPipeline / ParallelPipeline / LoopAgent
├── core/           AgentState, Message, events, termination algebra, Send
├── loop/           ReAct nodes (Think, Execute, Reflect)
├── router/         Cognitive router — GoalFrame, ProtocolRegistry, PolicyGate, CognitiveCompiler
├── deepagent/      create_deepagent + create_research_workflow + 6 node primitives
├── observability/  EventBus, run_context, agent yield bridge, EV_* constants
├── memory/         BaseCheckpointer + 8 backends
├── models/         Provider registry + OpenAI, Anthropic
├── multiagent/     Orchestrator, Swarm, Handoff, StateGraph, Functional
├── a2a/            Cross-process Agent-to-Agent protocol
├── reasoning/      Reflexion, Grounding, Causal, GSAR
├── rag/            Embeddings + 5 vector stores + rerankers + retrievers
├── providers/      Multi-modal: web search, web fetch, image, speech
├── tools/          @tool decorator, registry, builtins, executors
├── hooks/          Logging, telemetry, retry, guardrails, steering
├── skills/         AgentSkills.io filesystem-first capability disclosure
├── playbooks/      Declarative step plans + PlaybookEnforcer
├── server/         FastAPI AgentServer with thread persistence
├── evaluation/     EvalCase + EvalRunner + EvalReport
└── integrations/   MCP (client + server)

examples/           Progressive notebooks, each a single runnable file.
tests/unit/         Deterministic, no external deps. Runs in CI on every PR.
tests/integration/  Live OpenAI / Anthropic. Gated on credentials.
```

The documentation site and the browser workbench live in sibling repos:
[tuliplabs-ai/docs](https://github.com/tuliplabs-ai/docs) (published at
[tulipagents.ai](https://tulipagents.ai/)) and
[tuliplabs-ai/workbench](https://github.com/tuliplabs-ai/workbench).

---

## Contributing

```bash
git clone https://github.com/tuliplabs-ai/sdk-python.git
cd tulip-agents && pip install -e ".[dev,sdk]"
hatch run check        # ruff + mypy
hatch run test         # unit tests across Python 3.11–3.14
pre-commit install
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Every PR runs format, lint, mypy, unit tests, DCO sign-off.

---

## Citing GSAR

Paper: [GSAR: Typed Grounding for Hallucination Detection and Recovery in Multi-Agent LLMs](https://arxiv.org/abs/2604.23366) ([PDF](https://arxiv.org/pdf/2604.23366)), 2026.

```bibtex
@article{gsar2026,
  title   = {GSAR: Typed Grounding for Hallucination Detection and Recovery in Multi-Agent LLMs},
  journal = {arXiv preprint arXiv:2604.23366},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.23366},
}
```

---

## Security

Please consult the [security guide](./SECURITY.md) for our responsible security vulnerability disclosure process.

---

## License

Copyright 2026 Tulip Labs.

Released under the **Apache License, Version 2.0** — see [LICENSE](LICENSE) and
[NOTICE](NOTICE). Full text at <https://www.apache.org/licenses/LICENSE-2.0>.

Tulip began as a fork of an earlier project released under the Universal
Permissive License v1.0 (UPL-1.0); those original portions remain available
under the UPL-1.0, while all new contributions are licensed under Apache-2.0.
See [NOTICE](NOTICE) for details.
