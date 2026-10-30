<p align="center">
  <img src="https://locusagents.ai/img/og-card.png" alt="Oracle Generative AI Agentic Harness · Locus SDK · pip install locus-sdk · Built inside Oracle · github.com/oracle/locus" width="720">
</p>

<p align="center">
  <strong>Oracle Generative AI Agentic Harness</strong> · Locus, the <code>locus-sdk</code> package<br>
  <em>Multi-agent workflows built for production · Built inside Oracle · Open source (UPL-1.0)</em>
</p>

<p align="center">
  <strong>One <code>Agent</code> class, eight ways to combine agents, Oracle Cloud underneath. Around the model, the agentic harness;
  around the code an agent writes, an OCI GenAI Sandbox in your tenancy.</strong><br>
  <em>Tools, memory, RAG, observability, guardrails, <code>admit()</code> in front of every consequential action. Pipelines, fan-out,
  orchestrator, swarm, handoff, graphs, debate, A2A. OCI Generative AI on day 0, Dedicated AI Clusters, Oracle Database 26ai.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/locus-sdk/"><img src="https://img.shields.io/pypi/v/locus-sdk.svg?label=PyPI&color=blue" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/Python-3.11%E2%80%933.14-blue.svg" alt="Python 3.11–3.14">
  <img src="https://img.shields.io/badge/License-UPL--1.0-green.svg" alt="License: UPL-1.0">
  <a href="https://github.com/oracle/locus/actions/workflows/ci.yml"><img src="https://github.com/oracle/locus/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <a href="https://locusagents.ai/">Documentation</a> ·
  <a href="https://locusagents.ai/how-to/quickstart/">Quickstart</a> ·
  <a href="https://locusagents.ai/concepts/multi-agent/">Multi-agent shapes</a> ·
  <a href="https://locusagents.ai/how-to/oci-models/">OCI GenAI models</a> ·
  <a href="https://locusagents.ai/concepts/control-layer/">The admission gate</a> ·
  <a href="https://locusagents.ai/notebooks/">Notebooks</a>
</p>

---

## Your first agent

```bash
pip install "locus-sdk[oci,oracledb]"
export OCI_PROFILE=DEFAULT   # a profile in ~/.oci/config with Generative AI access
```

```python
from locus.agent import Agent

agent = Agent(model="oci:openai.gpt-5.5")
print(agent.run_sync("What is the capital of France?").text)
# → Paris
```

Locus runs on Python 3.11 to 3.14. Swap `oci:openai.gpt-5.5` for `openai:gpt-5.5`,
`anthropic:claude-sonnet-5-5` or `openrouter:x-ai/grok-4.7`, each reading its key from the
environment; the call stays the same. With an OCI profile in `~/.oci/config`, the notebooks in
`examples/` run against OCI Generative AI with no further setup.

→ [Quickstart](https://locusagents.ai/how-to/quickstart/) ·
[Notebook 02 — Basic agent](https://locusagents.ai/notebooks/notebook_02_basic_agent/) ·
[Notebook 03 — Agent with tools](https://locusagents.ai/notebooks/notebook_03_agent_with_tools/)

---

## One `Agent`, every shape

Every shape runs the same `Agent`, emits the same events, and checkpoints to the same backends.
Pick the shape; the agents do not change.

| Shape | What it is | Where |
|---|---|---|
| **Pipelines** | `SequentialPipeline`, `ParallelPipeline`, `LoopAgent`: chain, fan out, revise until a reviewer clears | [Composition](https://locusagents.ai/concepts/multi-agent/composition/) |
| **Orchestrator and specialists** | One coordinator delegates to typed specialists with their own tools and merges the results | [Orchestrator](https://locusagents.ai/concepts/multi-agent/orchestrator/) |
| **Swarm** | Peers hand work to each other on their own judgment, shared scratchpad, hard stop | [Swarm](https://locusagents.ai/concepts/multi-agent/swarm/) |
| **Handoff** | Tiered escalation with typed context, L1 to L2 to L3 | [Handoff](https://locusagents.ai/concepts/multi-agent/handoff/) |
| **StateGraph and functional API** | Nodes, edges, reducers, `Send` fan-out (a node scatters work to parallel workers), interrupts for a person, checkpointed state | [StateGraph](https://locusagents.ai/concepts/multi-agent/graph/) |
| **Debate and critic loops** | Two debaters and a judge; a skeptical reviewer that rejects unproven claims before they ship | [Notebook 26](https://locusagents.ai/notebooks/notebook_26_debate_with_judge/) |
| **Cognitive router** | Typed `GoalFrame` from the task, one of eight protocols, compiled onto real agents; a request too risky for its shape is held for approval, not refused | [Notebook 29](https://locusagents.ai/notebooks/notebook_29_cognitive_router/) |
| **Agent-to-agent (A2A) and DeepAgent** | Cross-process agents as spec-compliant peers; plan, explore, validate, report with grounding | [A2A](https://locusagents.ai/concepts/multi-agent/a2a/) · [DeepAgent](https://locusagents.ai/concepts/deepagent/) |

```python
import asyncio

from locus.agent import Agent
from locus.agent.composition import SequentialPipeline

model = "oci:openai.gpt-5.5"
diagnose = Agent(model=model, tools=[query_metrics, search_logs], system_prompt="Find what changed.")
mitigate = Agent(model=model, tools=[scale_up, rollback], system_prompt="Fix it, smallest step.")
brief = Agent(model=model, system_prompt="Write the status update.")

war_room = SequentialPipeline(agents=[diagnose, mitigate, brief])
result = asyncio.run(war_room.run("checkout p99 hit 4.2s: fix and brief"))
print(result.final_output)
```

The cognitive router picks the shape from the task: a `Router` extracts a typed `GoalFrame`,
selects one of the eight registered protocols and compiles it onto real agents (Notebook 29
builds one):

```python
result = await router.dispatch("Diagnose the checkout API slowdown.")
print(result.protocol_id)   # "specialist_fanout"
```

→ [Multi-agent overview](https://locusagents.ai/concepts/multi-agent/) ·
[Notebook 19 — Composition](https://locusagents.ai/notebooks/notebook_19_composition/) ·
[Notebook 22 — Orchestrator](https://locusagents.ai/notebooks/notebook_22_orchestrator_pattern/) ·
[Notebook 29 — Cognitive router](https://locusagents.ai/notebooks/notebook_29_cognitive_router/)

---

## Oracle Cloud, native

OCI Generative AI and Oracle Database are built into the runtime: same `Agent`, same events, same
admission gate.

| Primitive | What it gives you |
|---|---|
| **OCI Generative AI** | `model="oci:…"` picks the transport from the model id: the OpenAI-compatible V1 endpoint for the OpenAI, Meta, xAI, Mistral and Google families, the OCI SDK for Cohere and for Dedicated AI Cluster endpoints; the Responses transport is opt-in through `OCIResponsesModel`. A model OCI adds is a new string, with no Locus release needed (day 0). Auth from your OCI profile, a session token, or instance / resource principals. |
| **Dedicated AI Clusters** | `model="oci:ocid1.generativeaiendpoint…"` targets your own Dedicated AI Cluster (DAC) endpoint. |
| **OCI GenAI Sandboxes** | `OCISandboxBackend` runs the coding harness inside an isolated, expiring Linux workspace in your tenancy: the model's shell commands and file edits never touch the host, and every call still clears the gate. `pip install "locus-sdk[oci-sandbox]"`. |
| **OCI tools** | `use_oci()` / `describe_oci()` put the OCI control plane behind tools, so an agent can list, inspect and act on your tenancy. |
| **Oracle Database 26ai RAG** | `OracleVectorStore` on native `VECTOR(N, FLOAT32)` columns, `VECTOR_DISTANCE`, HNSW / IVF, hybrid search; `OracleInDBChunker` and `OracleInDBEmbeddings` so text never leaves the database; `OracleADBLoader` to ingest from ADB tables. |
| **26ai checkpoints and history** | `oracle_checkpointer` for durable threads, `OracleCheckpointSaver` for versioned history and time travel; `OCIBucketBackend` on Object Storage as the alternative. |
| **26ai long-term memory** | `OracleStore` for cross-thread memory and `OracleAgentMemoryManager` over Oracle's official `oracleagentmemory` client. |

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

→ [OCI GenAI models](https://locusagents.ai/how-to/oci-models/) ·
[Dedicated AI Clusters](https://locusagents.ai/how-to/oci-dac/) ·
[Oracle Database 26ai](https://locusagents.ai/concepts/oracle-26ai/) ·
[OCI GenAI Sandboxes](https://locusagents.ai/harness/oci-sandbox/) ·
[Notebook 01 — OCI transports](https://locusagents.ai/notebooks/notebook_01_oci_transports/) ·
[Notebook 38 — Oracle 26ai RAG](https://locusagents.ai/notebooks/notebook_38_oracle_26ai_rag/) ·
[Notebook 42 — Oracle 26ai checkpointer](https://locusagents.ai/notebooks/notebook_42_oracle_26ai_checkpointer/) ·
[Notebook 44 — Oracle agent memory](https://locusagents.ai/notebooks/notebook_44_oracle_agent_memory/)

---

## The harness

| Layer | What it gives you |
|---|---|
| **[The coding harness](https://locusagents.ai/harness/)** | The toolset a coding agent works with (read, write, edit, grep, a shell), written once against a workspace protocol and run unchanged on the host, inside an [OCI GenAI Sandbox](https://locusagents.ai/harness/oci-sandbox/) in your tenancy or inside [NVIDIA OpenShell](https://locusagents.ai/harness/openshell/); one `wrap` gates every call. `pip install "locus-sdk[oci-sandbox]"` adds the sandbox client. |
| **[Tools and MCP](https://locusagents.ai/concepts/tools/)** | `@tool` on any function; arguments coerced to the declared types whatever the model sends; `@tool(idempotent=True)` dedupes on `(name, args)`; `MCPClient` consumes Model Context Protocol (MCP) servers and `LocusMCPServer` exposes yours. |
| **[Memory](https://locusagents.ai/concepts/memory/)** | Checkpoint backends for Oracle Database 26ai (with its versioned saver), OCI Object Storage, PostgreSQL, MySQL, Redis, OpenSearch, S3-compatible object stores, HTTP, a local file, in-memory; `OracleStore`, Oracle agent memory, the portable `LLMMemoryManager` for long-term memory. |
| **[RAG](https://locusagents.ai/concepts/rag/)** | Six vector stores (Oracle 26ai, pgvector, Qdrant, Chroma, OpenSearch, in-memory); OCI, in-database, OpenAI and Cohere embeddings; OCI Cohere, local and Cohere rerankers; `retriever.as_tool()`. |
| **[Grounded reasoning](https://locusagents.ai/concepts/gsar/)** | Grounding-Stratified Adaptive Replanning (GSAR) partitions every claim as grounded, ungrounded, contradicted or complementary against typed evidence; below threshold the agent regenerates, replans or abstains. Reflexion, when enabled, scores every turn: the agent assesses its own progress from the tool results and adjusts. |
| **[Guardrails](https://locusagents.ai/concepts/safety/) and [the admission gate](https://locusagents.ai/concepts/control-layer/)** | Input, output and tool-call guardrails, steering (LLM-as-judge), trace hygiene, and `admit()` / `approve()`: allow, hold for a human, or deny, on a hash-chained audit trail. |
| **[Observability and streaming](https://locusagents.ai/concepts/observability/)** | Opt-in `EventBus`; one `run_context()` streams typed events from every layer; Server-Sent Events (SSE) and token streaming; OpenTelemetry hook. |
| **[Server](https://locusagents.ai/concepts/server/) and [deploy](https://locusagents.ai/how-to/deploy/)** | `locus serve` / `AgentServer` (FastAPI, bearer auth, `POST /invoke`, `POST /stream`, threads); a multi-stage `Dockerfile` for OCI Kubernetes Engine (OKE) or any container host; the [box runner](https://locusagents.ai/runner/) (`python -m locus.runner`) runs one agent's loop inside an NVIDIA OpenShell sandbox for a Locus gateway. |
| **[Skills](https://locusagents.ai/concepts/skills/), [playbooks](https://locusagents.ai/concepts/playbooks/), [evaluation](https://locusagents.ai/concepts/evaluation/), [testing](https://locusagents.ai/concepts/testing/)** | Progressive-disclosure skills, enforced playbooks, `EvalCase` / `EvalRunner` / `EvalReport` regression suites; `locus.testing` model doubles (`ScriptedModel`, `FunctionModel`) and `AgentTestClient` for deterministic tests with no API key. |
| **[Model providers](https://locusagents.ai/how-to/configure-providers/)** | `oci:`, `openai:`, `anthropic:` native; `gemini:`, `openrouter:`, `together:`, `runpod:`, `groq:`, `ollama:`, `vllm:` and 11 more OpenAI-compatible routes; failover chains, credential pooling, prompt-cache breakpoints. |

Every backend is an optional extra — install only what you use: `pip install "locus-sdk[oci,oracledb]"`
for the Oracle stack, `[oci-sandbox]` for OCI GenAI Sandboxes, `[openshell]` for NVIDIA OpenShell,
`[server]` for `AgentServer`, `[mcp]` for MCP, `[qdrant,s3,rerank-local]` and the other extras in
`pyproject.toml` as needed. A coding agent in an OCI GenAI Sandbox is one backend away from the
same harness that runs on your laptop:

```python
from locus.agent import Agent
from locus.control import ControlPolicy, gate_tool
from locus.harness import OCISandboxBackend, build_harness

backend = OCISandboxBackend.create("ocid1.generativeaiproject.oc1...", expiration="PT1H")
policy = ControlPolicy(
    require_verification_score=0.0, require_human_for=frozenset({"workspace.exec"})
)
harness = build_harness(
    backend, wrap=lambda tool, spec: gate_tool(tool, policy=policy, action=spec)
)
agent = Agent(
    model="oci:openai.gpt-5.5", tools=harness.tools, system_prompt=harness.prompt_fragment
)
print(agent.run_sync("Add pytest tests for calc/ and run them.").text)
backend.close()   # the sandbox is stopped and deleted
```

---

## Control, when you need it

A prompt rule is advisory; `admit()` is structural: the gate wraps the tool, so the side-effecting
call runs only after it clears a `ControlPolicy` you wrote, whatever the model asks for.

```python
from locus.control import governed_agent

agent = governed_agent(model="oci:openai.gpt-5.5", tools=[...])
print(agent.run_sync("Refund order 8812 if the return was received.").text)
assert agent.audit_trail.verify()   # the chain is intact
```

```bash
pip install locus-sdk && python -m locus.rogue
# an interactive challenge: talk the agent into wipe_database and watch the gate hold
#    🛡️  wipe_database → deny: labels ['irreversible'] are denied by policy; …
# FINAL SCORE  ·  dangerous attempts: 1  ·  💥 BREACHES: 0  ·  audit chain intact: ✓
```

Put the same gate around an agent loop you already have, whatever it is built on, or build on the
full runtime.

→ [How the admission gate works](https://locusagents.ai/concepts/control-layer/) ·
[First controlled action](https://locusagents.ai/how-to/first-controlled-action/) ·
[Notebook 71 — Payment refund gate](https://locusagents.ai/notebooks/notebook_71_payment_refund_gate/)

---

## Notebooks and deploy

The notebooks are numbered in reading order: start here (01–11), multi-agent shapes (12–32),
Oracle Cloud and Database 26ai (33–45), the harness (46–59), observability, streaming and
serving (60–67), control layer and security (68–76) and real-world workflows (77–80) live in
[`examples/`](https://github.com/oracle/locus/tree/main/examples); the security-operations notebooks (81–89) ship with the
opt-in [`locus-sdk-security`](https://github.com/oracle/locus/tree/main/packages/locus-sdk-security) package, installed from
the repository at the matching tag. Notebooks 01–80 run against OCI Generative AI with an OCI
profile, or the bundled mock model without one; 81–89 run on the mock model, or on OpenAI or
Anthropic through `LOCUS_MODEL_PROVIDER`.

```bash
git clone https://github.com/oracle/locus.git
cd locus && pip install -e ".[oci,oracledb]"

python examples/notebook_01_oci_transports.py        # the three OCI GenAI transports
python examples/notebook_02_basic_agent.py           # your first agent
python examples/notebook_19_composition.py           # pipelines
python examples/notebook_29_cognitive_router.py      # the cognitive router
python examples/notebook_38_oracle_26ai_rag.py       # RAG on Oracle Database 26ai
```

To serve an agent, `AgentServer` wraps it as a FastAPI app (`pip install "locus-sdk[server]"`) and
the repo ships a multi-stage `Dockerfile`:

```python
import os

from locus.server import AgentServer

AgentServer(agent=my_agent, api_key=os.environ["API_KEY"]).run(host="0.0.0.0", port=8080)
```

→ [Notebooks](https://locusagents.ai/notebooks/) · [Deploy](https://locusagents.ai/how-to/deploy/)

---

## Build it with Codex

[OpenAI Codex](https://developers.openai.com/codex) builds on Locus. The repository ships a Codex
skill, [`.agents/skills/locus/`](https://github.com/oracle/locus/tree/main/.agents/skills/locus), that
teaches it the SDK: `Agent` and `@tool`, the admission gate, the multi-agent shapes, memory and RAG,
the coding harness, the providers, and how to test an agent offline. `locus codex` puts the skill
where Codex looks and starts Codex with the sandbox settings it needs:

```bash
pip install "locus-sdk[oci]"
locus codex run "Build a support agent with an order lookup and a gated refund tool, with tests"
```

In a clone, [`codex/PROMPT.md`](https://github.com/oracle/locus/blob/main/codex/PROMPT.md) drives
Codex through a first agent with a typed tool, an admission gate allowing, holding and denying, and
the rogue-agent challenge, on OCI Generative AI, OpenAI, Anthropic or offline;
[`codex/SPEC.md`](https://github.com/oracle/locus/blob/main/codex/SPEC.md) is written for a person
to follow by hand as well:

```bash
git clone https://github.com/oracle/locus.git && cd locus
codex -s workspace-write -c sandbox_workspace_write.network_access=true "$(cat codex/PROMPT.md)"
```

Codex itself can run on OCI Generative AI:
[`examples/codex-on-oci/`](https://github.com/oracle/locus/tree/main/examples/codex-on-oci) is a
loopback gateway that signs Codex's calls with your OCI profile, so neither Codex nor the agent it
builds needs an account outside Oracle Cloud.

[`codex/scenarios/`](https://github.com/oracle/locus/tree/main/codex/scenarios) is a series of twelve
tasks Codex builds from the skill alone, each with an automatic check, and the runner that executes
them: the proof that the skill produces correct applications, and the way a change to the SDK or to
the skill is tried before a release.

→ [Build it with Codex](https://locusagents.ai/how-to/codex/)

---

## Repo layout

```text
src/locus/
├── agent/          Agent runtime, config, Sequential / Parallel / Loop pipelines
├── multiagent/     Orchestrator, Swarm, Handoff, StateGraph, Functional
├── router/         Cognitive router — GoalFrame, ProtocolRegistry, PolicyGate, CognitiveCompiler
├── a2a/            Cross-process Agent-to-Agent protocol
├── deepagent/      create_deepagent + create_research_workflow + node primitives
├── core/           AgentState, Message, events, termination algebra, Send
├── models/         Provider registry + OCI GenAI (V1 / SDK / Responses), OpenAI, Anthropic
├── memory/         BaseCheckpointer + backends (Oracle 26ai, OCI Object Storage, …), stores, memory managers
├── rag/            Embeddings + 6 vector stores (Oracle 26ai, …) + rerankers + loaders + chunkers
├── tools/          @tool decorator, registry, builtins, executors, OCI tools
├── harness/        Coding harness: workspace tools over LocalBackend, OCISandboxBackend, OpenShellBackend; build_harness(wrap=…)
├── reasoning/      Reflexion, Grounding, Causal, GSAR
├── control/        Admission gate, policy, audit trail, grounded findings, verification
├── decision/       Typed decisions (choice, yes/no, score) with probabilities
├── hooks/          Logging, telemetry, retry, guardrails, steering
├── observability/  EventBus, run_context, agent yield bridge, EV_* constants
├── skills/         AgentSkills.io filesystem-first capability disclosure
├── playbooks/      Declarative step plans + PlaybookEnforcer
├── providers/      Multi-modal: web search, web fetch, image, speech
├── server/         FastAPI AgentServer with thread persistence
├── evaluation/     EvalCase + EvalRunner + EvalReport
├── integrations/   MCP (client + server)
├── durable/        Durable execution for agent runs (Temporal, DBOS)
├── streaming/      Stream handlers: console, SSE, structured
├── voice/          Realtime speech-to-speech sessions; tool calls go through the gate
├── runner/         Box runner (`python -m locus.runner`): one agent's loop in an NVIDIA OpenShell sandbox
├── cli.py          The `locus` command line: run, resume, approvals, audit verify, serve
├── testing.py      Model doubles (ScriptedModel, FunctionModel) and AgentTestClient
└── rogue/          The rogue-agent challenge (`python -m locus.rogue`)
```

The docs site lives in [`docs/`](https://github.com/oracle/locus/tree/main/docs) (`mkdocs.yml`, theme in `overrides/`, build
hooks in `hooks/`) and publishes to [locusagents.ai](https://locusagents.ai/). The opt-in security
package lives in [`packages/locus-sdk-security/`](https://github.com/oracle/locus/tree/main/packages/locus-sdk-security); it is
not on PyPI and installs from the repository at the matching tag (see its
[README](https://github.com/oracle/locus/blob/main/packages/locus-sdk-security/README.md)).

---

## Contributing

```bash
git clone https://github.com/oracle/locus.git
cd locus && pip install -e ".[dev,sdk]"
hatch run check        # ruff format-check, ruff lint, mypy, unit tests under the coverage gate
hatch run test         # unit tests on the active Python (CI runs them on 3.11–3.14)
pre-commit install
```

See [CONTRIBUTING.md](https://github.com/oracle/locus/blob/main/CONTRIBUTING.md). Contributions need the
[Oracle Contributor Agreement](https://oca.opensource.oracle.com); every PR runs format,
lint, mypy, unit tests and the OCA check. Questions and bug reports go to
[GitHub Issues](https://github.com/oracle/locus/issues); [CHANGELOG.md](https://github.com/oracle/locus/blob/main/CHANGELOG.md) lists what changed in
each release. Please consult the [security guide](https://github.com/oracle/locus/blob/main/SECURITY.md) for vulnerability
disclosure.

---

## GSAR reference

Locus's GSAR layer is based on the method described in F. A. Kamelhar,
[GSAR: Typed Grounding for Hallucination Detection and Recovery in Multi-Agent LLMs](https://arxiv.org/abs/2604.23366)
([PDF](https://arxiv.org/pdf/2604.23366)), arXiv:2604.23366, 2026. The paper is separate from this
project; if you use the method, cite it directly:

```bibtex
@article{kamelhar2026gsar,
  author  = {Kamelhar, Federico A.},
  title   = {GSAR: Typed Grounding for Hallucination Detection and Recovery in Multi-Agent LLMs},
  journal = {arXiv preprint arXiv:2604.23366},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.23366},
}
```

---

## License

Copyright (c) 2025, 2026 Oracle and/or its affiliates.

Released under the **Universal Permissive License v1.0** — see [LICENSE](https://github.com/oracle/locus/blob/main/LICENSE) and
[THIRD_PARTY_LICENSES.txt](https://github.com/oracle/locus/blob/main/THIRD_PARTY_LICENSES.txt). Releases `1.0.0` – `2.28.0` were
published as `tulip-agents` under the Apache License 2.0 while the project was developed
outside Oracle; they remain available under that license from their tags. See
[PROVENANCE.md](https://github.com/oracle/locus/blob/main/PROVENANCE.md).
