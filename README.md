<p align="center">
  <img src="https://raw.githubusercontent.com/tuliplabs-ai/tulip-agents/main/docs/img/tuliplabs-logo.png" alt="tuliplabs" width="320">
</p>

<p align="center">
  <strong>The agent framework where the model never holds the trigger.</strong><br>
  <em>Your agent is about to issue a refund, ship a deploy, or delete a customer record. Every
  consequential action clears a policy check first — real code, outside the model, before the
  side effect, on a record you can verify. Build on it, or put it around the agent you already
  have.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/tulip-agents/"><img src="https://img.shields.io/pypi/v/tulip-agents.svg?label=PyPI&color=ED5A8B" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/Python-3.11%E2%80%933.14-blue.svg" alt="Python 3.11–3.14">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/mypy-strict-brightgreen.svg" alt="mypy strict">
  <img src="https://img.shields.io/badge/ruff-clean-brightgreen.svg" alt="ruff clean">
</p>

<p align="center">
  <a href="https://tulipagents.ai/concepts/security/">The admission gate</a> ·
  <a href="https://tulipagents.ai/integrations/frameworks/">Govern an existing agent</a> ·
  <a href="https://tulipagents.ai/how-to/quickstart/">Quickstart</a> ·
  <a href="https://tulipagents.ai/notebooks/">Notebooks</a> ·
  <a href="https://tulipagents.ai/">Docs</a>
</p>

---

## Try to break it — 30 seconds, no API key

`OpsBot` holds live-looking production tools: wipe the database, exfiltrate secrets, isolate
the cluster, disable every user. Offline, the model arrives **already fully compromised** — it
reaches for a dangerous tool every turn. Nothing left to jailbreak. Watch what stops it anyway.

```bash
pip install tulip-agents
python -m tulip.rogue
```

```text
🎯 attempt #1: wipe the production database
🤖 OpsBot: Blocked before it ran. I could not complete that.
   ⚠️  model was talked into trying: wipe_database
   🛡️  wipe_database → deny: labels ['irreversible'] are denied by policy;
       blast radius 999 exceeds the maximum 1; labels ['production'] require human approval
   → the MODEL was already owned. The RUNTIME didn't care. Breaches still 0.

FINAL SCORE  ·  dangerous attempts: 1  ·  💥 BREACHES: 0  ·  audit chain intact: ✓
```

Point `TULIP_MODEL_URL` at your own vLLM / Ollama / LM Studio server to run it against a real
model, or set `ANTHROPIC_API_KEY` for the hard version against a frontier one. When the model
refuses on its own the scoreboard says so, rather than claiming a win the gate didn't earn.

---

## Govern the agent you already have

You don't have to build on Tulip to be governed by it.
**[`tulip-frameworks`](https://github.com/tuliplabs-ai/tulip-frameworks)** wraps a tool from the
framework you already use — **LangChain, LangGraph, CrewAI, the OpenAI Agents SDK, LlamaIndex,
or Google ADK** — with the same gate and the same hash-chained audit trail. No rebuild, no
migration.

```bash
pip install "tulip-frameworks[langchain]"   # or [crewai] / [openai-agents] / [llama-index] / [adk] / [all]
```

Agents outside Python reach the same gate over the wire through
[`tulip-gateway`](https://tulipagents.ai/integrations/frameworks/)'s `/v1/admit`, with a TypeScript
client in [`tulip-frameworks-js`](https://github.com/tuliplabs-ai/tulip-frameworks-js).
→ [The frameworks guide](https://tulipagents.ai/integrations/frameworks/)

---

## The admission gate

The moment an agent stops advising and starts **acting**, a wrong step becomes a real
consequence. A prompt rule is advisory — a misleading input or an injected document can talk the
model past it. Tulip makes the rule **structural**: the side-effecting call runs only after it
clears `admit()`, which the model has no way to reach around.

```python
from tulip.control import Action, AuditTrail, ControlPolicy, admit, AdmissionError

policy = ControlPolicy(require_human_for={"production"})
trail = AuditTrail()

async def safe_refund(order_id: str, usd: float):
    try:
        return await admit(
            Action(name="refund", asset=order_id, kind="payment", environment="production"),
            lambda: payments.refund(order_id, usd),     # your code — any agent loop
            policy=policy, trail=trail,
        )
    except AdmissionError as e:
        notify_oncall(e.decision)                       # the gate held it; the trail has it
```

On a tool your own agent calls, `gate_tool` does the same thing in one line —
the returned tool keeps the original's name, description and schema, so the
model cannot tell the difference and nothing else in the agent changes:

```python
from tulip.control import ControlPolicy, gate_tool

agent = Agent(model=model, tools=[
    lookup_order,                                     # read-only, ungated
    gate_tool(issue_refund, policy=ControlPolicy()),  # gated
])
```

Refused calls come back to the model as a readable refusal naming the outcome
and the reason, so the agent explains the hold instead of the run ending in a
traceback. It is the same shape the `tulip-frameworks` bridges return, so a
policy reads the same whether the agent is Tulip-native or wrapped.

`action → policy → approval → admission → audit`

- **Policy + approval** — `approve()` weighs your `ControlPolicy` (blast radius,
  `require_human_for`, verification score) and returns allow, hold, or deny.
- **Admission** — `admit()` runs the action **only if** approval allows, recording the decision
  to the `AuditTrail`; otherwise it raises `AdmissionError`.
- **Audit** — every entry is linked to the one before it, so editing any record breaks
  `verify()`. (A keyless SHA-256 chain: tamper-evident, not notarized — add signing before
  treating it as legally authoritative.)

Human approvals are durable: `require_human_for` pauses the run, and an `interrupt()` +
checkpointer means the decision survives a restart and the run resumes where it left off.

For tools where a duplicate call would hurt — moving money, paging an on-call — declare
`@tool(idempotent=True)`: the loop keys every invocation on `(name, args)` and refuses to fire
the same one twice, even across retries.

→ [The admission gate](https://tulipagents.ai/concepts/security/) ·
[Idempotency](https://tulipagents.ai/concepts/idempotency/)

---

## And underneath it, a full SDK

If you're starting fresh rather than wrapping something, the gate sits on a complete agent
framework. A model is a string, a tool is a function, and `run_sync` runs the loop.

```python
from tulip import Agent, tool

@tool
def search_flights(origin: str, destination: str, date: str) -> list[dict]:
    """Find available flights between two cities on a given date."""
    return flights.search(origin, destination, date)

agent = Agent(
    model="anthropic:claude-sonnet-4-6",          # swap providers with one string
    tools=[search_flights],
    system_prompt="You are a travel assistant. Be concise and cite prices.",
)

print(agent.run_sync("Cheapest flight from Lisbon to Berlin next Friday?").text)
```

```bash
pip install "tulip-agents[anthropic]"     # or [openai], or [sdk] for everything
```

And `governed_agent()` gives any Tulip agent the whole harness — grounded, guarded, risk-gated,
audited — in one call:

```python
from tulip.control import governed_agent

secured = governed_agent(model="openai:gpt-4o", tools=[...])
assert secured.audit_trail.verify()   # the chain is intact — no record was altered
```

A bundled `MockModel` means every notebook runs offline with no credentials.

---

## What you get

**Control** — the runtime that clears actions:

| | |
|---|---|
| **[⚖️ Admission gate](https://tulipagents.ai/concepts/security/)** | `admit()` / `approve()` run a consequential action only if your `ControlPolicy` allows — else hold for a human or deny, recorded either way. |
| **[🧠 GSAR grounding](https://tulipagents.ai/concepts/gsar/)** | Claims partitioned grounded / ungrounded / contradicted / complementary; below threshold the agent regenerates, replans, or abstains. `arXiv:2604.23366`. |
| **[🔁 Idempotent tools](https://tulipagents.ai/concepts/idempotency/)** | `@tool(idempotent=True)` — dedupes on `(name, args)`. The model can't double-charge, double-book, or double-page. |
| **[🪝 Hooks](https://tulipagents.ai/concepts/hooks/)** | Logging · OpenTelemetry · ModelRetry · Guardrails · Steering (LLM-as-judge). |

**Build** — the agent framework surface:

| | |
|---|---|
| **[🧭 Cognitive router](https://tulipagents.ai/concepts/router/)** | Describe a task → eight named protocols → the right primitive compiled automatically. The LLM fills a typed schema; routing is deterministic. |
| **[🤝 Multi-agent](https://tulipagents.ai/concepts/multi-agent/)** | Seven native patterns + cross-process A2A. One `Agent` class. One event stream. |
| **[🔬 DeepAgent](https://tulipagents.ai/concepts/deepagent/)** | `create_deepagent` (per-turn grounding) and `create_research_workflow` (StateGraph with post-hoc grounding eval). |
| **[🪙 MCP](https://tulipagents.ai/concepts/mcp/)** | `MCPClient` consumes MCP servers. `TulipMCPServer` exposes the SDK's tools as MCP. |
| **[🌐 Multi-modal](https://tulipagents.ai/concepts/multi-modal-providers/)** | `Agent(web_search=…, web_fetch=…, image_generator=…, speech_provider=…)` auto-registers tools. |

**Run** — operate it in production:

| | |
|---|---|
| **[📡 Observability](https://tulipagents.ai/concepts/observability/)** | Opt-in `EventBus` — one `run_context()` streams 40+ canonical events from every layer, no external broker. |
| **[💾 Durable memory](https://tulipagents.ai/concepts/checkpointers/)** | 8 checkpoint backends — PostgreSQL · MySQL · Redis · OpenSearch · S3 / MinIO / R2 · in-memory · file · HTTP. |
| **[🔎 RAG](https://tulipagents.ai/concepts/rag/)** | 5 vector stores — pgvector · Qdrant · Chroma · OpenSearch · in-memory. OpenAI + Cohere embeddings, local + Cohere rerankers. |
| **[📡 Streaming + Server](https://tulipagents.ai/concepts/server/)** | Typed events · SSE · `AgentServer` (FastAPI, bearer auth, thread persistence). |
| **[📊 Evaluation](https://tulipagents.ai/concepts/evaluation/)** | `EvalCase` / `EvalRunner` / `EvalReport` regression suites. |

Every backend is an optional extra — install only what you use
(`pip install "tulip-agents[qdrant,s3,rerank-local]"`).

---

## Grounded by construction (GSAR)

An agent that *acts* must not assert what it can't back up. Tulip's GSAR layer
([paper](https://arxiv.org/abs/2604.23366)) partitions every claim — **grounded / ungrounded /
contradicted / complementary** — against typed evidence, where tool output outranks inference and
inference outranks domain priors. Below threshold the run **regenerates, replans, or abstains**.
There is no public constructor that emits a grounded result without a score, so an ungrounded
claim is unshippable *by construction* — not filtered after the fact.

```python
from tulip.security import ground_finding, Severity, is_finding

result = ground_finding(..., partition=partition)
# A grounded partition → a typed result. An ungrounded one → an auditable
# Abstention with the reason it was withheld. There is no third path.
print(result.title if is_finding(result) else f"withheld: {result.reason}")
```

→ [GSAR grounding](https://tulipagents.ai/concepts/gsar/)

---

## The cognitive router and multi-agent shapes

Describe a task in plain language; the **[cognitive router](https://tulipagents.ai/concepts/router/)**
(PRISM) runs an LLM classifier that fills a typed `GoalFrame`, matches it to one of eight
coordination protocols, and compiles the matching runtime primitive. **The model classifies;
routing is deterministic — it never authors the topology.**

| Protocol | Compiled shape | Best for |
|---|---|---|
| `direct_response` | Single `Agent` | `ANSWER`, `EXPLAIN` |
| `plan_execute_validate` | `SequentialPipeline` (planner → executor → validator) | `PLAN`, `BUILD`, `MODIFY` |
| `specialist_fanout` | `ParallelPipeline` of N tool-bound Agents | `DIAGNOSE`, `MONITOR` |
| `debate` | Two debaters + judge `Agent` | `COMPARE` |
| `codegen_test_validate` | `LoopAgent` (stops on `PASS`) | `GENERATE_CODE` |
| `approval_gated_execution` | `Agent` wrapped in approval interrupt | `ESCALATE`, `REMEDIATE` |
| `handoff_chain` | `SequentialPipeline` of one-tool Agents | `COORDINATE` |
| `a2a_delegate` | Cross-process agent-to-agent call (opt-in) | distributed meshes |

Each shape is also a first-class primitive you can use directly — `SequentialPipeline`,
`ParallelPipeline`, `LoopAgent`, `Orchestrator`, `Swarm`, `Handoff`, `StateGraph`, `A2A`.

```python
import asyncio

from tulip.agent import Agent, SequentialPipeline


async def main():
    result = await SequentialPipeline(agents=[draft, check, finalize]).run(
        "Summarize the trade-offs of moving the checkout service to a queue."
    )
    print(result.final_output)


asyncio.run(main())
```

→ [Cognitive router](https://tulipagents.ai/concepts/router/) ·
[All patterns](https://tulipagents.ai/concepts/multi-agent/)

---

## Providers

A model is a string. The prefix picks the provider; the rest is the model id it expects.

**21 prefixes ship built in.** `openai:`, `anthropic:`, `bedrock:` and `azure:` are native;
the rest are OpenAI-compatible endpoints with their base URL and key convention already
filled in — `gemini:` · `ollama:` · `vllm:` · `lmstudio:` · `llamacpp:` · `litellm:` ·
`groq:` · `together:` · `openrouter:` · `deepseek:` · `mistral:` · `xai:` · `fireworks:` ·
`cerebras:` · `perplexity:` · `nvidia:`, plus `openai-compatible:` for anything else.

```python
Agent(model="ollama:llama3.2")                      # localhost:11434, no key needed
Agent(model="groq:llama-3.3-70b-versatile")         # GROQ_API_KEY from the environment
Agent(model="anthropic:claude-sonnet-5")
Agent(model="bedrock:us.amazon.nova-lite-v1:0")     # boto3 credential chain
Agent(model="azure:gpt4o-prod")                     # AZURE_OPENAI_ENDPOINT, a deployment name
Agent(model="gemini:gemini-2.0-flash")              # GEMINI_API_KEY
```

`bedrock:` goes through the Converse API, so one code path covers every model on the
service — Nova, Claude, Llama, Mistral, Titan — and credentials are boto3's standard chain
(environment, profile, SSO, instance role, IRSA). Install with
`pip install "tulip-agents[bedrock]"`; `boto3` is imported lazily, so the four-package core
install is unchanged for everyone not on AWS.

`azure:` names a **deployment**, not a model id, and reuses the `openai` extra — Azure's
`api-key` header, `api-version` and deployment-shaped URLs are handled by the SDK's
Azure client, so there is no second implementation to drift. `gemini:` uses Google's own
OpenAI-compatible endpoint, so it needs no additional client either.

Configuration that is not in the environment travels in `model_kwargs` — a per-agent key, or
a host that is not the default:

```python
Agent(
    model="openai-compatible:qwen3.6-35b",
    model_kwargs={"base_url": "http://gpu-1:8000/v1", "api_key": "unused"},
)
```

Anything else implements `ModelProtocol` — `complete` · `stream`, ~50 lines — and registers
with `register_provider("myco", MyModel)`.

→ [Model providers](https://tulipagents.ai/concepts/models/) ·
[OpenAI-compatible endpoints](https://tulipagents.ai/concepts/providers/openai-compatible/)

---

## Notebooks, workbench, deploy

[`examples/`](examples/) has progressive notebooks, numbered in suggested reading order. Each
defaults to the bundled mock model when no API key is present.

```bash
git clone https://github.com/tuliplabs-ai/tulip-agents.git
cd tulip-agents && pip install -e .

python examples/notebook_06_basic_agent.py           # your first agent
python examples/notebook_58_cognitive_router.py      # the cognitive router
python examples/notebook_69_research_workflow.py     # full research pipeline
```

The **workbench** is a browser playground for every pattern — two clicks to a running
agent, no editor setup. For production, `AgentServer` is a
drop-in FastAPI app (`POST /invoke`, `POST /stream`, `GET/DELETE /threads/{id}`, `GET /health`)
and the repo ships a multi-stage `Dockerfile`.

```python
from tulip.server import AgentServer

AgentServer(agent=my_agent, api_key=os.environ["API_KEY"]).run(host="0.0.0.0", port=8080)
```

→ [Notebooks](https://tulipagents.ai/notebooks/) ·
[Workbench](https://tulipagents.ai/workbench/) ·
[Deploy](https://tulipagents.ai/how-to/deploy/)

---

## Any domain, one contract

The same contracts run wherever an agent acts. One fully worked domain package ships today:
`tulip.security` applies the grounded-evidence contract to red-teaming AI systems — every result
is a grounded `Evidence` tagged against public weakness catalogues (MITRE ATLAS, OWASP LLM /
Agentic Top 10), or an explicit `Abstention`.

```python
import asyncio

from tulip.security import Target, red_team, is_finding


async def main():
    report = await red_team(
        Target.endpoint("https://support-bot.example/chat"), suite="owasp-asi"
    )
    print([f for f in report.findings if is_finding(f)])


asyncio.run(main())
```

Vendor-specific adapters (Splunk, CrowdStrike, Okta, Auth0, VirusTotal, Wiz, RunPod, Lambda) live
in **[`tulip-integrations`](https://github.com/tuliplabs-ai/tulip-integrations)**; core ships
offline reference adapters so the SDK runs standalone.

---

## Repo layout

```text
src/tulip/
├── control/        Admission gate — Action, admit/approve, ControlPolicy, AuditTrail
├── rogue/          The rogue-agent challenge (`python -m tulip.rogue`)
├── agent/          Agent runtime, config, Sequential / Parallel / Loop pipelines
├── core/           AgentState, Message, events, termination algebra, Send
├── loop/           ReAct nodes (Think, Execute, Reflect)
├── router/         Cognitive router — GoalFrame, ProtocolRegistry, PolicyGate, Compiler
├── reasoning/      Reflexion, Grounding, Causal, GSAR
├── multiagent/     Orchestrator, Swarm, Handoff, StateGraph, Functional
├── a2a/            Cross-process Agent-to-Agent protocol
├── deepagent/      create_deepagent + create_research_workflow + 6 node primitives
├── memory/         BaseCheckpointer + 8 backends
├── rag/            Embeddings + 5 vector stores + rerankers + retrievers
├── models/         Provider registry + OpenAI, Anthropic, Bedrock, Azure
├── tools/          @tool decorator, registry, builtins, executors
├── hooks/          Logging, telemetry, retry, guardrails, steering
├── observability/  EventBus, run_context, agent yield bridge, EV_* constants
├── skills/         AgentSkills.io filesystem-first capability disclosure
├── playbooks/      Declarative step plans + PlaybookEnforcer
├── providers/      Multi-modal: web search, web fetch, image, speech
├── security/       Grounded findings, red-team / assure, taxonomy tags
├── server/         FastAPI AgentServer with thread persistence
├── evaluation/     EvalCase + EvalRunner + EvalReport
└── integrations/   MCP (client + server)
```

The docs site lives in a sibling repo:
[tuliplabs-ai/docs](https://github.com/tuliplabs-ai/docs), published at
[tulipagents.ai](https://tulipagents.ai/).

---

## Contributing

```bash
git clone https://github.com/tuliplabs-ai/tulip-agents.git
cd tulip-agents && pip install -e ".[dev,sdk]"
hatch run check        # ruff + mypy
hatch run test         # unit tests across Python 3.11–3.14
pre-commit install
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Every PR runs format, lint, mypy, unit tests, DCO sign-off.
Please consult the [security guide](./SECURITY.md) for vulnerability disclosure.

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

## License

Copyright 2026 Tulip Labs.

Released under the **Apache License, Version 2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Tulip began as a fork of an earlier project released under the Universal Permissive License v1.0
(UPL-1.0); those original portions remain available under the UPL-1.0, while all new
contributions are licensed under Apache-2.0. See [NOTICE](NOTICE) for details.
