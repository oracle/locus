---
name: locus
description: Build, govern, test and run AI agents with Locus (`locus-sdk`), Oracle's Python agent SDK. Use whenever code imports `locus`, the task names Locus or locus-sdk, or an agent has to run on OCI Generative AI with tools that are gated, audited and testable. Covers Agent and @tool, the admission gate (gate_tool, admit, ControlPolicy, AuditTrail, approvals), structured output, multi-agent shapes, memory and RAG on Oracle Database 26ai, the sandboxed coding harness, model providers, offline tests with model doubles, and the locus command line. Not for other agent frameworks.
license: UPL-1.0
metadata:
  author: Oracle
  version: "1.0"
compatibility: Python 3.11 to 3.14; pip install "locus-sdk[oci]" (or [openai], [anthropic]); a model credential only for live runs
---

# Locus

Locus is one `Agent` class that runs a think, act, observe loop over typed
Python tools, with an admission gate in front of every consequential action.
The same class is what every multi-agent shape instantiates, what the coding
harness drives inside a sandbox, and what `locus serve` exposes over HTTP.
Models are strings such as `oci:openai.gpt-5.5`; credentials come from the
environment or from `~/.oci/config`, never from code. The rule that matters
most: **a tool with a side effect runs only through `gate_tool()` or
`admit()`**, with a `ControlPolicy` that says what is allowed, held for a
person, or denied, and an `AuditTrail` whose chain the tests verify.

## How to work

1. Confirm the install: `python -c "import locus; print(locus.__version__)"`.
   Missing: `pip install "locus-sdk[oci]"` (or the extra for the provider in
   use, table below). Install into the project's own virtual environment.
2. Find the provider from the environment, never by asking for a key:
   `~/.oci/config` exists and a profile is named → `oci:`; `OPENAI_API_KEY` is
   set (`test -n "$OPENAI_API_KEY"`) → `openai:`; `ANTHROPIC_API_KEY` →
   `anthropic:`; nothing → build and test offline with the model doubles.
3. Read the reference for the area before writing code (table at the end).
   The API below is the shape; the references carry every option.
4. Write the agent, then an offline test with `ScriptedModel` or
   `FunctionModel` from `locus.testing`, then run both. A test that needs a
   key is not a unit test.
5. Gate every side effect. If a tool pays, deletes, deploys, sends or writes
   outside the workspace, it goes through `gate_tool()` with a policy; tests
   assert the allow, hold and deny outcomes and `trail.verify()`.
6. Never print, log or store a credential, and never run a command whose
   output would show one (`env`, `printenv`, `echo "$KEY"`, `cat ~/.oci/config`
   beyond its `[section]` names).

## Install

| Need | Install | Model id |
|---|---|---|
| OCI Generative AI | `pip install "locus-sdk[oci]"` | `oci:openai.gpt-5.5`, `oci:cohere.command-a-03-2025`, `oci:ocid1.generativeaiendpoint...` (a Dedicated AI Cluster) |
| OpenAI, and any OpenAI-compatible endpoint | `pip install "locus-sdk[openai]"` | `openai:gpt-5.5`, `openai:gpt-4o-mini` |
| Anthropic | `pip install "locus-sdk[anthropic]"` | `anthropic:claude-sonnet-5-5` |
| Oracle Database 26ai (checkpoints, memory, vectors) | `pip install "locus-sdk[oracledb]"` | |
| Embeddings, vector stores, rerankers | `pip install "locus-sdk[rag]"` | |
| HTTP server, MCP | `pip install "locus-sdk[server]"`, `"locus-sdk[mcp]"` | |
| Everything the tests use | `pip install "locus-sdk[sdk]"` | |

Environment for OCI: `OCI_PROFILE` (a `[section]` of `~/.oci/config`, API
key or session token), `OCI_REGION` (a region that serves Generative AI),
`OCI_COMPARTMENT_ID` (defaults to the profile's tenancy). On OCI compute,
`OCI_AUTH_TYPE=instance_principal` and no profile. Other providers:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`; `OPENAI_BASE_URL` points the OpenAI
client at a proxy.

## A first agent

```python
from locus import Agent, tool


@tool
def lookup_order(order_id: str) -> dict:
    """Return the current status of an order."""
    return {
        "order_id": order_id,
        "status": "delivered",
        "amount_usd": 42.50,
    }


agent = Agent(
    model="oci:openai.gpt-5.5",
    tools=[lookup_order],
    system_prompt="Answer support questions using the order lookup.",
)

result = agent.run_sync("What happened to order ORD-7842?")
print(result.message)
```

`@tool` builds the schema the model sees from the signature and the
docstring; type hints become JSON schema, defaults mark optional parameters.
A tool body may be sync (runs in a thread) or async (runs on the loop). A
raised exception becomes `ToolResult.error` and the model sees it; the agent
does not crash. `agent.run(prompt)` is the async form and yields typed events;
`run_sync` and `invoke` return an `AgentResult` with `message`, `parsed`,
`tool_executions`, `stop_reason` and usage.

Run it with the provider's environment on the command line, not exported
into a file:

```bash
OCI_PROFILE=DEFAULT OCI_REGION=us-chicago-1 python first_agent.py
```

## Gate every side effect

`gate_tool()` wraps a tool so each call is weighed against a `ControlPolicy`
before the body runs. `action=` is an `Action`, or a callable that derives
one from the call's name and arguments, which is how a label such as
`high_value` comes from the real amount. The policy is data: tags that
`require_human_for` holds and `deny_for` refuses, a `max_blast_radius`, a
`spend_limit_usd`. The decision and its reason land on the `AuditTrail`.

```python
from locus import Agent, tool
from locus.control import Action, AuditTrail, ControlPolicy, gate_tool
from locus.testing import FunctionModel, text, tool_call


@tool
def issue_refund(order_id: str, amount_usd: float) -> str:
    """Refund an order."""
    return f"refunded {order_id} ${amount_usd:.2f}"


def refund_action(name, kwargs):
    tags = {"refund"} | ({"high_value"} if kwargs.get("amount_usd", 0) > 100 else set())
    return Action(
        name=name,
        asset=kwargs.get("order_id", ""),
        kind="payment",
        environment="production",
        tags=frozenset(tags),
    )


policy = ControlPolicy(
    require_verification_score=0.0,
    require_human_for=frozenset({"high_value"}),
    deny_for=frozenset({"prohibited"}),
)
trail = AuditTrail()
refund = gate_tool(issue_refund, policy=policy, action=refund_action, trail=trail)


def handler(messages, tools):
    if any(m.role == "tool" for m in messages):
        return text("Done.")
    return tool_call("issue_refund", order_id="ORD-7842", amount_usd=42.5)


agent = Agent(model=FunctionModel(handler), tools=[refund], reflexion=False, grounding=False)
print(agent.run_sync("Refund ORD-7842").message)
print([r.payload["outcome"] for r in trail.records()], trail.verify())
```

For a side effect that is not a tool call, `admit(action, perform, policy=,
trail=)` is the same boundary around any coroutine: `perform` runs only on
allow, and `AdmissionError` carries the decision on hold or deny
(`exc.decision.outcome` is `"require_human"` or `"deny"`). The vocabulary is
allow, hold, deny; the API strings are `ApprovalOutcome.ALLOW`,
`REQUIRE_HUMAN`, `DENY`.

What the gate guarantees: the callable it was given does not run unless
policy allows it, and the refusal is on the trail. What it does not do: find
a second code path that calls the implementation directly (keep ungated
implementations private and test that every mutating tool is wrapped),
notice a mislabeled action (derive labels from the real arguments, with
`derive_labels` when a rule is declarative), or make a replay safe (pass an
idempotency key downstream; `@tool(idempotent=True)` stops a repeat within
one run).

## Hold for a person, then resume

`on_refusal="interrupt"` with an `ApprovalStore` pauses the run on a hold;
a person decides; the run resumes and re-issues the held call once.

```python
# docs: skip
from locus.control import FileApprovals, gate_tool
from locus.memory.backends.file import FileCheckpointer

store = FileApprovals("approvals.json")
refund = gate_tool(issue_refund, policy=policy, action=spec, trail=trail,
                   on_refusal="interrupt", approval=store)
agent = Agent(model=model, tools=[refund], checkpointer=FileCheckpointer("runs"))
agent.run_sync("Refund ORD-7842", thread_id="t1")          # pauses on the hold
store.decide(approval_id, approve=True, by="alice@example.com")
agent.resume("", thread_id="t1", perform_dangling=True)    # the refund happens once
```

The same flow from a terminal: `locus run app.py:agent "..." --thread t1`
exits 3 while paused, `locus approvals --store approvals.json list | approve
<id> --by <who>`, `locus resume app.py:agent --thread t1 --perform`. An
`ApprovalAuthority` restricts who may decide. `FileApprovals` and a
checkpointer survive a restart; `InMemoryApprovals` does not.

## Typed answers, stopping, streaming

```python
from pydantic import BaseModel

from locus import Agent
from locus.core.termination import MaxIterations, ToolCalled
from locus.testing import ScriptedModel, text


class Verdict(BaseModel):
    eligible: bool
    reason: str


agent = Agent(
    model=ScriptedModel([text('{"eligible": true, "reason": "returned in time"}')]),
    output_schema=Verdict,
    termination=MaxIterations(8) | ToolCalled("issue_refund"),
    reflexion=False,
    grounding=False,
)
result = agent.run_sync("Is ORD-7842 eligible?")
print(result.parsed_as(Verdict))
```

`output_schema=` parses the final message into the Pydantic model
(`result.parsed`, `result.parsed_as(Verdict)`); OpenAI-compatible routes,
including `oci:` ids other than `cohere.*`, use provider-strict
`response_format`. `termination=` composes conditions with `|` and `&`.
`async for event in agent.run(prompt)` streams `ThinkEvent`, `ToolStartEvent`,
`ToolCompleteEvent` and the rest; `locus.streaming` has console, SSE and
structured handlers. Budgets: `max_iterations`, `token_budget`,
`max_cost_usd`, `time_budget_seconds`; the `stop_reason` says which fired.

## Pick a shape

| The work is | Use | From |
|---|---|---|
| a fixed chain, fan-out, or repeat-until | `SequentialPipeline`, `ParallelPipeline`, `LoopAgent` | `locus.agent` |
| one coordinator assigning sub-tasks to experts | `Orchestrator`, `Specialist` | `locus.multiagent` |
| peers claiming tasks over shared context | `Swarm`, `SharedContext` | `locus.multiagent` |
| escalation, one agent passing the conversation on | `Handoff`, `HandoffAgent` | `locus.multiagent` |
| cycles, conditional routing, map-reduce, human pauses | `StateGraph` with `Send`, `interrupt`, `goto`, `END` | `locus.multiagent`, `locus.core` |
| your own asyncio glue | `@task`, `@entrypoint` | `locus.multiagent.functional` |
| agents in other processes or runtimes | `A2AServer`, `A2AClient`, `AgentCard` | `locus.a2a` |
| research with sub-agent dispatch and grounding | `create_deepagent`, `create_research_workflow` | `locus.deepagent` |

Every shape takes the same `Agent` instances and emits the same events; gate
the tools, not the shape.

## Memory, persistence, RAG

A `checkpointer=` on the agent plus a `thread_id` on each run gives
resumable conversations: `FileCheckpointer` for one machine,
`OracleCheckpointer` and the other backends in `locus.memory.backends` for a
service. A `conversation_manager=` prunes or compacts long threads; a
`memory_manager=` keeps long-term facts (`LLMMemoryManager`,
`OracleAgentMemoryManager`). `locus.rag` has embeddings, vector stores
(Oracle Database 26ai first), rerankers, loaders and chunkers; expose
retrieval to the agent as a `@tool` and turn on `grounding=True` when
answers must cite the evidence.

## The coding harness

`locus.harness` is read, write, edit, search and a shell, written against a
workspace protocol and run unchanged on the host, in memory, or in a sandbox.

```python
# docs: skip
from locus import Agent
from locus.control import ControlPolicy, gate_tool
from locus.harness import LocalBackend, build_harness

policy = ControlPolicy(
    require_verification_score=0.0, require_human_for=frozenset({"workspace.exec"})
)
harness = build_harness(
    LocalBackend("."),
    wrap=lambda tool, spec: gate_tool(tool, policy=policy, action=spec),
)
agent = Agent(
    model="oci:openai.gpt-5.5", tools=harness.tools, system_prompt=harness.prompt_fragment
)
```

`wrap=` is the one place governance is applied, to every tool; without it
the tools are ungated, which is right only for a throwaway sandbox. Set
`require_verification_score=0.0` on a policy that decides by label: with the
default score every call is held, reads included, because no verification
verdict accompanies a workspace tool call.
Backends: `LocalBackend` (host, files confined to a root), `MemoryBackend`
(tests), `SessionBackend` (any sandbox with `exec`, `upload_file`,
`download_file`), `OCISandboxBackend` (`locus-sdk[oci-sandbox]`),
`OpenShellBackend` (`locus-sdk[openshell]`). Every command produces an
`ExecRecord` on the event bus.

## Test offline

```python
from locus import Agent, tool
from locus.testing import AgentTestClient, ScriptedModel, text, tool_call


@tool
def lookup_order(order_id: str) -> dict:
    """Return the current status of an order."""
    return {"order_id": order_id, "status": "delivered"}


def test_the_order_is_looked_up() -> None:
    model = ScriptedModel([
        tool_call("lookup_order", order_id="ORD-7842"),
        text("ORD-7842 was delivered."),
    ])
    client = AgentTestClient(
        Agent(model=model, tools=[lookup_order], reflexion=False, grounding=False)
    )
    trace = client.run("What happened to ORD-7842?")
    trace.assert_tool_called("lookup_order", order_id="ORD-7842").assert_model_calls(2)
    trace.assert_succeeded()
    assert trace.message == "ORD-7842 was delivered."


test_the_order_is_looked_up()
```

`ScriptedModel` returns a fixed sequence of turns and raises when the agent
asks for more; `FunctionModel(handler)` decides per turn from the messages
so far. Both record `call_count`, `received_messages` and `offered_tools`.
`python -m locus.rogue` is the built-in check that the gate holds against a
model that is compromised by construction; its last line reads
`BREACHES: 0` and `audit chain intact: ✓`.

## Run and serve

`locus run app.py:agent "prompt" --thread t1` streams a run; `locus serve
app.py:agent --port 8000` serves it over HTTP (`AgentServer`, bearer key in
`LOCUS_SERVER_API_KEY`); `locus audit verify trail.jsonl --key id=pub.pem`
checks an exported trail. `AGENT` is `module:attr` or `file.py:attr` naming
an `Agent` or a zero-argument factory. Exit codes: 0, 1 a check failed, 2
usage, 3 paused for a person.

## Known behaviour

| Seen | Meaning | Do |
|---|---|---|
| OCI `NotFoundError: ... 404 ... Authorization failed or requested resource not found` | the model id is not available to that tenancy in that region, or the profile has no Generative AI policy | try another id the region serves, or another region; report, do not loop |
| OCI `NotAuthenticated` | the profile, key file or session token | the person refreshes with `oci session authenticate --profile-name <p> --region <r>`; never touch `~/.oci` |
| `No context window known for model ...` on standard error | a token-counting notice for a model without metadata | nothing |
| `ScriptedModel` raises "out of turns" | the agent asked for more turns than scripted, usually because it did not terminate | fix the termination or add the turn; `repeat_last=True` only when turn count is not the point |
| `LLMMemoryManager` stored nothing | its default extractor is a keyword heuristic | phrase facts as "never ..." or "I'm a ..." or supply an extractor |
| a gated tool ran with `require_human_for` set | the action's labels did not carry the tag | label from the real arguments; assert `trail.records()[-1].payload["outcome"]` in a test |

## Rules

- Credentials are never asked for, printed, logged or stored; `test -n "$VAR"`
  is the only way to check one exists.
- Every side effect goes through the gate; every gate has a trail; every
  trail is verified in a test.
- Prefer the provider the environment already has; make live calls only
  when the task needs the answer, and one at a time.
- Install only into the project's virtual environment; write only inside the
  project.
- When these files and the Locus source disagree, the source wins:
  `python -c "import locus.control, inspect; print(inspect.signature(locus.control.gate_tool))"`.

## Reference

| File | Read it for |
|---|---|
| `references/agent-and-tools.md` | `Agent` and `AgentConfig` options, `@tool`, `ToolContext`, structured output, streaming, hooks, termination, reflexion and grounding, decisions, skills, playbooks, evaluation, MCP, the testing doubles |
| `references/control-layer.md` | `ControlPolicy`, `Action`, `admit`, `gate_tool`, approvals and authorities, interrupt and resume, `AuditTrail` and signed exports, permission rules, shell parsing, spend ledgers, the governed wrapper |
| `references/multi-agent-harness-deploy.md` | each shape with code, `StateGraph` primitives, A2A, DeepAgent, the harness and its backends, the box runner, `AgentServer`, the command line, observability |
| `references/providers-memory-rag.md` | every provider prefix and its environment, OCI transports and errors, checkpointers and backends, memory managers, Oracle Database 26ai, embeddings, vector stores, rerankers |
| `references/extending-and-advanced.md` | subagents, voice sessions, external hooks, tool sandboxes, writing a model provider, the ReAct loop, core protocols, settings, events and errors, DeepAgent internals, graph evaluation, decision providers, router runnables, delta checkpointers, the runner, rogue and codex modules |

The documentation site is <https://locusagents.ai/>; <https://locusagents.ai/llms.txt>
lists every page and <https://locusagents.ai/llms-full.txt> holds the guides
in one file. In a clone of the repository, `codex/SPEC.md` is a complete,
checkpointed walk-through of a first agent, a controlled action and the
rogue-agent challenge.
