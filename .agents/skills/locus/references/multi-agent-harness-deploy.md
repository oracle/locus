# Multi-agent shapes, the coding harness, serving

This file covers the layers above one `Agent`: the multi-agent shapes Locus ships and when to pick each (pipelines,
orchestrator, swarm, handoff, `StateGraph`, functional API, cognitive router, A2A, DeepAgent); the coding harness in
`locus.harness` (tools, the single gate, labels, backends, evidence, the box runner); and serving and operating
(`AgentServer`, the `locus` command, the event bus, deployment, durable execution, the rogue-agent smoke test).
Examples run offline on doubles from `locus.testing`; where a live model id appears, `"oci:openai.gpt-5.5"` is the form.

## Pick a shape

| Need | Shape | Entry point |
| --- | --- | --- |
| Fixed chain, fan-out or revise-until loop | Pipelines | `locus.agent.SequentialPipeline`, `ParallelPipeline`, `LoopAgent` |
| One coordinator picks experts, correlates, summarizes | Orchestrator | `locus.multiagent.create_orchestrator`, `Specialist` |
| Peers claim queued tasks, share a blackboard, no router | Swarm | `locus.multiagent.create_swarm`, `create_swarm_agent` |
| One agent passes the case up a tier with typed findings | Handoff | `locus.multiagent.create_handoff_manager`, `create_handoff_agent` |
| Cycles, conditional routing, map-reduce, interrupts, checkpoints | StateGraph | `locus.multiagent.StateGraph`, `START`, `END` |
| Hand-written asyncio glue with task bookkeeping | Functional API | `locus.multiagent.functional.task`, `entrypoint` |
| Policy, not the model, must choose the topology | Cognitive router | `locus.router.Router`, `GoalFrame`, `PolicyGate` |
| Agents in other processes or frameworks | A2A | `locus.a2a.A2AServer`, `A2AClient` |
| Long-horizon research ending in a typed row | DeepAgent | `locus.deepagent.create_deepagent`, `create_research_workflow` |

Every shape runs the same `Agent`, takes the same tools and emits on the same event bus; a `StateGraph` node can await
any other shape. Shared state is `AgentState` (`locus.core.state`): immutable `messages`, `tool_executions`,
`confidence`, `metadata`; `with_message(Message.user("hi"))` returns a copy; `to_checkpoint()` /
`AgentState.from_checkpoint(d)` round-trip JSON; `Message` (`locus.core.messages`) has `.system`, `.user`, `.assistant`.
`ScriptedModel([turns], repeat_last=False)` returns fixed turns (`text(...)`, `tool_call(name, **args)`, or a bare
string) and raises past the script; `FunctionModel(handler)` decides each turn from `(messages, tools)`.

## Pipelines

Use when the flow is a plain function of its steps. Each class has one async `run(task) -> PipelineResult` (`success`,
`outputs`, `final_output`, `duration_ms`, `error`) and drives agents through `arun`. A pipeline has no `arun` or
`run_sync`, so pipelines do not nest; put one inside a `StateGraph` node instead.

| Class | Behaviour |
| --- | --- |
| `SequentialPipeline(agents=[...], prompt_template=...)` | output N feeds agent N+1; template vars `{previous_output}`, `{task}` |
| `ParallelPipeline(agents=[...], merge_strategy="concatenate" or "last", separator=...)` | `run(task, task_map={index: prompt})`; a failed agent leaves `""` in its slot and `success=False` |
| `LoopAgent(agent=..., condition=lambda out: bool, max_loops=5, loop_prompt=...)` | stops when `condition(latest_output)` is true or `max_loops` is hit |

```python
import asyncio

from locus.agent import Agent, SequentialPipeline
from locus.testing import ScriptedModel

research = Agent(model=ScriptedModel(["Finding: deploy-77 doubled p99 latency."]))
brief = Agent(model=ScriptedModel(["Status: deploy-77 is the cause; rollback proposed."]))
result = asyncio.run(SequentialPipeline(agents=[research, brief]).run("Why did checkout slow down?"))
print(result.success, len(result.outputs), result.final_output)  # True 2 Status: ...
```

## Orchestrator and specialists

Use when work splits into expert domains and one place owns the routing decision. `Specialist(name, specialist_type,
description, system_prompt, tools=[], model=None, max_iterations=10, confidence_threshold=0.85)`. `execute(task,
context=None)` is one model turn (tool schemas are offered, tool calls are not executed; a specialist that must act
wraps a tool-using `Agent` in a graph node) returning `SpecialistResult` (`output`, `confidence`, `error`, `success`),
an error result without a model. `create_orchestrator(name, specialists, model)` does not give specialists its model;
`orchestrator.with_model(model)` returns a copy that sets it everywhere. Routing is one model call answered with a
fenced ` ```json ` block of `specialists` (ids), `reasoning`, `subtasks`; with no model every specialist runs and
findings are concatenated. Picked specialists run concurrently (`max_parallel_specialists`, default 5), retried once on
an empty reply.

```python
import asyncio
import json

from locus.multiagent import Specialist, create_orchestrator
from locus.testing import FunctionModel, ScriptedModel

logs = Specialist(name="logs", specialist_type="log_analyst", description="Reads error logs.",
                  system_prompt="You read logs.", model=ScriptedModel(["502s start after deploy-77."]))

def coordinator(messages, tools):
    if "## Available Specialists" in (messages[-1].content or ""):
        return "```json\n" + json.dumps({"specialists": [logs.id], "reasoning": "logs", "subtasks": {}}) + "\n```"
    return "deploy-77 caused the incident; roll it back."

orchestrator = create_orchestrator(name="commander", specialists=[logs], model=FunctionModel(coordinator))
result = asyncio.run(orchestrator.execute("Why are checkout requests failing?"))
print(result.success, [r.output for r in result.specialist_results.values()], result.summary)
```

`execute` returns an `OrchestratorResult` (`success`, `summary`, `specialist_results`, `decisions`, `duration_ms`, `error`).
Check: `success` stays `True` when one specialist fails (read `result.specialist_results[<id>].error`);
`result.decisions` lists `RoutingDecision`s (`decision_type` in `invoke`, `correlate`, `summarize`; `specialists`,
`reasoning`, `context`). Factories: `create_log_analyst`, `create_metrics_analyst`, `create_trace_analyst`, `create_code_analyst`.

## Swarm

Use when peers should claim work from a shared queue without a router. `create_swarm_agent(name, capabilities=[...],
system_prompt="", model=None)`; `create_swarm(name, agents, model)` copies `model` into agents that have none. A task is
claimable by an agent whose capability word appears in the task description, or by one with no capabilities; each agent
answers a task once, in `### Findings`, `### Analysis`, `### Blackboard` sections parsed onto `SharedContext`
(`findings`, `blackboard`, `task_results`), so a double must answer in that format. `await swarm.execute(initial_task=None,
decompose_tasks=True)` asks the swarm model to split the task; seed the queue yourself with `swarm.add_task(description,
priority=0)` and `decompose_tasks=False`. It stops when nothing is pending or `max_iterations` (default 10) is hit.
`create_swarm_agent` returns a `SwarmAgent` (`name`, `capabilities`, `system_prompt`, `model`, `tasks_completed`;
`can_handle(task)`, `priority_for_task(task)`); `add_task` returns the `SwarmTask` (`id`, `description`, `priority`,
`required_tags`, `preferred_tags`, `status: TaskStatus` in `pending`, `claimed`, `in_progress`, `completed`, `failed`;
`claimed_by`, `result`, `error`); set `required_tags` on it and only an agent advertising every tag may claim it.
`execute` returns a `SwarmResult` (`success`, `completed_tasks`, `failed_tasks`, `context`, `summary`, `duration_ms`, `error`).

## Handoff

Use for tiered escalation where the next tier needs findings, not the transcript. `create_handoff_agent(name,
description="", system_prompt="", tools=None, model=None)` returns a `HandoffAgent` with `can_escalate_to` /
`can_delegate_to` id lists; `create_handoff_manager(agents, max_chain=5)` returns `Handoff`. `execute_handoff` returns
`HandoffResult` (`success`, `output`, `error`, `final_confidence`); `HandoffReason` is `SPECIALIZATION`, `ESCALATION`,
`DELEGATION`, `COMPLETION`, `FAILURE`; `chain_handoff(agent_chain=, task=)` runs several in order. The target reads a
rendered `HandoffContext` (`original_task`, `findings`, `conversation_summary` from `state`, `handoff_chain`; key
messages too when `manager.preserve_full_history`). An unknown target or exhausted chain gives `success=False`, `error`.
`HandoffEvent` (`source_agent_id`, `target_agent_id`, `reason`, `context_summary`) is the typed shape of a handoff; the
live publication is `multiagent.handoff.initiated` on the event bus with those fields, then `multiagent.handoff.completed`.

```python
import asyncio

from locus.multiagent import HandoffReason, create_handoff_agent, create_handoff_manager
from locus.testing import ScriptedModel

triage = create_handoff_agent(name="L1 Support", model=ScriptedModel(["Billing case."]))
billing = create_handoff_agent(name="L2 Billing", model=ScriptedModel(["Duplicate charge refunded."]))
triage.can_escalate_to = [billing.id]
manager = create_handoff_manager(agents=[triage, billing])
result = asyncio.run(manager.execute_handoff(
    source_agent=triage, target_agent_id=billing.id, reason=HandoffReason.ESCALATION,
    task="Customer reports a duplicate charge on ord-4821.", findings={"order": "ord-4821"},
))
print(result.success, result.output, len(manager.history))
```

## StateGraph

Use for cycles, conditional routing, `Send` fan-out, human interrupts and checkpoints.

| Call | Notes |
| --- | --- |
| `StateGraph(name="", state_schema=None, config=GraphConfig(...))` | `GraphConfig(parallel=True, allow_cycles=False, max_iterations=100, interrupt_before=[], interrupt_after=[], checkpointer=None, thread_id=None, stream_mode=StreamMode.VALUES)` |
| `add_node(id, fn, *, retry_policy=RetryPolicy(max_attempts=3, backoff_factor=2.0), cache_policy=CachePolicy(ttl_seconds=300), timeout_ms=None, condition=None, defer=False)` | `fn(state: dict)` returns a dict of updates, a `Command`, or `Send`s; sync functions run in a thread; a `StateGraph` as `fn` is a subgraph |
| `add_edge(source, target)`; `add_conditional_edges(source, router, targets=None, default=None)` | `START` and `END` are valid ends; `router(state)` returns a `targets` key, a node id, or `END`; pass `targets` so Mermaid draws the branches |
| `compile(checkpointer=, interrupt_before=, interrupt_after=, store=)`; `await graph.execute(inputs)` | `compile` configures and returns the same graph; `execute` returns `GraphResult` (`final_state`, `node_results`, `execution_order`, `iterations`, `interrupt`, `is_interrupted`) |
| `graph.run_sync(inputs)`, `graph.invoke(inputs)`, `await graph.ainvoke(inputs)` | return `final_state`; the sync forms raise inside a running event loop |
| `graph.stream(inputs, mode=StreamMode.X)`; `graph.get_mermaid()`, `draw_mermaid(direction="TD")`, `draw_ascii()` | `stream` yields `StreamEvent(node_id, data, mode)` as nodes finish; the others render the topology |

State is a dict; a node receives the whole state and its updates merge through reducers (`locus.core.reducers`). Cycles
are refused unless `allow_cycles=True`. An `Agent` is not callable: `async def plan(state): return {"plan": (await
planner.arun(state["prompt"])).message}`. The topology is data: `graph.nodes` is `dict[str, Node]` (`id`, `name`,
`executor`, `condition`, `retry_policy`, `cache_policy`, `timeout_ms`, `defer`), `graph.edges` is `list[Edge]` (`source`,
`target`) and `graph.conditional_edges` is `list[ConditionalEdge]` (`source_id`, `router`, `targets`, `default_target`);
`NodeResult.status` is a `NodeStatus` (`pending`, `running`, `completed`, `failed`, `skipped`, `interrupted`).
`create_graph(name="", description="", allow_cycles=False)` is a `StateGraph` with that `GraphConfig`; `Graph` is the
older DAG-only subclass, kept for existing code.

```python
from locus.multiagent import END, START, GraphConfig, StateGraph

def draft(state: dict) -> dict:
    revisions = state.get("revisions", 0) + 1
    return {"draft": f"v{revisions} for {state['prompt']}", "revisions": revisions}

def review(state: dict) -> dict:
    return {"approved": state["revisions"] >= 2}

graph = StateGraph(name="review-loop", config=GraphConfig(allow_cycles=True, max_iterations=10))
graph.add_node("draft", draft)
graph.add_node("review", review)
graph.add_edge(START, "draft")
graph.add_edge("draft", "review")
graph.add_conditional_edges("review", lambda s: "done" if s["approved"] else "revise", {"done": END, "revise": "draft"})
final = graph.run_sync({"prompt": "ticket t-8804"})
print(final["draft"], final["revisions"])
print(graph.get_mermaid())
```

Check: `final["revisions"]` is `2`; the Mermaid text lists `draft`, `review` and both branches.

- `Send` (`locus.core.send`): a node returns `[Send(node="score", payload={"ticket": t}), ...]`; targets run in
  parallel with the state plus their payload, need no incoming edge, and each result lands in state under its
  `send_id`; control then follows the node's own edge to the reducer node.
- `Command` (`locus.core.command`): return `Command(goto="sign_off", update={...})` to override an edge;
  `graph.execute(Command(resume=value))` resumes a run paused by `interrupt(payload)` (`locus.core.interrupt`,
  `GraphResult.is_interrupted` is `True`); persist the pause with `GraphConfig(checkpointer=..., thread_id=...)`.
- `async for event in graph.stream(inputs, mode=StreamMode.UPDATES)`; modes `VALUES` (state after each node, final
  state last), `UPDATES` (the node's output), `NODES` (`NodeResult`), `DEBUG`, `CUSTOM` (what `emit_custom(data,
  node_id=None)` from `locus.multiagent` pushes; a no-op under `execute()`). Breaking out cancels the run.

## Functional API

Use for asyncio glue you would otherwise write by hand. `@task(name=None, retry_attempts=1, cache=False)` marks a
coroutine (`async def summarize(doc): return (await agent.arun(doc)).message`); `@entrypoint(name=None)` marks the
workflow (`async def run_all(docs): return await asyncio.gather(*(summarize(d) for d in docs))`), which returns its raw
value; `run_all.get_result()` is an `EntrypointResult` (`value`, `tasks`, `duration_ms`, `error`) with one `TaskResult`
per task call. Use `await agent.arun(...)` inside tasks so `asyncio.gather` overlaps model calls.

## Cognitive router

Use when the model must not choose the topology. The model fills one typed `GoalFrame` (`primary_goal: TaskType`,
`domain`, `complexity`, `risk`, `requires_tools`, `required_capabilities`, `approval_required`); the rest is deterministic.

| Piece | Use |
| --- | --- |
| `ProtocolRegistry().register_many(builtin_protocols())` | ids `direct_response`, `plan_execute_validate`, `specialist_fanout`, `debate`, `codegen_test_validate`, `approval_gated_execution`, `a2a_delegate`, `handoff_chain`; `registry.select(frame, available_capabilities=set)` |
| `PolicyGate(max_risk=Risk.HIGH, require_approval_above=Risk.MEDIUM, denied_protocols=None)` | `gate.check(frame, protocol) -> PolicyVerdict(allow, require_approval, reason)` |
| `CapabilityIndex(tool_registry).annotate(cap_id, tool_name=, description=, domain=, risk=Risk.LOW)` | binds capability ids to tools |
| `CognitiveCompiler(protocols=, capabilities=, policy=, model=, hooks=None, on_approval=None)` | `compiler.explain(frame, goal="")` is model-free; `await compiler.compile(frame)` builds a `Runnable`; `hooks=` governs every agent a builder emits; a held frame is denied without `on_approval` |
| `Router(extractor=Agent(model=..., output_schema=GoalFrame), compiler=)` | `await router.dispatch(text) -> RunnableResult(text, parsed, raw, protocol_id, frame)`; `await router.explain(text)` runs nothing |

```python
from locus.router import (CapabilityIndex, CognitiveCompiler, Complexity, GoalFrame, PolicyGate,
                          ProtocolRegistry, Risk, TaskType, builtin_protocols)
from locus.tools import tool
from locus.tools.registry import create_registry

@tool
def search_runbooks(query: str) -> str:
    """Search the operations runbooks."""
    return f"runbook hit for {query}"

capabilities = CapabilityIndex(create_registry(search_runbooks))
capabilities.annotate("runbook_search", tool_name="search_runbooks", description="Runbooks.", domain="ops")
protocols = ProtocolRegistry()
protocols.register_many(builtin_protocols())
compiler = CognitiveCompiler(protocols=protocols, capabilities=capabilities, policy=PolicyGate(),
                             model="oci:openai.gpt-5.5")
frame = GoalFrame(primary_goal=TaskType.REMEDIATE, domain="ops", complexity=Complexity.MEDIUM,
                  risk=Risk.HIGH, requires_tools=True, required_capabilities=["runbook_search"])
explanation = compiler.explain(frame, goal="Roll back deploy-77 on checkout.")
print(explanation.selected_protocol, explanation.requires_approval)
```

Check: a `HIGH` risk frame selects `approval_gated_execution` with `requires_approval` `True`.

## A2A

Use when the peer is another process or framework. `A2AServer(agent, name, description, skills=[AgentSkill(id=, name=,
description=, tags=[])], url, provider=None, version="0.1.0", api_key=None, allow_unauthenticated=False)` serves the
card at `/.well-known/agent-card.json` and JSON-RPC at `POST /`; `server.run(host="127.0.0.1", port=8001)` needs
`uvicorn` and refuses a non-loopback host without `api_key` or `LOCUS_A2A_API_KEY`; set `url` to the public URL.
`A2AClient(url, api_key=None, timeout=None, protocol_version=A2A_V1_PROTOCOL_VERSION)`: `await client.get_agent_card()`;
`await client.send_message(Message(role="user", parts=[TextPart(text=...)], messageId="m-1"))` returns a `Task` whose
reply is `task.artifacts[-1].parts[0].text`; `send_message_streaming`, `get_task`, `list_tasks`, `cancel_task`; `await
client.invoke(prompt)` is the flat legacy call; `client.as_tool(name=, description=)` wraps the remote agent as a local
tool. The server runs the agent without a `thread_id`: `contextId` groups tasks but carries no conversation memory.
Check: `sorted(route.path for route in server.app.routes)` includes `/`, `/.well-known/agent-card.json` and
`/a2a/invoke` (needs the `server` extra); `A2AServer` also accepts a `GraphRunnable`.

## DeepAgent

Use for research that must end in a typed row. `create_deepagent(*, model, tools, system_prompt, output_schema=None,
submit_tool="submit_research", min_confidence=0.8, max_iterations=40, reflexion=True, grounding=True, checkpointer=None,
enable_filesystem=False, enable_todos=False, memory_files=None, subagents=None, datastores=None)` returns a plain
`Agent`: the loop ends when the model calls `submit_tool`, and `result.parsed` is the validated `output_schema`.
`tools` must include that tool (`@tool def submit_research(report: Report) -> str`, its argument the `output_schema`
model) beside the research tools. `create_research_workflow(*, model, tools, output_schema=None, grounding_threshold=0.65, max_replans=2,
max_regenerations=1, causal_inference=True, checkpointer=None, datastores=None)` returns a `StateGraph` (execute,
optional causal inference, summarize, grounding evaluation, then regenerate or replan below the threshold). Run `await
workflow.execute({"prompt": ...})` and read `result.final_state["structured_output"]`, `["grounding_score"]`,
`["replan_count"]` (`KEY_STRUCTURED_OUTPUT`, `KEY_GROUNDING_SCORE`, `KEY_REPLAN_COUNT`).

## Composition and production notes

A graph node may await `orchestrator.execute`, `swarm.execute`, `pipeline.run` or a graph; `Orchestrator.execute` and
`Swarm.execute` catch their errors and return `success=False`, so raise inside the node when a `retry_policy` should
fire. `Agent(model=..., output_schema=MyModel)` gives a typed `result.parsed`. Side-effecting tools in any shape go
behind `locus.control.gate_tool(tool, policy=, action=, trail=AuditTrail())` (hash-chained, verifiable trail);
`@tool(idempotent=True)` dedupes identical calls within a run; checkpoint with `Agent(checkpointer=...)` or
`GraphConfig(checkpointer=, thread_id=)`.

## The coding harness

`locus.harness` is the toolset a coding agent uses, written once against `WorkspaceBackend`. Tool names (`ALL_TOOLS`):
`read`, `ls`, `glob`, `grep`, `edit`, `multi_edit`, `write`, `apply_patch`, `notebook_edit`, `bash`, `bash_output`,
`kill_shell`, `write_stdin`, `todo_write`, `todo_read`. `DEFAULT_TOOLS` is everything but `apply_patch`; `PATCH_TOOLS`
swaps `edit`, `multi_edit`, `write` for `apply_patch` (`locus.models.profiles.profile_for(model).edit_format` says which
a model wants). Shell tools are dropped from the default set, and refused by name, on a backend with no shell.
`build_harness(backend, *, tools=None, wrap=None, config=None) -> Harness`:

| Attribute | Meaning |
| --- | --- |
| `harness.tools` | the built, wrapped tools for `Agent(tools=...)` |
| `harness.specs` | each tool's `ActionSpec` by name; `spec(tool_name, kwargs)` returns the `Action` a policy judges |
| `harness.prompt_fragment` | workspace root, host-shell warning, read-before-edit rule, shell timeouts; append to the system prompt |
| `harness.context` | `HarnessContext(backend, config, ledger, todos)` shared by the tools |
| `harness.preview(name, arguments)` | the diff a `write` / `edit` / `multi_edit` / `apply_patch` / `notebook_edit` call would make, or its refusal; `None` otherwise |
| `harness.tool(name)`, `harness.gated`, `harness.backend` | one built tool; whether `wrap` was applied; the backend |

`HarnessConfig` (frozen dataclass): `require_read=True`, `bash_timeout=120`, `bash_max_timeout=600`,
`on_timeout="background"` or `"kill"`, `output_chars=30_000`, `output_head_chars=8_000`, `vision=False`,
`result_store=None`, `on_exec=None`, `on_change=None`, `environment="unknown"`; `on_exec` receives each `ExecRecord`,
`on_change` each `FileChange` (`path`, `shown`, `before`, `after`, `action`). `context.ledger` is the `ReadLedger`
(`require_read=True`) that refuses a change to a file not read in this session, or changed since it was read.

### The single gate

Tool bodies never gate. `wrap(tool, spec)` is applied to every tool and is the one place a policy goes; `wrap=None`
leaves the tools ungated and `build_harness` warns when that leaves a host shell open. The default `ControlPolicy()`
holds every call (`require_verification_score=0.8` with no verdict supplied): pass `verdict=` to `gate_tool` or set
`require_verification_score=0.0` and decide by labels. Add `trail=AuditTrail()` to record decisions and `approval=` to
hand holds to an approval store.

```python
import asyncio

from locus.control import ControlPolicy, gate_tool
from locus.harness import MemoryBackend, action_spec, build_harness

print([tool.name for tool in build_harness(MemoryBackend()).tools])
bash = action_spec("bash", environment="dev")
print(bash("bash", {"command": "ls -la src"}).kind, sorted(bash("bash", {"command": "pytest -q"}).tags))
policy = ControlPolicy(require_verification_score=0.0, require_human_for=frozenset({"workspace.write"}))
harness = build_harness(MemoryBackend({"app.py": "print('hi')\n"}), tools=["read", "write"],
                        wrap=lambda tool, spec: gate_tool(tool, policy=policy, action=spec))

async def main() -> None:
    print(await harness.tool("read").execute(path="app.py"))
    print(await harness.tool("write").execute(path="new.py", content="x = 1\n"))
    print(harness.preview("write", {"path": "new.py", "content": "x = 1\n"}))

asyncio.run(main())
```

Check: ten tools and no shell tools; `ls -la src` is `workspace.read`, `pytest -q` carries `bash` and `exec:check`; the
gated read returns the file; the write returns JSON with `"status": "held_for_approval"`; the preview is a unified diff.

### Labels

A policy matches `Action.labels()` = `{environment, kind, *tags}`; every harness action carries the tool name as a tag
and the path or command as `asset`.

| Kind | Tools |
| --- | --- |
| `workspace.read` | `read`, `ls`, `glob`, `grep`, `bash_output`, `todo_write`, `todo_read`, and a `bash` line made only of read-only programs |
| `workspace.write` | `write`, `edit`, `multi_edit`, `apply_patch`, `notebook_edit` |
| `workspace.exec` | every other `bash` line, `write_stdin`, `kill_shell` |
| `network` | reserved for web tools; a fetching shell line is `workspace.exec` tagged `exec:network` |

`classify_command(line) -> CommandClass` (`kind`, `tags`, `reasons`, `.never`; kinds are the constants `KIND_READ`,
`KIND_WRITE`, `KIND_EXEC`, `KIND_NETWORK`) reads the programs a shell line runs (behind `sudo`, `env`, `xargs`, `sh -c`, `$(...)`) and
tags it: `exec:destructive`, `exec:vcs-push`, `exec:force-push`, `exec:network`, `exec:remote-code`, `exec:shutdown`,
`exec:check` (tests, linters), `exec:never` (the never-allow floor), `exec:unparsed` (fails closed), `exec:stdin`,
`exec:background`; `rm -rf build` is `exec:destructive` + `exec:never`. Gate by label:
`ControlPolicy(require_verification_score=0.0, require_human_for={"workspace.exec"}, deny_for={"exec:never"})`.

### Backends

| Backend | Install | Where | Isolated | Shell |
| --- | --- | --- | --- | --- |
| `LocalBackend(root=".", env=None, base_env=None, label=...)` | core | the host, files confined to `root` | no (`UNISOLATED: host shell`) | yes |
| `MemoryBackend(files={path: text})` | core | a dict, root `/` | yes | no |
| `SessionBackend(session, root="/workspace", label="session", isolated=True, env=None)` | core | any sandbox through `SessionLike` | yes | over `exec` |
| `OCISandboxBackend.create(project_id, region=, profile=, auth_type=, expiration="PT1H", runtime=, shape=)`; `.connect(sandbox_id, region=, profile=)` | `locus-sdk[oci-sandbox]` | an OCI GenAI Sandbox, root `/workspace` | yes | yes |
| `OpenShellBackend.create(workspace="default", endpoint=, image=, providers=)`; `.connect(sandbox, workspace=, endpoint=)` | `locus-sdk[openshell]` | an OpenShell sandbox behind a gateway, root `/sandbox` | yes | yes |

`SessionLike` is three methods: `exec(command, *, timeout) -> (exit_code or None, output)`, `upload_file(path, data)`,
`download_file(path) -> bytes`; everything else is a POSIX script run over `exec`. `create()` owns the sandbox and
`close()` deletes it (call it in a `finally`); `connect()` never deletes. OCI: `auth_type` is `api_key`,
`security_token`, `instance_principal` or `resource_principal`; `profile` names an `~/.oci/config` entry; `expiration`
is an ISO 8601 duration up to 48 hours; `backend.extend("PT30M")`. OpenShell: without `endpoint` the client comes from
the active gateway configuration; `providers=` attaches credentials the sandbox sees only as placeholders. With a client
of your own, `OCISandboxBackend(client, sandbox_id, region=, owns_sandbox=False)` and `OpenShellBackend(client, sandbox,
workspace=)` wrap it in the `SessionLike` adapters they build internally, `OCISandboxSession` and `OpenShellSession`.

The `WorkspaceBackend` surface: `capabilities` is a `BackendCapabilities` (`isolated`, `label`, `has_rg`, `root`,
`can_exec`); `stat(path)` returns a `FileStat` (`size`, `mtime`, `sha256`, `is_dir`), `read_lines(path, offset=, limit=)`
a `LineWindow` (`lines`, `total`, `first`); `ls`, `glob` and `grep` skip `SKIP_DIRS` (`.git`, `node_modules`, `.venv`,
`dist`, `build`, ...); file errors raise `BackendError` (`code` in `file_not_found`, `permission_denied`, `is_directory`,
`invalid_path`). `exec(command, *, timeout, env=None, cwd=None, on_output=None)` returns an `ExecResult` (`exit_code`,
`output` bytes, `truncated`, `timed_out`, `duration_s`), or raises `ExecUnsupportedError` where there is no shell; the
`*_background` methods return a `JobStatus` (`handle`, `command`, `running`, `exit_code`, `elapsed_s`) or, for
`read_background(handle, since=0)`, a `JobOutput` (`status`, `data`, `offset`, `lost`); a bad handle raises `JobError`.

### Evidence and the box runner

Every command produces an `ExecRecord` (`command_sha256`, `exit_code`, `duration_s`, `timed_out`, `output_sha256`,
`output_bytes`, `truncated`, `backend_label`, `background`): digests, never text, emitted as `harness.exec`
(`EV_HARNESS_EXEC`) inside a `run_context` and handed to `HarnessConfig(on_exec=callback)`. `python -m locus.runner`
(also `locus-runner`) is one agent's loop inside an OpenShell sandbox, started by a Locus gateway: it reads
`LOCUS_ADMIT_URL`, `LOCUS_ADMIT_TOKEN`, `LOCUS_RUN_ID`, asks the gateway for `start`, `resume` or `stop`, fetches the
run manifest, builds the agent, streams events and reports the result; commands never inherit the token or credential
placeholders. Exit codes: `0` done, parked on a person, or told to stop; `2` refused (the manifest asks for a tool,
provider or credential the runner lacks); `1` error.

## AgentServer

`AgentServer(agent, title=..., description=..., api_key=None, allow_unauthenticated=False, *, stream_tokens=True,
metadata_resolver=None, decision_handler=None)` needs `locus-sdk[server]`. `server.app` is the FastAPI application
(`app = server.app` at module scope serves under `uvicorn server:app`); `server.run(host="127.0.0.1", port=8000,
**uvicorn_kwargs)` refuses a non-loopback host without `api_key` or `LOCUS_SERVER_API_KEY`; serving `server.app`
yourself skips that check, so set the key anyway. With a key, every route but `/health` needs `Authorization: Bearer
<key>`, and thread ids are prefixed with a principal derived from it (one key, one namespace, not per-tenant isolation).
`/stream` is POST: browsers use `fetch` with a stream reader, not `EventSource`. A graph is served as
`agent=GraphRunnable(graph, input_key="prompt", output_key="answer")`; `/resume` and `/threads` do not apply to it.

| Route | Body | Returns |
| --- | --- | --- |
| `POST /invoke` | `{"prompt", "thread_id"?, "metadata"?}` | `InvokeResponse`: `message`, `success`, `stop_reason`, `iterations`, `tool_calls`, `duration_ms` |
| `POST /stream` | same | SSE `data: {json}` frames with `type` in `think`, `tool_start`, `tool_complete`, `model_chunk`, `interrupt`, `done`, `error`; then `data: [DONE]` |
| `POST /resume` | `{"thread_id", "response"?, "decision"?, "perform_dangling"?}` | SSE of the continued thread; 409 while an approval is undecided, 404 for an unknown thread |
| `GET` / `DELETE /threads/{tid}` | | thread state; needs a checkpointer on the agent |
| `GET /health` | | 200 |

## The locus command line

`AGENT` is `module:attr` or `path/to/file.py:attr`, naming an `Agent` or a zero-argument factory.

| Command | What it does |
| --- | --- |
| `locus run AGENT "prompt" [--thread ID]` | runs, printing `-> tool(args)` / `<- tool: result` and the final message; a pause prints the `resume` command |
| `locus resume AGENT --thread ID [--answer TEXT] [--perform]` | continues a paused thread; `--perform` re-issues a held call once approved |
| `locus approvals --store FILE [--authority MODULE:ATTR] list [--json] / show ID / approve ID --by WHO / reject ID --by WHO` | reads and decides holds in a `FileApprovals` store |
| `locus audit verify FILE.jsonl [--key KEY_ID=PEM ...] [--head HASH]` | verifies an exported audit trail |
| `locus serve AGENT [--host H] [--port P] [--api-key K]` | `AgentServer` over the named agent |
| `locus codex init [DIR] / run ["prompt"] / path` | installs or runs the bundled Codex skill |

Exit codes: `0` success, `1` a check or decision failed, `2` usage error, `3` paused waiting for a person.

## Observability

`locus.observability`: `EventBus` (`get_event_bus()`), `StreamEvent(run_id, event_type, data, timestamp, event_id)`,
`run_context(run_id=None)`, `emit(event_type, **data)`, `emit_sync`, `EventBusHook(run_id=None)`; `current_run_id()`
reads the bound id (or `None`) from a tool or hook, `set_run_id(run_id)` pins it in a subprocess. Nothing publishes
outside an active `run_context`; inside one every `Agent.run` event is mirrored as `agent.*` (`agent.think`,
`agent.tool.started` / `.completed` paired by `span_id`, `agent.interrupt`, `agent.terminate`) and each shape emits its
`EV_*` constants from `locus.observability.emit` (`multiagent.orchestrator.*`, `multiagent.specialist.*`,
`multiagent.handoff.*`, `multiagent.graph.node.*`, `composition.*`, `deepagent.*`, `research.*`, `harness.exec`).

```python
import asyncio

from locus.agent import Agent, SequentialPipeline
from locus.observability import get_event_bus, run_context
from locus.testing import ScriptedModel

pipeline = SequentialPipeline(agents=[Agent(model=ScriptedModel(["a"])), Agent(model=ScriptedModel(["b"]))])

async def main() -> None:
    async with run_context() as rid:
        await pipeline.run("go")
        bus = get_event_bus()
        await bus.close_stream(rid)
        print(sorted({event.event_type async for event in bus.subscribe(rid)}))

asyncio.run(main())
```

Check: the set holds `composition.stage.started`, `composition.stage.completed`, `agent.think`, `agent.terminate`.
`subscribe(run_id)` replays up to 500 retained events, then goes live, and ends only after `close_stream(run_id)`;
`subscribe_global()` is live only. The bus is in-process, bounded and lossy: a live feed, not an audit log. For OTLP
use `locus.hooks.builtin.TelemetryHook(service_name=...)` with the standard `OTEL_EXPORTER_OTLP_ENDPOINT` variables.

## Deploy, durable execution, rogue-agent challenge

The deploy how-to ships one shape: a `server.py` that builds the `Agent`, wraps it in
`AgentServer(api_key=os.environ["LOCUS_SERVER_API_KEY"])`, exposes `app = server.app` at module scope and runs under
`uvicorn server:app --host 0.0.0.0 --port 8080`: in a container from the repository's multi-stage `Dockerfile`
(non-root, `HEALTHCHECK` on `/health`; derive an image that copies `server.py`), behind a serverless ASGI adapter, in a
Kubernetes `Deployment` with the provider credential in a Secret and a readiness probe on `/health`, or on a VM under
systemd. Two secrets: the provider credential (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_BASE_URL`, an OCI profile
named by `OCI_PROFILE`, or on OCI infrastructure `model_kwargs={"auth_type": "instance_principal", "compartment_id":
...}`) and `LOCUS_SERVER_API_KEY` for callers; an empty key turns auth off. Disable proxy buffering for SSE.

`locus.durable` imports nothing until used; both backends run the agent in segments, each at most once, a held approval
becomes a wait of any length, and workflow ids default to `locus-<thread_id>`. `locus.durable.temporal`
(`locus-sdk[temporal]`): `create_worker(client, task_queue=, agents={name: factory})`, `await start_agent_run(client,
agent=, prompt=, thread_id=, task_queue=)`, `await signal_decided(client, workflow_id, answer="approved", perform=True)`.
`locus.durable.dbos` (`locus-sdk[dbos]`): `register_agents({name: factory})`, `await start_agent_run(agent=, prompt=,
thread_id=)`, `await pending_decision(workflow_id)`, `await signal_decided(workflow_id, answer=, perform=)`; for graphs
`register_graphs`, `start_graph_run(graph=, inputs=, thread_id=)`, `pending_interrupt`, `resume_graph(workflow_id, value)`.

`python -m locus.rogue` is a smoke test of the gate: an agent with dangerous tools behind
`ControlPolicy(deny_for={"irreversible", "data-exfil"})`, reading one attack per line from stdin (`quit` ends it). With
no key it uses `CompromisedModel`, a double that calls a dangerous tool every turn, so what holds is the gate.
`LOCUS_MODEL_URL` points it at a self-hosted chat-completions server (`LOCUS_MODEL_NAME` overrides the model);
`ANTHROPIC_API_KEY` runs it against a hosted model. The scoreboard reports attempts, breaches (expected `0`) and an intact audit chain;
`CompromisedModel` is importable from `locus.testing` for your own rogue suite.
