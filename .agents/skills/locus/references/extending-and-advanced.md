# Extending Locus and the advanced surface

This file covers the public names the other four references leave out: delegating to subagents (`run_subagent`, `Subagent`, `task_tool`, `AgentSpec`) and the completion check, realtime voice sessions, operator-configured command hooks and the hook registry, sandboxed tools, writing and registering a model provider, the deprecated `locus.loop`, the `locus.core` protocols, settings, events, errors and graph primitives, the DeepAgent building blocks, evaluating a `StateGraph`, decision providers, the router's runnables, delta checkpointing and store types, run-id plumbing, and the operational modules (`locus.runner`, `locus.rogue`, `locus.codex`, the A2A wire types). Every example runs offline on doubles from `locus.testing`; where a live model id appears, `"oci:openai.gpt-5.5"` is the form. The installed source wins over this file.

## Subagents

Use `run_subagent` when a tool body should hand a focused task to an isolated child loop instead of building an `Agent` by hand: the child's tokens fold into the parent's budgets, `parent.cancel()` stops it, and its events reach the parent's stream. The child gets only the tools you list; the parent's toolset is never inherited, and a `gate_tool()` copy stays gated inside the child.

| Name (`locus.agent`) | Signature / fields |
|---|---|
| `await run_subagent(prompt, *, model, tools=None, system_prompt=..., name="subagent", max_iterations=10, hooks=None, on_event=None, cancel_signal=None, **agent_kwargs)` | one-shot child; `agent_kwargs` are `AgentConfig` fields (`token_budget`, `temperature`, `termination`); reflexion and grounding are off in the child |
| `await agent.run_subagent(prompt, *, tools=None, system_prompt=..., model=None, name=..., max_iterations=10, hooks=None, on_event=None, **agent_kwargs)` | the same, defaulting `model` to the parent's and linked to `agent.cancel()` |
| `SubagentResult` | frozen: `text`, `stop_reason` (`StopReason`), `iterations`, `tool_calls`, `usage` (`None` means unmetered, never free), `agent_name`, `task_id`, `.success` |
| `Subagent(*, model, tools=None, system_prompt=..., name=..., max_iterations=10, hooks=None, task_id=None, **agent_kwargs)` | keeps its conversation under `task_id`; `await sub.send(prompt, on_event=None, cancel_signal=None)` is one more turn; `turns` counts them; passing `checkpointer` raises `ValueError` |
| `task_tool(agents, *, tools, model, resolve_model=None, hooks=(), inherit_hooks=True, max_depth=2, max_iterations=20, registry=None, name="task", default_type=None, agent_kwargs=None)` | builds the delegation tool from `AgentSpec`s: arguments `description`, `prompt`, `subagent_type`, `task_id`; several calls in one turn run concurrently; the parent's hooks run inside the child unless `inherit_hooks=False` |
| `TaskRegistry(max_tasks=64)` | the resumable children of one session by `task_id`: `add`, `get`, `ids`, `len`; least recently used dropped past the cap |
| `AgentSpec(name, description="", prompt="", model=None, tools=None, disallowed_tools=(), mode="all", max_turns=None, token_budget=None, temperature=None, source=None, metadata={})` | frozen; `allows(tool_name)`, `select_tools(available)` (only ever a subset), `unknown_tools(available)`, `is_primary`, `is_subagent`; tool patterns are globs compared with case, `_` and `-` ignored |
| `load_agent_specs(sources, *, on_error=None)` | reads every `*.md` with YAML frontmatter and the body as the prompt from directories, later sources winning; a `(directory, default_mode)` pair sets the mode for files that carry none; `disable: true` removes an earlier spec of the same name |

```python
import asyncio

from locus.agent import Agent, run_subagent
from locus.core.events import SubagentEvent
from locus.core.messages import Role
from locus.testing import FunctionModel, ScriptedModel, text, tool_call
from locus.tools import tool


@tool
def read_file(path: str) -> str:
    """Return a file's contents."""
    return "def refund(order_id): ...\n"


@tool(emits_progress=True)
async def delegate(prompt: str) -> str:
    """Hand a read-only question to a subagent and return its answer."""
    child = ScriptedModel([
        tool_call("read_file", path="billing.py"),
        text("billing.py:1 defines refund(order_id)."),
    ])
    result = await run_subagent(
        prompt, model=child, tools=[read_file], name="explorer", max_iterations=4
    )
    return f"{result.text} [stop={result.stop_reason}, iterations={result.iterations}]"


def parent(messages, tools):
    if any(m.role == Role.TOOL for m in messages):
        return text("The refund entry point is billing.py:1.")
    return tool_call("delegate", prompt="Where is refund defined?")


agent = Agent(model=FunctionModel(parent), tools=[delegate], reflexion=False, grounding=False)


async def main() -> None:
    nested = []
    async for event in agent.run("Find the refund entry point."):
        if isinstance(event, SubagentEvent):
            nested.append((event.agent_name, event.event.event_type))
    print(nested)


asyncio.run(main())
print(agent.run_sync("Find the refund entry point.").tool_executions[0].result)
```

Prints `[('explorer', 'think'), ('explorer', 'tool_start'), ('explorer', 'tool_complete'), ('explorer', 'think'), ('explorer', 'terminate')]`, then `billing.py:1 defines refund(order_id). [stop=complete, iterations=2]`. The child's events arrive wrapped in `SubagentEvent` (`tool_call_id`, `tool_name`, `task_id`, `event`) between the delegating call's start and complete events, only because the tool declared `emits_progress=True`; a child's `TerminateEvent` is never mistaken for the parent's. A stream that ends without a terminate event (the child paused on `ask_user`) reads as `stop_reason="interrupted"`. The DeepAgent variant is `locus.deepagent.task_tool(subagents, *, parent_model)` over `SubAgentDef`s; the harness variant is `locus.agent.task_tool` over `AgentSpec`s, with `task_id` resumption and nesting capped by `max_depth`.

`CompletionCheck` (`locus.agent`) is a `final_answer_verifier` for models that announce a step and stop: `CompletionCheck(max_nudges=3, announcements=True, needs_changes=None, unchecked_edits=False, on_continuation=None, insist_on_changes=False)`; `check.agent_options()` returns the `final_answer_verifier` and `final_answer_verifier_max_replans` keywords, `check.continuations(run_id)` counts nudges. Each nudge is a `Continuation(text, reason)` (a `str` subclass with `reason` in `announced_step`, `no_changes`, `unchecked_edits`); the reply stays in the conversation and the loop emits `FinalAnswerVerificationEvent(continuation=True, reason=...)`. `chain_verifiers(*verifiers)` folds several verifiers into one, the first rejection deciding; give the agent the sum of their replans (`max_replans_for` in `locus.agent.completion`).

```python
import asyncio

from locus.agent import Agent, CompletionCheck
from locus.core.events import FinalAnswerVerificationEvent
from locus.testing import ScriptedModel, text

check = CompletionCheck(max_nudges=2)
model = ScriptedModel([
    text("Let me check the tests first:"),
    text("All tests pass; the fix is in."),
])
agent = Agent(model=model, reflexion=False, grounding=False, **check.agent_options())


async def main() -> None:
    async for event in agent.run("Fix the failing test."):
        if isinstance(event, FinalAnswerVerificationEvent):
            print(event.passed, event.continuation, event.reason)
    print(model.call_count)


asyncio.run(main())
```

Prints `False True announced_step`, `True False None`, `2`: the announced step was sent back once, the second reply was accepted. `ObservationPackConfig` (`locus.agent`, `Agent(observation_pack=True)` or an instance: `enabled`, `directory`, `threshold_bytes=10240`, `max_inline_chars=128000`, `full_sends=2`, `excerpt_bytes`, `recall_max_bytes`, `recall_max_lines`, `min_batch_bytes`, `cache_read_cost`, `cache_write_cost`, `horizon_requests`, `min_horizon_requests`) sends a large old tool output whole for its first `full_sends` requests and then as a placeholder that the auto-registered `obs_recall` tool pages back from a per-session archive; children started under it inherit a pack. `StopReason` is the `Literal` of the fourteen stop strings; `StreamingResult(state, partial_content, iteration, is_complete, final)` is the partial-result shape.

## Voice

Use `locus.voice` when a speech-to-speech model must call the same gated tools as a text agent. `RealtimeSession(connection, *, tools=(), instructions="", voice=None)` in `locus.voice.realtime` drives any socket with `async send(event)` and `async recv()` (`RealtimeConnection`): `await session.start()` declares instructions and tools, `send_audio(pcm)` and `send_text(text)` speak for the user, and `async for event in session.events()` yields `AudioDelta(audio)`, `Transcript(role, text)`, `ToolRan(name, arguments, result)`, `ActionHeld(name, arguments, question, approval_id, metadata)`, `ResponseDone()` and `SessionError(message)` (`RealtimeEvent` is their union). Each model tool call runs `tool.execute(**arguments)`, so `gate_tool`, hooks, approval stores and the trail apply unchanged; a held call is not performed, the model is told it is waiting for a person, and the application gets `ActionHeld` with the `approval_id` to route. An unknown tool or malformed arguments become a `SessionError` and an error payload to the model. `realtime_tool_schema(tool)` is the declaration a session sends. Two connectors open sessions: `async with connect_openai_realtime(*, model="gpt-realtime", tools=(), instructions="", voice=None, api_key=None) as session` (the `openai` extra) and `async with connect_gemini_live(*, model, tools=(), instructions="", voice=None, api_key=None, config=None) as session` (`locus.voice.gemini`, the `gemini` extra; input audio is 16 kHz 16-bit mono PCM, output 24 kHz; `GeminiLiveConnection(session)` is the socket adapter and `gemini_live_config(...)` the connect config). Both need a live realtime endpoint, so there is no offline example; unit-test the gate on the tools themselves.

## External hooks and the registry

Use `ExternalHooks` (`locus.hooks`) when an operator configures hooks as programs in a settings file rather than as Python: a command (or an HTTP URL) run at a lifecycle event, handed the event as JSON on stdin, answering with an exit code and optional JSON on stdout. Exit 0 succeeds and stdout is read for `decision` / `reason`, `continue` / `stopReason`, `systemMessage` and `hookSpecificOutput` (`permissionDecision`, `permissionDecisionReason`, `updatedInput`, `additionalContext`); exit 2 blocks with stderr as the reason; any other exit, a timeout or a crash is a non-blocking error that is reported and otherwise ignored. Events: `PreToolUse` (deny cancels the call, `updatedInput` rewrites the arguments, allow or ask goes to `on_permission`), `PostToolUse` (a block reason or context is appended to the result the model reads), `UserPromptSubmit` (a block raises `HookBlockedError(event, reason)`; context is appended to the prompt), `Stop` / `SubagentStop` (a final-answer verifier through `hooks.verifier(event="Stop")`), and `SessionStart`, `SessionEnd`, `PreCompact`, `Notification`, which the host fires itself with `hooks.run(event, payload, subject=None)` or `await hooks.arun(...)`.

| Name | Fields / arguments |
|---|---|
| `HookConfig.from_settings(block, *, source="")` | `{"Event": [{"matcher": "regex", "hooks": [{"type": "command", "command": "...", "timeout": 30}]}]}`; `type: "http"` takes `url` and `headers`; malformed entries land in `config.problems`; `merged(other)`, `for_event(event, subject)`, `has(event)` |
| `HookCommand(type, target, timeout=60.0, matcher="", source="", headers={})` | frozen; `matches(subject)`: empty or `*` matches all, else a whole-subject case-insensitive regex |
| `ExternalHooks(config, *, cwd=None, session_id="", env=None, on_run=None, guard=None, on_permission=None, payload=None)` | a `HookProvider` at `HookPriority.SECURITY_DEFAULT`; `guard(hook, event) -> refusal or None` is consulted before each run; `on_run(HookRun)` sees every execution, refused ones included |
| `HookRun(event, hook, exit_code, duration_ms, stdout, stderr, error, decision, reason)` | one execution; `as_dict(limit=400)` for logs; output is capped at 20000 characters |
| `HookOutcome(event, runs, decision, reasons, context, updated_input, system_messages, stop, stop_reason)` | every hook of one event folded, strongest decision wins; `blocked`, `reason`, `additional_context` |
| `ToolCallVerdict(arguments, cancel=None)` | what `hooks.pre_tool_use(tool_name, arguments, *, tool_use_id="", session_id=None)` returns for a nested call; `hooks.post_tool_use(...)` returns the notes to append |

```python
import sys

from locus.agent import Agent
from locus.hooks import ExternalHooks, HookConfig
from locus.testing import ScriptedModel, text, tool_call
from locus.tools import tool


@tool
def run_shell(command: str) -> str:
    """Run a shell command."""
    return f"ran {command}"


guard = (
    f'{sys.executable} -c "import json, sys; call = json.load(sys.stdin); '
    "pushing = 'push' in call['tool_input']['command']; "
    "sys.stderr.write('agents do not push' if pushing else ''); "
    'sys.exit(2 if pushing else 0)"'
)
config = HookConfig.from_settings(
    {"PreToolUse": [{"matcher": "run_shell", "hooks": [{"type": "command", "command": guard}]}]},
    source="settings.json",
)
runs = []
hooks = ExternalHooks(config, on_run=runs.append)
model = ScriptedModel([
    tool_call("run_shell", command="git push origin main"),
    tool_call("run_shell", command="git status"),
    text("Done."),
])
agent = Agent(model=model, tools=[run_shell], hooks=[hooks], reflexion=False, grounding=False)
result = agent.run_sync("ship it")
print([t.result for t in result.tool_executions])
print([(r.exit_code, r.decision) for r in runs], config.problems)
```

Prints `['Refused by a PreToolUse hook: agents do not push', 'ran git status']` and `[(2, 'deny'), (0, None)] ()`: the hook saw `tool_name`, `tool_input`, `tool_use_id`, `session_id`, `cwd` and `hook_event_name` on stdin, its exit 2 cancelled the first call with stderr as the reason the model reads, and both runs were reported. Command hooks run with `LOCUS_PROJECT_DIR` and `LOCUS_HOOK_EVENT` in the environment and their process group is killed on timeout. A hook is a side effect the operator configured: gate it with `guard` and record `on_run` on the trail.

`HookRegistry` (`locus.hooks`) is the priority-ordered dispatcher for a host that runs hooks itself rather than through `Agent(hooks=[...])`, which takes a plain list of providers: `add_provider(provider)` (duplicate names raise `ValueError`), `remove_provider(name)`, `get_provider(name)`, `providers`, `name in registry`, and `await registry.emit_before_invocation(prompt, state)`, `emit_after_invocation(state, success)`, `emit_before_tool_call(tool_name, arguments)`, `emit_after_tool_call(tool_name, result, error, *, tool_call_id="", arguments=None)`, `emit_iteration_start(iteration, state)`, `emit_iteration_end(...)`, `emit_before_model_call(messages, tools)`, `emit_after_model_call(response, messages)`, `emit(event_name, *args, **kwargs)`; `create_registry(*providers)` builds one. Before-hooks run in priority order and raise through; after-hooks run in reverse, and the after-invocation and after-tool emitters collect provider errors into one `RuntimeError`. `ProtectedEvent` is the base of the writable hook events `BeforeModelCallEvent`, `AfterModelCallEvent`, `BeforeToolCallEvent`, `AfterToolCallEvent` and `BeforeCompactionEvent`: each names its `_writable` fields, any other assignment raises `AttributeError`, `_redacted` fields show as `***` in `repr`, `event.run` is the `RunInfo` (`run_id`, `thread_id`, `metadata`, `agent_name`), and `event.emit(CustomEvent(...))` reaches the stream. The plain info events `HookEvent`, `BeforeInvocationEvent`, `AfterInvocationEvent`, `IterationStartEvent(iteration, agent_id)` and `IterationEndEvent(iteration, agent_id, tool_calls_made, confidence)` and the `HookResult(provider_name, success, result, error)` container are in `locus.hooks.events`.

## Tool sandboxes

Use `@tool(sandbox=...)` when a tool body must not run in the host process: the function's source is shipped into an isolated interpreter and executed there. The body must be synchronous, module-level, with its imports inside, taking and returning JSON-serializable values and no `ctx` parameter; `validate_sandboxable` enforces that at decoration time and raises `SandboxError`. `sandbox=True` picks the default provider (`LOCUS_SANDBOX`, else `subprocess`), a name selects `"subprocess"` / `"local"`, `"docker"`, or a provider registered under the `locus.sandbox_providers` entry-point group, and a `SandboxSpec(provider=None, timeout=30.0, env={}, deps=[], workspace=None)` or a provider object configures it. Direct calls (`tool(**kwargs)`) are sandboxed too.

| Name (`locus.tools`) | Meaning |
|---|---|
| `ToolSandbox` | the protocol: `run_tool(code, func, args=None, *, manifest=None, timeout=30.0)` returning an object with `ok`, `stdout`, `stderr`, `value`, `exit_code`, `timed_out`, `duration_ms` |
| `SubprocessSandbox(python=None)` | fresh working directory, `python -I`, environment scrubbed to `PATH` and `LANG` plus the manifest's `env`; process and environment isolation only, not a network or filesystem boundary; `run_code(code, *, manifest=None, timeout=30.0)` runs a program |
| `SandboxManifest(files={}, env={}, deps=[], workspace=None)` | what the box receives: seeded files (paths may not escape the workspace), the only environment it sees, packages installed into a local `.deps`, a persistent directory |
| `SandboxResult(ok, stdout, stderr, value, exit_code, timed_out, duration_ms)` | one run; a timeout is `exit_code=124`, `timed_out=True` |
| `SandboxError` / `SandboxExecutionError(message, *, result)` | the sandbox itself failed (bad provider, escaped path, unshippable function) / the code ran and failed, with the box's diagnostics on `result` |
| `SandboxEnforcerHook(policy, tools, *, priority=HookPriority.SECURITY_DEFAULT)` | cancels a call to a tool whose `labels` intersect `policy.require_sandbox_for` and whose `sandbox` is unset; emits `tool.sandbox.denied` |
| `generate_schema(fn, description=None)` / `pydantic_to_json_schema(model)` | the OpenAI-format tool schema `@tool` builds (type hints to JSON schema, `Args:` descriptions, `ctx` skipped) / `model.model_json_schema()` |

```python
import json

from locus.agent import Agent
from locus.testing import ScriptedModel, text, tool_call
from locus.tools import SandboxSpec, SubprocessSandbox, tool


@tool(sandbox=SandboxSpec(timeout=20), labels={"code-exec"})
def word_count(text: str) -> dict:
    """Count words in a text, inside an isolated interpreter."""
    import collections

    return dict(collections.Counter(text.split()))


print(word_count(text="a b a"))
model = ScriptedModel([tool_call("word_count", text="x y x"), text("Counted.")])
agent = Agent(model=model, tools=[word_count], reflexion=False, grounding=False)
print(json.loads(agent.run_sync("count").tool_executions[0].result))
run = SubprocessSandbox().run_tool("def double(n):\n    return n * 2", "double", {"n": 21})
print(run.ok, run.value, run.exit_code, run.timed_out)
```

Prints `{'a': 2, 'b': 1}`, `{'x': 2, 'y': 1}` and `True 42 0 False`. `Tool.execute` emits `tool.sandbox.started` / `tool.sandbox.completed` on the event bus and raises `SandboxExecutionError` (`sandboxed tool 'word_count' failed: ...`, or `timed out`) when the box fails; inside an agent the executor records that as the tool's error and the run continues. `DockerSandbox(image="python:3.12-slim", *, docker="docker", network=False, memory="512m", cpus=1.0, pids_limit=256, user=None)` in `locus.tools.sandbox` adds a network and filesystem boundary (read-only root, no capabilities, a fresh container per call); `manifest.deps` needs `network=True`. Pair the hook with `ControlPolicy(require_sandbox_for={"code-exec"})` and an `action` that adds `SANDBOXED_TAG` when `tool.sandbox is not None`, as the control reference describes.

## Custom model providers

Use `register_provider` when the model lives behind a transport the registered prefixes do not speak, or when every prompt must pass through your own proxy. A provider is any object satisfying `ModelProtocol` (`locus.models`, also `locus.core.protocols`): `async complete(messages, tools=None, **kwargs) -> ModelResponse` and `stream(messages, tools=None, **kwargs)`, an async generator of `ModelChunkEvent`. Return `locus.models.ModelResponse(message, usage={}, stop_reason=None, provider_state=None, reasoning=None, logprobs=None, candidates=[], metadata={}, cost_usd=None)`; the loop reads `message.content`, `message.tool_calls` (`ToolCall(id, name, arguments)` on the assistant `Message`), `usage` (`prompt_tokens`, `completion_tokens`) and `stop_reason`. `register_provider(prefix, factory)` takes `factory(model_id, **kwargs)`; `get_model("prefix:id", **kwargs)` and `Agent(model="prefix:id", model_kwargs={...})` then build it, and `list_providers()` shows the prefix.

```python
from collections.abc import AsyncIterator

from locus.agent import Agent
from locus.core.events import ModelChunkEvent
from locus.core.messages import Message
from locus.models import ModelResponse, get_model, list_providers, register_provider


class FixedModel:
    """A provider that answers every request with one line."""

    def __init__(self, model: str, *, answer: str = "fixed answer") -> None:
        self.model = model
        self.answer = answer

    async def complete(self, messages, tools=None, **kwargs) -> ModelResponse:
        return ModelResponse(
            message=Message.assistant(self.answer),
            usage={"prompt_tokens": 1, "completion_tokens": 1},
            stop_reason="end_turn",
        )

    async def stream(self, messages, tools=None, **kwargs) -> AsyncIterator[ModelChunkEvent]:
        yield ModelChunkEvent(content=self.answer)
        yield ModelChunkEvent(done=True, stop_reason="end_turn")


register_provider("fixed", lambda model_id, **kwargs: FixedModel(model_id, **kwargs))
print("fixed" in list_providers(), type(get_model("fixed:demo")).__name__)
agent = Agent(model="fixed:demo", model_kwargs={"answer": "42"}, reflexion=False, grounding=False)
result = agent.run_sync("What is the answer?")
print(result.message, result.stop_reason, result.metrics.total_tokens)
```

Prints `True FixedModel` and `42 complete 2`. Expose the id as `model` (or `config.model`) so `locus.models.metadata.model_id_of` can name it; register a `ModelMetadata` for it to silence the context-window notice and to price `max_cost_usd`. Tool calls come back as `Message.assistant(content, tool_calls=[ToolCall(...)])`; set `stop_reason` as the provider reports it. `RequestBuilder` (`build(messages, tools, **kwargs)`) and `ResponseParser` (`parse(response) -> ModelResponse`) are the two protocols a transport is split into when you want the wire shape testable apart from the client. `ModelConfig(model, max_tokens=4096, temperature=0.7, top_p=0.9, stop_sequences=[])` is the base the shipped providers extend with `extra="allow"`: `OpenAIConfig` adds `api_key`, `base_url`, `organization`, `default_headers`, `api`, `max_retries`, `request_timeout`, `stream_reconnects`, `keepalive`, `frequency_penalty`, `presence_penalty`, `seed`, `stream_usage`, `extra_body`; `AnthropicConfig` adds `api_key`, `base_url`, `auth_token`, `default_headers`, `prompt_cache`, `max_retries`, `request_timeout`; `OCIChatCompletionsConfig` extends `OpenAIConfig` with `region`, `profile`, `auth_type`, `config_file`, `compartment_id`; `OCIConfig` (the SDK transport) has `model_id`, `compartment_id`, `profile_name`, `config_file`, `auth_type` (`OCIAuthType`), `service_endpoint`, `frequency_penalty`, `presence_penalty`, `stop_sequences`; `OCIResponsesConfig` has `model`, `region`, `profile`, `auth_type`, `config_file`, `compartment_id`, `project_ocid`, `base_url`, `max_output_tokens`, `temperature`, `request_timeout`, `store`. All are importable from `locus.models`; the provider classes take the same names as keywords.

## The ReAct loop (`locus.loop`)

Do not build on `locus.loop`. It is a second ReAct implementation that `Agent` never used, kept for compatibility: every attribute access emits `LocusDeprecationWarning` (`python -W error::DeprecationWarning` finds them). `ReActLoop(model=, registry=, config=ReActLoopConfig(max_iterations=20, confidence_threshold=0.85, enable_reflection=True, reflect_interval=1, system_prompt=None, terminal_tools=...))`, `create_react_loop(model, registry, *, ...)`, the nodes `Node` (`async execute(state) -> NodeResult(state, events)`), `ThinkNode(model, registry, system_prompt)`, `ExecuteNode(registry, executor)`, `ReflectNode(confidence_adjustments)`, the `Router` / `ConditionalRouter(custom_conditions).add_condition(name, fn)` returning `RouteDecision(next_node: NodeType, reason, metadata)`, and the runners `LoopRunner`, `BatchRunner`, `StreamingCollector`, `create_runner` map onto `Agent`, `Agent.arun`, `EvalRunner`, `async for event in agent.run(...)`, hooks, and `StateGraph` conditional edges. A builder touches it only to migrate code that imports it.

## Core protocols, settings, events, errors

`locus.core` holds the contracts the rest of the SDK is written against; implement them to substitute a part.

| Name | Contract |
|---|---|
| `ModelProtocol` | `complete` and `stream`, as above; `locus.core.protocols.ModelResponse` is a plain class, use `locus.models.ModelResponse` |
| `ToolProtocol` | `name`, `description`, `parameters` (JSON schema) and `async execute(**kwargs)`; `Tool` from `@tool` satisfies it |
| `CheckpointerProtocol` | `capabilities` (`CheckpointerCapabilities`), `async save(state, thread_id, checkpoint_id=None, metadata=None) -> str`, `load(thread_id, checkpoint_id=None)`, `list_checkpoints(thread_id, limit=10)`; subclass `locus.memory.BaseCheckpointer` for the optional methods |
| `LocusSettings` | `pydantic-settings` root read from `LOCUS_*` and `.env`: `env`, `debug`, nested `model`, `agent`, `telemetry`, `checkpointer` (`LOCUS_MODEL__...` with `__`); `get_settings()` and `configure(settings)` in `locus.core.config`; the HTTP servers read `debug`, `Agent` reads only its `AgentConfig` |
| `ToolCall(id, name, arguments={}, malformed_arguments=None)` | a model's request; `args` is accepted as an alias of `arguments`; `to_openai_format()` |
| `SubagentEvent` | `tool_call_id`, `tool_name`, `task_id`, `event` (the child's event, `agent_name` set); see Subagents |
| `FinalAnswerVerificationEvent` | `passed`, `attempt`, `replanning`, `feedback`, `error` (the verifier raised and the draft was accepted), `replaced` (`final_answer_fallback` answered), `continuation`, `reason` |
| `RAGError` / `EmbeddingError` / `VectorStoreError` | `kind` `rag_error`, `embedding_error`, `vector_store_error`; the rest of the hierarchy is in the agent reference |

Graph primitives beyond `Send`, `Command` and `interrupt`: `End(update={})` ends the graph (`goto="__END__"`), `Continue(update={})` updates and follows the normal edges, `goto(target, **updates)`, `end(**updates)` and `resume_with(value, **updates)` build commands, `is_command(value)` tests one, and `normalize_node_output(output)` is what the executor applies to a node's return (`dict` is an update, `Command` splits into update and routing, a `BaseModel` is dumped, anything else becomes `{"result": value}`). Interrupts: `interrupt(payload, **metadata)` raises `InterruptException(value: InterruptValue)` on the first pass and returns the resume value on the second; `InterruptValue(interrupt_id, payload, node_id, graph_id, created_at, metadata)` with `to_display()`, `InterruptState(interrupt, node_id, pending_nodes, partial_results, state_snapshot)` is what a checkpoint keeps; the executor itself reports a pause on `GraphResult.interrupt`, and `GraphInterrupted(interrupt_state, checkpoint_id=None)` plus `InterruptHandler` (`async handle(interrupt) -> response`, `can_handle(interrupt)`; `AutoApproveHandler(response="approved")` for tests) are the exception and handler shapes for a host that drives interrupts itself. Fan-out: `send(node, **payload)`, `broadcast(nodes, payload=None)`, `scatter(node, items, key="item", include_index=True)`, `SendResult(send_id, node, success, result, error, duration_ms)`, `SendBatch(sends, source_node, aggregator_node)`, `is_send`, `is_send_list`, `normalize_sends(value) -> list[Send] | None`, `extract_send_results(results, key="result") -> {send_id: result}` (successes only) and `aggregate_send_results(results, reducer=None)`. Reducers (`locus.core.reducers`) satisfy the `Reducer` protocol, `(current, update) -> merged`, and are declared with `Annotated[list[Message], add_messages]` on a state model: `add_messages` (append, replace by id, `AddMessages.REMOVE_ALL` clears), `merge_dict`, `deep_merge_dict`, `append_list`, `unique_append_list`, `add_numbers`, `max_value`, `min_value`, `last_value` (the default), `first_value`, `set_union`; `get_reducer(annotation)` reads one, `apply_reducers(current, update, reducers)` merges a state dict, `@reducer` wraps a function.

## DeepAgent internals

Use these when `create_deepagent(...)` is not enough and you assemble the pieces yourself; the factory only composes them.

| Name (`locus.deepagent`) | Meaning |
|---|---|
| `BackendProtocol` | `read(path, *, offset=0, limit=100) -> str`, `write(path, contents)`, `edit(path, old_str, new_str)`, `exists(path)`, `ls(path="/", *, recursive=False) -> list[FileInfo]`, `glob(...)`, `grep(...) -> list[Match]`; failures raise `BackendError(code, path="")` with `code` in `file_not_found`, `permission_denied`, `is_directory`, `invalid_path` |
| `StateBackend()` / `FilesystemBackend(root)` | in-memory scratch space that dies with the run / a directory on disk, paths confined to `root` |
| `FileInfo(path, is_dir, size, modified_at)` / `Match(path, line, text)` | what `ls`, `glob` and `grep` return |
| `make_filesystem_tools(backend)` | the six tools `write_file`, `read_file`, `ls`, `edit_file`, `glob`, `grep`; errors come back to the model as the code string; `deepagent.fs.read` / `deepagent.fs.write` on the bus |
| `Todo(content, status="pending")`, `TodoState()`, `make_todo_tools(state=None)` | `write_todos(todos_json)` replaces the list, `read_todos()` returns it; `state.snapshot()` reads it from outside; `deepagent.todo.added` / `deepagent.todo.completed` |
| `load_agents_md(paths)` | joins the Markdown files that exist, each labelled, for `memory_files=` |
| `SubAgentDef(name, description, system_prompt, tools=[], model=None, max_iterations=10)` and `task_tool(subagents, *, parent_model)` | the one-shot `task(subagent_type, description)` tool, built on `run_subagent`; `deepagent.subagent.spawned` / `deepagent.subagent.completed` |
| `KnowledgeProvider` | a research domain: `async open()`, `close()`, `discover(query=None) -> list[ItemRef]`, `ground(item) -> Grounding`, `tools_for_agent()`, `output_schema()`, `merge_to_row(item, grounding, research, *, model_id, prompt_hash) -> KnowledgeRow` |
| `ItemRef(name, provider, key="", tags={})` / `KnowledgeRow(name, provider, short_description, long_description, domains, tags, confidence, embedding, search_text)` | a discoverable item (`key` defaults to `provider:name`) / one catalog row, subclass it for typed fields |

The research workflow's nodes are plain `StateGraph` node factories over a `dict` state with well-known keys: `make_execute_node(model, tools, system_prompt="", reflexion=True, max_iterations=20)`, `make_causal_inference_node(model, max_nodes=10)`, `make_summarize_node(model, output_schema=None)`, `make_grounding_eval_node(model)`, `make_regenerate_summary_node(model, output_schema=None)`, `make_replan_node()`, and `route_after_grounding(threshold=0.65, max_replans=2, max_regenerations=1)` for the conditional edge; the keys are `KEY_PROMPT`, `KEY_EXECUTE_PROMPT`, `KEY_EVIDENCE`, `KEY_GROUNDING_FACTS`, `KEY_CAUSAL_CHAIN`, `KEY_CAUSAL_HYPOTHESIS`, `KEY_CAUSAL_CONFIDENCE`, `KEY_SUMMARY`, `KEY_UNGROUNDED_CLAIMS`, `KEY_REGENERATION_COUNT`, `KEY_STOP_REASON` beside the three the multi-agent reference names. Each node emits a `research.*` event. `create_deepagent` also takes `backend=`, `todo_state=`, `total_token_budget=`, `max_output_tokens=`, `datastore_top_k=5`, `summarize_after_messages=`, `summarize_keep_recent=10`; with `output_schema` it sets `completion_mode="explicit"` and an empty `terminal_tools`, so the only exits are `ToolCalled(submit_tool, require_success=True) & ConfidenceMet(min_confidence)` and the budgets.

## Evaluating graphs

Use `as_eval_target` when a regression suite must run against a `StateGraph`: `EvalRunner` calls `run_sync(prompt)` and reads `message`, `iterations` and `tool_executions`, which a graph does not have. `as_eval_target(graph, *, input_key="prompt", output_key=None, output_node=None, initial_state=None)` (also `GraphEvalTarget(...)`) puts the case prompt under `input_key`, grades `final_state[output_key]` or one node's output (`output_node`; passing both raises `ValueError`; neither grades the whole state minus `_node_*` entries), and presents the nodes that ran, in order, as tool executions, so `expected_tools` and `expected_tool_sequence` match node ids. `check_trajectory(actual, expected, *, exact=False) -> (passed, reason)` is the ordering assertion on its own: in-order containment by default, equality with `exact=True`.

```python
from locus.evaluation import EvalCase, EvalRunner, as_eval_target, check_trajectory
from locus.multiagent import END, START, StateGraph


def classify(state: dict) -> dict:
    return {"route": "billing" if "charged" in state["question"] else "general"}


def billing(state: dict) -> dict:
    return {"answer": "Refund issued for the duplicate charge."}


def general(state: dict) -> dict:
    return {"answer": "Forwarded to support."}


graph = StateGraph(name="triage")
graph.add_node("classify", classify)
graph.add_node("billing", billing)
graph.add_node("general", general)
graph.add_edge(START, "classify")
graph.add_conditional_edges(
    "classify", lambda s: s["route"], {"billing": "billing", "general": "general"}
)
graph.add_edge("billing", END)
graph.add_edge("general", END)

target = as_eval_target(graph, input_key="question", output_key="answer")
report = EvalRunner(agent=target).run([
    EvalCase(
        name="billing-route",
        prompt="I was charged twice",
        expected_tool_sequence=["classify", "billing"],
        expected_output_contains=["Refund"],
    )
])
print(report.passed, report.failed, report.results[0].checks)
print(check_trajectory(["classify", "lookup", "billing"], ["classify", "billing"]))
print(check_trajectory(["billing", "classify"], ["classify", "billing"]))
```

Prints `1 0 {'output_contains:Refund': True, 'tool_sequence': True}`, `(True, 'trajectory contains the expected order')` and `(False, "expected ['classify', 'billing'] in order; ['billing'] did not follow, got ['billing', 'classify']")`. The real `GraphResult` stays on `result.graph_result` for node-level detail. `JudgeUnavailableError` is raised by `LLMJudge.score(...)` and out of `EvalRunner` when the judge model cannot be called: a rubric that could not be graded is not a failing agent, so it propagates instead of scoring zero; an unparseable verdict is a failed check labelled `judge_unparseable`.

## Decisions

The provider seam under `LogprobDecider` is `DecisionProvider` (`locus.decision`): `async decide(text, fields) -> Decision`, raising `DecisionError` rather than guessing; `decide_sync(provider, text, fields)` runs it from synchronous code. `TenantDecisionRouter(providers, *, audit_for=None, public=None, record_text=False)` serves each tenant only from its own head (`decide(text, fields, *, tenant=)`, `decide_sync(...)`, `for_tenant(tenant)` returns a bound provider for a `DecisionAdvisor`), refuses an unknown tenant with `UnknownTenantError` unless a `public` head is named, and records every decision on that tenant's `AuditTrail` from `audit_for(tenant)`, failing the decision when the record cannot be written; `per_tenant_trails(directory)` is the one-JSONL-file-per-tenant `audit_for`. The wire contract is `SYSTEM_PROMPT`, `LETTERS` (`A` to `Z`, one per option), `YES_NO` (`("yes", "no")`) and `answer_from_logprobs(question, top_logprobs) -> Answer`. For the gate, `ADMIT_FIELD` is `Choice("admit", ADMIT_QUESTION, (allow, require_human, deny))`, `describe_action(action)` renders an `Action` for a head, and `verification_from_decision(decision, thresholds)` turns yes/no safety heads into the `VerificationResult` a `ControlPolicy` weighs (a head at or over its threshold is a fatal refutation).

## Router

`locus.router` builders return a `Runnable` (`async execute(task) -> RunnableResult(text, parsed, raw, protocol_id, frame)`), one adapter per primitive: `AgentRunnable(agent, protocol_id, frame)`, `PipelineRunnable(pipeline, ...)` for the three composition classes, `OrchestratorRunnable(orchestrator, ...)`, `DebateRunnable(debaters, judge, ...)` (parallel debaters, then a judge agent whose verdict is the text) and `A2ARunnable(client, ...)` for a remote peer. A `Protocol(id, description, handles, primary_for=[], requires_capabilities=[], risk_max=Risk.MEDIUM, cost="medium", latency="medium", supports_streaming=True, supports_repair=True, builder)` is built by `builder(frame, capabilities, ctx: BuilderContext)`, where `BuilderContext(model, capabilities, skills=None, a2a_endpoint=None, hooks=[])` carries the model, the `CapabilityIndex`, a `SkillIndex` (`register(skill, *, domain="")`, `for_domain(domain)` returns the domain's skills plus the global ones) and the hooks every emitted agent must carry. `HUMAN_SENTINEL` (`"$human"`) is the `tool_name` of a capability only a person performs. `CognitiveCompiler(..., on_approval=, protocol_picker=, risk_fallback="approval_gated_execution")`: `ApprovalCallback` is `async (frame, verdict) -> bool`, consulted before a runnable the gate held executes (default denies, raising `PolicyDeniedError`); `LLMProtocolPicker(*, model, system_prompt=None)` lets a model choose among the already-filtered candidates as a `PickedProtocol(protocol_id, rationale)`, falling back to the rule ranking on `PickerError`; `NoMatchingProtocolError` means nothing handles the frame. `Router.extract` raises `FrameExtractionError` when the extractor returns no `GoalFrame`.

## Memory extras

`DeltaCheckpointer(storage=None, max_chain_depth=5, compression_level=6)` (`locus.memory`) stores each save as a zlib-compressed delta from the previous checkpoint and a full snapshot every `max_chain_depth` deltas; `save(state, thread_id, checkpoint_id=None)`, `load(thread_id, checkpoint_id=None)`, `list_checkpoints`, `get_metadata(thread_id, checkpoint_id) -> CheckpointMetadata(checkpoint_id, thread_id, parent_id, is_full, chain_depth, created_at, size_bytes, compressed_size_bytes)`, `delete`, `get_storage_stats(thread_id)`. Storage is a `DeltaStorage` (`store`, `retrieve`, `list_checkpoints`, `delete` over `DeltaCheckpoint(metadata, data, is_delta)`); `InMemoryDeltaStorage()` is the test one. It has no `capabilities`, so it is not a `BaseCheckpointer`; drive it yourself.

```python
import asyncio

from locus.core.messages import Message
from locus.core.state import AgentState
from locus.memory import DeltaCheckpointer, InMemoryDeltaStorage, list_checkpointers


async def main() -> None:
    checkpointer = DeltaCheckpointer(InMemoryDeltaStorage(), max_chain_depth=3)
    state = AgentState()
    for turn in ("open case 4821", "customer confirms the address", "refund approved"):
        state = state.with_message(Message.user(turn))
        await checkpointer.save(state, "case-4821")
    restored = await checkpointer.load("case-4821")
    stats = await checkpointer.get_storage_stats("case-4821")
    print(len(restored.messages), stats["full_checkpoints"], stats["delta_checkpoints"])
    print(len(list_checkpointers()), "oracle" in list_checkpointers())


asyncio.run(main())
```

Prints `3 1 2` and `10 True`. `register_checkpointer(name, factory)` adds a name to the `get_checkpointer("name:hint")` registry; the factory receives `config_hint=` plus keywords. Store types: `StoreCapabilities(search, semantic_search, embedding_dimension, ttl, list_namespaces, batch_operations, transactions)` is what `store.capabilities` reports; `StoreCapabilityError(capability, store_class, hint=None)` (a `NotImplementedError`) is raised for an unsupported operation; `StoreItem(namespace, key, value, metadata, created_at, updated_at, version)` is a stored record; `SemanticSearchResult(item, score, distance=None)` comes from vector search. `MemoryScope(namespace, recall=())` is what a `namespace_resolver` returns when a run should also recall shared memories it never writes to; every `recall` prefix must share `namespace[0]`.

## Observability and streaming extras

`current_run_id()` returns the run id bound on the current context or `None`; `set_run_id(run_id)` returns a token and `reset_run_id(token)` restores the previous value, for a host that carries the id across a process boundary where `run_context()` cannot; `reset_event_bus()` discards the singleton bus and every subscription, for tests only. `StreamHandler` (`locus.streaming`) is the protocol (`async on_event(event)`, `on_complete()`, `on_error(error)`) and `BaseStreamHandler` the base that needs only `on_event`; `SSEMessage(event=None, data, id=None, retry=None).format()` renders one Server-Sent Event (`event:`, `id:`, `retry:`, one `data:` line per line of data, a blank line).

## Operational modules

- `locus.runner` is the box runner's side of the gateway protocol, by family: the pinned manifest (`RunManifest` with its `ToolEntry`, `ModelRoute`, `McpMount` and `ApiConnector` entries, `canonical_digest`), the gateway client (`GatewayClient`, `RunnerConfig.from_env()` reading the admit URL, token and run id from the environment; `GatewayError`, `GatewayUnavailable`), admission (`RemoteGate`: every tool call decided by the gateway's admit endpoint, failing closed; `AdmitResult`), events and checkpoints kept by the gateway (`GatewayEvents`, `GatewayCheckpointer`), tools performed server-side or through connectors (`RemoteTool`, `McpClient`, `api_request` / `call_api`, `ConnectorError`), the start-up handshake and the outcome (`next_op`, `fetch_manifest`, `report_result`, `mint_child` for a subagent's child run), and the runtime (`build_runtime` builds the agent from the manifest and nothing else, `RunnerRefused` when it cannot, `run_box` is `python -m locus.runner`). Exit codes: `0` done, parked or told to stop; `2` refused; `1` error. The module's `__all__` lists the constants and the remaining result types.
- `locus.rogue`: `pick_mode() -> (mode, description)` chooses `local` when `LOCUS_MODEL_URL` points at a chat-completions server, `frontier` when `ANTHROPIC_API_KEY` is set, else `compromised` (the offline `CompromisedModel`); `build_agent(mode) -> Agent` builds the challenge agent with its dangerous tools behind `ControlPolicy(deny_for={"irreversible", "data-exfil"})`; `main()` is what `python -m locus.rogue` runs.
- `locus.codex`: `skill_source()` is the bundled skill directory (`SKILL_DIR` is `.agents/skills/locus`, `SKILL_NAME` `locus`), `install(dest)` copies it and returns the files written (unchanged files are skipped), `add_agents_section(project)` appends `AGENTS_SECTION` to `AGENTS.md` once, `codex_argv(prompt, *, exec_mode=False, codex_profile=None, extra=())` builds the command line with `CODEX_FLAGS` (workspace-write sandbox with network), `launch(argv)` replaces the process, and `CodexError` is a problem the person fixes, reported without a traceback.
- `locus.a2a` exports the `A2AV1*` wire types (`A2AV1Message`, `A2AV1Task`, `A2AV1SendMessageRequest`, `A2AV1StreamResponse`, `A2AV1TaskState`, and the rest) that mirror the A2A v1 JSON-RPC schema; import them from `locus.a2a` (defined in `locus.a2a.spec_v1`, with `A2A_V1_PROTOCOL_VERSION`). The same package holds the card and message models the client calls use (`AgentCard`, `AgentCapabilities`, `AgentProvider`, `AgentInterface`, `Message`, `Part` with `TextPart`, `DataPart`, `FilePart`, `Artifact`, `Task`, `TaskState`, `TaskStatus`, the `JsonRpc*` envelopes, the `Task*Params`, `*UpdateEvent` and `PushNotification*` types) and the legacy flat `A2AMessage`, `A2ARequest`, `A2AResponse`.
