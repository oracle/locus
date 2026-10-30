# Agent, tools and the loop

This file covers the single-agent surface of `locus-sdk`: constructing an `Agent` (explicit parameters and every `AgentConfig` field reachable through `**kwargs`), the three ways to run it, `AgentResult`, tools (`@tool`, `ToolContext`, `ToolOutput`, registry, executors), structured output, the event stream and `locus.streaming` handlers, hooks, the termination algebra, reasoning add-ons (reflexion, grounding, GSAR, causal), typed decisions, skills, playbooks, evaluation, conversation management, the error hierarchy, offline testing with `locus.testing`, and MCP in both directions. Every example below runs offline on a model double; replace the double with `"oci:openai.gpt-5.5"` (or any `provider:model` string) for a live run.

## Build an agent

Use `Agent` directly; `AgentConfig` is the validated form of the same options and is what `config=` takes. Pydantic hides the `**kwargs` from `inspect.signature(Agent)`, but every `AgentConfig` field is accepted as a keyword. An unknown keyword raises `ValidationError` naming the field; provider options for a string model go in `model_kwargs`.

```python
from locus.agent import Agent
from locus.tools import tool


@tool
def lookup_order(order_id: str) -> dict:
    """Return an order's status.

    Args:
        order_id: The order identifier.
    """
    return {"order_id": order_id, "status": "shipped"}


agent = Agent(
    model="oci:openai.gpt-5.5",
    tools=[lookup_order],
    system_prompt="You are a support agent. Look the order up before answering.",
    max_iterations=12,
    tool_execution="concurrent",
    termination=None,
)
```

An `oci:` model string authenticates from the `OCI_PROFILE` environment variable, or from `model_kwargs={"profile": "<section>"}`, or `model_kwargs={"auth_type": "instance_principal", "compartment_id": "<ocid>"}`; construction raises `ValueError` when none is set, so the block above needs OCI credentials to run. `cohere.*` ids and `ocid1.generativeaiendpoint.` ids take the OCI SDK transport instead.

Explicit constructor parameters: `model`, `tools`, `system_prompt`, `reflexion`, `grounding`, `max_iterations`, `conversation_manager`, `checkpointer`, `hooks`, `config`. The `AgentConfig` fields most often passed as keywords:

| Keyword | Type / default | Effect |
|---|---|---|
| `model` | `"provider:model"` string or a model instance | Required. A string without `:` is rejected. |
| `model_kwargs` | `dict`, `{}` | Forwarded to `get_model()` for a string model (`base_url`, `api_key`, `temperature`). Error when `model` is an instance. |
| `auxiliary_model` | string or instance, `None` | Cheaper model for summaries, grounding and compaction; `None` uses `model`. |
| `system_prompt` | `str` or `callable(context) -> str` | Dynamic prompts take a context dict. |
| `max_iterations` | `int`, 20 (1 to 500) | Loop cap; stop reason `max_iterations`. |
| `token_budget` / `max_cost_usd` / `time_budget_seconds` | `None` | Stop reasons `token_budget`, `cost_budget`, `time_budget`. `max_cost_usd` needs priced model metadata. |
| `budget_nudge_at` | `float`, 0.8 | Once this fraction of any budget is used, one note asks the model to converge. `None` turns it off. |
| `termination` | `TerminationCondition`, `None` | Replaces the default stop logic (see Termination). |
| `completion_mode` | `"auto"` or `"explicit"` | `"explicit"` stops only on a terminal tool, `max_iterations` or a budget. |
| `terminal_tools` | `set[str]`, `{"submit", "done", "finish", "complete", "task_complete"}` | Tool names that end the run with `terminal_tool`. |
| `require_verification` / `verification_tools` / `verify_tools` | `True` / run-style names / write-style names | In explicit mode `task_complete` is blocked until a verification tool ran after the last write. |
| `output_schema` / `output_schema_strict` / `output_schema_retries` | `BaseModel` subclass, `True`, 2 | Structured final answer (see Structured output). |
| `tool_execution` / `max_concurrency` / `tool_event_order` | `"concurrent"`, 10, `"sequential"` | Parallel tool calls, their cap, and whether `ToolCompleteEvent` fires in call order or on completion. |
| `max_tool_result_length` / `tool_result_head_fraction` / `tool_result_store` | 32000 chars, 0.4, `None` | Oversized results keep head and tail; a `ToolResultStore` offloads them instead. |
| `text_tool_calls` | `"auto"` | Parse tool calls written as text only for models that declare no native tool calling. |
| `tool_loop_threshold` | 3 | Identical consecutive steps that count as a loop (warn once, then stop with `tool_loop`). |
| `reflexion` / `grounding` | `True`, a config, or `None` | See Reasoning. |
| `gsar` | `GSARConfig`, `None` | Post-run typed grounding judge. |
| `planning` | `False` | Explicit plan on the first iteration, revised when stuck. |
| `memory_manager` | `BaseMemoryManager`, `None` | Long-term memory retrieved at start, extracted at end. |
| `conversation_manager` / `context_window` / `compaction` | `None` / `None` / `CompactionConfig` | Context control (see Conversation management). |
| `checkpointer` / `checkpoint_every_n_iterations` | `None` | Required for `thread_id` to carry history. |
| `hooks` / `plugins` / `callback_handler` | lists, `None` | `HookProvider` instances, `Plugin` instances, or one `fn(event)`. |
| `skills` / `playbook` | `list[Skill or path]`, `Playbook` | Progressive-disclosure skills; a playbook installs `PlaybookEnforcerHook`. |
| `mcp_servers` | `list[MCPClient]` | Tools attached on the first run (see MCP). |
| `model_retry` | `ModelRetryConfig`, on | Backoff on 429/5xx/timeouts; `False` disables. |
| `web_search` / `web_fetch` / `image_generator` / `speech_provider` | providers, `None` | Each registers a matching tool. |
| `name` / `agent_id` / `metadata` / `temperature` / `max_tokens` | | Labels, tool metadata, per-call model overrides. |

`AgentConfig` also exposes `with_reflexion(...)`, `with_grounding(...)` and `with_hooks(*hooks)` that return copies. Tools added after construction go through `agent.add_tool(tool)` or `agent.add_tools([...])`; mutating `config.tools` later is a no-op.

## Run it

| Call | Returns | Notes |
|---|---|---|
| `agent.run(prompt, *, thread_id=None, metadata=None, model_kwargs=None, stream_tokens=False)` | async generator of `LocusEvent` | `stream_tokens=True` adds `ModelChunkEvent` per chunk. |
| `await agent.arun(prompt, *, thread_id=None, metadata=None, model_kwargs=None)` | `AgentResult` | Async; leaves model clients open for reuse. |
| `agent.run_sync(prompt, *, thread_id=None, metadata=None, model_kwargs=None)` | `AgentResult` | Blocking; runs its own loop (a worker thread when one is already running) and closes clients. |
| `agent.invoke(prompt, *, thread_id=None, metadata=None)` | `AgentResult` | Alias of `run_sync`. |

Each call starts from an empty conversation unless the agent has a `checkpointer` and you pass `thread_id`; then earlier turns on that thread are reloaded and the new turn is saved. `metadata` reaches tools through `ToolContext.get_metadata`. `agent.cancel()` (any thread) stops every in-flight run with `stop_reason="cancelled"`; `agent.cancel(thread_id="t1")` stops only that thread's runs. A run paused by `ask_user` or a gated hold yields `InterruptEvent`; continue it with `async for event in agent.resume(response, thread_id=..., perform_dangling=False)`.

```python
import asyncio

from locus.agent import Agent
from locus.core.events import TerminateEvent, ToolCompleteEvent, ToolStartEvent
from locus.core.messages import Role
from locus.memory.backends import MemoryCheckpointer
from locus.testing import FunctionModel, text, tool_call
from locus.tools import tool


@tool
def lookup_order(order_id: str) -> dict:
    """Return an order's status."""
    return {"order_id": order_id, "status": "shipped"}


def handler(messages, tools):
    if any(m.role == Role.TOOL for m in messages):
        return text("Order ord-4821 has shipped.")
    return tool_call("lookup_order", order_id="ord-4821")


agent = Agent(model=FunctionModel(handler), tools=[lookup_order], checkpointer=MemoryCheckpointer())


async def main() -> None:
    async for event in agent.run("Where is ord-4821?", thread_id="t1"):
        if isinstance(event, ToolStartEvent):
            print("tool:", event.tool_name, event.arguments)
        elif isinstance(event, ToolCompleteEvent):
            print("result:", event.result, "error:", event.error)
        elif isinstance(event, TerminateEvent):
            print("stop:", event.reason, "->", event.final_message)


asyncio.run(main())
result = agent.run_sync("Where is ord-4821?", thread_id="t1")
print(result.message, result.stop_reason, [t.tool_name for t in result.tool_executions])
```

Prints `tool: lookup_order {'order_id': 'ord-4821'}`, the JSON result, `stop: complete -> Order ord-4821 has shipped.`, then `Order ord-4821 has shipped. complete []`: the second run on thread `t1` reloaded the first turn, so the handler saw the tool result already in history and answered without calling the tool again. Drop `thread_id` (or the checkpointer) and each run starts clean.

## `AgentResult`

| Attribute | Meaning |
|---|---|
| `message` / `text` | Final assistant text; with `output_schema`, the canonical JSON of `parsed`. Empty when a `ToolCalled` condition stopped the run before a text turn. |
| `stop_reason` | One of `complete`, `terminal_tool`, `confidence_met`, `max_iterations`, `tool_loop`, `no_tools`, `grounding_failed`, `token_budget`, `cost_budget`, `time_budget`, `context_exhausted`, `interrupted`, `error`, `cancelled`. |
| `success` | `stop_reason in ("complete", "terminal_tool", "confidence_met")`. A failing tool does not make a run unsuccessful. |
| `parsed` / `parse_error` / `parsed_as(Schema)` | Structured output; `parsed_as` raises `ValueError` when nothing parsed, `TypeError` on the wrong type. |
| `metrics` | `ExecutionMetrics`: `iterations`, `tool_calls`, `tool_errors`, `total_tokens`, `prompt_tokens`, `completion_tokens`, `duration_ms`, `cost_usd`, cache token counts, `tools_success_rate`. |
| `tool_executions` | `ToolExecution` tuple: `tool_name`, `tool_call_id`, `arguments`, `result`, `error`, `duration_ms`, `idempotent_cache_hit`. |
| `messages` / `state` / `iterations` / `confidence` | Full `AgentState`; `state.metadata` holds `output_key` values. |
| `error` / `grounding_score` / `ungrounded_claims` / `gsar_judgment` / `gsar_score` / `gsar_decision` | Set by the matching feature, else `None`. |

## Tools

`@tool` builds a `Tool` from a function: the name is the function name, the description is the docstring, and the JSON schema comes from the type hints with per-parameter descriptions parsed from an `Args:` block. Options: `@tool(name=..., description=..., idempotent=False, labels=None, sandbox=None, deferred=False, emits_progress=False)`. The body may be sync (run in a worker thread) or `async`. A parameter named `ctx` or `context` typed `ToolContext` is injected and left out of the schema. Model-supplied arguments are coerced to the annotations (`"17"` to `17`); a missing required argument becomes an error the model reads. Return values: `str` as is, `BaseModel` as JSON, `dict`/`list` as indented JSON, `None` as `"Success (no output)"`.

An exception inside a tool never escapes the loop: the executor records `ToolResult.error` as `"ValueError: boom"`, the model sees `Error: ValueError: boom`, `ToolCompleteEvent.error` is set, `metrics.tool_errors` increments, and the run continues. Return `ToolOutput(text, structured_content=..., content_blocks=..., is_error=False)` to hand the application machine-readable data the model never sees (`ToolCompleteEvent.structured_content`); `is_error=True` reports a failure without raising.

```python
from locus.agent import Agent
from locus.testing import ScriptedModel, text, tool_call
from locus.tools import ToolContext, tool


@tool(idempotent=True)
def page_oncall(case_id: str, summary: str, ctx: ToolContext) -> str:
    """Page the on-call engineer once per case.

    Args:
        case_id: Case to page about.
        summary: One-line summary for the page.
    """
    tenant = ctx.get_metadata("tenant", "unknown")
    return f"paged {tenant} about {case_id}: {summary}"


model = ScriptedModel([
    tool_call("page_oncall", case_id="c-1", summary="disk full"),
    tool_call("page_oncall", case_id="c-1", summary="disk full"),
    text("Paged once."),
])
result = Agent(model=model, tools=[page_oncall]).run_sync("page about c-1", metadata={"tenant": "acme"})
print([(t.result, t.idempotent_cache_hit) for t in result.tool_executions])
```

Prints `paged acme about c-1: disk full` twice, with `idempotent_cache_hit` `False` then `True`: the second identical call reused the first result and the body did not run. `idempotent=True` is per run, not across processes. An `async def` tool returning `ToolOutput(f"2 hotels in {city}", structured_content={"count": 2})` is handled the same way, with the dict on `ToolCompleteEvent.structured_content`.

`ToolContext` fields: `tool_call_id`, `tool_name`, `agent_id`, `run_id`, `iteration`, `state`, `invocation_metadata`, `tool_config`, plus `get_metadata(key, default)`, `get_config(key, default)`, `messages`, `confidence`. From inside a running tool, `current_tool_context()` returns the same object and `report_progress(progress, total=None, message=None)` emits `ToolProgressEvent` live for a tool declared `emits_progress=True`.

`Tool` (`locus.tools.decorator.Tool`) exposes `name`, `description`, `parameters`, `fn`, `idempotent`, `labels`, `to_openai_schema()`, `await tool.execute(ctx=None, **kwargs)` and plain `tool(**kwargs)` for direct calls. `agent.tools` is the `ToolRegistry` (`register`, `register_many`, `unregister`, `get`, `get_or_raise`, `list_tools`, `to_openai_schemas`, `search(query, limit=5)`, `in`, `len`). Builtins in `locus.tools`: `get_today_date` (idempotent), `use_oci` and `describe_oci` (OCI CLI access).

Executors in `locus.tools`: `ConcurrentExecutor(max_concurrency=...)` is selected by `tool_execution="concurrent"`, `SequentialExecutor` by `"sequential"`; `ToolExecutor` is the base for a custom one. Concurrent execution keeps `state.tool_executions` in call order; `tool_event_order="completion"` only changes event timing. Pick `"sequential"` when tools share a resource or must observe each other's side effects.

## Structured output

Set `output_schema=` to a `pydantic.BaseModel` subclass. The schema always rides the system prompt as a JSON instruction; providers with native structured output (OpenAI-compatible transports, including `oci:` ids on the `/openai/v1` transport) also receive a strict `response_format` when a call offers no tools (`output_schema_strict=False` for providers that reject strict mode). The final answer is extracted and validated client-side; on failure the model is re-prompted up to `output_schema_retries` times with the validation errors, metered but not written to the returned state. Tool iterations are unaffected: the schema applies to the final answer only.

```python
from pydantic import BaseModel

from locus.agent import Agent
from locus.testing import ScriptedModel, text


class OrderStatus(BaseModel):
    order_id: str
    shipped: bool


model = ScriptedModel([text("not json"), text('{"order_id": "ord-4821", "shipped": true}')])
agent = Agent(model=model, output_schema=OrderStatus, output_schema_retries=1)
result = agent.run_sync("Report ord-4821 as JSON.")
print(result.parsed, result.parse_error, model.call_count)
print(result.parsed_as(OrderStatus).shipped, result.message)
```

Prints `order_id='ord-4821' shipped=True None 2` then `True {"order_id":"ord-4821","shipped":true}`: the first answer failed validation, one repair call fixed it. For streaming UIs wrap the event stream: `StructuredStream(agent.run(prompt, stream_tokens=True), schema=OrderStatus)` (or `stream_structured(...)`) yields partial instances as required fields appear and exposes `.final` after `TerminateEvent`. `locus.core.structured.parse_structured` is the standalone parser.

## Streaming events

`agent.run(...)` yields `LocusEvent` subclasses from `locus.core.events`; match on type or on `event.event_type`.

| Event | Key fields |
|---|---|
| `ThinkEvent` | `iteration`, `reasoning`, `tool_calls` (one per model turn) |
| `ToolStartEvent` / `ToolCompleteEvent` | `tool_name`, `tool_call_id`, `arguments`; `result`, `error`, `duration_ms`, `structured_content`, `content_blocks` |
| `ToolProgressEvent` | `progress`, `total`, `message` |
| `ModelChunkEvent` | `content`, `reasoning`, `tool_calls`, `done`, `usage` (only with `stream_tokens=True`) |
| `ReflectEvent` / `GroundingEvent` | `assessment`, `new_confidence`, `guidance`; `score`, `ungrounded_claims`, `requires_replan` |
| `CompactionEvent` / `ModelRetryEvent` | context compaction stages; retry `attempt`, `delay_seconds`, `reason` |
| `InterruptEvent` | `question`, `options`, `interrupt_id`, `metadata` (run paused) |
| `CustomEvent` | `name`, `data` (emitted by hooks for UIs, never seen by the model) |
| `TerminateEvent` | `reason`, `iterations_used`, `final_message`, `usage`, `cost_usd`, `error` |

Handlers in `locus.streaming` implement `async on_event(event)`, `on_complete()`, `on_error(exc)`; drive one with `async for event in agent.run(...): await handler.on_event(event)` and `await handler.on_complete()` after the loop. Available: `ConsoleHandler(output=None, show_reasoning=True, show_tool_args=False, show_tool_results=True, use_color=True, use_emoji=True, max_result_length=500)`, `MinimalConsoleHandler`, `BufferingHandler` (`get_events()`), `FilteringHandler`, `CompositeHandler([...])`, `SSEHandler` (`format_all()`, `pop_messages()`) and `AsyncSSEHandler` (`stream()` of SSE strings for a web response, with `create_sse_response_headers()`). With `stream_tokens=True` on a `ScriptedModel` the stream is `model_chunk` events, then `think`, then `terminate`, so a UI test needs no live model.

## Hooks

Subclass `HookProvider` (`locus.hooks`), set `priority` (lower runs first; `HookPriority.SECURITY_DEFAULT` 50, `OBSERVABILITY_DEFAULT` 150, `BUSINESS_DEFAULT` 250, `DEFAULT` 300) and override any of: `on_before_invocation(prompt, state) -> state`, `on_after_invocation(state, success)`, `on_iteration_start(iteration, state)`, `on_iteration_end(iteration, state)`, `on_before_model_call(event)` (writable `messages`), `on_after_model_call(event)` (writable `retry`, `retry_feedback`, `response`), `on_before_tool_call(event)` (writable `arguments`, `secret_arguments`, `cancel`), `on_after_tool_call(event)` (writable `result`, `retry`), `on_before_compaction(event)` (writable `instructions`, `cancel`). Events are write-protected: setting a read-only field raises `AttributeError`. `event.run` gives `run_id`, `thread_id` and run metadata; `event.emit(CustomEvent(...))` sends UI-only data to the stream. Pass instances in `hooks=[...]`.

```python
from locus.agent import Agent
from locus.hooks import HookPriority, HookProvider
from locus.testing import ScriptedModel, text, tool_call
from locus.tools import tool


@tool
def issue_refund(order_id: str, amount_usd: float) -> str:
    """Refund an order."""
    return f"refunded {amount_usd} on {order_id}"


class ReadOnlyEnvironment(HookProvider):
    priority = HookPriority.SECURITY_DEFAULT

    async def on_before_tool_call(self, event) -> None:
        if event.tool_name == "issue_refund":
            event.cancel = "refunds are read-only in this environment"


model = ScriptedModel([tool_call("issue_refund", order_id="o1", amount_usd=10.0), text("Blocked.")])
result = Agent(model=model, tools=[issue_refund], hooks=[ReadOnlyEnvironment()]).run_sync("refund o1")
print([(t.tool_name, t.result, t.error) for t in result.tool_executions])
```

Prints `[('issue_refund', 'refunds are read-only in this environment', None)]`: a cancelled call is recorded with the reason as its result and the body never runs. Built-ins in `locus.hooks.builtin`: `LoggingHook(level=..., log_arguments=False, log_results=False)` and `StructuredLoggingHook` (JSON), `TelemetryHook(service_name=...)` (needs the `telemetry` extra), `GuardrailsHook(config=GuardrailConfig(...), on_violation=...)` with `block_dangerous_tools`, `allow_only_tools`, `pii_patterns`, `blocked_content_patterns`, `ContentFilterHook`, `ModelRetryHook(max_retries=3, retry_on_empty=True)`, and `SteeringHook(model=judge_model, policy="...", interrupt_tools=None)` for model-as-judge tool approval. The hard admission gate for consequential tools is `locus.control.gate_tool` (see the control reference); a `HookProvider` is for observation, argument shaping and soft policy.

## Termination

`locus.core.termination` conditions implement `check(state, **context) -> (stop, reason)` and compose with `|` (stop when any holds) and `&` (stop when all hold). The eight conditions: `MaxIterations(n)`, `TokenLimit(max_tokens)`, `TimeLimit(seconds)`, `TextMention(text, case_sensitive=False)`, `ToolCalled(tool_name, require_success=False)`, `ConfidenceMet(threshold=0.9)`, `NoToolCalls()`, `CustomCondition(fn)`. Pass the composition as `termination=`; it replaces the default stop logic, so include a cap. The reason string is mapped onto `stop_reason` (`tool_called:x` becomes `terminal_tool`, `text_mention:x` becomes `complete`).

```python
from locus.agent import Agent
from locus.core.termination import MaxIterations, TextMention, ToolCalled
from locus.testing import ScriptedModel, text, tool_call
from locus.tools import tool


@tool
def lookup_order(order_id: str) -> str:
    """Return an order's status."""
    return "shipped"


@tool
def submit_report(summary: str) -> str:
    """Submit the final report."""
    return "submitted"


stop = ToolCalled("submit_report", require_success=True) | TextMention("ESCALATE") | MaxIterations(5)
model = ScriptedModel([
    tool_call("lookup_order", order_id="o1"),
    tool_call("submit_report", summary="o1 shipped"),
    text("Finished."),
])
result = Agent(model=model, tools=[lookup_order, submit_report], termination=stop).run_sync("Report on o1.")
print(result.stop_reason, [t.tool_name for t in result.tool_executions], model.call_count, repr(result.message))

looping = ScriptedModel([tool_call("lookup_order", order_id="o1")], repeat_last=True)
capped = Agent(model=looping, tools=[lookup_order], termination=MaxIterations(3)).run_sync("Loop.")
print(capped.stop_reason, capped.iterations)
```

Prints `terminal_tool ['lookup_order', 'submit_report'] 2 ''` (the run stopped right after the terminal tool, before the scripted text turn, so `message` is empty) and `max_iterations 3`.

## Reasoning add-ons

- Reflexion: `reflexion=True` or `ReflexionConfig(confidence_threshold=0.85, diminishing_returns=True, evaluate_every_n_iterations=1, include_guidance=True, model=None)`. The `Reflector` (`locus.reasoning`) scores progress after iterations, emits `ReflectEvent`, and can stop the run with `confidence_met`; `metrics.reflexion_evaluations` counts evaluations.
- Grounding: `grounding=True` or `GroundingConfig(threshold=0.65, max_replans=2, check_before_final=True, model=None)`. A `GroundingEvaluator` checks the final claims against tool evidence, emits `GroundingEvent`, replans up to `max_replans`, and fills `result.grounding_score` / `result.ungrounded_claims`; the stop reason `grounding_failed` appears when replans run out.
- GSAR: `gsar=GSARConfig(judge=StructuredOutputGSARJudge(model=judge_model), tau_proceed=0.80, tau_regenerate=0.65, contradiction_penalty=0.5, fail_on_low_score=False)` from `locus.agent.config` and `locus.reasoning.gsar_judge`. After the loop the judge partitions the answer's claims by evidence type; `result.gsar_score` and `result.gsar_decision` (`proceed`, `regenerate`, `replan`, `abstain`) are set, and `fail_on_low_score=True` raises `GSARValidationError` (carrying `decision` and `score`) instead of returning. Pass an explicit judge on a different model from the generator. The full regenerate/replan loop is `locus.reasoning.gsar_evaluator.GSAREvaluator`.
- Causal: `build_causal_chain([{"label": ..., "causes": [...]}, ...])` from `locus.reasoning.causal` returns a `CausalChain` with `identify_root_causes()` and `get_chain_summary()` (`root_causes`, `symptoms`, `conflicts`); it is a post-processing utility, not an agent option.

## Decisions

Use `locus.decision` when the output is a label with a probability, not prose: `Choice(name, question, options)`, `YesNo(name, question)`, `Score(name, question, levels)` (an `int` means levels `"1"` to `"n"`). `LogprobDecider(base_url, model=..., api_key=None, timeout=10.0, extra_body=None, top_logprobs=20)` answers every field in one forward pass per field against any OpenAI-compatible server that returns logprobs; `await decider.decide(text, fields)` returns a `Decision` indexed by field name, and `decide_sync(decider, text, fields)` is the blocking form. Each `Answer` has `label`, `probability`, `margin`, `coverage`, `p_yes` (yes/no), `expected` (scores), `p(label)` and `flagged(threshold)`. `DecisionAdvisor(decider)` plugs an admit head into the control gate as `advisor=`. Example fields: `Choice("intent", "What does the customer want?", ("refund", "status", "complaint"))`, `YesNo("personal", "Does the message contain personal data?")`, `Score("urgency", "How urgent is the request?", 5)`. Deciding needs a live logprobs endpoint and there is no offline double; `decider.request_body(text, field)` returns the exact request for a server check.

## Skills and playbooks

A `Skill` (`locus.skills`) is an AgentSkills-style bundle: `Skill(name=..., description=..., instructions=..., allowed_tools=None)` in code, `Skill.from_file(path)` for one `SKILL.md` folder, `Skill.from_directory(parent)` for many. Pass `skills=[...]` (instances or paths) and the agent gets a catalog system message after `system_prompt` plus one `skills` tool; calling `skills(skill_name=...)` loads that skill's full instructions. `allowed_tools` is advisory outside a playbook step. For host-side routing build `SkillsPlugin(skills=[...], active=["name"], catalog=None, enforce_allowed_tools=False)` and pass it in `plugins=`.

A `Playbook` (`locus.playbooks`) is an ordered, enforced plan: `Playbook(id=..., name=..., steps=[PlaybookStep(id=..., description=..., expected_tools=[...], hints=[...], required=True, uses=[...], required_probes=[...], min_tool_calls=None, max_tool_calls=None)], strict_sequence=True, allow_extra_tools=False)`, or `load_playbook("plan.yaml")`. `playbook=` installs `PlaybookEnforcerHook` (`locus.playbooks.hook`), which blocks a call outside the current step with a message the model reads, advances when a step's `expected_tools` are exhausted, and records violations. With `hooks=[PlaybookEnforcerHook(playbook=..., block_violations=True, skills={...})]` you keep the hook handle: `hook.enforcer.violations`, `hook.enforcer.plan.completed_steps`, `hook.enforcer.adherence_score()`.

```python
from locus.agent import Agent
from locus.playbooks import Playbook, PlaybookStep
from locus.playbooks.hook import PlaybookEnforcerHook
from locus.testing import ScriptedModel, text, tool_call
from locus.tools import tool


@tool
def lookup_order(order_id: str) -> str:
    """Return an order's status."""
    return "shipped"


@tool
def issue_refund(order_id: str, amount_usd: float) -> str:
    """Refund an order."""
    return f"refunded {amount_usd} on {order_id}"


plan = Playbook(id="refund", name="Refund", steps=[
    PlaybookStep(id="verify", description="Look up the order.", expected_tools=["lookup_order"]),
    PlaybookStep(id="pay", description="Issue the refund.", expected_tools=["issue_refund"]),
])
hook = PlaybookEnforcerHook(playbook=plan)
model = ScriptedModel([
    tool_call("issue_refund", order_id="o1", amount_usd=1.0),
    tool_call("lookup_order", order_id="o1"),
    tool_call("issue_refund", order_id="o1", amount_usd=1.0),
    text("Refunded."),
])
Agent(model=model, tools=[lookup_order, issue_refund], hooks=[hook]).run_sync("refund o1")
print([(v.violation_type, v.step_id) for v in hook.enforcer.violations], hook.enforcer.plan.completed_steps)
```

Prints `[('unexpected_tool', 'verify')] ['verify', 'pay']`: the early refund was blocked, the ordered calls completed both steps.

## Evaluation

`locus.evaluation` runs structural checks over an agent: `EvalCase(name=..., prompt=..., expected_tools=[...], expected_output_contains=[...], expected_output_not_contains=[...], expected_tool_sequence=[...], exact_tool_sequence=False, max_iterations=None, max_duration_ms=None, rubric=None, tags=[...])`. `EvalRunner(agent=agent, judge=None, concurrency=4).run(cases)` (blocking) or `await runner.arun(cases)` returns an `EvalReport` with `results` (`EvalResult`: `passed`, `score`, `checks`, `tools_called`, `output`, `error`, `timed_out`), `passed`, `failed`, `avg_score`, `summary()`. `rubric` is graded only when `judge=LLMJudge(model=...)` is given. Use a double for structural regression tests and a live model for rubric grading.

```python
from locus.agent import Agent
from locus.core.messages import Role
from locus.evaluation import EvalCase, EvalRunner
from locus.testing import FunctionModel, text, tool_call
from locus.tools import tool


@tool
def lookup_order(order_id: str) -> str:
    """Return an order's status."""
    return "shipped"


def handler(messages, tools):
    if any(m.role == Role.TOOL for m in messages):
        return text("Order ord-4821 status: shipped.")
    return tool_call("lookup_order", order_id="ord-4821")


case = EvalCase(
    name="lookup",
    prompt="Where is ord-4821?",
    expected_tools=["lookup_order"],
    expected_output_contains=["shipped"],
    max_iterations=5,
)
report = EvalRunner(agent=Agent(model=FunctionModel(handler), tools=[lookup_order])).run([case])
print(report.passed, report.failed, report.results[0].checks)
```

Prints `1 0 {'tool_called:lookup_order': True, 'output_contains:shipped': True, 'within_iteration_budget': True}`.

## Prompts and conversation management

`system_prompt` is a string or a callable receiving a context dict; render templates yourself and pass the string. Each run's prompt becomes the user turn. Without a `conversation_manager` the agent manages context itself: when the window is known (`context_window=`, `LOCUS_CONTEXT_WINDOW`, or model metadata) `CompactionConfig` clears old tool output, then summarises older history with `compaction.summary_model` (default: the agent's model), ending with `context_exhausted` when that is not enough; `compaction=False` keeps pruning only. Explicit managers from `locus.memory`: `SlidingWindowManager(window_size=20, preserve_system=True, preserve_first_user=True)`, `SummarizingManager(threshold=30, keep_recent=10, summarize_fn=None)`, and `locus.memory.compactor.LLMCompactor(context_length=..., trigger_fraction=0.8, summarize_fn=...)`. Multi-turn memory across runs is the `checkpointer` plus `thread_id`; `MemoryCheckpointer` (`locus.memory.backends`) is the in-process one for tests, `FileCheckpointer(path)` for a local file.

## Errors

Everything the SDK raises derives from `locus.core.errors.LocusError`, which carries a stable `kind` string for logs and metrics: `ToolError` (`ToolNotFoundError`, `ToolValidationError`, `ToolExecutionError`), `ModelError` (`ModelAuthError`, `ModelThrottledError`, `ModelResponseError`), `CheckpointError` (`CheckpointNotFoundError`, `CheckpointSerializationError`), `ConfigError`, `ValidationError`, `GSARValidationError` (`decision`, `score`), `ApprovalPendingError` (`thread_id`, `interrupt_id`, `question`, `metadata`). A tool exception is not raised to the caller; it becomes `ToolResult.error`. Provider SDK exceptions arrive unwrapped; transient ones are retried inside the loop by `model_retry`, and `locus.models.failover.classify(exc)` names the reason when you need to handle one yourself. A model failure that exhausts retries ends the run with `stop_reason="error"` and `result.error` set rather than raising.

## Testing offline

`locus.testing` replaces the model, the one part a test does not own. `text(content)` and `tool_call(name, **arguments)` build turns (`tool_call("write", arguments={...})` when an argument is named `name`, `content`, `call_id` or `arguments`). `ScriptedModel([...], repeat_last=False)` returns turns in order and raises `AssertionError` when the agent asks for more than scripted; it is consumed by one run, so build a new one per run. `FunctionModel(handler)` (alias `MockModel`) decides each turn from `(messages, tools)`. Both record `call_count`, `received_messages`, `offered_tools`, `last_prompt`. `CompromisedModel(name, arguments)` always attacks, for gate tests. `AgentTestClient(agent).run(prompt)` returns an `AgentTrace` with chainable `assert_tool_called(name, **args)`, `assert_tool_not_called`, `assert_tools_called(*names)`, `assert_model_calls(n)`, `assert_tool_offered(name)`, `assert_succeeded()` and plain data `message`, `tool_names`, `tool_calls`, `failed_tools`, `result`, `model`. A refused or cancelled call still appears in the trace (it was sent), so test a gate on the refusal the call returned, not with `assert_tool_not_called`.

```python
from locus.agent import Agent
from locus.control import Action, AuditTrail, ControlPolicy, gate_tool
from locus.testing import AgentTestClient, ScriptedModel, text, tool_call
from locus.tools import tool


@tool
def issue_refund(order_id: str, amount_usd: float) -> str:
    """Refund an order."""
    return f"refunded {amount_usd} on {order_id}"


def test_payment_refunds_are_denied() -> None:
    trail = AuditTrail()
    gated = gate_tool(
        issue_refund,
        policy=ControlPolicy(require_verification_score=0.0, deny_for=frozenset({"payment"})),
        action=lambda name, kwargs: Action(name=name, asset=kwargs["order_id"], kind="payment"),
        trail=trail,
    )
    model = ScriptedModel([tool_call("issue_refund", order_id="o1", amount_usd=10.0), text("Refused.")])
    trace = AgentTestClient(Agent(model=model, tools=[gated])).run("refund o1")

    trace.assert_tools_called("issue_refund").assert_model_calls(2).assert_succeeded()
    refusal = trace.result.tool_executions[0].result
    assert '"outcome": "deny"' in refusal
    assert "refunded" not in refusal
    assert trail.verify()


test_payment_refunds_are_denied()
print("gate held")
```

The gated call returns a JSON refusal (`{"status": "denied", "outcome": "deny", ...}`) to the model, the body never runs, and the audit trail verifies. Keep `pytest-asyncio` for `await client.arun(...)`; `client.run` is a plain blocking call. `await check_model_conformance(model)` runs five live checks against a real model before trusting it in a fallback chain.

## MCP

Consume a server with `MCPClient` (`locus.integrations`; requires the `mcp` extra): `MCPClient(server_command=[...])` for stdio or `MCPClient(base_url=..., access_token=None, headers={})` for HTTP, with `allowed_tools=[...]` or `tool_filter=` to narrow what the model sees. Either call `await client.connect()`, `tools = client.to_locus_tools(await client.list_tools())` and pass them in `tools=`, or pass `mcp_servers=[client]` and let the agent connect, list and attach on its first run (an unreachable server is logged and retried later, not fatal). Each remote tool is an ordinary `Tool` returning `ToolOutput`, so hooks, gates and `ToolCompleteEvent.structured_content` apply. Per-user credentials ride in run metadata: `agent.run(prompt, metadata={"mcp_headers": {"Authorization": "..."}})` with `metadata_headers_key="mcp_headers"` on the client; they never reach state or checkpoints.

Expose an agent with `LocusMCPServer(agent=agent, name="billing-tools")` (or `create_mcp_server(agent, name=...)`) and `server.run(transport="stdio")`; `"http"`, `"streamable-http"` and `"sse"` select the HTTP transports. The server publishes every registered tool plus `run_agent` and `run_agent_stream`. `run()` blocks, so start it in its own process. Minimal consumer shape: `billing = MCPClient(server_command=["python", "-m", "billing.mcp_server"], allowed_tools=["lookup_invoice"])`, then `Agent(model="oci:openai.gpt-5.5", mcp_servers=[billing])` and `await agent.arun(...)`; call `await billing.close()` when done. MCP needs a live server, so unit-test the agent with a plain `@tool` double for each remote tool and keep MCP for integration tests.
