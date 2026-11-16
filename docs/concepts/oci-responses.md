# OCI Responses API — when to use it, what changes

The Oracle Generative AI – Multi-Agent Locus SDK exposes the OCI Generative
AI **Responses** endpoint as a separate transport, `OCIResponsesModel`,
alongside the default `OCIChatCompletionsModel` (which speaks
`/openai/v1/chat/completions`). Picking it means opting into server-side
conversation state — the OCI server holds the message thread between
turns, and the SDK references it via a continuation token instead of
re-sending the full history each call.

Use it for:

- **Responses-only OCI models** such as `openai.gpt-5.5-pro`, which the
  chat/completions endpoint can't reach.
- **Long conversations** where re-sending the full history per turn is
  wasteful and you trust OCI to retain it server-side for ~30 days.

If those don't apply, use `OCIChatCompletionsModel` — it covers every OCI
model family served on `/openai/v1` (Llama, Mistral, GPT, Grok, Gemini;
Cohere goes through `OCIModel`) in one transport with no Project OCID, no
server-state lock-in.

## Quick start

```python
from locus.agent import Agent
from locus.models.providers.oci import OCIResponsesModel

model = OCIResponsesModel(
    model="openai.gpt-5.5-pro",
    profile="MY_PROFILE",      # ~/.oci/config
    region="us-chicago-1",
)
agent = Agent(model=model, tools=[my_tool])
result = agent.run_sync("Plan a trip to Tokyo.")
```

Auth surface is identical to `OCIChatCompletionsModel`:

| Mode | Constructor |
|---|---|
| API key / session token (laptop / CI) | `profile="MY_PROFILE"` |
| Instance principal (compute) | `auth_type="instance_principal"`, `compartment_id=...` |
| Resource principal (functions / pipelines) | `auth_type="resource_principal"`, `compartment_id=...` |

`project_ocid=` is **optional**. Pass it only if a specific Responses
feature needs one; the model raises `OCIProjectRequiredError` at request
time if the endpoint signals that requirement. Non-OpenAI models served
through the Responses endpoint are one such case. The OCID is sent as the
`OpenAI-Project` header (and as `opc-project-id`).

## Reasoning controls

Reasoning models take an effort level. Set it once on the model with
`reasoning_effort=`, or per run through the agent's `model_kwargs`; the
per-run value wins. It goes on the wire as `reasoning={"effort": ...}`.

```python
from locus.agent import Agent
from locus.models.providers.oci import OCIResponsesModel

model = OCIResponsesModel(
    model="openai.gpt-5.5",
    profile="MY_PROFILE",
    region="us-chicago-1",
    reasoning_effort="low",
)
agent = Agent(model=model)
quick = agent.run_sync("Summarise the incident in one line.")
careful = agent.run_sync(
    "Find the root cause of the incident.",
    model_kwargs={"reasoning_effort": "high"},
)
```

The full `reasoning` object of the Responses API (`effort`, `summary`,
and whatever the model accepts) goes through `reasoning=`, on the model
or in `model_kwargs`, and is sent as given. When both are set the dict
wins and `reasoning_effort` only fills `effort` if the dict lacks it:

```python
model = OCIResponsesModel(
    model="openai.gpt-5.5",
    profile="MY_PROFILE",
    region="us-chicago-1",
    reasoning={"effort": "high", "summary": "auto"},
)
agent = Agent(model=model)
result = agent.run_sync(
    "Find the root cause of the incident.",
    model_kwargs={"reasoning": {"summary": "detailed"}, "reasoning_effort": "medium"},
)
# sent: reasoning={"summary": "detailed", "effort": "medium"}
```

Left unset, no `reasoning` field is sent and the server's default applies.
The accepted values depend on the model and the server validates them; a
model that does not reason rejects the field. Reasoning summaries the
endpoint returns surface as `ModelResponse.reasoning`.

**Reasoning across tool calls without server state.** With `store=False`
(or after the Zero Data Retention switch) nothing is kept server-side, so
when a reasoning control is set the request also asks for
`include=["reasoning.encrypted_content"]`. The reasoning items that come
back stay on the assistant message (`metadata["oci_responses_items"]`,
together with the turn's `function_call` items) and the next request
replays them verbatim ahead of the tool result, so a tool-calling turn
keeps the reasoning that led to the call. Whether the endpoint returns
reasoning items depends on the model; when it returns none, the turn is
reconstructed from the message as before. Streamed turns are
reconstructed too.

## Native compaction

The endpoint can compact a thread itself: `POST /responses/compact`
keeps the messages a continuation needs and folds the rest into an
encrypted `compaction` item that later requests carry in their `input`.
`OCIResponsesModel.compact_history()` exposes it, and the agent's
`ContextCompactor` can hand its summary stage to it instead of writing a
summary client-side. Opt in with `compaction=CompactionConfig(native=True)`
(or `ContextCompactor(native_compaction=True)`); it applies on the
stateless path, where the client holds the history:

```python
from locus.agent import Agent, CompactionConfig
from locus.models.providers.oci import OCIResponsesModel

model = OCIResponsesModel(
    model="openai.gpt-5.5",
    profile="MY_PROFILE",
    region="us-chicago-1",
    store=False,
)
agent = Agent(
    model=model,
    tools=[read_log],
    context_window=400_000,
    compaction=CompactionConfig(native=True),
)
```

Old tool output is still cleared first. When that is not enough, the
system prompt and the task message stay verbatim and the rest of the
history goes to the endpoint; what comes back replaces it in
`state.messages` and in the checkpoint, and the model replays the
compaction item on every later request. The run yields a
`CompactionEvent` with `stage="native"`; the compaction call's tokens
count against the run's budgets. If the endpoint call fails the
client-side summary takes over for that compaction. With `store=True`
the server holds the thread and no client-side compaction runs.

## What changes vs `OCIChatCompletionsModel`

**One thing only:** `ConversationManager` strategies (window,
summarize, truncate) don't apply. They operate on the full message
list, and on the Responses path that list lives server-side — there's
nothing for them to trim. The runtime loop reads the `server_stateful`
class flag and skips the strategy step; you don't have to configure
anything.

**Everything else works identically:**

| SDK primitive | Status on `OCIResponsesModel` |
|---|---|
| `MemoryStore` (cross-run facts in system prompt) | ✅ Works — system prompt threads into the first Responses call, server carries forward |
| `Reflexion` | ✅ Works — additional refinement turns in the same Responses thread |
| `GSAR` (self-assessment) | ✅ Works — operates on the final answer |
| `Grounding` evaluators | ✅ Works — operates on the final answer |
| `Checkpointer` | ✅ Works — `provider_state` (the continuation token) is persisted alongside other state |
| `@tool(idempotent=True)` dedup | ✅ Works — client-side tool execution gates it |
| `on_before_tool_call` / `on_after_tool_call` hooks | ✅ Works — user tools still execute client-side |
| `on_before_model_call` / `on_after_model_call` hooks | ✅ Works — see the Responses wire request/response |
| `on_before_invocation` / `on_after_invocation` hooks | ✅ Works |
| Output schema / structured output | ✅ Works — passed through as `response_format` |
| Streaming | ✅ Works — Responses SSE events translated to `ModelChunkEvent` |
| Custom termination conditions | ✅ Works |

That's why the bypass is one line, not a matrix.

## Tools on the Responses path

User `@tool` functions work identically. The model emits tool calls,
the SDK's `ToolExecutor` runs them client-side, results are sent back in
the next turn as `function_call_output` items (carrying `call_id` for
correlation). Tool hooks fire normally; idempotency dedup applies; the
`AfterToolCallEvent.arguments` and `.tool_call_id` fields you'd expect
from [hooks](hooks.md#on_after_tool_call-what-the-event-carries) are
all populated.

OCI built-in Responses tools (`file_search`, `web_search`,
`code_interpreter`) are not exposed in this release. The SDK's hook /
guardrail layers can't see server-side tool execution, so wrapping
those tools in the SDK would be misleading. If you need them, call OCI
directly. We may add an opt-in pass-through in a later release.

## Continuation state and checkpointing

After each turn, the model returns `provider_state =
{"previous_response_id": "resp_..."}`. The SDK stores it on
`AgentState.provider_state` and threads it into the next `complete()`
call. The `Checkpointer` snapshots `provider_state` alongside the
message history, so `agent.resume(...)` works across process restarts
as long as the OCI-side thread hasn't expired (~30 days).

If the thread is unknown or expired when the agent resumes, the model
raises `OCIResponsesStateLostError`. The agent should usually restart the
run rather than silently dropping the conversation; catch it explicitly
if you need different behavior.

## `store`: stateful, stateless, or decided by the tenancy

`store` controls whether OCI keeps the thread. It takes three values:

- `"auto"` (default). The model starts stateful. Some tenancies enforce
  Zero Data Retention (ZDR): the endpoint then answers every request that
  carries `previous_response_id` with a 400 saying so, and nothing can be
  continued server-side. On that error the model switches itself to the
  stateless path for the rest of its lifetime (`store: false` and the
  full history on every request), warns once with a `LocusWarning`, and
  the agent runtime replays the interrupted turn from the history it
  holds, so the run completes with the same result. The runtime reads
  `server_stateful` before every model call, so the switch takes effect on
  that very turn; `model.zdr_detected` tells you it happened, and the
  run's event stream carries a `CustomEvent` named `server_state_fallback`.
- `True`. Always stateful. On a ZDR tenancy the first continuation raises
  `OCIResponsesZeroDataRetentionError`.
- `False`. Always stateless. The right choice when you know the tenancy
  is ZDR: it skips the one stateful attempt and the warning. The
  Responses-only models stay reachable either way.

With `store=False`, or after the switch, the runtime takes the same
client-side path as `OCIChatCompletionsModel`: `ConversationManager`,
message-pair validation, the observation pack and streaming all apply.

## Errors you'll see

| Exception | When |
|---|---|
| `OCIProjectRequiredError` | OCI returned 403/404 with a project-related error body, or 400 saying the model needs the `OpenAI-Project` / `opc-conversation-store-id` header. Pass `project_ocid=` to the constructor. |
| `OCIResponsesStateLostError` | `previous_response_id` is unknown or expired. Restart the run. |
| `OCIResponsesZeroDataRetentionError` | The tenancy enforces Zero Data Retention, so `previous_response_id` can never be used. A subclass of `OCIResponsesStateLostError`. Under `store="auto"` the model switches to the stateless path and the agent runtime replays the turn; under `store=True` it propagates — build the model with `store=False`. |
| `RuntimeError` | Generic 5xx / non-JSON body / transport error. Status code + first 300 chars of body included in the message. |

No fallback to chat/completions. Picking the Responses
transport is explicit; an error on this path stays on this path.

## See also

- [Hooks](hooks.md) — what `on_after_tool_call` sees on either transport
- [OCI models — provider page](../how-to/oci-models.md) — when to pick which OCI transport
- [Streaming](streaming.md) — how `ModelChunkEvent` works on streamed runs

### Oracle reference docs

- [OCI Responses API](https://docs.oracle.com/iaas/Content/generative-ai/responses-api.htm)
  — the `/openai/v1/responses` endpoint this transport targets, plus
  the full list of supported tool types (file search, code interpreter,
  function calling, MCP).
- [OCI OpenAI-compatible endpoints](https://docs.oracle.com/iaas/Content/generative-ai/openai-compatible-api.htm)
  — base URL shape, auth modes, and how Responses fits alongside
  Conversations and Chat Completions.
- [OCI Generative AI — concepts](https://docs.oracle.com/iaas/Content/generative-ai/concepts.htm)
  — serving modes (on-demand vs Dedicated AI Cluster) that apply to
  the Responses endpoint too.
