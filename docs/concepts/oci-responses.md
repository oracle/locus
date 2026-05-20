# OCI Responses API — when to use it, what changes

Locus exposes the OCI Generative AI **Responses** endpoint as a separate
transport, `OCIResponsesModel`, alongside the default `OCIChatCompletionsModel`
(which speaks `/openai/v1/chat/completions`). Picking it means opting
into server-side conversation state — the OCI server holds the message
thread between turns, and Locus references it via a continuation token
instead of re-sending the full history each call.

Use it for:

- **Responses-only OCI models** such as `openai.gpt-5.5-pro`, which the
  chat/completions endpoint can't reach.
- **Long conversations** where re-sending the full history per turn is
  wasteful and you trust OCI to retain it server-side for ~30 days.

If those don't apply, use `OCIChatCompletionsModel` — it covers every OCI model
family (Cohere, Llama, Mistral, GPT, Grok, Gemini) in one transport with
no Project OCID, no server-state lock-in.

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
time if the endpoint signals that requirement.

## What changes vs `OCIChatCompletionsModel`

**One thing only:** `ConversationManager` strategies (window,
summarize, truncate) don't apply. They operate on the full message
list, and on the Responses path that list lives server-side — there's
nothing for them to trim. The runtime loop reads the `server_stateful`
class flag and skips the strategy step; you don't have to configure
anything.

**Everything else works identically:**

| Locus primitive | Status on `OCIResponsesModel` |
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
Locus's `ToolExecutor` runs them client-side, results are sent back in
the next turn as `function_call_output` items (carrying `call_id` for
correlation). Tool hooks fire normally; idempotency dedup applies; the
`AfterToolCallEvent.arguments` and `.tool_call_id` fields you'd expect
from [hooks](hooks.md#on_after_tool_call-what-the-event-carries) are
all populated.

OCI built-in Responses tools (`file_search`, `web_search`,
`code_interpreter`) are not exposed in this release. Locus's hook /
guardrail layers can't see server-side tool execution, so wrapping
those tools in Locus would be misleading. If you need them, call OCI
directly. We may add an opt-in pass-through in a later release.

## Continuation state and checkpointing

After each turn, the model returns `provider_state =
{"previous_response_id": "resp_..."}`. Locus stores it on
`AgentState.provider_state` and threads it into the next `complete()`
call. The `Checkpointer` snapshots `provider_state` alongside the
message history, so `agent.resume(...)` works across process restarts
as long as the OCI-side thread hasn't expired (~30 days).

If the thread is unknown or expired when the agent resumes, the model
raises `OCIResponsesStateLostError`. The agent should usually restart the
run rather than silently dropping the conversation; catch it explicitly
if you need different behavior.

## Errors you'll see

| Exception | When |
|---|---|
| `OCIProjectRequiredError` | OCI returned 403/404 with a project-related error body. Pass `project_ocid=` to the constructor. |
| `OCIResponsesStateLostError` | `previous_response_id` is unknown or expired. Restart the run. |
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
