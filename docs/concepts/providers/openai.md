# OpenAI

The OpenAI provider connects Locus directly to OpenAI's API (`api.openai.com`). It's what you reach
for when you want **the latest OpenAI model the day it ships** without
going through any gateway, translation layer, or middleware.

It's also the **fastest way to try the SDK** — one env var, one line of
code, you're talking to GPT-5 or the o-series reasoning models.

## When to pick OpenAI

| You want… | This is the right provider |
|---|---|
| GPT-5, GPT-4o, or any latest OpenAI release | ✓ |
| The o-series reasoning models (`o3`, `o4-mini`) | ✓ |
| To go through Portkey / LiteLLM / vLLM | ✓ — same class, different `base_url` |

## Getting started

### 1. Set your API key

```bash
export OPENAI_API_KEY=sk-...
```

That's the only setup. The SDK reads the env var on import.

### 2. Pick a model

```python
from locus.agent import Agent
agent = Agent(model="openai:gpt-5.5", system_prompt="You are a support agent.")
```

The string `"openai:gpt-5.5"` does two things: tells the SDK to use
the OpenAI provider (`openai:` prefix), and which model id to call
(`gpt-5.5`). Any model id OpenAI accepts, the SDK accepts.

### 3. Run it

```python
result = agent.run_sync("Summarise open orders for customer 4821.")
print(result.message)
# → 'Customer 4821 has two open orders — ord-4821 (refund under review) and ord-4907 (shipped).'
```

Done. Streaming, tool calls, structured output — all of it works
without further configuration.

## What you get out of the box

### Chat completions across the GPT family

Every chat-shaped OpenAI model: `gpt-4o`, `gpt-4.1`, `gpt-5`, `gpt-5.5`.
Vision input (image URLs / base64), audio input, and
function calling work the same way you'd use them on the OpenAI SDK
directly — the SDK just normalises the events the model emits.

Image *generation* (`gpt-image-2`) is a separate provider, not a chat
model — see [Multi-modal providers](../multi-modal-providers.md).

### Reasoning models — the o-series

`o1`, `o3`, `o4-mini` route through the same `Agent(model="openai:o3")`
call. They're slower and more expensive but think before they answer.
The SDK detects these models and adapts the request automatically —
it sends `max_completion_tokens` instead of `max_tokens` and drops the
`temperature` / `top_p` sampling params the o-series rejects.

```python
agent = Agent(model="openai:o3", system_prompt="You are a support agent.")
```

Each turn's assistant message is surfaced as a `ThinkEvent` (carrying
`response.message.content`) so your UI can show the model's working as
it goes. Note: OpenAI does not return separate reasoning traces over the
API, so the `ThinkEvent` carries the model's normal per-turn message
text, not hidden chain-of-thought.

### The gpt-5.6 family and the Responses API

`OpenAIModel` speaks both OpenAI wire APIs. `api="auto"` (the default) sends
chat-completions for everything except model ids starting with `gpt-5.6`
(`gpt-5.6-sol`, `-terra`, `-luna`): that family rejects function tools on
chat-completions whenever reasoning is active, so Locus routes it to
`/v1/responses`. Auto-routing fires only when no `base_url` is passed to the
model. With a custom `base_url` the request stays on chat-completions, because
an OpenAI-compatible gateway typically does not serve the Responses path. A
gateway set only through the `OPENAI_BASE_URL` environment variable is not
detected, so pass `base_url` explicitly or set `api="chat_completions"`.

Force either wire API with `api="responses"` / `api="chat_completions"`. It is
a model argument, not an agent one, so `Agent(..., api=...)` raises. Build the
model first, or pass it through `model_kwargs` with the string form
(`Agent(model="openai:gpt-5.6-sol", model_kwargs={"api": "responses"})`):

```python
from locus.agent import Agent
from locus.models import get_model

agent = Agent(model=get_model("openai:gpt-5.6-sol", api="responses"))
```

### Real SSE streaming

Token-level streaming over Server-Sent Events. The model emits
deltas, the SDK turns them into `ModelChunkEvent`s, your `async for`
loop reads them as they arrive — no buffering, no fake chunking.

```python
async for event in agent.run("Summarise the history of order ord-4821."):
    if isinstance(event, ModelChunkEvent) and event.content:
        print(event.content, end="", flush=True)
```

### Tool calling — the OpenAI protocol

`@tool` functions are converted to OpenAI's tool-call schema and
the structured `tool_calls` field in the response is parsed back into
SDK `ToolCall` objects. Parallel tool calls are supported (the
model can request multiple tools per turn; the SDK runs them
concurrently via the `ConcurrentExecutor`).

### Structured output — Pydantic models in, validated objects out

```python
from pydantic import BaseModel

class Finding(BaseModel):
    summary: str
    confidence: float

agent = Agent(
    model="openai:gpt-5.5",
    output_schema=Finding,
    system_prompt="Reply as JSON matching the schema.",
)
result = agent.run_sync("Was order ord-4821 charged twice?")
print(result.parsed)        # Finding(summary='...', confidence=0.83)
```

Under the hood, the SDK sends an OpenAI `response_format` with the
schema and a strict-mode flag; if the model produces invalid JSON,
the SDK retries with the validation errors in the prompt
(`output_schema_retries=2` by default).

## Going through a gateway

A `base_url` override turns `OpenAIModel` into a client for any
OpenAI-compatible endpoint:

| Gateway | When to use it | `base_url` |
|---|---|---|
| **Portkey** | Virtual keys, request routing across providers, retries | `https://api.portkey.ai/v1` |
| **LiteLLM Proxy** | Self-hosted control plane in front of N providers | `https://<your-litellm-host>/v1` |
| **vLLM** | Self-hosted inference for open models with the OpenAI shape | `http://localhost:8000/v1` |
| **together.ai / fireworks / groq** | Hosted open-model inference at OpenAI-shape | their published `/v1` |

Build an `OpenAIModel` with the `base_url` (and `api_key`) you want and
hand it to the agent:

```python
from locus.agent import Agent
from locus.models.native.openai import OpenAIModel

agent = Agent(
    model=OpenAIModel(model="gpt-4o", base_url="https://api.portkey.ai/v1"),
)
```

The `api_key` your `OPENAI_API_KEY` provides is forwarded — for Portkey
it's the Portkey virtual key, etc.

## Handling sensitive data

Provider traffic carries whatever your tools touch — customer PII,
order data, internal hostnames — plus any untrusted user input that
entered the conversation. Two defaults keep it out of the wrong places:

- **Tool args/results aren't traced by default.** `record_arguments` and
  `record_results` are off, so order records and account details don't
  leak into your tracing backend unless you opt in (and verify its
  retention and access controls first). See [Observability](../observability.md).
- **Redact before the reply leaves the box.** An `OutputFilterHook`
  strips PII or blocks topics in the model's output, so nothing
  untrusted round-trips into a downstream ticket or chat.

Every model call and tool invocation still lands on the typed SSE event
stream — the faithful trace you forward to your logging backend — and
admission decisions land on the hash-chained
[`AuditTrail`](../agentic-ai-security.md) (hash-chained: each entry
commits to the one before it, so editing any record breaks
verification). Routing through a gateway (Portkey /
LiteLLM) keeps your `OPENAI_API_KEY` off the egress path when policy
requires it.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `401 Unauthorized` | `OPENAI_API_KEY` not set, or set to the wrong project's key |
| `429 Rate limit exceeded` | OpenAI quota; `ModelRetryHook` (if installed) retries with backoff |
| `model_not_found` | Model id doesn't exist for your tier — check `https://platform.openai.com/docs/models` |
| Empty `tool_calls` | Model decided not to call a tool; check the system prompt |
| `temperature`/`top_p` rejected on o-series | The SDK drops these for reasoning models automatically; remove any manual override that re-adds them |

## Source

[`OpenAIModel` in `src/locus/models/native/openai.py`](https://github.com/oracle/locus/blob/main/src/locus/models/native/openai.py)

## See also

- [Models overview](../models.md) — the full provider tree.
- [Anthropic](anthropic.md) — Claude family direct.
