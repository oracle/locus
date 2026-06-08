# OpenAI

The OpenAI provider connects Tulip directly to OpenAI's API (`api.openai.com`). It's what you reach
for when you want **the latest OpenAI model the day it ships** without
going through any gateway, translation layer, or middleware.

It's also the **fastest way to try the SDK** — one env var, one line of
code, you're talking to GPT-5 or the o-series reasoning models.

## When to pick OpenAI

| You want… | This is the right provider |
|---|---|
| GPT-5, GPT-4o, or any latest OpenAI release | ✓ |
| The o-series reasoning models (`o3`, `o4-mini`) | ✓ |
| To go through Azure / Portkey / LiteLLM / vLLM | ✓ — same class, different `base_url` |

## Getting started

### 1. Set your API key

```bash
export OPENAI_API_KEY=sk-...
```

That's the only setup. The SDK reads the env var on import.

### 2. Pick a model

```python
from tulip.agent import Agent
agent = Agent(model="openai:gpt-5.5", system_prompt="You are helpful.")
```

The string `"openai:gpt-5.5"` does two things: tells the SDK to use
the OpenAI provider (`openai:` prefix), and which model id to call
(`gpt-5.5`). Any model id OpenAI accepts, the SDK accepts.

### 3. Run it

```python
result = agent.run_sync("What is two plus two?")
print(result.message)
# → 'Four.'
```

Done. Streaming, tool calls, structured output — all of it works
without further configuration.

## What you get out of the box

### Chat completions across the GPT family

Every chat-shaped OpenAI model: `gpt-4o`, `gpt-4.1`, `gpt-5`, `gpt-5.5`,
`gpt-image-1`. Vision input (image URLs / base64), audio input, and
function calling work the same way you'd use them on the OpenAI SDK
directly — the SDK just normalises the events the model emits.

### Reasoning models — the o-series

`o1`, `o3`, `o4-mini` route through the same `Agent(model="openai:o3")`
call. They're slower and more expensive but think before they answer.
The SDK surfaces the model's thinking blocks as `ThinkEvent`s so your
UI can show "thinking…" without parsing the response yourself.

```python
agent = Agent(
    model="openai:o3",
    model_config={"reasoning_effort": "high"},   # low | medium | high
)
```

`reasoning_effort` is OpenAI's knob for how long the model spends
thinking. Default is `medium`.

### Real SSE streaming

Token-level streaming over Server-Sent Events. The model emits
deltas, the SDK turns them into `ModelChunkEvent`s, your `async for`
loop reads them as they arrive — no buffering, no fake chunking.

```python
async for event in agent.run("Write a haiku about latency."):
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

class Answer(BaseModel):
    summary: str
    confidence: float

agent = Agent(
    model="openai:gpt-5.5",
    output_schema=Answer,
    system_prompt="Reply as JSON matching the schema.",
)
result = agent.run_sync("Was the meeting productive?")
print(result.parsed)        # Answer(summary='...', confidence=0.83)
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
| **Azure OpenAI** | Enterprise / regulated workloads, Azure billing | `https://<resource>.openai.azure.com/openai/deployments/<deployment-id>` |
| **Portkey** | Virtual keys, request routing across providers, retries | `https://api.portkey.ai/v1` |
| **LiteLLM Proxy** | Self-hosted control plane in front of N providers | `https://<your-litellm-host>/v1` |
| **vLLM** | Self-hosted inference for open models with the OpenAI shape | `http://localhost:8000/v1` |
| **together.ai / fireworks / groq** | Hosted open-model inference at OpenAI-shape | their published `/v1` |

```python
agent = Agent(
    model="openai:gpt-4o",
    model_config={"base_url": "https://api.portkey.ai/v1"},
)
```

The `api_key` your `OPENAI_API_KEY` provides is forwarded — for Azure
that's the Azure resource key, for Portkey it's the Portkey virtual
key, etc.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `401 Unauthorized` | `OPENAI_API_KEY` not set, or set to the wrong project's key |
| `429 Rate limit exceeded` | OpenAI quota; `ModelRetryHook` (if installed) retries with backoff |
| `model_not_found` | Model id doesn't exist for your tier — check `https://platform.openai.com/docs/models` |
| Empty `tool_calls` | Model decided not to call a tool; check the system prompt |
| `reasoning_effort` rejected | Only valid for o-series models, not GPT-4o / GPT-5 |

## Source

[`OpenAIModel` in `src/tulip/models/native/openai.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/models/native/openai.py)

## See also

- [Models overview](../models.md) — the full provider tree.
- [Anthropic](anthropic.md) — Claude family direct.
