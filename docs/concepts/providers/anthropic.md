# Anthropic

The Anthropic provider connects Locus directly to Anthropic's API (`api.anthropic.com`). Use it when you want **the Claude family** —
Opus for the hardest problems, Sonnet as the everyday workhorse,
Haiku for high-volume cheap calls — and want to talk to Anthropic
without going through an intermediary.

What makes this provider distinct is **prompt caching**: a long
product-catalog, policy, or playbook block reused across a session pays
1/10th the input cost on repeat turns once you opt in with
`prompt_cache=True`. Each turn's assistant message is
also surfaced as a typed `ThinkEvent`, so a UI can show the model's
working as it streams.

## When to pick Anthropic

| You want… | This is the right provider |
|---|---|
| Claude Opus / Sonnet / Haiku from Anthropic directly | ✓ |
| Long shared context (catalogs, policies, playbooks) amortised across a session | ✓ — built-in prompt caching |
| Each turn's reasoning surfaced for the operator to follow | ✓ — `ThinkEvent` stream |

## Getting started

### 1. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Pick a Claude model

```python
from locus.agent import Agent
agent = Agent(
    model="anthropic:claude-sonnet-5-5",
    system_prompt="You are a support agent.",
)
```

The string `"anthropic:claude-sonnet-5-5"` tells the SDK the
provider (`anthropic:`) and the exact model id. Any model id
Anthropic accepts, the SDK accepts.

### 3. Run it

```python
result = agent.run_sync("Summarise the open orders for customer 4821 in three bullets.")
print(result.message)
```

That's the full setup. Tool calling works without extra configuration;
token streaming is one argument away (`stream_tokens=True` on
`agent.run`), and prompt caching is opt-in (`prompt_cache=True`, below).

## What you get out of the box

### The whole Claude family

Whatever Anthropic ships, you can address by name. The ids below are
examples, not a maintained roster:

| Model | When to pick it |
|---|---|
| `claude-fable-5-1` | Hardest problems — long multi-step plans, timeline reconstruction |
| `claude-opus-5-5` | Complex reasoning where cost matters less than quality |
| `claude-sonnet-5-5` | Everyday workhorse — fast enough, smart enough, cheap enough for routine work |
| `claude-haiku-5-5` | High-volume classification and routing — cheap calls, log summaries |

### Real SSE streaming

Token-level streaming, opted into with `stream_tokens=True` on
`agent.run`. The model emits content deltas; the SDK
converts them to `ModelChunkEvent`s; your `async for` loop reads
them as they arrive.

```python
async for event in agent.run("Summarise the history of order ord-4821.", stream_tokens=True):
    if isinstance(event, ModelChunkEvent) and event.content:
        print(event.content, end="", flush=True)
```

### Tool calling — the Anthropic tool-use protocol

`@tool` functions are translated into Anthropic's `tools` schema; the
model's structured `tool_use` blocks are parsed back into SDK
`ToolCall`s. Parallel tool calls are supported (the model can
request multiple tools per turn; the SDK runs them concurrently via
the `ConcurrentExecutor`).

### Structured output — tool-as-schema

Anthropic doesn't expose a `response_format` field, so the SDK uses
the standard "single-tool" trick: define the schema as a tool, force
the model to call it. From your side, the API is identical to the
other providers:

```python
from pydantic import BaseModel

class Triage(BaseModel):
    severity: str
    needs_human: bool

agent = Agent(
    model="anthropic:claude-sonnet-5-5",
    output_schema=Triage,
)
result = agent.run_sync("Customer 4821 reports a duplicate charge on ord-4821.")
print(result.parsed)        # Triage(severity='high', needs_human=True)
```

### Prompt caching — opt in for long prompts

This is the biggest cost saver if your system prompt or tool block is
long (product catalogs, policy docs, SOC playbooks). Re-feeding
the same context on every turn of a session is the
common case — and the expensive one. Anthropic's prompt-caching
mechanism marks a span of the request as cacheable; subsequent turns
within the cache window pay **1/10th** the input cost on the cached
span.

Opt in with `prompt_cache=True` on `AnthropicModel`. The SDK then sends
the system prompt as a block list with `cache_control: ephemeral` and
tags the last entry of the tool catalog the same way (Anthropic walks
markers in order — the last tag anchors the cache point).
The same markers go on the streaming path: `AnthropicModel.stream()`,
which `agent.run(..., stream_tokens=True)` uses, builds its request through
the same code as `complete()`, so streaming turns hit the cache too.

```python
from locus.agent import Agent
from locus.models import AnthropicModel

agent = Agent(
    model=AnthropicModel(
        model="claude-sonnet-5-5",
        prompt_cache=True,
    ),
    tools=[...],
    system_prompt="<a long system prompt — product catalog, policy docs, playbooks>",
)

result = agent.run_sync("...")
print(f"cache writes: {result.metrics.cache_creation_input_tokens}")
print(f"cache reads:  {result.metrics.cache_read_input_tokens}")
# → cache writes: 4092      (turn 1, written once)
# → cache reads:  4092       (turn 2 — same prefix, ~10× cheaper input)
```

When it kicks in:

- A 5-minute "ephemeral" cache (rolling window).
- Subsequent turns reusing the same prefix pay `0.1× input rate` on
  the cached portion.
- Most effective when system prompts ≥ ~1024 tokens, or you've loaded
  a big skill / playbook / RAG block.

`cache_creation_input_tokens` and `cache_read_input_tokens` surface
on `AgentResult.metrics` so observability hooks can chart cache hits
and the cost saved.

### Per-turn reasoning — visible message stream

Each agent turn surfaces the model's assistant message as a
`ThinkEvent` before any tool calls run, so a human can follow how
the model is working through a decision. The SDK builds the
`ThinkEvent` from the turn's `message.content` (the Anthropic response
parser handles `text` and `tool_use` blocks); it does **not** send an
extended-thinking request param, so this is the normal per-turn message
text, not a separate hidden chain-of-thought channel:

```python
async for event in agent.run("...", stream_tokens=True):
    match event:
        case ThinkEvent(reasoning=r) if r:
            print(f"💭 {r}")
        case ModelChunkEvent(content=c) if c:
            print(c, end="", flush=True)
```

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `401 authentication_error` | `ANTHROPIC_API_KEY` not set, or set to a key without console access |
| `404 not_found_error` on the model id | Model id is wrong; check `https://docs.anthropic.com/en/docs/about-claude/models/all-models` |
| `429 overloaded_error` | Anthropic capacity; the `ModelRetryHook` re-tries with backoff if installed |
| Prompt caching not visible in usage stats | Cache window expired (5 min ephemeral) or prompt below the threshold |
| `ThinkEvent`s never fire | The turn produced no assistant `message.content` (e.g. tool-call-only turn) |
| `temperature` appears to have no effect | Anthropic deprecated `temperature` from `claude-opus-4-7` onward and across the Claude 5 family. The provider matches those ids by prefix (`claude-opus-4-7`, `claude-opus-4-8`, `claude-opus-4-9`, `claude-sonnet-5`, `claude-opus-5`, `claude-haiku-5`, `claude-fable-5`, `claude-mythos-5`) and omits the parameter rather than taking a 400 — it is dropped silently, with no warning, even though `AnthropicConfig.temperature` defaults to `0.7`. Separately, token streaming (`agent.run(..., stream_tokens=True)`) goes through `AnthropicModel.stream()`, which sends no `temperature` for any Claude model, so while streaming the setting has no effect on `claude-sonnet-5-5` or `claude-haiku-4-5` either. Steer with the system prompt instead. |

## Source

[`AnthropicModel` in `src/locus/models/native/anthropic.py`](https://github.com/oracle/locus/blob/main/src/locus/models/native/anthropic.py)

## See also

- [Models overview](../models.md) — the full provider tree.
- [OpenAI](openai.md) — GPT family direct.
