# Ollama

The Ollama provider is **Oracle Generative AI – Multi-Agent Locus SDK
pointed at a local model runtime**. Ollama runs open-weight models on
your laptop or a shared GPU box; the SDK calls it over HTTP exactly
the way it would call OpenAI or Anthropic. No API key, no network
egress, no per-token billing.

This is the right pick for **offline development**, **reproducible
tests**, and **iterating on agent design before you spend a dollar
on hosted inference**.

## When to pick Ollama

| You want… | This is the right provider |
|---|---|
| To develop offline — laptop, plane, isolated network | ✓ |
| Reproducible tests — same prompt + seed → same output | ✓ |
| Cost-free agent iteration before swapping to a paid API | ✓ |
| Privacy-sensitive prototyping where data can't leave the machine | ✓ |
| A frontier model (GPT-5, Claude Opus 4) | use [OpenAI](openai.md) or [Anthropic](anthropic.md) |
| Production-scale concurrency | use [OCI](oci.md), [OpenAI](openai.md), [Anthropic](anthropic.md) |

## Getting started

### 1. Install Ollama and pull a model

Ollama itself isn't a Python package — it's a small binary that runs
a local HTTP server.

```bash
# macOS (Homebrew) — or download from ollama.com
brew install ollama

# Start the server (it backgrounds itself):
ollama serve &

# Pull a model with native tool-calling support:
ollama pull llama3.3
```

`ollama list` will show what you've pulled. Anything in that list is
addressable from the SDK immediately.

### 2. Wire the SDK

```python
from locus.agent import Agent
agent = Agent(model="ollama:llama3.3", system_prompt="You are helpful.")
```

That's it. No env vars, no auth — Ollama is local-first by default.

### 3. Run it

```python
result = agent.run_sync("Sum 7 plus 35 in one word.")
print(result.message)
# → '42.'
```

Done. Streaming and tool calling work the same as for any other
provider — provided the model you pulled supports them.

## What you get out of the box

### Any pulled local model — no SDK change needed

The `model_id` after `ollama:` is whatever appears in `ollama list`.
The SDK doesn't maintain an allow-list; if Ollama can run it, the SDK
can address it.

```bash
ollama list
# llama3.3:latest
# qwen2.5-coder:32b
# deepseek-r1:14b
```

```python
agent_a = Agent(model="ollama:llama3.3")
agent_b = Agent(model="ollama:qwen2.5-coder:32b")
agent_c = Agent(model="ollama:deepseek-r1:14b")
```

### Real local streaming

Ollama emits SSE-shaped chunks; the SDK reads them as `ModelChunkEvent`s
just like any other provider. Token-level streaming over localhost
is fast — typically <5 ms per chunk.

```python
async for event in agent.run("Write a haiku about caching."):
    if isinstance(event, ModelChunkEvent) and event.content:
        print(event.content, end="", flush=True)
```

### Tool calling — model-dependent

Ollama supports tool calling for models that emit it natively. As of
writing:

| Model family | Tool calling |
|---|---|
| `llama3.1` / `llama3.2` / `llama3.3` | ✓ |
| `llama4` | ✓ |
| `qwen2.5` / `qwen2.5-coder` / `qwen3` | ✓ |
| `mistral` / `mixtral` | ✓ |
| `deepseek-r1` | ✓ (with reasoning) |
| `phi3` | ✗ — no native tool calling |

If a model doesn't support tool calling, the agent will still **run** —
it just won't be able to invoke any `@tool` you defined. The loop
then terminates after the first turn (no tools called, no follow-up
needed).

### No auth — by design

Ollama listens on `localhost:11434` with no authentication. That's
intentional for the local-first use case. To run against a shared
remote Ollama:

```bash
export OLLAMA_HOST=http://gpu-box.internal:11434
```

The same `OllamaModel` class talks to any HTTP-reachable Ollama
endpoint. (If you're exposing a remote Ollama, put it behind a VPN
or auth proxy yourself — Ollama doesn't ship one.)

## Practical workflow — develop local, ship hosted

A common pattern: prototype an agent against Ollama for free, then
swap one line to point at OCI / OpenAI / Anthropic for production.

```python
# Development:
agent = Agent(model="ollama:llama3.3", tools=[...], system_prompt="...")

# Production — same agent, swap the model id:
agent = Agent(model="oci:openai.gpt-5.5", tools=[...], system_prompt="...")
```

Everything else — tools, hooks, checkpointers, termination, RAG —
stays identical. You're not coupled to the local runtime; Ollama is
just a model address.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `Connection refused` on `localhost:11434` | Ollama server isn't running. `ollama serve &` in another terminal. |
| `model 'X' not found` | Haven't pulled it yet. `ollama pull X`. |
| Slow first response after hours of idle | Ollama unloads models from VRAM after inactivity. The first call after a long pause re-loads (a few seconds). |
| Tool calls never fire | The model you pulled doesn't support tools (e.g. `phi3`). Switch to `llama3.3` or `qwen2.5`. |
| `tool_calls` parsed as text instead of structured | Some Ollama versions emit XML-style `<tool_call>{...}</tool_call>` blocks. Update Ollama (`brew upgrade ollama`) or use a model with stable structured tool-call output. |
| Different output every run despite the same prompt | Set `temperature=0` and pin `seed` in `model_config`. |

## Source

[`OllamaModel` in `src/locus/models/native/ollama.py`](https://github.com/oracle-samples/locus/blob/main/src/locus/models/native/ollama.py)

## See also

- [Models overview](../models.md) — the full provider tree.
- [OpenAI](openai.md) — GPT family direct.
- [OCI Generative AI](oci.md) — production-scale OCI inference.
