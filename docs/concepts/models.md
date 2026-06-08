# Model providers

A model is a string. The prefix before the colon (`openai:`,
`anthropic:`) tells the Tulip SDK which provider to use;
the rest is the model id that provider expects. `get_model()` parses
the string and returns a ready client.

```python
# tools, system_prompt, and other kwargs are the same across all providers
Agent(model="openai:gpt-4o")                     # OpenAI direct
Agent(model="anthropic:claude-sonnet-4-6")       # Anthropic direct
Agent(model="anthropic:claude-sonnet-4-6")                   # local
```

The same `Agent` works against any provider — only the model id and
the credentials change.

## The provider tree at a glance

```text
tulip.models
│
├── openai:                                ── OpenAI direct · OpenAIModel
│   ├─ chat completions       — gpt-* family
│   ├─ reasoning models       — o-series (adds reasoning_effort)
│   └─ base_url override      — Azure · Portkey · LiteLLM · vLLM ·
│                               together.ai · fireworks · groq
│
├── anthropic:                             ── Anthropic direct · AnthropicModel
│   ├─ Claude family          — opus · sonnet · haiku
│   ├─ prompt caching         — long blocks marked cacheable;
│   │                           subsequent turns pay 1/10th input cost
│   └─ extended thinking      — thinking blocks → ThinkEvent
│
│   └─ any pulled local model — llama, mistral, qwen, deepseek-r1 …
│
└── custom:                                ── register_provider("myco", MyModel)
    └─ implement BaseModel    — complete · stream · count_tokens
```

Pick the prefix that matches your auth surface. If you have an OpenAI
or Anthropic API key, jump straight to the matching provider page. For
OpenAI-compatible gateways via a `base_url`.

| Provider | Detail page |
|---|---|
| **OpenAI** | [OpenAI →](providers/openai.md) |
| **Anthropic** | [Anthropic →](providers/anthropic.md) |

## Custom providers

Implement the `BaseModel` Protocol — three methods (`complete`,
`stream`, `count_tokens`) — and you are a first-class provider. No
adapter layer, no inheritance from `OpenAIModel`. Register the class
with the prefix you want; it becomes a valid model id.

```python
from tulip.models import register_provider
from tulip.models.base import BaseModel

class MyModel(BaseModel):
    async def complete(self, request): ...
    async def stream(self, request): ...
    def count_tokens(self, text): ...

register_provider("myco", lambda model_id, **kw: MyModel(model_id, **kw))

agent = Agent(model="myco:my-model-id")
```

Source: [`register_provider` in `models/registry.py:21`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/models/registry.py#L21).

## Provider failover & pooling

For high-availability deployments, wrap the model in a pool:

```python
from tulip.models.pooled import PooledModel

agent = Agent(
    model=PooledModel(
        primary="anthropic:claude-sonnet-4-6",
        fallbacks=["openai:gpt-4o", "anthropic:claude-sonnet"],
    ),
    # tools=..., system_prompt=...,
)
```

The pool tries the primary first; on `RateLimitError`, `TimeoutError`,
or persistent 5xx it fails over to the next entry. Source:
[`PooledModel` in `models/pooled.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/models/pooled.py).

## Notebook

[`notebook_56_model_providers.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_56_model_providers.py)
exercises all three providers with the same agent.

## Source

| Area | Path |
|---|---|
| Provider registry | [`models/registry.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/models/registry.py) |
| `OpenAIModel` | [`models/native/openai.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/models/native/openai.py) |
| `AnthropicModel` | [`models/native/anthropic.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/models/native/anthropic.py) |
| `PooledModel` | [`models/pooled.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/models/pooled.py) |
