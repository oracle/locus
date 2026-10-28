# Model providers

A model is a string. The prefix before the colon names the provider —
`oci:`, `openai:`, `anthropic:`, or one of the
OpenAI-compatible prefixes such as `ollama:`, `vllm:`, `groq:` or
`gemini:` — and the rest is the model id that provider expects.
`get_model()` parses the string and returns a ready client.

```python
# tools, system_prompt, and other kwargs are the same across providers
Agent(model="oci:openai.gpt-5.5", tools=[lookup_order, issue_refund])           # OCI Generative AI
Agent(model="openai:gpt-4o", tools=[lookup_order, issue_refund])                # OpenAI direct
Agent(model="anthropic:claude-sonnet-5-5", tools=[lookup_order, issue_refund])  # Anthropic direct
Agent(model="openrouter:provider/model-id", tools=[lookup_order, issue_refund]) # OpenRouter
Agent(model="together:organization/model-id", tools=[lookup_order, issue_refund]) # Together.ai
```

To reach a **self-hosted model**, use its prefix. `ollama:` and `vllm:`
default to the server's conventional local endpoint
(`http://localhost:11434/v1` and `http://localhost:8000/v1`) and need no
API key — run the server on your own host and the model call stays
inside your network; prompts never reach a hosted model API:

```python
# Self-hosted servers have their own prefixes: local default endpoint, no key.
# On-prem: the model call never leaves the box.
Agent(model="ollama:qwen3", tools=[lookup_order, issue_refund])
```

To run the server somewhere other than localhost, set
`LOCUS_OLLAMA_BASE_URL` or `LOCUS_VLLM_BASE_URL` (the conventional
`OLLAMA_BASE_URL` / `VLLM_BASE_URL` also work; the `LOCUS_` variable wins
when both are set). An OpenAI-compatible server with no
named prefix goes through `openai-compatible:`, which has no default
endpoint, so you supply one:

```python
from locus.models import get_model

# Any other OpenAI-compatible server: no default endpoint, so pass base_url.
local = get_model("openai-compatible:llama-3.3-70b",
                  base_url="http://inference.internal:8000/v1")
Agent(model=local, tools=[lookup_order, issue_refund])     # model call stays on your network
```

The same agent works against any provider — only the model id, the
`base_url`, and the credentials change. Provider choice is a
**data-governance decision**: it decides where your prompts — which may
carry customer records, order data, or internal hostnames — are sent, who
logs them, and which jurisdiction holds them.

## The provider tree at a glance

```text
locus.models
│
├── oci:                                   ── OCI Generative AI · auto-routed transport
│   ├─ V1 (default)            — OCIChatCompletionsModel on /openai/v1:
│   │                           OpenAI · Meta · xAI · Google — SSE
│   │                           streaming, new models as OCI lists them
│   ├─ SDK                     — OCIModel for every cohere.* id (Command R
│   │                           and Command A; OCI's /openai/v1 rejects
│   │                           them) and Dedicated AI Cluster endpoint OCIDs
│   └─ Responses (opt-in)      — OCIResponsesModel, server-stateful
│
├── openai:                                ── OpenAI direct · OpenAIModel
│   ├─ chat completions       — gpt-* family
│   └─ reasoning models       — o-series
│
├── anthropic:                             ── Anthropic direct · AnthropicModel
│   ├─ Claude family          — opus · sonnet · haiku
│   └─ prompt caching         — opt-in (prompt_cache=True), on complete()
│                               and stream() alike: marks the system prompt
│                               + tool catalog; once that prefix passes the
│                               model's minimum cacheable length, later
│                               turns read it at ~1/10th input cost while
│                               the ~5-min cache holds
│
├── OpenAI-compatible prefixes             ── OpenAIModel at a routed base_url
│   ├─ self-hosted            — ollama · vllm · lmstudio · llamacpp
│   │                           (model call stays on-network)
│   ├─ gateway                — litellm (on-network only if its upstreams are)
│   ├─ managed                — groq · together · openrouter · deepseek ·
│   │                           mistral · xai · fireworks · cerebras ·
│   │                           perplexity · nvidia · gemini
│   └─ openai-compatible:     — any other endpoint; you supply base_url
│
└── custom:                                ── register_provider("myco", MyModel)
    └─ implement ModelProtocol — complete · stream
```

Pick the prefix that matches both your auth surface and your data-handling
rules. The hosted API endpoints send prompts off-box — fine when the
vendor's audit logging and data-residency terms cover your data. For
**air-gapped or strictly data-residency-bound environments**, use a
self-hosted prefix (`ollama:`, `vllm:`, or `openai-compatible:` with your
own `base_url`) so prompt payloads never reach a hosted model API.

Other surfaces you configure egress on their own paths — a web-search or
web-fetch provider, a hosted embedding backend for RAG, an
`auxiliary_model` or grounding model on a hosted provider, or trace
export — so audit those separately.

| Provider | Detail page |
|---|---|
| **OCI Generative AI** | [OCI Generative AI →](providers/oci.md) · [Responses API →](oci-responses.md) · [DAC →](../how-to/oci-dac.md) |
| **OpenAI** | [OpenAI →](providers/openai.md) |
| **Anthropic** | [Anthropic →](providers/anthropic.md) |
| **OpenRouter** | [OpenAI-compatible providers →](providers/openai-compatible.md#openrouter-and-togetherai) |
| **Together.ai** | [OpenAI-compatible providers →](providers/openai-compatible.md#openrouter-and-togetherai) |
| **Other hosted or self-hosted endpoints** | [OpenAI-compatible providers →](providers/openai-compatible.md) |
| **Failover, credential pools, rate limits** | [Resilience →](providers/resilience.md) |

## Custom providers

Implement the `ModelProtocol` interface — `complete()` returns one
`ModelResponse`, `stream()` yields `ModelChunkEvent`s — and you are a
first-class provider. No adapter layer, no
inheritance from `OpenAIModel`. Register the class with the prefix you
want; it becomes a valid model id. This is the hook for a self-hosted
model behind your own audit proxy — every prompt logged to your own audit
store before it reaches the LLM.

```python
from collections.abc import AsyncIterator
from locus.core.events import ModelChunkEvent
from locus.models import ModelResponse, register_provider

class AuditedModel:                          # duck-typed ModelProtocol
    async def complete(self, messages, tools=None, **kw) -> ModelResponse: ...  # tee → audit store, then forward
    async def stream(self, messages, tools=None, **kw) -> AsyncIterator[ModelChunkEvent]: ...

register_provider("audited", lambda model_id, **kw: AuditedModel(model_id, **kw))

agent = Agent(model="audited:internal-llm", tools=[lookup_order, issue_refund])
```

Source: [`register_provider` in `models/registry.py`](https://github.com/oracle/locus/blob/main/src/locus/models/registry.py).

## Credential pooling & rotation

Always-on agents can't stall on a rate-limited credential. Wrap the
model in a `CredentialPoolModel` that rotates through a pool of API
keys for the **same** provider:

```python
import os
from pydantic import SecretStr
from locus.models.credentials import Credential, CredentialPool
from locus.models.pooled import CredentialPoolModel
from locus.models import AnthropicModel

pool = CredentialPool([
    Credential(label="primary", api_key=SecretStr(os.environ["KEY_A"])),
    Credential(label="backup",  api_key=SecretStr(os.environ["KEY_B"])),
])

def _build(cred: Credential) -> AnthropicModel:
    return AnthropicModel(model="claude-sonnet-5-5", api_key=cred.api_key.get_secret_value())

agent = Agent(
    model=CredentialPoolModel(pool=pool, build_model=_build),
    tools=[lookup_order, issue_refund],
)
```

Each call picks the active credential; when the error classifier says
rotation should help (rate-limit / auth errors), the credential is
marked bad with a cooldown and the next one is tried. It rotates
**credentials**, not providers — to fail over across *providers*,
compose the failover classifier (`locus.models.failover`) yourself.
Source:
[`CredentialPoolModel` in `models/pooled.py`](https://github.com/oracle/locus/blob/main/src/locus/models/pooled.py).

## Notebook

[`notebook_11_model_providers.py`](https://github.com/oracle/locus/blob/main/examples/notebook_11_model_providers.py)
runs the same cloud-operations agent against OCI Generative AI, OpenAI and
Anthropic by swapping one string.

## Source

| Area | Path |
|---|---|
| Provider registry | [`models/registry.py`](https://github.com/oracle/locus/blob/main/src/locus/models/registry.py) |
| `OpenAIModel` | [`models/native/openai.py`](https://github.com/oracle/locus/blob/main/src/locus/models/native/openai.py) |
| `AnthropicModel` | [`models/native/anthropic.py`](https://github.com/oracle/locus/blob/main/src/locus/models/native/anthropic.py) |
| `CredentialPoolModel` | [`models/pooled.py`](https://github.com/oracle/locus/blob/main/src/locus/models/pooled.py) |
