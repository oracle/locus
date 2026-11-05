# Model providers

The SDK supports OCI Generative AI, OpenAI and Anthropic directly, and
OpenRouter, Together.ai and other hosted or self-hosted services through
named OpenAI-compatible prefixes. The same `Agent` code works against any of
them; only the model object changes. This notebook builds one
cloud-operations agent and swaps the model underneath it with a string, so a
platform team can move to another provider (for cost, latency, data
residency or an outage failover) without rewriting the agent.

The model object is the seam. Fleet-wide controls such as virtual keys,
per-team budgets, fallback chains and shared observability belong at an AI
gateway in front of the providers rather than in each agent; the
[LiteLLM AI Gateway](notebook_66_litellm_gateway.md) notebook runs the SDK
behind one.

Provider matrix:

| Provider | Model class | Notes |
| --- | --- | --- |
| OCI Generative AI | `OCIChatCompletionsModel` (default) or `OCIModel`, picked from the model id | `oci:<vendor>.<model>` reaches the OCI Generative AI catalog; see [OCI transports](notebook_01_oci_transports.md) |
| OpenAI | `OpenAIModel` | `openai:<model-id>` against the OpenAI API; key from `OPENAI_API_KEY` |
| Anthropic | `AnthropicModel` | `anthropic:<model-id>` against the Anthropic API; key from `ANTHROPIC_API_KEY` |
| OpenRouter | `OpenAIModel` at the routing-table base URL | `openrouter:provider/model-id`; key from `OPENROUTER_API_KEY` |
| Together.ai | `OpenAIModel` at the routing-table base URL | `together:organization/model-id`; key from `TOGETHER_API_KEY` |
| OpenAI-compatible | `OpenAIModel(base_url=…)` | Named prefixes for vLLM, Ollama, LiteLLM and others; `openai-compatible:` takes an explicit `base_url` |

In application code the registry string goes straight into `Agent`:

```python
from locus import Agent

openrouter_agent = Agent(model="openrouter:provider/model-id")
together_agent = Agent(model="together:organization/model-id")
```

What this notebook covers:

- the provider registry: `list_providers()` and the `prefix:model-id` string
  that `get_model()` resolves
- `OCIChatCompletionsModel`, `OpenAIModel` and `AnthropicModel` constructed
  directly, without the registry, and what "ready" means before a client has
  made a request
- one live call through a plain `Agent` on whichever provider
  `examples/config.py` resolves
- request parameters: the model's own configuration against per-run
  `model_kwargs` on `agent.arun`

The registry the first excerpt lists is the routing table the diagram shows:
`get_model()` splits `prefix:model-id` at the first colon and hands the id to
the factory registered for the prefix. `oci:` enters OCI Generative AI, where
the model id picks the transport; `openai:` and `anthropic:` build their
native clients; each OpenAI-compatible prefix builds an `OpenAIModel` at the
base URL the routing table resolves for it; and a prefix of your own is added
with `register_provider()`:

{{ locus_diagram provider-routing }}

## Key lines

`list_providers()` returns every prefix the registry accepts, in registration
order, which is why `oci` prints last; the usage lines show three model
strings in the form the registry resolves:

{{ locus_source examples/notebook_11_model_providers.py::example_providers }}

Each provider class can also be built directly, without the registry.
Constructing a client does not call the provider, and neither an unusable OCI
profile nor a missing API key is detected until the first request, so this
part runs offline and "ready" means built, not authenticated:

{{ locus_source examples/notebook_11_model_providers.py::example_direct }}

Request parameters have two homes. Values set when the model is built
(`temperature`, `top_p`, penalties, `extra_body`) hold for its lifetime;
`model_kwargs` on `agent.arun` applies to that run only and takes precedence
over both the model's configuration and the agent's, so on a live model the
second call is pinned to `temperature=0` without changing the agent or its
model:

{{ locus_source examples/notebook_11_model_providers.py::example_request_parameters }}

## Run it

```bash
python examples/notebook_11_model_providers.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one. `LOCUS_MODEL_ID` pins
a model for the chosen provider, and any other prefix the registry lists
(`openrouter`, `together`, …) is accepted as `LOCUS_MODEL_PROVIDER` when
`LOCUS_MODEL_ID` names the model:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_11_model_providers.py
LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_11_model_providers.py
```

## See also

- [Notebook 01 — OCI transports](notebook_01_oci_transports.md) — the three
  OCI model classes and the transport the `oci:` prefix picks
- [Notebook 66 — LiteLLM AI Gateway](notebook_66_litellm_gateway.md) — the SDK
  behind a gateway that carries keys, budgets and fallbacks
- [Concepts: Model providers](../concepts/models.md) — the provider tree,
  self-hosted prefixes and custom providers
- [Concepts: Resilience](../concepts/providers/resilience.md) — failover
  between models, credential pools and rate-limit headroom
- [How-to: Configure a model provider](../how-to/configure-providers.md) —
  per-provider environment variables and model ids

## Source

??? note "Full source — `examples/notebook_11_model_providers.py`"

    ````python
    --8<-- "examples/notebook_11_model_providers.py"
    ````
