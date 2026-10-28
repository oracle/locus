# Model providers

Locus supports OpenAI and Anthropic directly, and OpenRouter, Together.ai, and
other hosted or self-hosted services through named OpenAI-compatible prefixes.
The same `Agent` code works against any of them — only the model object
changes. A cloud platform team can re-platform a provider (for cost,
latency, data residency, or an outage failover) without rewriting a
single runbook: the Gateway is the seam, and routing, rate limits, and
audit live there rather than in each agent.

Provider matrix:

| Provider | Model class | Notes |
| --- | --- | --- |
| OpenAI | `OpenAIModel` | GPT-4o, o1, o3, gpt-5.x against the direct API |
| Anthropic | `AnthropicModel` | Claude models (opus / sonnet / haiku) |
| OpenRouter | `get_model("openrouter:provider/model-id")` | Uses `OPENROUTER_API_KEY` and OpenRouter's model id |
| Together.ai | `get_model("together:organization/model-id")` | Uses `TOGETHER_API_KEY` and Together's model id |
| OpenAI-compatible | `OpenAIModel(base_url=…)` | Any compatible endpoint — vLLM, Ollama, LiteLLM, and others |

The registry helper `get_model("provider:model_name")` returns the right
client for the prefix.

This particular notebook harness supports `mock`, `openai`, and `anthropic`.
OpenRouter and Together.ai are available in application code through the SDK's
provider registry:

```python
from locus import Agent

openrouter_agent = Agent(model="openrouter:provider/model-id")
together_agent = Agent(model="together:organization/model-id")
```

Run the notebook (OCI Generative AI with an OCI profile, otherwise the bundled mock model; set `LOCUS_MODEL_PROVIDER` to force a provider):

    python examples/notebook_11_model_providers.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_11_model_providers.py

Pin a specific model:

    LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_11_model_providers.py

## Source

````python
--8<-- "examples/notebook_11_model_providers.py"
````
