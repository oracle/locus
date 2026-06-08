# Model Providers

Tulip supports OpenAI, Anthropic as first-class providers.
The same `Agent` code works against any of them — only the model object
changes.

Provider matrix:

| Provider | Model class | Notes |
| --- | --- | --- |
| OpenAI | `OpenAIModel` | GPT-4o, o1, o3, gpt-5.x against the direct API |
| Anthropic | `AnthropicModel` | Claude models (opus / sonnet / haiku) |

The registry helper `get_model("provider:model_name")` returns the right
client for the prefix.

Run it (defaults to the bundled mock model; set `TULIP_MODEL_PROVIDER` to `openai` / `anthropic` for a live model):

    python examples/notebook_56_model_providers.py

Offline:

    TULIP_MODEL_PROVIDER=mock python examples/notebook_56_model_providers.py

Pin a specific model:

    TULIP_MODEL_PROVIDER=openai TULIP_MODEL_ID=gpt-4o python examples/notebook_56_model_providers.py

## Source

```python
--8<-- "examples/notebook_56_model_providers.py"
```
