# Model Providers

Locus targets OCI Generative AI as its primary provider, and OpenAI,
Anthropic, and Ollama as first-class alternatives. The same `Agent`
code works against any of them — only the model object changes.

Provider matrix (in order of priority):

| Provider | Model class | Notes |
| --- | --- | --- |
| **OCI Generative AI** (default) | `OCIOpenAIModel` | `/openai/v1` transport — OpenAI, Meta, xAI, Mistral, Gemini, non-R Cohere |
| **OCI Generative AI** | `OCIModel` | OCI SDK transport — required for Cohere R-series |
| OpenAI | `OpenAIModel` | GPT-4o, o1, o3, gpt-5.x against the direct API |
| Anthropic | `AnthropicModel` | Claude models |
| Ollama | `OllamaModel` | Local LLMs (Llama, Mistral, Gemma) |

The registry helper `get_model("provider:model_name")` routes OCI ids
to the right transport automatically.

Run it (OCI Generative AI is the default; auto-detected from `~/.oci/config`):

    python examples/notebook_56_model_providers.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_56_model_providers.py

Pin a specific OCI model:

    LOCUS_MODEL_ID=meta.llama-3.3-70b-instruct python examples/notebook_56_model_providers.py

## Source

```python
--8<-- "examples/notebook_56_model_providers.py"
```
