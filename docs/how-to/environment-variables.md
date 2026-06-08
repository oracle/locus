# Environment variables

Tulip configures itself from a small set of environment variables. The
notebook harness in `examples/config.py` reads them with a consistent
fallback chain; the SDK itself reads provider API keys directly.

## Provider keys

| Variable           | Used by                       |
|--------------------|-------------------------------|
| `OPENAI_API_KEY`   | `OpenAIModel` (and OpenAI-compatible gateways) |
| `ANTHROPIC_API_KEY`| `AnthropicModel`              |
| `COHERE_API_KEY`   | `CohereEmbeddings`, `CohereReranker` |

## Minimum set for a notebook run

```bash
# Pick a model + provider, then run any notebook.
export TULIP_MODEL_PROVIDER=openai
export TULIP_MODEL_ID=gpt-4o-mini
export OPENAI_API_KEY=sk-...

python examples/notebook_06_basic_agent.py
```

When no key is set, the notebooks fall back to a bundled mock model so
they still run offline.

## Other env vars

| Variable             | Used by                       |
|----------------------|-------------------------------|
| `TULIP_MODEL_PROVIDER` | `examples/config.py:get_model` |
| `TULIP_MODEL_ID`       | model factory dispatch         |
| `TULIP_MODEL_ID_B`     | secondary "model B" slot       |
| `TULIP_MODEL_ID_C`     | tertiary "model C" slot        |
| `TULIP_A2A_API_KEY`    | `A2AServer.__init__` bearer    |
| `OPENAI_API_KEY`       | `OpenAIModel`                  |
| `ANTHROPIC_API_KEY`    | `AnthropicModel`               |
