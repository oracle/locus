# OpenAI-compatible providers

Most of the model ecosystem speaks the OpenAI wire protocol. Managed
services (Groq, Together, OpenRouter, DeepSeek, Mistral, xAI, Fireworks,
Cerebras, Perplexity, NVIDIA NIM) and self-hosted servers (Ollama, vLLM,
LM Studio, llama.cpp, LiteLLM) differ only in **base URL** and **which
environment variable holds the key** — so the SDK reaches all of them through
the same `OpenAIModel`, with no extra client dependency.

Address one by prefix (per-provider variables and examples:
[Configure a model provider](../../how-to/configure-providers.md)):

```python
from locus.agent import Agent

Agent(model="groq:llama-3.3-70b-versatile")
Agent(model="ollama:qwen3")
Agent(model="deepseek:deepseek-chat")
```

## OpenRouter and Together.ai

Both are supported directly through named prefixes. Install the OpenAI extra,
set the provider's own key, and use the exact model id shown in that provider's
catalog:

=== "OpenRouter"

    ```bash
    python -m pip install "locus-sdk[openai]"
    export OPENROUTER_API_KEY="your-key"
    ```

    ```python
    from locus import Agent

    agent = Agent(model="openrouter:provider/model-id")
    result = agent.run_sync("Explain the control decision.")
    ```

=== "Together.ai"

    ```bash
    python -m pip install "locus-sdk[openai]"
    export TOGETHER_API_KEY="your-key"
    ```

    ```python
    from locus import Agent

    agent = Agent(model="together:organization/model-id")
    result = agent.run_sync("Explain the control decision.")
    ```

The prefix selects the endpoint and credential variable; the SDK passes the part
after the colon through as the provider's model id. No OpenAI account or
`OPENAI_API_KEY` is required for these two routes. Tool calling, structured
output, vision, context size, pricing, and availability remain properties of
the selected model and provider.

## Provider routes in this build

This table is generated from the provider registry in SDK
**{{ locus_sdk_version }}** during the documentation build.

{{ locus_provider_table }}

Anything not listed is still reachable without a code change — give the base
URL explicitly:

```python
from locus.models import get_model

model = get_model("openai-compatible:my-model", base_url="https://host/v1")
Agent(model=model)
```

## Resolution order

For both the endpoint and the key, the first value found wins:

**Endpoint** — explicit `base_url=` → `LOCUS_<PREFIX>_BASE_URL` → the
vendor's own variable where one exists (`OLLAMA_BASE_URL`, `VLLM_BASE_URL`,
`LMSTUDIO_BASE_URL`, `LLAMACPP_BASE_URL`, `LITELLM_GATEWAY_URL`) → the
default in the table.

**Key** — explicit `api_key=` → the provider's variable from the table. A
hosted provider with no key raises immediately, naming the variable to set.
Local servers need none.

```bash
# Point the ollama prefix at a GPU box instead of localhost
export LOCUS_OLLAMA_BASE_URL=http://gpu-box:11434/v1
```

!!! note "Passing configuration inline"
    `AgentConfig` rejects unknown keyword arguments, so
    `Agent(model="groq:x", api_key=...)` raises. Build the model first when
    the configuration is not in the environment:

    ```python
    Agent(model=get_model("groq:llama-3.3-70b", api_key="..."))
    ```

## What this does not change

- **The Responses API is never auto-selected against a custom base URL.**
  `api="auto"` routes to `/v1/responses` only for model families that
  require it, and only against `api.openai.com` itself — a gateway serves
  chat-completions and would 404 on the Responses path. Set
  `api="responses"` explicitly if your endpoint does serve it.
- **Capability still varies by model.** A prefix makes an endpoint
  reachable; it does not promise that the model behind it supports tool
  calling, structured output, or vision. See
  [Structured output](../structured-output.md) for the fallbacks the SDK
  applies when a model cannot constrain its own decoding.
- **Sampling defaults are sent.** `temperature` and `top_p` default to
  `0.7` / `0.9` on the model config and go out with chat-completions
  requests (model ids the SDK treats as reasoning or search-preview models
  are the exception). `AgentConfig.temperature` defaults to `None` and is
  forwarded on the agent's regular turns only when set, so it does not
  override them there. (The iteration-limit summary, empty-answer recovery
  and structured-output repair calls pass the agent's `temperature`
  explicitly, so with the default `None` they send `top_p` but no
  `temperature`.) `AgentConfig` has no `top_p` field. To let a
  self-hosted server's own `generation_config.json` defaults apply, clear
  them when you build the model —
  `get_model("vllm:my-model", temperature=None, top_p=None)`, or
  `Agent(model="vllm:my-model", model_kwargs={"temperature": None, "top_p": None})`
  — since `None` means "do not send this parameter".

→ [Models overview](../models.md) · [OpenAI provider](openai.md) ·
[LiteLLM gateway](../../how-to/litellm-gateway.md)
