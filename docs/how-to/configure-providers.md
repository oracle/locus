---
title: Configure a model provider
description: "Where each provider's credentials and endpoint come from — OCI Generative AI, OpenAI, Anthropic, OpenRouter, Together, RunPod, Gemini, the other hosted routes, self-hosted servers and any OpenAI-compatible endpoint — and how to check a route in ten lines."
---

# Configure a model provider

Every provider is one environment variable or two and a prefix in the model
string. Nothing is read from code: keys come from the environment (or an
explicit `api_key=`), endpoints from the vendor's default, an environment
override, or an explicit `base_url=`. The order is always the same:

1. `base_url=` / `api_key=` passed to `get_model()` or the model class.
2. `LOCUS_<PREFIX>_BASE_URL` (for example `LOCUS_OPENROUTER_BASE_URL`).
3. The vendor's own variable, where one exists (`OLLAMA_BASE_URL`, `RUNPOD_ENDPOINT_ID`, …).
4. The provider's default endpoint.

Install the extra for the client you use: `pip install "locus-sdk[oci]"` for
OCI Generative AI, `"locus-sdk[openai]"` for OpenAI and every
OpenAI-compatible route, `"locus-sdk[anthropic]"` for Anthropic.

## OCI Generative AI

```bash
export OCI_PROFILE=DEFAULT                 # a profile in ~/.oci/config (API key or session token)
export OCI_REGION=eu-frankfurt-1            # any region that serves Generative AI
export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..xxx
```

```python
from locus.agent import Agent

Agent(model="oci:openai.gpt-5.5")                       # V1 transport (OpenAI, Google, Meta, xAI models)
Agent(model="oci:cohere.command-a-03-2025")             # SDK transport, chosen from the id
Agent(model="oci:ocid1.generativeaiendpoint.oc1...")    # your Dedicated AI Cluster
```

The profile decides the auth: an API-key profile, or a session-token profile
from `oci session authenticate` (Locus detects `security_token_file`). On OCI
compute, OKE or Functions pass the principal through `model_kwargs` and drop
the profile:
`Agent(model="oci:openai.gpt-5.5", model_kwargs={"auth_type": "instance_principal", "compartment_id": os.environ["OCI_COMPARTMENT_ID"]})`
(`OCI_AUTH_TYPE` in the environment is honoured only by the SDK transport,
`OCIModel`). Details, regions and the transports:
[OCI Generative AI](../concepts/providers/oci.md),
[model support on OCI](oci-models.md), [Dedicated AI Clusters](oci-dac.md).

## OpenAI

```bash
export OPENAI_API_KEY=sk-...
```

```python
Agent(model="openai:gpt-5.5")
Agent(model="openai:gpt-6.1-sol")     # gpt-6 and gpt-5*-pro go to /v1/responses automatically
```

`OPENAI_BASE_URL` points the client at a proxy; an explicit `base_url=` (or
`api="chat_completions"`) keeps the Responses-only families on
chat-completions — the environment variable alone does not. Model families,
caching and the Responses API: [OpenAI](../concepts/providers/openai.md).

## Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
Agent(model="anthropic:claude-sonnet-5-5")
Agent(model="anthropic:claude-fable-5-1")
```

Prompt caching is opt-in on the model (`prompt_cache=True`) and applies to
streaming and non-streaming calls alike: [Anthropic](../concepts/providers/anthropic.md).

## OpenRouter

```bash
export OPENROUTER_API_KEY=sk-or-...
```

```python
Agent(model="openrouter:anthropic/claude-sonnet-5.5")
Agent(model="openrouter:x-ai/grok-4.7")
```

The model id is the one in OpenRouter's catalog (`vendor/model`). Rate limits
and 429s are the provider's; Locus's retry policy absorbs transient ones.

## Together

```bash
export TOGETHER_API_KEY=tgp_v1_...
```

```python
Agent(model="together:openai/gpt-oss-120b")
```

Use the exact id from Together's catalog. Many ids there are
dedicated-endpoint only; an account without a dedicated endpoint gets
"Unable to access non-serverless model" for them, which is an account limit,
not a Locus one.

## RunPod

RunPod's OpenAI-compatible URL embeds the endpoint id, so give Locus the id
and it builds the URL:

```bash
export RUNPOD_API_KEY=rpa_...
export RUNPOD_ENDPOINT_ID=qwen3-32b-awq      # a public endpoint, or the id of your own serverless endpoint
```

```python
Agent(model="runpod:Qwen/Qwen3-32B-AWQ")     # the model id the endpoint serves
```

`RUNPOD_BASE_URL` or `LOCUS_RUNPOD_BASE_URL` set the full URL instead
(`https://api.runpod.ai/v2/<endpoint-id>/openai/v1`), for example for a
dedicated endpoint behind a proxy.

## Gemini and the other hosted routes

Each one is a key and a prefix; the endpoint is the vendor's default.

| Prefix | Key variable | Endpoint |
|---|---|---|
| `gemini:` | `GEMINI_API_KEY` | `https://generativelanguage.googleapis.com/v1beta/openai/` (`GEMINI_BASE_URL` overrides) |
| `groq:` | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` |
| `mistral:` | `MISTRAL_API_KEY` | `https://api.mistral.ai/v1` |
| `xai:` | `XAI_API_KEY` | `https://api.x.ai/v1` |
| `deepseek:` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/v1` |
| `fireworks:` | `FIREWORKS_API_KEY` | `https://api.fireworks.ai/inference/v1` |
| `cerebras:` | `CEREBRAS_API_KEY` | `https://api.cerebras.ai/v1` |
| `perplexity:` | `PERPLEXITY_API_KEY` | `https://api.perplexity.ai` |
| `nvidia:` | `NVIDIA_API_KEY` | `https://integrate.api.nvidia.com/v1` |

```python
Agent(model="gemini:gemini-2.5-pro")
Agent(model="groq:llama-3.3-70b-versatile")
```

## Self-hosted servers

No key. The prefix defaults to the server's conventional local port; move it
with the vendor variable or `LOCUS_<PREFIX>_BASE_URL`.

| Prefix | Default endpoint | Endpoint variable |
|---|---|---|
| `ollama:` | `http://localhost:11434/v1` | `OLLAMA_BASE_URL` |
| `vllm:` | `http://localhost:8000/v1` | `VLLM_BASE_URL` |
| `lmstudio:` | `http://localhost:1234/v1` | `LMSTUDIO_BASE_URL` |
| `llamacpp:` | `http://localhost:8080/v1` | `LLAMACPP_BASE_URL` |

```bash
export VLLM_BASE_URL=http://inference.internal:8000/v1
```

```python
Agent(model="vllm:meta-llama/Llama-3.3-70B-Instruct")
Agent(model="ollama:qwen3")
```

## A LiteLLM gateway

```bash
export LITELLM_GATEWAY_URL=https://gateway.internal:4000/v1
export LITELLM_API_KEY=sk-...            # a virtual key from the gateway
```

```python
Agent(model="litellm:gpt-4o")            # the alias the gateway exposes
```

Routing, budgets and audit then live at the gateway: [LiteLLM gateway](litellm-gateway.md).

## Any other OpenAI-compatible endpoint

```python
from locus.models import get_model

model = get_model("openai-compatible:my-model", base_url="https://host/v1",
                  api_key=os.environ["MY_KEY"])
Agent(model=model)
```

`openai-compatible:` has no default endpoint, so `base_url` is required;
`OPENAI_COMPATIBLE_API_KEY` is read when no `api_key` is passed.

## Check a route in ten lines

The same two calls Locus's own live tests make: a completion and a tool call.

```python
from locus.agent import Agent
from locus.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


model = "openrouter:openai/gpt-4.1-mini"          # any prefix:model
print(Agent(model=model).run_sync("Reply with exactly: pong").text)
print(Agent(model=model, tools=[add]).run_sync("What is 17 + 25?").text)   # 42
```

## Keep the keys out of the code

Read them from the environment or a secrets manager at start-up, never from
source or notebooks. For always-on agents, `CredentialPoolModel` rotates
between several keys for one provider, and `FallbackChain` moves to another
provider when one is down: [Resilience](../concepts/providers/resilience.md).
