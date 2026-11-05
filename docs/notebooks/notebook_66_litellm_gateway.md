# LiteLLM AI Gateway

This notebook is the runnable companion to the
[LiteLLM AI Gateway how-to](../how-to/litellm-gateway.md). A cloud-operations agent talks to a LiteLLM AI Gateway through the existing
`OpenAIModel(base_url=...)`, and the gateway takes on the concerns the
how-to assigns to it: provider authentication, vendor adapters, fallbacks,
virtual keys, budgets, observability, cost tracking, caching and
guardrails.

No new model class is involved. The gateway exposes the OpenAI API,
so the client the SDK already has is sufficient, and the agent host holds
only a gateway-issued key, never a provider key.

What this notebook covers:

- Health-checking the gateway at `LITELLM_GATEWAY_URL` and printing the
  model aliases it exposes, which surfaces configuration drift before any
  agent code runs.
- Running an `Agent` built around `OpenAIModel(base_url=..., api_key=...)`
  against the alias in `LITELLM_GATEWAY_MODEL` (default `gpt-4o`, defined
  in
  [`examples/litellm-gateway/config.yaml`](https://github.com/oracle/locus/blob/main/examples/litellm-gateway/config.yaml)).
  The agent answers two cloud-operations questions: what IaC stands for
  (Infrastructure as Code), and the first triage step for an Amazon EC2
  instance that is failing its health checks.
- Running a second agent, built the same way, through the event-streaming
  `agent.run(..., stream_tokens=True)` loop, printing each `ModelChunkEvent`
  as the gateway relays it and counting the chunks at the end.

## Prerequisites

```bash
# 1. Start the gateway (in another shell).
cd examples/litellm-gateway/
export OPENAI_API_KEY="sk-..."                        # upstream provider key
export ANTHROPIC_API_KEY="sk-ant-..."                 # optional second provider
export LITELLM_MASTER_KEY="$(openssl rand -hex 32)"
export LITELLM_DB_PASSWORD="$(openssl rand -hex 16)"  # Postgres sidecar password
docker compose up -d

# 2. Point this notebook at the gateway.
export LITELLM_GATEWAY_URL="http://localhost:4000"
export LITELLM_GATEWAY_KEY="$LITELLM_MASTER_KEY"      # or a virtual key from /key/generate
export LITELLM_GATEWAY_MODEL="gpt-4o"                 # alias from config.yaml
```

The diagram shows the path every model call in this notebook takes: the
agent's `OpenAIModel` sends OpenAI-shaped requests to the gateway at
`LITELLM_GATEWAY_URL` carrying only the gateway-issued key from
`LITELLM_GATEWAY_KEY`, and the gateway resolves the `gpt-4o` alias through
`model_list` in `config.yaml`, records virtual keys and spend in Postgres,
and calls OpenAI, Anthropic or any other provider added there with
provider keys that only it holds, so budgets, fallback chains and provider
credentials never reach the agent host:

{{ locus_diagram litellm-gateway }}

## Key lines

`_run_agent` is the whole integration. `OpenAIModel` takes the gateway
URL as `base_url`, the key from `LITELLM_GATEWAY_KEY` as `api_key` and
the alias from `LITELLM_GATEWAY_MODEL` as `model`; because `base_url` is
set, the default `api="auto"` sends every request to the gateway's
chat-completions endpoint rather than the Responses API. The `Agent`
built around it answers each prompt through `arun` and reports the prompt
and completion token counts from `result.metrics`. The function holds no
provider key, and the only provider name in it, the `OpenAIModel` class,
refers to the gateway's OpenAI-shaped API; which upstream serves the alias
is decided by `model_list` in `config.yaml`:

{{ locus_source examples/notebook_66_litellm_gateway.py::_run_agent }}

`_run_streaming` builds a second model and agent the same way, with a
shorter system prompt, and iterates `agent.run(..., stream_tokens=True)`,
printing the `content` of every `ModelChunkEvent` it receives and the
number of chunks at the end. The flag matters: without it `Agent.run`
yields no `ModelChunkEvent`, so the request would still go through the
gateway but the loop would report `[0 streamed chunks]`;
[Agent streaming](notebook_06_agent_streaming.md) covers the flag:

{{ locus_source examples/notebook_66_litellm_gateway.py::_run_streaming }}

## Run it

```bash
python examples/notebook_66_litellm_gateway.py
```

The script talks to a running LiteLLM AI Gateway and does not use the
shared `examples/config.py` harness, so `LOCUS_MODEL_PROVIDER` plays no
part and there is no mock fallback. It reads `LITELLM_GATEWAY_URL` and
`LITELLM_GATEWAY_KEY` from the environment set under Prerequisites;
`LITELLM_GATEWAY_MODEL` defaults to `gpt-4o`, the alias the sample
`config.yaml` ships. The upstream provider keys are read by the gateway
container, never by this script. If either required variable is missing,
the script prints a short setup reminder and exits with status 0; if both
are set but the gateway does not answer `GET /v1/models`, the health check
reports the error and exits with status 1.

## See also

- [LiteLLM AI Gateway how-to](../how-to/litellm-gateway.md) — when to
  choose the gateway over direct providers, issuing per-team virtual keys,
  cost tracking and Kubernetes deployment.
- [LiteLLM cost guardrails](notebook_67_litellm_gateway_cost.md) — the
  next notebook: per-team virtual keys and the gateway's spend endpoints.
- [`examples/litellm-gateway/`](https://github.com/oracle/locus/tree/main/examples/litellm-gateway) — the working sample: `config.yaml`, `docker-compose.yml`, `helm-values.yaml`.
- [Model providers](../concepts/models.md) — the direct (no-gateway)
  providers, the right default for single-tenant deployments.

## Source

??? note "Full source — `examples/notebook_66_litellm_gateway.py`"

    ````python
    --8<-- "examples/notebook_66_litellm_gateway.py"
    ````
