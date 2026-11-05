# LiteLLM cost guardrails

This notebook is the companion to the [LiteLLM AI Gateway
notebook](notebook_66_litellm_gateway.md) for the question a platform
owner asks once agents run unattended: who spent what on which model. It
issues budget-capped virtual keys for two sample payments teams, drives
traffic on each, and reads the gateway's spend records back. None of it
needs SDK integration code: the gateway is the source of truth.

The gateway prices every request a virtual key makes and writes it to its
Postgres store, and it enforces the `max_budget` on each key from the same
store, so a looping automation is stopped at its cap rather than on the
invoice ([OWASP LLM10, Unbounded
Consumption](../concepts/threat-scenarios.md#owasp-llm-top-10-2025)).
Finance charge-back and per-model cost drill-down read from the same
records; the how-to's [enterprise patterns
section](../how-to/litellm-gateway.md#how-enterprises-use-this-pattern)
describes the deployment shape this supports.

What this notebook covers:

- Issuing per-team virtual keys with `/key/generate`: a model allowlist,
  a `max_budget` hard cap, a one-hour expiry and `team` metadata.
- Driving ordinary OpenAI-shaped chat completions under each key with
  `httpx`, so the gateway attributes the tokens and cost of every request
  to the key that made it.
- Reading three of the gateway's spend endpoints: `/spend/logs` for
  per-request rows (model, tokens, cost in USD, metadata),
  `/global/spend/keys` for the total per virtual key and
  `/global/spend/models` for the total per upstream model.

## Prerequisites

This notebook needs the Postgres-backed gateway from
[`examples/litellm-gateway/`](https://github.com/oracle/locus/tree/main/examples/litellm-gateway).
The gateway from the LiteLLM AI Gateway notebook already includes the
Postgres sidecar that `/key/generate` and the spend endpoints require; a
gateway started without `DATABASE_URL` does not suffice. If that gateway
is still running, skip step 1 and export the same `LITELLM_MASTER_KEY` in
this shell.

```bash
# 1. From the repository root, start the gateway with its Postgres sidecar.
#    `docker compose up -d` detaches, so the exports stay in this shell.
export OPENAI_API_KEY="sk-..."                        # upstream provider key
export ANTHROPIC_API_KEY="sk-ant-..."                 # optional second provider
export LITELLM_MASTER_KEY="sk-master-$(openssl rand -hex 16)"
export LITELLM_DB_PASSWORD="$(openssl rand -hex 16)"
docker compose -f examples/litellm-gateway/docker-compose.yml up -d

# 2. Point this notebook at the gateway. It reads LITELLM_MASTER_KEY as
#    exported in step 1.
export LITELLM_GATEWAY_URL="http://localhost:4000"
```

The drawing is shared with the LiteLLM AI Gateway notebook, so it accents
the agent. The callers in this notebook are the "Other services"
box above it in the "Your services" lane: plain OpenAI-shaped requests
from `httpx` that carry only a team's virtual key. The element to look at
is the Postgres store beneath the gateway. The gateway resolves the
`gpt-4o` alias through `model_list` in `config.yaml`, calls the provider
with keys that only it holds, and records the cost of every request
against the key that made it; that store is what `/spend/logs`,
`/global/spend/keys` and `/global/spend/models` read, and the gateway
enforces each key's `max_budget` against the spend recorded there:

{{ locus_diagram litellm-gateway }}

## Key lines

`issue_virtual_key` calls `/key/generate` with the master key and returns
a per-team token. `models` is the allowlist of aliases the key may call,
`max_budget` is the hard cap that stops a runaway automation at the
gateway, `duration` expires the key after an hour, and the `team` entry in
`metadata` is what the script reads back from each spend row:

{{ locus_source examples/notebook_67_litellm_gateway_cost.py::issue_virtual_key }}

`chat` is the traffic each team drives: an ordinary OpenAI-shaped chat
completion against the gateway, authenticated with the team's virtual key
alone, so the gateway attributes the tokens and cost of every request to
that key:

{{ locus_source examples/notebook_67_litellm_gateway_cost.py::chat }}

`fetch_spend_logs` reads the per-request rows back from `/spend/logs`
with the master key, filtered to one virtual key through the `api_key`
query parameter when `virtual_key` is given; `fetch_spend_by_key` and
`fetch_spend_by_model` read the two rollups through the same `_admin_get`
helper:

{{ locus_source examples/notebook_67_litellm_gateway_cost.py::fetch_spend_logs }}

## Run it

```bash
python examples/notebook_67_litellm_gateway_cost.py
```

The script talks to a running LiteLLM AI Gateway over HTTP with `httpx`
and does not use the shared `examples/config.py` harness or any model class, so `LOCUS_MODEL_PROVIDER` plays no part and there is no mock
fallback. It reads `LITELLM_GATEWAY_URL` and `LITELLM_MASTER_KEY` from the
Prerequisites; the master key is required because `/key/generate` and the
spend endpoints are admin calls. The upstream provider keys are read by
the gateway container, never by this script. If either variable is
missing, the script prints the wiring snippet and exits cleanly with
status 0. With both set, it issues the two keys, drives five chat
completions, waits fifteen seconds for the gateway's asynchronous spend
logger to flush, and then prints the three spend views.

## See also

- [LiteLLM AI Gateway](notebook_66_litellm_gateway.md) — the gateway
  integration SDK consumers see: `OpenAIModel(base_url=...)` against
  the same gateway.
- [LiteLLM AI Gateway how-to](../how-to/litellm-gateway.md) — when to
  use the gateway, the authentication boundary, the
  [cost tracking](../how-to/litellm-gateway.md#cost-tracking) endpoints
  at the `curl` level, and the enterprise patterns the spend records
  support.

## Source

??? note "Full source — `examples/notebook_67_litellm_gateway_cost.py`"

    ````python
    --8<-- "examples/notebook_67_litellm_gateway_cost.py"
    ````
