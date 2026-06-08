# LiteLLM AI Gateway — per-team cost tracking

Companion to [notebook 71](notebook_71_litellm_gateway.md) for the
enterprise piece: **who spent what on which model**.

Issues virtual keys for two pretend teams, drives traffic on each,
then walks the gateway's full spend surface:

- `/spend/logs` — per-request rows (model, tokens, USD cost, team metadata)
- `/global/spend/keys` — aggregate per virtual key
- `/global/spend/models` — aggregate per upstream model

All four endpoints are SQL-backed (Postgres sidecar from the sample
`docker-compose.yml`) and require zero Tulip integration glue —
the gateway is the source of truth.

## What enterprises use this for

- **Charge-back / showback to business units.** Finance pulls a monthly
  report; teams see what they cost.
- **"What did Cohere Command cost across all teams this week?"** Drill
  per upstream model.
- **"Who's about to blow their budget?"** Aggregate-per-key view +
  `max_budget` field.
- **Audit.** Append-only spend log keyed by virtual key + metadata,
  one place for SOC-2 / ISO-27001 review.

See the [LiteLLM AI Gateway how-to](../how-to/litellm-gateway.md#cost-tracking)
for the curl-level API and the [enterprise patterns
section](../how-to/litellm-gateway.md#how-enterprises-use-this-pattern)
for the deployment shape.

## Prerequisites

The Postgres-backed gateway from
[`examples/litellm-gateway/`](https://github.com/tuliplabs-ai/sdk-python/tree/main/examples/litellm-gateway).
The stateless gateway from notebook 71 won't work for this notebook —
`/key/generate` and `/spend/*` both require Postgres.

```bash
cd examples/litellm-gateway/
export LITELLM_MASTER_KEY="sk-master-$(openssl rand -hex 16)"
export LITELLM_DB_PASSWORD="$(openssl rand -hex 16)"
docker compose up -d

export LITELLM_GATEWAY_URL="http://localhost:4000"
export LITELLM_MASTER_KEY="$LITELLM_MASTER_KEY"
python examples/notebook_72_litellm_gateway_cost.py
```

## See also

- [Notebook 71 — LiteLLM AI Gateway](notebook_71_litellm_gateway.md) —
  the gateway happy path Tulip consumers see.
- [LiteLLM AI Gateway how-to](../how-to/litellm-gateway.md) — when to
  use the gateway, auth boundary, scope, and the enterprise patterns
  the cost surface unlocks.
