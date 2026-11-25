# Running the Oracle Generative AI – Multi-Agent Locus SDK behind the LiteLLM AI Gateway

[LiteLLM](https://litellm.ai) ships an open-source proxy — variously
branded the **LiteLLM Proxy Server** and the **LiteLLM AI Gateway** —
that fronts 100+ model providers behind one OpenAI-shaped HTTP API.

When you route your agents' model traffic through it, the SDK consumes it
through its existing
[`OpenAIModel`](../concepts/providers/openai.md) with no SDK-side code
change. The gateway carries the parts of the integration that genuinely
belong in a gateway: virtual keys, per-team budgets, fallback chains,
centralised observability, cost reporting, caching, and guardrails.

{{ locus_diagram litellm-gateway }}

!!! note "The SDK has zero `litellm` dependency"
    The `litellm` package only lives inside the gateway's Docker
    container. Your services only need `openai` (already pulled by
    `OpenAIModel`).

## When to choose this over direct providers

The SDK's [direct model providers](../concepts/models.md) remain the right
default for **single-tenant production, dev / CI** — they're simpler,
in-process, lower-latency, and have no extra service to operate.

**Reach for the gateway when** you need:

- **Multi-tenant key management** — issue virtual keys per team /
  agent / tenant with per-key budgets, RPM/TPM (requests/tokens per
  minute) limits, expiry, and model allowlists.
- **Fallback chains across regions or providers** — "OpenAI →
  Anthropic" defined in `config.yaml`, no restart.
- **Centralised observability** — one Langfuse / OpenTelemetry /
  Datadog / Helicone hook configured in the gateway, every service feeds it.
- **Centralised cost tracking** — Postgres-backed per-key / per-team /
  per-model spend reporting across every consumer.
- **Polyglot consumers** — Python agents built with the SDK, a JS web console,
  Ruby / Go enrichment services all talk OpenAI to the same gateway.
- **Caching across services** — Redis / S3 / Qdrant in-flight, shared
  across every consumer.

If none of those apply, **prefer the direct providers**. The
gateway is an extra deployment, not a shortcut.

## Quickstart — local Docker

The `examples/litellm-gateway/` directory ships a working sample:

```bash
cd examples/litellm-gateway/

# Provider credentials for upstream calls, plus the gateway's admin token
# and Postgres password. These live in the *gateway's* environment, not in
# your Locus app.
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export LITELLM_MASTER_KEY="sk-master-$(openssl rand -hex 32)"  # admin token for /key/generate
export LITELLM_DB_PASSWORD="$(openssl rand -hex 16)"          # postgres pw

# Start detached, so the exports above stay in this shell for the curls below.
docker compose up -d
# Optional: follow the gateway's startup log (Ctrl-C stops following only).
# docker compose logs -f litellm
```

The compose file reads `OPENAI_API_KEY`, `LITELLM_MASTER_KEY`, and
`LITELLM_DB_PASSWORD` with the `${VAR:?}` form, so it refuses to start
if any of them is unset.

The gateway listens on `http://localhost:4000` and exposes the model
aliases declared in `config.yaml`. The sample ships six:
`gpt-4o`, `gpt-4o-mini`, `gpt-5-mini`, `claude-sonnet-4-6`,
`claude-haiku`, and `text-embedding-3-small`. Add more by extending
`model_list`.

`config.yaml` needs nothing beyond the gateway and its Postgres. A second
file, `config.platform.yaml`, is the same gateway plus the two add-ons that
need a backing service — the OpenTelemetry callback and the Redis response
cache. Setting `REDIS_URL` switches the compose file to it and the `cache`
profile adds the Redis sidecar; the "Observability" and "Caching" sections
below walk through both.

Verify with a `curl`:

```bash
curl -s http://localhost:4000/v1/models \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" | jq '.data[].id'
```

Then issue a scoped virtual key in "Issuing per-team virtual keys"
below, and re-run this with `$LITELLM_VIRTUAL_KEY` to confirm the key's
model allowlist.

## Issuing per-team virtual keys

The gateway's master key (`LITELLM_MASTER_KEY`) is the admin token —
treat it as a high-value secret and **never hand it to an agent**. Each service built on the SDK should carry a scoped **virtual key**
issued via the gateway's `/key/generate` endpoint:

```bash
curl http://localhost:4000/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "models":   ["gpt-4o"],
    "max_budget": 5.00,
    "duration": "24h",
    "metadata": {"team": "payments", "owner": "platform"}
  }'
```

Response (truncated):

```json
{
  "key": "sk-<example-virtual-key-here>",
  "models": ["gpt-4o"],
  "max_budget": 5.0,
  "spend": 0.0,
  "metadata": {"team": "payments", "owner": "platform"}
}
```

Export the returned `key`; the rest of this page uses it:

```bash
export LITELLM_VIRTUAL_KEY="sk-..."   # the "key" field from the response
```

The gateway enforces every field at request time:

- **Model allowlist** — a key with `models: ["gpt-4o"]`
  trying to call `gpt-5-mini` gets rejected:
  `key not allowed to access model. This key can only access
  models=['gpt-4o']. Tried to access gpt-5-mini`.
- **Budget** — when cumulative spend exceeds `max_budget`, subsequent
  calls 429.
- **Expiry** — `duration: "24h"` automatically deactivates the key
  after 24 hours.
- **Metadata** is attached to every request the key makes, so spend
  reporting and audit logs can group by `team` / `owner` / whatever
  fields you put there.

!!! note "`/key/generate` requires Postgres"
    The `docker-compose.yml` in this sample includes a Postgres sidecar
    for virtual-key storage. Without it the gateway returns
    `{"error": "DB not connected"}` for `/key/generate`. In production
    point `DATABASE_URL` at an external managed Postgres so the gateway
    pod itself stays stateless.

## Cost tracking

The same Postgres backend logs every request automatically with token
counts and computed cost. No extra config beyond connecting the DB.
The full admin / analytics API is documented at
[docs.litellm.ai/docs/proxy/cost_tracking](https://docs.litellm.ai/docs/proxy/cost_tracking);
the snippets below cover the three endpoints the sample deployment
relies on.

```bash
# Per-request spend log (flushed asynchronously every ~10s by default).
curl http://localhost:4000/spend/logs \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"

# Aggregate spend grouped by virtual key.
curl http://localhost:4000/global/spend/keys \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"

# Aggregate spend grouped by model.
curl http://localhost:4000/global/spend/models \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

Sample output:

```text
/spend/logs
  · model=openai/gpt-4o  tokens=11  cost=$0.000017
  · model=openai/gpt-4o  tokens=10  cost=$0.000016
  · model=openai/gpt-4o  tokens=9   cost=$0.000014

/global/spend/keys
  · key=sk-<example-vkey-1>...  total_spend=$0.000034
  · key=sk-<example-vkey-2>...  total_spend=$0.000014

/global/spend/models
  · model=openai/gpt-4o  total_spend=$0.000048
```

LiteLLM ships an internal pricing table covering every model it
routes (so each provider's per-token pricing is applied automatically). Spend
is keyed by `api_key`, `user`, `team_id`, and any custom field in
`metadata`, so the same SQL surface answers "what did team X spend
this week?" and "what did model Y cost across all teams?".

## Observability

`config.platform.yaml` turns on LiteLLM's OpenTelemetry callback, so the
gateway emits one trace per request:

```yaml
litellm_settings:
  callbacks: [otel]
```

Spans leave over OTLP to the collector named by the standard variables
`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_PROTOCOL` and
`OTEL_EXPORTER_OTLP_HEADERS` (the collector's auth). The compose file sets
the protocol to `http/protobuf` whenever an endpoint is set and leaves it
empty otherwise, which makes LiteLLM print the spans to the gateway log —
a way to see them before a collector exists.

```bash
export REDIS_URL=redis://redis:6379                                   # selects config.platform.yaml
export OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4318   # your collector's OTLP/HTTP port
docker compose --profile cache up -d
```

A chat completion produces a `litellm_request` span with, among its
attributes:

```text
gen_ai.request.model          gpt-4o
gen_ai.response.id            chatcmpl-…
gen_ai.usage.input_tokens     20
gen_ai.usage.output_tokens    1
litellm.model_group           gpt-4o
litellm.provider.model        openai/gpt-4o
litellm.cache_hit             False
metadata.applied_guardrails   ['pii-mask']
metadata.user_api_key_hash    <hash of the virtual key>
metadata.user_api_key_team_id, metadata.user_api_key_alias, …
```

and child spans for the auth step, the router, the Redis lookup and each
guardrail (`guardrail_name`, `guardrail_mode`, `guardrail_response`). A
request served from the cache shows `litellm.cache_hit = True` and no
upstream call.

For Langfuse, replace `otel` with `langfuse` in `config.platform.yaml`
and set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST`
in the gateway's environment; the compose file passes the three through.
LiteLLM's [logging page](https://docs.litellm.ai/docs/proxy/logging)
lists the other backends it can push the same spans to.

## Caching

`config.platform.yaml` also turns on the response cache, backed by the
Redis sidecar the `cache` profile starts:

```yaml
litellm_settings:
  cache: true
  cache_params:
    type: redis
    url: os.environ/REDIS_URL
    ttl: 600
```

An identical request — same model alias, messages and parameters — within
the TTL is answered from Redis without an upstream call. The second response
is the stored first one, `id` included, and carries an `x-litellm-cache-key`
header the first did not; its `x-litellm-response-cost` is `0.0`, the trace
shows `litellm.cache_hit = True`, and with Postgres connected the
`/spend/logs` row records the hit with no spend:

```text
call 1  1.730s  id=chatcmpl-…94c33  x-litellm-response-cost: 0.000017
call 2  0.050s  id=chatcmpl-…94c33  x-litellm-response-cost: 0.0  x-litellm-cache-key: f24b15…
```

The saving on identical prompts is the whole upstream cost of the repeat. Agent loops repeat less than it seems — each turn carries new tool results — so the cache pays off on the stable parts of a workload: classification prompts, retrieval rewrites, evaluation rubrics, anything many users ask the same way.

A cached answer is the earlier sample, so a request that wants a fresh one opts out with `"cache": {"no-cache": true}` in the body. The gateway does not start with `cache: true` and no reachable Redis, which is why the cache lives in its own config file rather than behind a switch in `config.yaml`.

## Guardrails

Both config files run one guardrail inside the gateway process — LiteLLM's
own content filter, so no further service is needed:

```yaml
guardrails:
- guardrail_name: pii-mask
  litellm_params:
    guardrail: litellm_content_filter
    mode: pre_call
    default_on: true
    patterns:
    - {pattern_type: prebuilt, pattern_name: us_ssn, action: MASK}
    - {pattern_type: prebuilt, pattern_name: credit_card, action: MASK}
```

`mode: pre_call` rewrites the prompt before the upstream call, so the
provider, the spend log and the trace only ever see the masked text.
Asking the gateway to echo a line back shows it:

```bash
curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_VIRTUAL_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content":
       "Return the text inside the square brackets exactly as written, nothing else: [reference 123-45-6789 / 4111 1111 1111 1111]"}]}' \
  -D -
```

```text
x-litellm-applied-guardrails: pii-mask
…
"content": "[reference [US_SSN_REDACTED] / [CREDIT_CARD_REDACTED]]"
```

The `guardrail` span in the trace lists each match with its
`pattern_name` and `action`; `action: BLOCK` rejects the request instead
of masking. Presidio is the alternative when you need entity recognition
rather than patterns: it runs as two services, analyzer and anonymizer,
named by `PRESIDIO_ANALYZER_API_BASE` and `PRESIDIO_ANONYMIZER_API_BASE`
in the gateway's environment, with `guardrail: presidio` in place of the
content filter. LiteLLM's [guardrails page](https://docs.litellm.ai/docs/proxy/guardrails)
covers it, Lakera, and the per-key and per-request controls.

## Pointing the SDK at the gateway

Use the existing `OpenAIModel` — that's the LiteLLM-compatible client:

```python
import os

from locus.agent import Agent
from locus.models.native.openai import OpenAIModel

model = OpenAIModel(
    model="gpt-4o",                  # alias from gateway config.yaml
    api_key=os.environ["LITELLM_VIRTUAL_KEY"],  # virtual key issued by the gateway
    base_url="http://localhost:4000",            # the LiteLLM AI Gateway
)

agent = Agent(model=model, system_prompt="You are a concise support assistant.")
print(agent.run_sync("Summarise ticket T-4821.").message)
```

No new class is needed. The gateway handles provider auth,
vendor adapters, fallback, budgets, and observability internally.
The SDK only ever sees the OpenAI-shaped HTTP contract.

## Running existing notebooks through the gateway

Every `examples/notebook_*.py` already routes model construction
through `examples/config.py:get_model()`, which honors
`LOCUS_MODEL_PROVIDER=openai` plus the standard `OPENAI_BASE_URL` /
`OPENAI_API_KEY` env vars. So pointing every notebook at the gateway
is a four-line shell change — no code edits:

```bash
docker compose -f examples/litellm-gateway/docker-compose.yml up -d

export LOCUS_MODEL_PROVIDER=openai
export LOCUS_MODEL_ID=gpt-4o          # alias from config.yaml
export OPENAI_BASE_URL=http://localhost:4000
export OPENAI_API_KEY=$LITELLM_VIRTUAL_KEY                # gateway virtual key

python examples/notebook_02_basic_agent.py
python examples/notebook_03_agent_with_tools.py
# …
```

## Deploying on Kubernetes

The sample [`helm-values.yaml`](https://github.com/oracle/locus/blob/main/examples/litellm-gateway/helm-values.yaml)
in `examples/litellm-gateway/` plugs into LiteLLM's official Helm chart
([`ghcr.io/berriai/litellm-helm`](https://github.com/BerriAI/litellm/tree/main/helm/litellm-helm)).
The recommended deployment shape is:

- One LiteLLM gateway Deployment per environment.
- Provider credentials wired in via Kubernetes secrets (or via a
  workload-identity binding if your platform supports keyless cloud
  auth — see "Authentication" below).
- Postgres for virtual-key state and spend logs.
- Service exposed cluster-internal only — services built on the SDK hit it via
  the in-cluster DNS name (`litellm-gateway.litellm.svc.cluster.local:4000`).

Don't expose the gateway publicly — issuing virtual keys is your
auth boundary, but the provider credentials inside the gateway are not.

The values file uses the keys the chart reads: `environmentSecrets` exports
every key of the `provider-credentials` and `litellm-master` Secrets as
environment variables for `os.environ/<NAME>` in `config.yaml`,
`masterkeySecretName` / `masterkeySecretKey` point the chart's own master
key at the same Secret, and `db.useExisting: true` with
`db.deployStandalone: false` connects to your Postgres instead of bundling
one. Render it locally with:

```bash
helm template gateway oci://ghcr.io/berriai/litellm-helm --version 1.109.1 \
  --namespace litellm \
  --values examples/litellm-gateway/helm-values.yaml \
  --set-file proxy_config=examples/litellm-gateway/config.yaml
```

!!! warning "Not verified on a cluster by the maintainers"
    `helm lint` and `helm template` pass with the sample values against
    chart 1.109.1, and the rendered Deployment carries both Secrets, the
    external Postgres and a ClusterIP Service. An install on a live
    cluster — OKE or any other — is not something the Locus maintainers
    have run. To verify it yourself:

    ```bash
    kubectl create namespace litellm
    kubectl -n litellm create secret generic provider-credentials \
      --from-literal=OPENAI_API_KEY="sk-..." \
      --from-literal=ANTHROPIC_API_KEY="sk-ant-..."
    kubectl -n litellm create secret generic litellm-master \
      --from-literal=LITELLM_MASTER_KEY="$(openssl rand -hex 32)"
    kubectl -n litellm create secret generic litellm-db \
      --from-literal=username=litellm --from-literal=password="..."   # your Postgres

    helm -n litellm upgrade --install gateway oci://ghcr.io/berriai/litellm-helm \
      --version 1.109.1 \
      --values examples/litellm-gateway/helm-values.yaml \
      --set-file proxy_config=examples/litellm-gateway/config.yaml
    kubectl -n litellm rollout status deployment/gateway-litellm
    helm -n litellm test gateway                        # the chart's connection test

    kubectl -n litellm port-forward svc/gateway-litellm 4000:4000 &
    curl -s http://localhost:4000/health/liveness
    export LITELLM_GATEWAY_URL=http://localhost:4000
    export LITELLM_GATEWAY_KEY="$(kubectl -n litellm get secret litellm-master \
      -o jsonpath='{.data.LITELLM_MASTER_KEY}' | base64 -d)"
    pytest tests/integration/test_litellm_gateway_live.py \
           tests/integration/test_litellm_gateway_platform.py -v
    ```

    The tests cover the model list, auth, completions, streaming, tool
    calls and the PII guardrail through the forwarded port. For
    `config.platform.yaml` on a cluster, mount it with `--set-file` instead
    and add a Secret holding `REDIS_URL` and `OTEL_EXPORTER_OTLP_ENDPOINT`
    to `environmentSecrets`.

## Authentication

The gateway changes the credential boundary:

| Without gateway | With gateway |
|---|---|
| SDK → provider directly. The SDK carries the provider API key. | SDK → gateway with a **virtual key**. Gateway → provider with the provider API key. |

So **the SDK no longer needs provider credentials at all** — the gateway
is the only thing that does. The SDK only needs the virtual API key the
gateway issued it. This is the central reason to deploy the gateway
on a multi-tenant platform: agents from different teams use different
virtual keys with different budgets, all hitting the same underlying
provider accounts.

## What lives in `config.yaml`

The sample `examples/litellm-gateway/config.yaml` declares the provider
entries (one per model you want to expose), a virtual-key section (mock
or Postgres-backed), and the global gateway settings.
The full schema is documented at
[docs.litellm.ai/docs/proxy/configs](https://docs.litellm.ai/docs/proxy/configs).
Highlights:

- **`model_list`** — every model alias the gateway exposes. The same
  alias is what the SDK passes as `model=` to `OpenAIModel`.
- **`general_settings.master_key`** — the admin key that creates
  per-team virtual keys via `/key/generate`.
- **`router_settings.fallbacks`** — fallback chains across model
  aliases (e.g. `[{"gpt-5-mini": ["claude-sonnet-4-6", "gpt-4o"]}]`).
- **`guardrails`** — the in-process PII mask from "Guardrails" above;
  each entry names a guardrail, its mode and whether it is on by default.
- **`litellm_settings.callbacks`** — observability hooks (OpenTelemetry,
  Langfuse, Datadog, …). In `config.platform.yaml`.
- **`litellm_settings.cache`** — Redis / S3 / Qdrant caching config. In
  `config.platform.yaml`, because the gateway needs the backend to start.

## How enterprises use this pattern

The recurring deployment shape inside large organisations adopting
LLMs across many teams is *one gateway per environment, owned by a
platform team, fronting every provider, accessed by every service*.

The platform-grade pieces it earns them:

- **Charge-back / showback** — finance pulls a SQL report keyed on
  virtual key + `team` metadata; per-team costs roll up without
  manual reconciliation.
- **Compliance, audit, data residency** — append-only spend log
  (ISO-27001 / SOC-2 / PCI-friendly); PII redaction via guardrails
  *before* prompts leave the tenancy.
- **Centralised governance** — security/IT control which providers,
  models, and regions are approved; engineering can't bypass.
- **Vendor diversification** — declarative fallback chains across
  regions and providers; application code stays one `OpenAIModel` call.
- **Quota arbitration** — per-key `rpm_limit` / `tpm_limit` /
  `max_budget` lets the platform team fair-share shared vendor quotas.
- **Observability** — `success_callback` / `failure_callback` push
  LLM spans into the existing Datadog / OTel / Splunk pipeline.
- **Cost optimisation that compounds** — cache identical prompts,
  route cheap requests to cheap models, identify top-spend prompts
  and rewrite them. All require centralised visibility.
- **Polyglot consumers** — Python services on the SDK, JS clients, Go / Ruby /
  Java services all talk the same OpenAI-shaped HTTP.

### Deployment-shape table

| Layer | Owner | Lives in |
|---|---|---|
| Provider accounts + API keys | Cloud / security team | Secret manager, workload identity |
| Gateway pod + Postgres + Redis + obs backends | Platform / SRE team | Kubernetes, one deployment per env |
| Gateway `config.yaml` (model catalog, fallbacks, callbacks, guardrails) | Platform team | GitOps repo, change-controlled |
| Virtual keys + per-team budgets | Platform team issues; security reviews | Postgres; admin UI for issuance |
| Agents built with the SDK / other consumers | Application teams | Their own services, talking to `litellm-gateway.<env>.svc.cluster.local:4000` |
| Spend reports + audit + alerts | Finance + security | SQL on the gateway's Postgres; obs dashboards |

The pattern lets the platform team **set policy once** and application
teams **consume it through a single contract** — without anyone writing
provider-specific integration code or holding provider credentials.
LiteLLM's own [enterprise documentation](https://docs.litellm.ai/docs/enterprise)
covers each surface (callbacks, cache, guardrails, audit) in depth.

## See also

- [Model providers](../concepts/models.md) — the direct providers
  (`OpenAIModel`, `AnthropicModel`). The default for
  single-tenant deployments.
- [`examples/litellm-gateway/`](https://github.com/oracle/locus/tree/main/examples/litellm-gateway)
  — working `config.yaml`, `config.platform.yaml`, `docker-compose.yml`,
  and `helm-values.yaml`.
- [`examples/codex-on-oci/`](https://github.com/oracle/locus/tree/HEAD/examples/codex-on-oci)
  — the same gateway in front of OCI Generative AI, signed with an OCI
  profile, so that [Codex runs on OCI](codex.md#codex-on-oci-generative-ai).
- [LiteLLM AI Gateway quickstart](https://docs.litellm.ai/docs/proxy/quick_start)
- [LiteLLM `config.yaml` reference](https://docs.litellm.ai/docs/proxy/configs)
- [LiteLLM Helm chart](https://github.com/BerriAI/litellm/tree/main/helm/litellm-helm)
