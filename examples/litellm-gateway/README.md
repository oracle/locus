# LiteLLM AI Gateway — sample in front of OpenAI + Anthropic

Working sample for deploying the [LiteLLM AI Gateway](https://litellm.ai)
in front of your upstream providers (OpenAI, Anthropic, and any other
LiteLLM-supported backend), with Locus pointed at it via its existing
`OpenAIModel(base_url=...)`.

| File | What it is |
|---|---|
| [`config.yaml`](config.yaml) | Gateway model catalog: providers, fallback chains, drop_params, the in-process PII guardrail. Needs nothing beyond the gateway and its Postgres. |
| [`config.platform.yaml`](config.platform.yaml) | The same gateway plus the two add-ons that need a backing service: the OpenTelemetry callback and the Redis response cache. |
| [`docker-compose.yml`](docker-compose.yml) | One-command local-dev gateway with a Postgres sidecar. Reads provider credentials from env vars on the host; switches to `config.platform.yaml` when `REDIS_URL` is set and adds a Redis sidecar under the `cache` profile. |
| [`helm-values.yaml`](helm-values.yaml) | Kubernetes deployment via the official `litellm-helm` chart. |

## Local quickstart

```bash
# 1. Set the provider credentials the gateway will use upstream.
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export LITELLM_MASTER_KEY="$(openssl rand -hex 32)"  # admin key for /key/generate
export LITELLM_DB_PASSWORD="$(openssl rand -hex 16)"  # postgres pw — change for non-throwaway

# 2. Start the gateway.
docker compose up

# 3. From your Locus app — no Locus-side code change needed:
#
#   from locus.models.native.openai import OpenAIModel
#   model = OpenAIModel(
#       model="gpt-4o",                      # alias from config.yaml
#       api_key="<gateway-virtual-key>",     # issued via /key/generate
#       base_url="http://localhost:4000",
#   )
```

### Cache and traces

Set `REDIS_URL` and start with the `cache` profile. The gateway then runs
`config.platform.yaml`: identical requests are answered from Redis, and
every request becomes an OpenTelemetry trace. Spans go to the collector in
`OTEL_EXPORTER_OTLP_ENDPOINT`, or to the gateway log when that is unset.

```bash
export REDIS_URL=redis://redis:6379
export OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4318   # your collector; optional
docker compose --profile cache up -d
```

For Langfuse, replace `otel` with `langfuse` in `config.platform.yaml` and
export `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST`;
the compose file passes them through.

### Behind a corporate proxy?

If your Docker daemon can't reach Docker Hub directly (TLS interception
on a corporate egress proxy is the usual culprit), the `postgres` pull
fails with `tls: failed to verify certificate: x509: certificate signed
by unknown authority`. Override the image with any mirror you can
reach:

```bash
# Use Google's container-registry mirror — typically reachable through
# corporate networks that block Docker Hub.
export LITELLM_DB_IMAGE="mirror.gcr.io/library/postgres:17-alpine"
export LITELLM_REDIS_IMAGE="mirror.gcr.io/library/redis:7-alpine"   # with the cache profile
docker compose up
```

The `litellm` image itself ships from `ghcr.io/berriai/litellm` (GitHub
Container Registry, separate proxy story) — if *that* one fails too,
override `LITELLM_IMAGE` similarly or pull all of them via your internal
registry. Every image variable is honored by `docker-compose.yml`.

## Kubernetes quickstart

```bash
# 1. Provider keys, master key and the external Postgres credentials as secrets.
kubectl create namespace litellm
kubectl -n litellm create secret generic provider-credentials \
  --from-literal=OPENAI_API_KEY="sk-..." \
  --from-literal=ANTHROPIC_API_KEY="sk-ant-..."
kubectl -n litellm create secret generic litellm-master \
  --from-literal=LITELLM_MASTER_KEY="$(openssl rand -hex 32)"
kubectl -n litellm create secret generic litellm-db \
  --from-literal=username=litellm --from-literal=password="..."

# 2. Install the chart with the sample values + mount the config.yaml.
helm -n litellm upgrade --install gateway oci://ghcr.io/berriai/litellm-helm \
  --values helm-values.yaml \
  --set-file proxy_config=config.yaml
```

`helm lint` and `helm template` pass with these values; an install on a
cluster is not something the Locus maintainers have verified — the guide
below has the sequence to do so.

## Full documentation

See the [LiteLLM gateway guide](https://locusagents.ai/how-to/litellm-gateway/)
for when to choose this path over direct providers, the auth boundary
diagram, the observability, caching and guardrail sections, and the
notebook-run-via-gateway recipe.
