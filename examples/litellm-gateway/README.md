# LiteLLM AI Gateway — sample in front of OpenAI + Anthropic

Working sample for deploying the [LiteLLM AI Gateway](https://litellm.ai)
in front of your upstream providers (OpenAI, Anthropic, and any other
LiteLLM-supported backend), with Tulip pointed at it via its existing
`OpenAIModel(base_url=...)`.

| File | What it is |
|---|---|
| [`config.yaml`](config.yaml) | Gateway model catalog. Providers, fallback chains, drop_params. Mounted into the container. |
| [`docker-compose.yml`](docker-compose.yml) | One-command local-dev gateway. Reads provider credentials from env vars on the host. |
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

# 3. From your Tulip app — no Tulip-side code change needed:
#
#   from tulip.models.native.openai import OpenAIModel
#   model = OpenAIModel(
#       model="gpt-4o",                      # alias from config.yaml
#       api_key="<gateway-virtual-key>",     # issued via /key/generate
#       base_url="http://localhost:4000",
#   )
```

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
docker compose up
```

The `litellm` image itself ships from `ghcr.io/berriai/litellm` (GitHub
Container Registry, separate proxy story) — if *that* one fails too,
override `LITELLM_IMAGE` similarly or pull both via your internal
registry. Both env vars are honored by `docker-compose.yml`.

## Kubernetes quickstart

```bash
# 1. Provider keys + master key as secrets.
kubectl create namespace litellm
kubectl -n litellm create secret generic provider-credentials \
  --from-literal=OPENAI_API_KEY="sk-..." \
  --from-literal=ANTHROPIC_API_KEY="sk-ant-..."

kubectl -n litellm create secret generic litellm-master \
  --from-literal=LITELLM_MASTER_KEY="$(openssl rand -hex 32)"

# 2. Install the chart with the sample values + mount the config.yaml.
helm repo add litellm oci://ghcr.io/berriai/litellm-helm
helm -n litellm upgrade --install gateway litellm/litellm \
  --values helm-values.yaml \
  --set-file proxy_config=config.yaml
```

## Full documentation

See [`docs/how-to/litellm-gateway.md`](../../docs/how-to/litellm-gateway.md)
for when to choose this path over direct providers, the auth boundary
diagram, and the notebook-run-via-gateway recipe.
