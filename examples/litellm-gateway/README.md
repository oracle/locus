# LiteLLM AI Gateway — sample in front of OCI Generative AI

Working sample for deploying the [LiteLLM AI Gateway](https://litellm.ai)
in front of Oracle Generative AI Infrastructure, with Locus pointed at
it via its existing `OpenAIModel(base_url=...)`.

| File | What it is |
|---|---|
| [`config.yaml`](config.yaml) | Gateway model catalog. OCI providers, fallback chains, drop_params. Mounted into the container. |
| [`docker-compose.yml`](docker-compose.yml) | One-command local-dev gateway. Reads OCI credentials from env vars on the host. |
| [`helm-values.yaml`](helm-values.yaml) | OKE / Kubernetes deployment via the official `litellm-helm` chart. |

## Local quickstart

```bash
# 1. Set OCI credentials the gateway will sign upstream calls with.
export OCI_REGION="us-chicago-1"
export OCI_USER="ocid1.user.oc1..xxx"
export OCI_FINGERPRINT="aa:bb:cc:..."
export OCI_TENANCY="ocid1.tenancy.oc1..xxx"
export OCI_KEY_FILE="$HOME/.oci/keys/your_api_key.pem"
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..xxx"
export LITELLM_MASTER_KEY="$(openssl rand -hex 32)"  # admin key for /key/generate

# 2. Start the gateway.
docker compose up

# 3. From your Locus app — no Locus-side code change needed:
#
#   from locus.models.native.openai import OpenAIModel
#   model = OpenAIModel(
#       model="oci-cohere-command",          # alias from config.yaml
#       api_key="$LITELLM_VIRTUAL_KEY",              # virtual key
#       base_url="http://localhost:4000",
#   )
```

## OKE quickstart

```bash
# 1. OCI credentials + master key as secrets.
kubectl create namespace litellm
kubectl -n litellm create secret generic oci-credentials \
  --from-literal=OCI_REGION="us-chicago-1" \
  --from-literal=OCI_USER="ocid1.user.oc1..xxx" \
  --from-literal=OCI_FINGERPRINT="aa:bb:cc:..." \
  --from-literal=OCI_TENANCY="ocid1.tenancy.oc1..xxx" \
  --from-literal=OCI_COMPARTMENT_ID="ocid1.compartment.oc1..xxx" \
  --from-file=OCI_KEY_FILE=/path/to/api_key.pem

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
for when to choose this path over the direct OCI providers, the auth
boundary diagram, and the notebook-run-via-gateway recipe.
