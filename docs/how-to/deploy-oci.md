# Deploy to OCI with `locus deploy`

`locus deploy` puts an `AgentServer` on Oracle Cloud Infrastructure from one
values file. It writes a deployment directory with Terraform roots for three
targets, builds the server image from a Dockerfile it generates, pushes the
image to OCI Registry (OCIR) and applies the target you chose. The three
targets share one identity model, one secrets model and one set of variables
for the managed services an agent needs: the Oracle AI Database checkpointer,
OCI Cache, OCI Object Storage and OCI Application Performance Monitoring.

| Field | Value |
|---|---|
| Status | Supported CLI; the Terraform roots and the Helm chart are reference configurations to adapt |
| Tested SDK | `locus-sdk` {{ locus_sdk_version }} |
| Execution mode | Live OCI tenancy |
| Requirements | `locus-sdk[server,oci,deploy]`, Terraform 1.9+, Docker, the OCI CLI configuration Terraform reads (`~/.oci/config`), an OCIR auth token for `docker login`, an OCI Vault with a key for the server's bearer key |
| Secrets | OCI Vault secret OCIDs in the values file; nothing secret in the repository or the image |

Pick the target by how the agent is called:

| Target | Suits | Identity | Scales | Operate |
|---|---|---|---|---|
| **Container Instances** | one agent, steady traffic, the first deployment | resource principal | one instance per root; put a load balancer in front for more | nothing: no cluster, no node |
| **OKE** (Kubernetes Engine) | several agents, replicas, autoscaling, an existing platform team | workload identity (resource principal per service account) or the node's instance principal | Horizontal Pod Autoscaler on CPU | the cluster you already run |
| **Functions** | bursty or event-driven calls, scale to zero | resource principal | per invocation, up to the application's concurrency | nothing; cold starts and a 300 s ceiling per call |

## Quickstart

The server answers only callers that present a bearer key, so the key comes
first: generate it, keep it in OCI Vault and hand `init` the secret's OCID.

```bash
pip install "locus-sdk[server,oci,deploy]"

LOCUS_SERVER_API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
SECRET_OCID=$(oci vault secret create-base64 \
    --compartment-id "$OCI_COMPARTMENT_ID" --vault-id "$VAULT_ID" --key-id "$KMS_KEY_ID" \
    --secret-name concierge-server-key \
    --secret-content-content "$(printf %s "$LOCUS_SERVER_API_KEY" | base64)" \
    --query 'data.id' --raw-output)

locus deploy init concierge \
    --target container-instances \
    --compartment-id "$OCI_COMPARTMENT_ID" \
    --region eu-frankfurt-1 \
    --profile "$OCI_PROFILE" \
    --server-api-key-secret-id "$SECRET_OCID"
cd concierge
```

`init` writes `locus-deploy.yaml`, the file you edit, and `app/server.py`,
the agent the image serves. The generated server reads its model from
`LOCUS_MODEL` and authenticates to OCI Generative AI with the principal the
platform gives it; put `LOCUS_MODEL: mock` under `environment:` for a first
deployment that proves the path before a model is wired, then remove it.

`init` also says which SDK the image installs: `app/requirements.txt` pins
`locus-sdk[server,oci]` to the version running `locus deploy`. To ship a
build that is not on PyPI yet, pass `--sdk-wheel dist/locus_sdk-<version>-py3-none-any.whl`
(from `python -m build --wheel`): the wheel is copied to `app/wheels/` and the
Dockerfiles install it before `requirements.txt`, whose pin then matches the
wheel's version.

```bash
docker login eu-frankfurt-1.ocir.io      # <namespace>/<user>, the auth token as password
locus deploy plan                        # terraform plan: registry, then the target
locus deploy apply                       # repository, image build + push, then the target
```

`apply` ends with the target's outputs. For Container Instances that is a
`url`:

```bash
curl -sf "$URL/health"
curl -sS -X POST "$URL/invoke" \
  -H "Authorization: Bearer $LOCUS_SERVER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello"}'
```

`locus deploy destroy` removes the target, then the repository and its
images. `plan`, `apply` and `destroy` take `--dry-run` to print what they
would run. A usage problem (a bad values file, a missing variable) exits 2; a
Terraform or Docker step that fails exits 1 after that tool's own output.

## The deployment directory

```
concierge/
├── locus-deploy.yaml          # target, compartment, image, variables, environment
├── Dockerfile                 # Container Instances and OKE image (Oracle Linux 9, Python 3.12)
├── Dockerfile.functions       # the same server behind the Fn FDK
├── .dockerignore              # the build context is app/ alone: no Terraform state or tfvars
├── app/
│   ├── server.py              # your agent: `app = server.app`
│   ├── func.py                # Functions entry, two lines
│   ├── requirements.txt       # locus-sdk[server,oci]==<version> plus what your agent imports
│   └── wheels/                # a local locus-sdk build, with --sdk-wheel
├── terraform/
│   ├── registry/              # the OCIR repository
│   ├── container-instances/
│   ├── oke/
│   ├── functions/
│   └── modules/agent-environment/   # one place that turns variables into environment
└── helm/locus-agent/          # the chart the oke root installs; usable with helm alone
```

`terraform/` and `helm/` are refreshed by every `init`; `locus-deploy.yaml`, `app/server.py` and `app/requirements.txt` are yours and are only rewritten with `--force`. A `Dockerfile`, `Dockerfile.functions` or `.dockerignore` that already exists with other content (an existing project's, or one you edited) stops `init` until you pass `--force`, and an existing `.gitignore` only gains the missing lines under a `# locus deploy` marker.

Commit the directory: the state files and the generated `locus.auto.tfvars.json` are ignored by that `.gitignore`, because they name your tenancy, and `.dockerignore` keeps them, the `.terraform/` provider caches and every `*.tfvars` out of the image.

### `locus-deploy.yaml`

| Key | Meaning |
|---|---|
| `target` | `container-instances`, `oke` or `functions` |
| `name` | Prefix for every resource, the repository name and the Helm release |
| `region`, `compartment_id` | Where everything lands |
| `oci_profile` | Profile in `~/.oci/config` for Terraform; `null` uses `DEFAULT` or the `OCI_CLI_*` environment |
| `app` | `MODULE:ATTR` of the ASGI app in `app/`, `server:app` by default |
| `port` | What the server listens on in the container |
| `image.tag` | The tag `apply` pushes; bump it for every release |
| `image.architecture` | `amd64` or `arm64`: the Docker build platform and the shape family (`CI.Standard.E4.Flex` or `CI.Standard.A1.Flex`, `GENERIC_X86` or `GENERIC_ARM`, the `kubernetes.io/arch` node selector) |
| `image.registry_public` | Create the repository public; a private one needs a pull secret on the compute side |
| `image.repository` | Push to an existing repository path instead; skips the registry root |
| `variables` | Terraform variables of `terraform/<target>`, passed through unchanged; `null` leaves the module default |
| `environment` | Extra environment variables for the server; these win over the derived ones |

Every root declares the same managed-service variables; `init` seeds
`variables:` with the knobs a first deployment touches. Two of them decide
who can call the agent:

| Variable | Meaning |
|---|---|
| `server_api_key_secret_id` | The Vault secret holding the bearer key (`--server-api-key-secret-id` at `init`). The generated server refuses to start without the key it resolves to. |
| `allow_unauthenticated` | `true` runs the server with no bearer key (`--allow-unauthenticated`); the container gets `LOCUS_SERVER_ALLOW_UNAUTHENTICATED=1`. For an agent behind an API Gateway or a private network that authenticates callers itself. |

A Container Instance with the root's defaults has a public IP open to
`0.0.0.0/0`, so `plan` and `apply` refuse that target until one of the two is
set, and the root's own validation says the same.

## Identity

The container never holds a credential. It authenticates to OCI services with
the principal the platform gives it, and the roots export the pieces the SDK
needs: `OCI_COMPARTMENT_ID`, `OCI_REGION`, and `LOCUS_OCI_AUTH_TYPE` with
`OCI_AUTH_TYPE` (`resource_principal` on Container Instances and Functions,
`oke_workload_identity` on OKE workload identity, `instance_principal` on OKE
node identity). The generated `server.py` passes them to the model:

```python
import os

from locus.agent import Agent

agent = Agent(
    model=os.environ.get("LOCUS_MODEL", "oci:openai.gpt-4.1"),
    model_kwargs={
        "auth_type": os.environ.get("LOCUS_OCI_AUTH_TYPE", "resource_principal"),
        "compartment_id": os.environ["OCI_COMPARTMENT_ID"],
    },
    tools=[],
    system_prompt="Answer concisely.",
)
```

A tenancy administrator grants the principal what the agent uses. Each root's
`resource_principal_rule` output prints the matching rule for the dynamic
group; the policy then names the group:

```text
# Dynamic group (one per target; the root prints the rule)
ALL {resource.type = 'computecontainerinstance', resource.compartment.id = '<compartment>'}
ALL {resource.type = 'fnfunc', resource.compartment.id = '<compartment>'}
ALL {instance.compartment.id = '<compartment>'}                 # OKE node identity

# Policy, in the compartment the agent runs in
Allow dynamic-group <group> to use generative-ai-family in compartment <name>
Allow dynamic-group <group> to read secret-bundles in compartment <name>
Allow dynamic-group <group> to manage objects in compartment <name> where target.bucket.name = '<bucket>'
```

OKE workload identity (enhanced clusters) needs no dynamic group; the policy
subject is the service account the chart creates:

```text
Allow any-user to use generative-ai-family in compartment <name> where all {
  request.principal.type = 'workload',
  request.principal.namespace = 'locus',
  request.principal.service_account = '<release>-locus-agent',
  request.principal.cluster_id = '<cluster-id>' }
```

Add `read secret-bundles` and the Object Storage statements the same way. `<release>` is the Helm release name, the deployment's `name`; the `service` and `service_account` outputs of the OKE root print the exact value (a release name that already contains `locus-agent` is used as is, so the default `locus-agent` release creates a service account named `locus-agent`).

The OKE root's `workload_identity` variable switches between the two models. With workload identity the chart sets `LOCUS_OCI_AUTH_TYPE` and `OCI_AUTH_TYPE` to `oke_workload_identity` next to `OCI_RESOURCE_PRINCIPAL_VERSION=2.2` and the region: a pod holds no resource principal session token, so the SDK exchanges the pod's Kubernetes service-account token through the OKE workload-identity signer.

The SDK's `resource_principal` auth type detects that token on its own and does the same, so an agent written for Functions runs unchanged on OKE; `oke_workload_identity` names the path outright.

## Secrets

A secret reaches the container as an OCI Vault secret OCID in a variable named
`<VAR>_SECRET_OCID`. The image starts with `python -m locus.deploy.entrypoint`,
which reads each one with the container's principal and sets `<VAR>` before
uvicorn starts. A `<VAR>` that is already set wins, so a value injected another
way is never overwritten, and an image with no `*_SECRET_OCID` variable never
imports the OCI SDK at start-up.

The roots take the common ones as variables:

| Variable | Becomes | Read by |
|---|---|---|
| `server_api_key_secret_id` | `LOCUS_SERVER_API_KEY` | `AgentServer`, the bearer key callers send |
| `oracle_password_secret_id` | `ORACLE_PASSWORD` | the Oracle AI Database checkpointer |
| `apm_data_key_secret_id` | `LOCUS_APM_DATA_KEY`, then `OTEL_EXPORTER_OTLP_HEADERS` | the OTLP exporter behind `TelemetryHook` |
| `image_pull_secret_id` (Container Instances) | the registry pull secret | the Container Instances service, for a private repository |

Any other secret goes through `environment:` the same way:

```yaml
environment:
  SLACK_WEBHOOK_SECRET_OCID: ocid1.vaultsecret.oc1..example
```

The server key is the one the quickstart creates first; `init` takes its OCID
as `--server-api-key-secret-id` and the roots hand it to the container as
`LOCUS_SERVER_API_KEY_SECRET_OCID`. The policy line `read secret-bundles`
above is what lets the container read it. On OKE a Kubernetes Secret works
too: `serverApiKey.existingSecret` in the chart maps a Secret key onto
`LOCUS_SERVER_API_KEY`. Without a key the generated server exits at start-up
with the reason; `allow_unauthenticated: true` is the explicit way to run it
open.

## Managed services

The variables below exist on every root and become environment variables
through `terraform/modules/agent-environment`. Wire them in `app/server.py`;
the server template only reads the model settings.

### Oracle AI Database — the checkpointer

`oracle_dsn`, `oracle_user` and `oracle_password_secret_id` become
`ORACLE_DSN`, `ORACLE_USER` and `ORACLE_PASSWORD`. Use the database's one-way
TLS connect string (Autonomous Database: *TLS* authentication with an access
control list or a private endpoint) so the container needs no wallet. The
container reaches a private endpoint when its subnet routes to the database's
VCN; a public Autonomous Database needs the container's egress IP on the ACL.

```python
import os

from locus.memory.backends.oracle import OracleBackend

checkpointer = OracleBackend(
    dsn=os.environ["ORACLE_DSN"],
    user=os.environ["ORACLE_USER"],
    password=os.environ["ORACLE_PASSWORD"],
)
```

Pass it to `Agent(checkpointer=...)` and `thread_id` survives restarts and
replicas. [Persist conversations](persist-conversations.md) covers the
backends; the Oracle Database notebooks cover the schema it creates.

### OCI Cache — hot threads and rate limits

`cache_url` becomes `REDIS_URL`. OCI Cache speaks the Redis protocol over TLS,
so the value is `rediss://<cache-endpoint>:6379`. The cluster must be in a
VCN the container reaches: the same VCN, or a peered one.

```python
import os

from locus.memory.backends.redis import RedisBackend

checkpointer = RedisBackend(url=os.environ["REDIS_URL"])
```

### OCI Object Storage — checkpoints and audit anchors

`object_storage_bucket` and `object_storage_namespace` become
`LOCUS_BUCKET_NAME` and `LOCUS_OS_NAMESPACE`. `OCIBucketBackend` keeps
checkpoints there; `OCIBucketAnchorStore` anchors an audit trail's chain head
into a bucket with a retention rule. Both take the principal through the OCI
SDK's default signer chain, so no extra configuration is needed inside the
container; the policy statement is `manage objects` on that bucket.

### OCI Application Performance Monitoring — traces

`apm_data_upload_endpoint` and `apm_data_key_secret_id` set
`OTEL_EXPORTER_OTLP_ENDPOINT` to the domain's OTLP path,
`OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf` and, once the data key is
resolved, `OTEL_EXPORTER_OTLP_HEADERS`. Add `TelemetryHook` to the agent and
every run, tool call and model call arrives as a trace in APM Trace Explorer.
Install the `telemetry` extra in `app/requirements.txt`.

## Target notes

### Container Instances

The root creates one container instance in a VCN. With `create_network: true` it also creates the VCN: a public subnet, an internet gateway and a security list that admits the agent port from `ingress_cidr`; the instance pulls its image and reaches OCI services over its public IP. Narrow `ingress_cidr` to your callers.

For an existing network, set `create_network: false` and `subnet_id`; a private subnet needs a NAT gateway or a service gateway for OCIR and the OCI APIs, and `assign_public_ip: false` with an OCI load balancer or an API Gateway in front.

The health check is `GET /health` every 30 s; the service restarts the
container after three failures. The instance has no autoscaler: one instance
serves one agent at the concurrency uvicorn handles, which is many idle chat
sessions but few simultaneous long tool runs. Run more roots behind a load
balancer, or move to OKE, when one is not enough.

### OKE

The root installs `helm/locus-agent` on an existing cluster through the Helm
provider. Give it `cluster_id` (the kubeconfig is fetched from the API) or
`kubeconfig_path`. The chart creates a Deployment with readiness and liveness
probes on `/health`, a Service (a flexible OCI load balancer by default;
`ClusterIP` for an ingress controller), a ServiceAccount, a
PodDisruptionBudget and, when `max_replicas` exceeds `replicas`, a
HorizontalPodAutoscaler at 70% CPU. Pods run as user 10001 with a read-only
root filesystem.

The chart works without Terraform:

```bash
helm upgrade --install concierge ./helm/locus-agent \
  --namespace locus --create-namespace \
  --set image.repository=eu-frankfurt-1.ocir.io/<namespace>/concierge \
  --set image.tag=0.1.0 \
  --set oci.region=eu-frankfurt-1 \
  --set env.OCI_COMPARTMENT_ID="$OCI_COMPARTMENT_ID" \
  --set env.LOCUS_SERVER_API_KEY_SECRET_OCID="$SECRET_OCID"
```

Installed this way the chart derives `LOCUS_OCI_AUTH_TYPE` and
`OCI_AUTH_TYPE` from `oci.workloadIdentity`; a value under `env:` wins.

A private repository needs a docker-registry Secret in the namespace, named in
`image_pull_secret_name`. Server-Sent Events from `/stream` pass through the
OCI load balancer unbuffered; an ingress controller in front must not buffer
responses either.

### Functions

The root creates an application in `subnet_id` and one function from
`Dockerfile.functions`. Inside the image the Fn FDK calls
`locus.deploy.functions.asgi_handler`, which runs each invocation as one
request on the `AgentServer` app:

- A direct invocation (`oci fn function invoke`, an Events rule, a Notifications
  subscription) carries no HTTP method or path. It becomes `POST /invoke` with
  the body as sent, authenticated with the function's own `LOCUS_SERVER_API_KEY`
  because the Functions API already authenticated the caller through IAM.
- Behind an API Gateway or an HTTP trigger, the caller's method, path, headers
  and bearer token reach the server unchanged. Set `LOCUS_FN_PATH_PREFIX` to the
  gateway deployment's prefix (`/agent`) so `/agent/invoke` reaches `/invoke`.

```bash
oci fn function invoke --function-id "$FUNCTION_ID" --file - \
  --body '{"prompt": "hello"}'
```

Streaming returns the whole response at once: a function returns one body. An
invocation ends at `timeout_seconds`, 300 at most, so an agent turn with
several tools must fit, and a cold start adds a few seconds of image start-up
to the first call; `provisioned_concurrency` keeps warm instances ready at a
cost. The application's subnet needs a route to the model endpoint and to the
services the agent uses, through a NAT gateway or a service gateway.

## Sizing and cost

Price lists are the authority; the shape of the bill is stable.

| Target | Starting size | Billed on | Notes |
|---|---|---|---|
| Container Instances | 1 OCPU, 4 GB (`ocpus`, `memory_gbs`) | OCPU-hours and GB-hours of the shape, as for a compute instance of the same family; no per-instance fee | The SDK with the OCI provider idles near 300 MB. A1 (arm64) OCPUs cost a fraction of E4 (amd64); build for arm64 when every dependency has a wheel. |
| OKE | 2 replicas at 500m CPU / 1 Gi requested, 2 CPU / 4 Gi limit | Worker nodes at compute rates; the control plane is free on basic clusters and billed per cluster-hour on enhanced clusters (which workload identity needs); the load balancer on its bandwidth range (`load_balancer_min_mbps`, `load_balancer_max_mbps`) | A 10 Mbps flexible load balancer is the smallest; SSE streams are small. |
| Functions | 1024 MB, 300 s (`memory_mbs`, `timeout_seconds`) | Invocations and GB-seconds; a monthly always-free allowance covers light use | Memory below 512 MB does not fit the SDK with the OCI provider. Provisioned concurrency is billed while provisioned. |

Model calls are billed by OCI Generative AI separately and are the dominant
cost of an agent that is actually used; the Oracle AI Database, OCI Cache and
APM are billed on their own units. A Container Instance running around the
clock is cheaper than Functions above a few hundred thousand invocations a
month and more expensive below a few thousand.

## Deploy on a tag from CI

A workflow that applies the deployment when a version tag is pushed. It is a
template to copy into your project; the secrets are the OCI API key of a
deployment user (or an OIDC federation if your tenancy has one), the OCIR
auth token, and the deployment's values.

```yaml
name: deploy
on:
  push:
    tags: ["v*"]

jobs:
  apply:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - uses: hashicorp/setup-terraform@v3
        with: { terraform_version: "1.9.*" }

      - name: OCI configuration
        run: |
          mkdir -p ~/.oci
          printf '%s\n' "${{ secrets.OCI_CONFIG }}" > ~/.oci/config
          printf '%s\n' "${{ secrets.OCI_API_KEY }}" > ~/.oci/key.pem
          chmod 600 ~/.oci/config ~/.oci/key.pem

      - name: Registry login
        run: |
          echo "${{ secrets.OCIR_AUTH_TOKEN }}" | docker login "${{ vars.OCI_REGION }}.ocir.io" \
            -u "${{ vars.OCIR_USERNAME }}" --password-stdin

      - name: Deploy
        run: |
          pip install "locus-sdk[server,oci,deploy]"
          # The tag is the image tag.
          sed -i "s/^  tag: .*/  tag: ${GITHUB_REF_NAME#v}/" deploy/locus-deploy.yaml
          locus deploy plan deploy
          locus deploy apply deploy
```

Keep Terraform state out of the runner: add a `backend.tf` next to each root
(an object-storage backend on an OCI Object Storage bucket) before the first
CI apply, and commit it. Run
`locus deploy plan` on pull requests the same way, without the apply step, to
review changes before they are tagged.

## Removing a deployment

```bash
locus deploy destroy
```

The target root is destroyed first, then the registry root deletes the
repository and every image in it. The Vault secrets, the database, the cache
and the bucket are not managed by these roots, so they stay.

## See also

- [Deploy](deploy.md): the server, the base image and generic container,
  serverless, Kubernetes and VM paths.
- [Agent Server](../concepts/server.md): the routes, auth and thread scoping.
- [OCI Generative AI models](oci-models.md): the `oci:` route, principals and
  compartments.
- [Observability](../concepts/observability.md): `TelemetryHook` and OTLP.
- Notebook 65, [Agent server](../notebooks/notebook_65_agent_server.md): the
  server this guide deploys.
