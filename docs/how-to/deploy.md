# Deploy

`AgentServer` is a FastAPI wrapper that deploys anywhere FastAPI runs. The
runnable path below carries one provider choice from application code through
the container and Kubernetes configuration: pick the provider in the tabs, and
inject that provider's credential wherever the manifests below show a provider
secret.

| Field | Value |
|---|---|
| Status | Supported server API; deployment manifests are starting points |
| Tested SDK | `locus-sdk` {{ locus_sdk_version }} |
| Execution mode | Live model provider |
| Requirements | Python 3.11+, `locus-sdk[server]` plus the extra for your provider tab: `oci`, `openai` (also for Self-hosted), or `anthropic` |
| Secrets | The provider credential for your tab (table below); `LOCUS_SERVER_API_KEY` for callers |

Those two credentials serve different trust boundaries. The provider
credential lets the application call its model. The server key is the
bearer token clients send to `/invoke`, `/stream`, and thread routes.

## The shape you ship

{{ locus_model_tabs }}

```python
# server.py
import os

from locus.agent import Agent
from locus.server import AgentServer

agent = Agent(
    model="{{ locus_model }}",
    tools=[],
    system_prompt="Answer concisely.",
)

server = AgentServer(
    agent=agent,
    title="Booking concierge",
    api_key=os.environ["LOCUS_SERVER_API_KEY"],
)

# Module-level ASGI app, so a process manager can serve it directly:
#   uvicorn server:app
app = server.app

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=8080)
```

Each tab needs one provider credential; the table says which, and the
deployment paths below show where to inject it.

| Tab | Extra | Provider credential |
|---|---|---|
| OCI | `oci` | A principal on OCI infrastructure (below), or an OCI config profile at the service user's `~/.oci/config`, named by `OCI_PROFILE` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| Self-hosted | `openai` | `OLLAMA_BASE_URL`, the URL of your Ollama server; no key |

On OCI compute, OKE, or OCI Functions, the OCI tab authenticates with the
platform's principal instead of a stored credential. The `oci:` route forwards
`model_kwargs` to the model, so pass the principal type and the compartment
there; this snippet is OCI-specific, so it names the OCI model id directly:

```python
agent = Agent(
    model="oci:openai.gpt-5.5",
    model_kwargs={
        "auth_type": "instance_principal",
        "compartment_id": os.environ["OCI_COMPARTMENT_ID"],
    },
    tools=[],
    system_prompt="Answer concisely.",
)
```

`instance_principal` serves OCI VMs and OKE node identity;
`resource_principal` serves OCI Functions and OKE workload identity. See
[model support on OCI](oci-models.md#on-oci-compute-oke-functions-workload-identity).

You get out of the box:

- `POST /invoke` — synchronous run, returns the final `message` plus
  run metrics (`success`, `stop_reason`, `iterations`, `tool_calls`,
  `duration_ms`) as JSON.
- `POST /stream` — Server-Sent Events of every typed event.
- `GET / DELETE /threads/{id}` — conversation persistence.
- `GET /health` — liveness probe.

Provider credentials are read from the environment, or supplied by the
platform as a principal. Inject secrets at runtime; never bake one into the
image. Add a durable checkpointer before relying on thread continuity across
restarts or replicas.

Serving `app` with uvicorn, as the container and VM examples below do, skips
`AgentServer.run()` and with it the check that refuses a non-loopback host
when no key is configured. A `server.app` built without a key only logs a
warning, then serves every route without authentication. Keep `api_key` or
`LOCUS_SERVER_API_KEY` set to a non-empty value: the `os.environ[...]` lookup
above fails at import when the variable is missing, but an empty value turns
authentication off.

## Container — the universal target

The locus-sdk repo ships a multi-stage [`Dockerfile`](https://github.com/oracle/locus/blob/main/Dockerfile)
(non-root user, `HEALTHCHECK` on `/health`). It is the library base image:
built from the locus-sdk repo root, it installs `locus-sdk` with the
`openai`, `server`, and `checkpoints` extras but copies no application code,
and its `CMD` names a placeholder `app:server.app` for you to replace. The
`checkpoints` extra also pulls in the OCI SDK, so the image as built serves
the OCI, OpenAI, and Self-hosted tabs; for the Anthropic tab, add `anthropic`
to the extras on the `pip install` line before building. Build it once, then
derive your own image that adds `server.py`:

```bash
# In a locus-sdk checkout, from the repo root, at the tested release
git checkout v{{ locus_sdk_version }}
docker build -t registry.example.com/locus-sdk:{{ locus_sdk_version }} .
```

```dockerfile
# Dockerfile.app, next to your server.py
FROM registry.example.com/locus-sdk:{{ locus_sdk_version }}
COPY server.py /app/server.py
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
```

`server:app` is the `app = server.app` callable from the top of this page.
Build, push to any registry, and run anywhere that runs containers:

```bash
docker build -f Dockerfile.app -t registry.example.com/locus-concierge:0.1.0 .
docker push    registry.example.com/locus-concierge:0.1.0

# provider.env holds the provider credential for your tab, for example
# OPENAI_API_KEY=sk-... or OLLAMA_BASE_URL=http://ollama.internal:11434/v1
docker run -p 8080:8080 \
  --env-file provider.env \
  -e LOCUS_SERVER_API_KEY=replace-with-a-long-random-token \
  registry.example.com/locus-concierge:0.1.0
```

For the OCI tab, prefer the principal route from the table above, which needs
no mounted credential; off OCI infrastructure, mount an OCI config profile into
the container at the service user's `~/.oci/config` and set `OCI_PROFILE` in
`provider.env`. This single image runs unchanged on OCI Container Instances,
OKE, or any other container host.

## Serverless — scale to zero

This option suits low-frequency or bursty traffic. Wrap the FastAPI app in
an ASGI adapter for your functions platform, or deploy the container image
directly to a scale-to-zero container runtime such as OCI Container
Instances. The adapter wraps `server.app`; the FastAPI app is built on first
access to it. Set the provider credential and `LOCUS_SERVER_API_KEY` as
function secrets; on OCI Functions, the OCI tab uses `resource_principal`
and stores no provider credential.

## Kubernetes — for production

This option suits multi-replica, autoscaled, multi-region production on OKE
or any conformant cluster. A minimal deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: concierge }
spec:
  replicas: 3
  selector: { matchLabels: { app: concierge } }
  template:
    metadata: { labels: { app: concierge } }
    spec:
      containers:
      - name: concierge
        image: registry.example.com/locus-concierge:0.1.0
        ports: [{ containerPort: 8080 }]
        # locus-provider holds the provider credential for your tab
        envFrom:
        - secretRef: { name: locus-provider }
        env:
        - name: LOCUS_SERVER_API_KEY
          valueFrom: { secretKeyRef: { name: locus-secrets, key: server-api-key } }
        readinessProbe:
          httpGet: { path: /health, port: 8080 }
        resources:
          requests: { cpu: 500m, memory: 1Gi }
          limits:   { cpu: 2,    memory: 4Gi }
---
apiVersion: v1
kind: Service
metadata: { name: concierge }
spec:
  type: LoadBalancer
  selector: { app: concierge }
  ports: [{ port: 80, targetPort: 8080 }]
```

The `locus-provider` Secret carries the variable for your tab from the table
above (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `OLLAMA_BASE_URL`). For the
OCI tab on OKE, the pod authenticates with the node's instance principal or
with workload identity, so drop the `envFrom` entry and set
`OCI_COMPARTMENT_ID` as a plain `env` variable instead.

For SSE streaming, ensure your ingress / load balancer doesn't buffer the
response (`X-Accel-Buffering: no` on nginx, or the buffering-off
equivalent on your load balancer).

## Plain VM — full control

This option suits raw VM access, or running the agent alongside other local
services.

```bash
# The extra matches your tab: oci here, openai (also for Self-hosted), or anthropic
pip install "locus-sdk[oci,server]"

# Your own server.py — the one from the top of this page. It ends in
# `app = server.app`, which is the ASGI callable uvicorn loads below.
mkdir -p ~/concierge && cd ~/concierge
# ... put server.py here ...

# Launch under systemd
sudo tee /etc/systemd/system/concierge.service <<'EOF'
[Unit]
Description=Locus concierge agent
After=network.target

[Service]
Type=simple
User=app
WorkingDirectory=/home/app/concierge
EnvironmentFile=/etc/concierge.env
ExecStart=/home/app/.local/bin/uvicorn server:app --host 0.0.0.0 --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now concierge
```

Create `/etc/concierge.env` with permissions readable only by the service
account and define `LOCUS_SERVER_API_KEY` and the provider credential for
your tab there. For the OCI tab on an OCI VM, the instance principal replaces
a stored credential: define `OCI_COMPARTMENT_ID` there instead.

## Sessions — `thread_id` for chat UIs

When the underlying agent has a checkpointer, pass a `thread_id` in the
request body for cross-request continuity. Requests from the same browser
tab should carry the same `thread_id` so they share context. Omit it, and
each request starts fresh.

```http
POST /invoke
Content-Type: application/json

{"prompt": "What were we discussing?", "thread_id": "user-c42-support"}
```

When `api_key` is set, the authenticated principal is prefixed onto the
`thread_id` server-side, so threads are scoped to the caller that owns
the key. The server takes a single shared `api_key`, so this is
single-principal scoping — not per-tenant isolation.

## Observability

Wire `TelemetryHook` to your OTLP collector for traces and metrics.
Set the exporter target via the standard OpenTelemetry environment
variables before the agent starts:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
```

```python
from locus.hooks.builtin import TelemetryHook

agent = Agent(
    ...,
    hooks=[TelemetryHook(service_name="my-agent")],
)
```

OCI Application Performance Monitoring accepts OTLP, as do Datadog,
Honeycomb, Grafana Tempo, and other OTLP-compatible backends. See
[Observability](../concepts/observability.md).

## See also

- [Agent Server](../concepts/server.md) — the FastAPI wrapper in detail.
- [Conversation Management](../concepts/conversation-management.md) —
  how `thread_id` survives across requests and restarts.
- [Model providers](../concepts/models.md) — providers and configuration.
