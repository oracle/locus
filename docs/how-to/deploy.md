# Deploy

`AgentServer` is a drop-in FastAPI wrapper. Deploys anywhere FastAPI
runs. This guide covers the four most common targets: OCI Functions,
OCI Container Instances, OKE / Kubernetes, and OCI Compute.

## The shape you ship

```python
# server.py
from locus.agent import Agent
from locus.server import AgentServer
from locus.memory.backends import OCIBucketBackend

agent = Agent(
    model="oci:openai.gpt-5.5",
    tools=[...],
    system_prompt="...",
    checkpointer=OCIBucketBackend(
        bucket_name="locus-threads",
        namespace="<your-namespace>",
    ),
)

server = AgentServer(
    agent=agent,
    title="Booking concierge",
    cors_origins=["https://app.example.com"],
)

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=8080)
```

You get out of the box:

- `POST /invoke` — synchronous run, full `AgentResult` JSON.
- `POST /stream` — Server-Sent Events of every typed event.
- `GET / DELETE /threads/{id}` — conversation persistence.
- `GET /health` — liveness probe.

## OCI Functions — serverless, scale to zero

Best for low-frequency or bursty traffic. Pay only when the function
runs.

```dockerfile
# Dockerfile
FROM fnproject/python:3.11-fdk
COPY requirements.txt /function/
RUN pip install -r /function/requirements.txt
COPY server.py /function/
CMD ["server.handler"]
```

```yaml
# func.yaml
schema_version: 20180708
name: locus-concierge
version: 0.1.0
runtime: python
build_image: fnproject/python:3.11-fdk-build
run_image:   fnproject/python:3.11-fdk
entrypoint: /python/bin/fdk /function/server.py handler
```

Deploy:

```bash
fn deploy --app concierge-app
```

Functions inherit OCI workload identity, so the agent
authenticates to OCI Generative AI without explicit credentials. Set
`OCI_AUTH_TYPE=resource_principal` in the function configuration.

## OCI Container Instances — managed, no cluster

Best when you want a long-running container without operating
Kubernetes.

```bash
# 1. Build and push
docker build -t \
  iad.ocir.io/$NAMESPACE/locus-concierge:0.1.0 .
docker push \
  iad.ocir.io/$NAMESPACE/locus-concierge:0.1.0

# 2. Create the container instance
oci container-instances container-instance create \
  --availability-domain "$AD" \
  --compartment-id "$COMPARTMENT" \
  --containers '[{
    "image-url": "iad.ocir.io/'$NAMESPACE'/locus-concierge:0.1.0",
    "display-name": "concierge",
    "command": ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
  }]' \
  --shape "CI.Standard.E4.Flex" \
  --shape-config '{"ocpus":1,"memoryInGBs":4}'
```

Container Instances also support `instance_principal` auth — the
running container can call OCI services without a stored API key.
Set `OCI_AUTH_TYPE=instance_principal` in the container env.

## OKE — Kubernetes for production

Best for multi-replica, autoscaled, multi-region production. The
container image is built from the [root `Dockerfile`](https://github.com/oracle-samples/locus/blob/main/Dockerfile) — multi-stage, non-root user, `HEALTHCHECK` on `/health`. Build it with:

```bash
docker build -t iad.ocir.io/$NAMESPACE/locus-concierge:0.1.0 .
docker push    iad.ocir.io/$NAMESPACE/locus-concierge:0.1.0
```

A minimal Kubernetes deployment looks like:

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
      serviceAccountName: locus-workload          # OCI workload identity
      containers:
      - name: concierge
        image: iad.ocir.io/$NAMESPACE/locus-concierge:0.1.0
        ports: [{ containerPort: 8080 }]
        env:
        - { name: OCI_AUTH_TYPE,    value: instance_principal }
        - { name: LOCUS_THREAD_BUCKET, value: locus-threads-prod }
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

For SSE streaming, ensure your ingress / load balancer doesn't
buffer the response (`X-Accel-Buffering: no` on nginx,
`response_buffering: off` equivalent on OCI Load Balancer).

## OCI Compute — full VM control

Best when you need raw VM access or run the agent alongside other
local services.

```bash
# On the compute instance:
pip install "locus-sdk[oci]"
git clone https://github.com/oracle-samples/locus.git ~/concierge
cd ~/concierge

# Launch under systemd
sudo tee /etc/systemd/system/concierge.service <<EOF
[Unit]
Description=Locus concierge agent
After=network.target

[Service]
Type=simple
User=opc
Environment=OCI_AUTH_TYPE=instance_principal
ExecStart=/home/opc/.local/bin/uvicorn server:app --host 0.0.0.0 --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now concierge
```

## Auth — one surface, four targets

| Deployment target | `OCI_AUTH_TYPE` |
|---|---|
| OCI Functions | `resource_principal` |
| OCI Container Instances | `instance_principal` |
| OKE | `instance_principal` (via workload identity) |
| OCI Compute | `instance_principal` |
| Laptop / CI | `api_key` or `session_token` |

Locus picks the right OCI signer from the `OCI_AUTH_TYPE` env var.
You don't change application code between environments.

## Sessions — `X-Session-ID` for chat UIs

When the underlying agent has a checkpointer, `AgentServer`
honours the `X-Session-ID` header (or `thread_id` in the body) for
cross-request continuity. Same browser tab → same thread → same
context. Drop the header, drop the thread.

```http
POST /invoke
X-Session-ID: user-c42-support
Content-Type: application/json

{"prompt": "What were we discussing?"}
```

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

OCI APM accepts OTLP. So do Honeycomb, Tempo, Grafana Cloud, and
every other backend that speaks the spec. See
[Observability](../concepts/observability.md).

## See also

- [Agent Server](../concepts/server.md) — the FastAPI wrapper in detail.
- [Conversation Management](../concepts/conversation-management.md) —
  how `thread_id` survives across requests and restarts.
- [OCI GenAI models](oci-models.md) — auth and transport selection.
