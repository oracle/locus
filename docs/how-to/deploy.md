# Deploy

`AgentServer` is a drop-in FastAPI wrapper. It deploys anywhere FastAPI
runs. This guide covers the most common targets: a container, Kubernetes,
serverless, and a plain VM. In every case the agent authenticates to its
model provider with an API key supplied via an environment variable — no
cloud-specific identity wiring required.

## The shape you ship

```python
# server.py
from tulip.agent import Agent
from tulip.server import AgentServer
from tulip.memory.backends import S3Backend

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[...],
    system_prompt="...",
    checkpointer=S3Backend(bucket="tulip-threads"),  # or RedisBackend(...)
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

Provider keys are read from the environment (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`). Inject them as secrets, never bake them into the
image.

## Container — the universal target

The repo ships a multi-stage [`Dockerfile`](https://github.com/tuliplabs-ai/sdk-python/blob/main/Dockerfile)
(non-root user, `HEALTHCHECK` on `/health`). Build, push to any registry,
and run anywhere that runs containers:

```bash
docker build -t registry.example.com/tulip-concierge:0.1.0 .
docker push    registry.example.com/tulip-concierge:0.1.0

docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  registry.example.com/tulip-concierge:0.1.0
```

This single image drops straight into Cloud Run, ECS / Fargate, Fly.io,
Azure Container Apps, or any other container host.

## Serverless — scale to zero

Best for low-frequency or bursty traffic. Wrap the FastAPI app in an
adapter for your platform — [Mangum](https://mangum.io/) for
AWS Lambda, or deploy the container image directly to a
scale-to-zero container runtime (Cloud Run, Container Apps). `Agent` is
constructed lazily, so cold starts stay cheap. Set the provider key as a
function secret.

## Kubernetes — for production

Best for multi-replica, autoscaled, multi-region production. A minimal
deployment:

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
        image: registry.example.com/tulip-concierge:0.1.0
        ports: [{ containerPort: 8080 }]
        env:
        - name: OPENAI_API_KEY
          valueFrom: { secretKeyRef: { name: tulip-secrets, key: openai-api-key } }
        - { name: TULIP_THREAD_BUCKET, value: tulip-threads-prod }
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

For SSE streaming, ensure your ingress / load balancer doesn't buffer the
response (`X-Accel-Buffering: no` on nginx, or the buffering-off
equivalent on your load balancer).

## Plain VM — full control

Best when you need raw VM access or run the agent alongside other local
services.

```bash
pip install "tulip-agents[openai,server]"
git clone https://github.com/tuliplabs-ai/sdk-python.git ~/concierge
cd ~/concierge

# Launch under systemd
sudo tee /etc/systemd/system/concierge.service <<'EOF'
[Unit]
Description=Tulip concierge agent
After=network.target

[Service]
Type=simple
User=app
Environment=OPENAI_API_KEY=sk-...
ExecStart=/home/app/.local/bin/uvicorn server:app --host 0.0.0.0 --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now concierge
```

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
from tulip.hooks.builtin import TelemetryHook

agent = Agent(
    ...,
    hooks=[TelemetryHook(service_name="my-agent")],
)
```

Datadog accepts OTLP. So do Honeycomb, Tempo, Grafana Cloud, and
every other backend that speaks the spec. See
[Observability](../concepts/observability.md).

## See also

- [Agent Server](../concepts/server.md) — the FastAPI wrapper in detail.
- [Conversation Management](../concepts/conversation-management.md) —
  how `thread_id` survives across requests and restarts.
- [Model providers](../concepts/models.md) — providers and configuration.
