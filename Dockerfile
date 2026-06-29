# syntax=docker/dockerfile:1.7
#
# Multi-stage image for a tulip AgentServer deployment.
#
# Base: Amazon Linux 2023 (EKS / Graviton-arm64). A venv is built in the builder
# and copied into a minimal runtime; non-root 10001, readOnlyRootFilesystem-safe.
#
# Build:
#   docker build -t tulip-agent:latest .
# Run (with a model-provider key and a bearer server key):
#   docker run -p 8080:8080 -e OPENAI_API_KEY=sk-... -e TULIP_SERVER_API_KEY=secret \
#     tulip-agent:latest
#
# The image is built around `tulip.server.AgentServer`. Replace
# `app:server.app` in the CMD with the import path of your own AgentServer.

ARG AL2023=public.ecr.aws/amazonlinux/amazonlinux:2023
ARG PYTHON=python3.12

# ─── builder ──────────────────────────────────────────────────────────
FROM ${AL2023} AS builder
ARG PYTHON
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/app/.venv/bin:$PATH"

# gcc + libpq-devel for native checkpointer drivers (psycopg) on arm64.
RUN dnf -y install --setopt=install_weak_deps=False \
        ${PYTHON} ${PYTHON}-pip ${PYTHON}-devel gcc libpq-devel \
    && dnf clean all && rm -rf /var/cache/dnf

WORKDIR /app
RUN ${PYTHON} -m venv /app/.venv

COPY pyproject.toml README.md LICENSE.txt ./
COPY src ./src

# tulip + the OpenAI provider + AgentServer + checkpoint backends.
# Add [telemetry,rag,anthropic] to taste.
RUN --mount=type=cache,target=/root/.cache/pip pip install ".[openai,server,checkpoints]"

# ─── runtime ──────────────────────────────────────────────────────────
FROM ${AL2023} AS runtime
ARG PYTHON
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# libpq for asyncpg/psycopg at runtime; ca-certificates for outbound TLS.
RUN dnf -y install --setopt=install_weak_deps=False \
        ${PYTHON} libpq ca-certificates shadow-utils \
    && groupadd -g 10001 tulip \
    && useradd -u 10001 -g 10001 -M -s /sbin/nologin tulip \
    && dnf -y remove shadow-utils \
    && dnf clean all && rm -rf /var/cache/dnf

WORKDIR /app
COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv

USER 10001
EXPOSE 8080

# Liveness — the AgentServer /health endpoint returns 200 unless the process
# is dead. Use a stdlib probe so the image needs no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/health').read()"]

# Replace `app:server.app` with the import path of your own AgentServer instance.
CMD ["uvicorn", "app:server.app", "--host", "0.0.0.0", "--port", "8080"]
