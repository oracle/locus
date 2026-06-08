# Multi-stage Dockerfile for a tulip AgentServer deployment.
#
# Build:
#     docker build -t tulip-agent:latest .
#
# Run (with a model-provider API key and a bearer-token server key):
#     docker run -p 8080:8080 \
#       -e OPENAI_API_KEY=sk-... \
#       -e TULIP_SERVER_API_KEY=secret \
#       tulip-agent:latest
#
# The image is built around `tulip.server.AgentServer`. Replace
# `your_app:server.app` in the CMD with the import path of your own
# AgentServer instance.

# -----------------------------------------------------------------------------
# Stage 1 — builder. Resolves wheels for tulip[server,checkpoints,openai].
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build deps only — gcc for native checkpointer drivers (psycopg, etc.).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy minimal package metadata first to maximize layer caching.
COPY pyproject.toml README.md LICENSE.txt ./
COPY src/ ./src/

# Install tulip + the OpenAI provider + the AgentServer extras + the
# checkpoint backends. Add ``[telemetry,rag,anthropic]`` to taste.
RUN pip install --user --no-cache-dir ".[openai,server,checkpoints]"

# -----------------------------------------------------------------------------
# Stage 2 — runtime. Slim image; non-root user.
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/tulip/.local/bin:$PATH

# Runtime deps only (libpq for asyncpg). curl is used by HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN useradd --create-home --shell /bin/bash --uid 10001 tulip
USER tulip
WORKDIR /home/tulip

# Copy the wheels installed in stage 1.
COPY --from=builder --chown=tulip:tulip /root/.local /home/tulip/.local

# Default port — override with `-p` or in your orchestrator's manifest.
EXPOSE 8080

# Liveness check — the AgentServer's /health endpoint always returns
# 200 unless the process is dead. Override with a readiness probe at
# the orchestrator layer if you want richer signals.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent http://localhost:8080/health || exit 1

# Replace ``your_app:server.app`` with the import path of your own
# AgentServer instance. The placeholder example below assumes a module
# called ``app.py`` at /home/tulip/app.py exposing a ``server`` symbol.
CMD ["uvicorn", "app:server.app", "--host", "0.0.0.0", "--port", "8080"]
