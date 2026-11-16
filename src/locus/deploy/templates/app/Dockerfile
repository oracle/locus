# syntax=docker/dockerfile:1.7
#
# The agent server image `locus deploy` builds for Container Instances and OKE.
# Oracle Linux 9, Python 3.12, a venv built in one stage and copied into a
# slim runtime; non-root user 10001; works with a read-only root filesystem.
#
# Build context is this directory; .dockerignore admits app/ alone. app/ holds
# requirements.txt, the module that defines the ASGI app (__LOCUS_APP__) and,
# after `locus deploy init --sdk-wheel`, wheels/ with a locus-sdk build that is
# installed before requirements.txt.

ARG BASE_IMAGE=container-registry.oracle.com/os/oraclelinux:9-slim

# ─── builder ──────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN microdnf -y install python3.12 python3.12-pip \
    && microdnf clean all

WORKDIR /app
COPY app/ /app/
RUN python3.12 -m venv /opt/venv \
    && if ls /app/wheels/*.whl >/dev/null 2>&1; then /opt/venv/bin/pip install /app/wheels/*.whl; fi \
    && /opt/venv/bin/pip install -r requirements.txt

# ─── runtime ──────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/tmp

RUN microdnf -y install python3.12 ca-certificates shadow-utils \
    && groupadd -g 10001 locus \
    && useradd -u 10001 -g 10001 -M -s /sbin/nologin locus \
    && microdnf -y remove shadow-utils \
    && microdnf clean all

WORKDIR /app
COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv
COPY --from=builder --chown=10001:10001 /app /app

USER 10001
EXPOSE __LOCUS_PORT__

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:__LOCUS_PORT__/health').read()"]

# locus.deploy.entrypoint resolves <VAR>_SECRET_OCID variables from OCI Vault
# with the instance or resource principal, then starts uvicorn.
CMD ["python", "-m", "locus.deploy.entrypoint", "uvicorn", "__LOCUS_APP__", "--host", "0.0.0.0", "--port", "__LOCUS_PORT__"]
