# syntax=docker/dockerfile:1.7
#
# The Tulip box runner: one agent's loop, inside an NVIDIA OpenShell sandbox.
#
# A Tulip gateway starts this image as a sandbox's command for one run. The
# runner reads TULIP_ADMIT_URL, TULIP_ADMIT_TOKEN and TULIP_RUN_ID from its
# environment, asks the gateway what to do, and runs the agent the run's
# manifest describes (see docs/runner.md).
#
# Built by the release workflow from tulip-runner.pyz (scripts/build_runner_pyz.sh):
#   docker build -f runner.Dockerfile --build-arg PYZ=dist-runner/tulip-runner.pyz .
#
# Any image with CPython 3.12 can run the same .pyz instead:
#   python3 tulip-runner.pyz
#
# Non-root (uid 10001), no capabilities needed. The workspace is /sandbox, the
# sandbox's persistent volume.
ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE}
ARG PYZ=dist-runner/tulip-runner.pyz

# git and ripgrep: what the harness tools reach for in a workspace.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ripgrep ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --home-dir /home/runner --shell /bin/bash runner \
    && mkdir -p /sandbox /opt/tulip \
    && chown runner:runner /sandbox

COPY --chmod=0644 ${PYZ} /opt/tulip/tulip-runner.pyz

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SHIV_ROOT=/home/runner/.shiv

USER 10001
WORKDIR /sandbox
ENTRYPOINT ["python3", "/opt/tulip/tulip-runner.pyz"]
