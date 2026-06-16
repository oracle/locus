# Locus workbench — single-image build of the three-tier playground.
#
# Layout (mirrors optic + webapp Dockerfiles in observai):
#   • Oracle Linux 9-slim base, microdnf for installs.
#   • Python 3.12 hosts the FastAPI runner (workbench/backend).
#   • Node 20 runs the Express BFF (workbench/bff) and Vite UI (workbench/web).
#   • All three tiers boot via one CMD; the container exposes 5173/3101/8100.
#
# Build (from repo root):
#   docker build -t locus-workbench -f workbench/Dockerfile .
#
# Run (BYO model key — paste it in the UI's Provider settings):
#   docker run --rm -p 5173:5173 -p 3101:3101 -p 8100:8100 locus-workbench
#   open http://localhost:5173

FROM container-registry.oracle.com/os/oraclelinux:9-slim

ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

WORKDIR /app

# Toolchain + Python 3.12 + Node 20.
RUN microdnf install -y \
        gcc gcc-c++ make \
        wget tar gzip xz findutils \
        openssl-devel bzip2-devel libffi-devel \
        zlib-devel readline-devel sqlite-devel \
        xz-devel ncurses-devel \
        python3.12 python3.12-pip \
    && microdnf module enable nodejs:20 -y \
    && microdnf install -y nodejs \
    && microdnf clean all

# Install locus + the runtime extras the workbench actually needs:
#   server  — fastapi + uvicorn (the runner is a FastAPI app)
#   oci     — OCI SDK for the GenAI / embeddings transports
#   openai  — openai SDK (used by the OpenAI provider + OCI v1 transport)
#   anthropic — anthropic SDK (Anthropic provider)
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
RUN python3.12 -m pip install --no-cache-dir ".[server,oci,openai,anthropic]"

# Workbench source. .dockerignore keeps node_modules / dist / .vite out
# of the COPY context so we install fresh inside the image.
COPY workbench/ /app/workbench/

# Notebook catalog. The runner walks `<repo>/examples/notebook_*.py`
# to populate the sidebar — without this COPY the catalog is empty
# and every Run hits a "no such notebook" error.
COPY examples/ /app/examples/

# Node dependency install for both BFF and web.
RUN cd workbench/bff && npm ci --no-audit --no-fund \
    && cd ../web   && npm ci --no-audit --no-fund

EXPOSE 5173 3101 8100

# Entrypoint runs all three tiers and blocks on `wait`. Container exits
# when any of the three dies, surfacing failures to `docker logs`.
# Hosts are bound to 0.0.0.0 so the published ports are reachable from
# outside the container.
CMD ["bash", "-c", "\
set -e; \
python3.12 -m uvicorn --app-dir workbench/backend runner:app --host 0.0.0.0 --port 8100 & \
( cd workbench/bff && npm run dev ) & \
( cd workbench/web && npx vite --host 0.0.0.0 --port 5173 ) & \
wait -n"]
