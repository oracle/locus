# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 71: Run a Locus agent behind the LiteLLM AI Gateway.

This notebook is the runnable companion to
``docs/how-to/litellm-gateway.md``. It does **not** add a new Locus
model class — it uses Locus's existing :class:`OpenAIModel` pointed at
a LiteLLM AI Gateway URL. The gateway is what fronts Oracle Generative
AI Infrastructure (and optionally other providers); Locus only ever
sees the OpenAI-shaped HTTP contract the gateway exposes.

Key concepts:

- The LiteLLM Proxy Server (a.k.a. **LiteLLM AI Gateway**) is the
  product. The Python ``litellm.acompletion()`` function is internal
  scaffolding. The gateway is what carries the platform-grade pieces:
  virtual keys, per-team budgets, fallback chains, centralised
  observability, cost tracking, caching, guardrails.
- Locus consumes the gateway through ``OpenAIModel(base_url=...)``.
  No new Locus class is needed; the gateway is OpenAI-shaped by design.
- The gateway holds the OCI signing credentials. Locus only holds the
  gateway-issued virtual key. On OKE, the gateway pod can use Workload
  Identity so the OCI signing key never lands on disk at all.

Run it::

    # 1. Start the gateway. The sample ships at examples/litellm-gateway/.
    cd examples/litellm-gateway/
    export OCI_REGION="us-chicago-1"
    export OCI_USER="ocid1.user.oc1..xxx"
    export OCI_FINGERPRINT="aa:bb:cc:..."
    export OCI_TENANCY="ocid1.tenancy.oc1..xxx"
    export OCI_KEY_FILE="$HOME/.oci/keys/your_api_key.pem"
    export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..xxx"
    export LITELLM_MASTER_KEY="$(openssl rand -hex 32)"
    docker compose up -d

    # 2. Point this notebook at the gateway:
    export LITELLM_GATEWAY_URL="http://localhost:4000"
    export LITELLM_GATEWAY_KEY="$LITELLM_MASTER_KEY"   # or any virtual key issued by /key/generate
    export LITELLM_GATEWAY_MODEL="oci-cohere-command"  # alias from config.yaml

    python examples/notebook_71_litellm_gateway.py

Without ``LITELLM_GATEWAY_URL`` and ``LITELLM_GATEWAY_KEY`` set, the
notebook prints the wiring snippet and exits cleanly — no traceback,
no half-initialised state.

Difficulty: Beginner
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


_REQUIRED_ENV = (
    "LITELLM_GATEWAY_URL",
    "LITELLM_GATEWAY_KEY",
)
_OPTIONAL_ENV = ("LITELLM_GATEWAY_MODEL",)


def _print_skip_banner(missing: list[str]) -> None:
    print("=" * 72)
    print(" LiteLLM AI Gateway not configured — skipping the live demo.")
    print("=" * 72)
    print(
        f"\n Missing environment variables: {', '.join(missing)}\n\n"
        " This notebook expects a LiteLLM AI Gateway running in front of OCI\n"
        " Generative AI. Start the sample gateway in another terminal:\n\n"
        "     cd examples/litellm-gateway/\n"
        '     export OCI_REGION="us-chicago-1"\n'
        '     export OCI_USER="ocid1.user.oc1..xxx"\n'
        '     export OCI_FINGERPRINT="aa:bb:cc:..."\n'
        '     export OCI_TENANCY="ocid1.tenancy.oc1..xxx"\n'
        '     export OCI_KEY_FILE="$HOME/.oci/keys/your_api_key.pem"\n'
        '     export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..xxx"\n'
        '     export LITELLM_MASTER_KEY="$(openssl rand -hex 32)"\n'
        "     docker compose up -d\n\n"
        " Then export the gateway URL and key in this shell:\n\n"
        '     export LITELLM_GATEWAY_URL="http://localhost:4000"\n'
        '     export LITELLM_GATEWAY_KEY="$LITELLM_MASTER_KEY"\n'
        '     export LITELLM_GATEWAY_MODEL="oci-cohere-command"\n\n'
        " Full how-to: docs/how-to/litellm-gateway.md\n"
    )


def _check_prerequisites() -> tuple[str, str, str]:
    missing = [v for v in _REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        _print_skip_banner(missing)
        sys.exit(0)

    url = os.environ["LITELLM_GATEWAY_URL"].rstrip("/")
    key = os.environ["LITELLM_GATEWAY_KEY"]
    # Default to a Cohere alias because the sample config.yaml ships it.
    model = os.environ.get("LITELLM_GATEWAY_MODEL", "oci-cohere-command")
    return url, key, model


# ---------------------------------------------------------------------------
# Part 1 — Health check against the gateway
# ---------------------------------------------------------------------------


def _print_gateway_health(url: str, key: str) -> None:
    """Print the gateway's reachable models. Surfaces config drift early."""
    import httpx

    print("=== Gateway health ===\n")
    try:
        resp = httpx.get(
            f"{url}/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5.0,
        )
        resp.raise_for_status()
    except httpx.RequestError as exc:
        print(f"  could not reach {url}/v1/models: {exc}")
        print("  (is the gateway running? — see docker compose logs)")
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        print(f"  {url}/v1/models returned {exc.response.status_code}")
        print(f"  body: {exc.response.text[:200]}")
        sys.exit(1)

    payload = resp.json()
    aliases = [m["id"] for m in payload.get("data", [])]
    print(f"  {url} is up. {len(aliases)} model alias(es) reachable:")
    for a in aliases[:10]:
        print(f"    · {a}")
    if len(aliases) > 10:
        print(f"    … and {len(aliases) - 10} more")
    print()


# ---------------------------------------------------------------------------
# Part 2 — A Locus agent talking to the gateway via OpenAIModel
# ---------------------------------------------------------------------------


async def _run_agent(url: str, key: str, model_alias: str) -> None:
    """Build a Locus Agent pointed at the gateway and run a basic prompt."""
    from locus.agent import Agent
    from locus.models.native.openai import OpenAIModel

    print(f"=== Agent vs. {model_alias} (via gateway) ===\n")

    # The OPENAI-COMPATIBLE client. ``base_url`` is the gateway endpoint;
    # ``api_key`` is a gateway-issued virtual key. Locus carries NO OCI
    # credentials here — the gateway does the OCI signing internally.
    model = OpenAIModel(
        model=model_alias,
        api_key=key,
        base_url=url,
    )

    agent = Agent(model=model, system_prompt="You are concise.")

    prompts = [
        "What is the capital of Japan? One word.",
        "What is 7 times 8?",
    ]
    for prompt in prompts:
        print(f"  > {prompt}")
        result = await asyncio.to_thread(agent.run_sync, prompt)
        print(f"  < {result.message.strip()}")
        print(f"    [{result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]")
        print()


# ---------------------------------------------------------------------------
# Part 3 — Streaming through the gateway
# ---------------------------------------------------------------------------


async def _run_streaming(url: str, key: str, model_alias: str) -> None:
    from locus.agent import Agent
    from locus.core.events import ModelChunkEvent
    from locus.models.native.openai import OpenAIModel

    print(f"=== Streaming vs. {model_alias} (via gateway) ===\n")

    model = OpenAIModel(model=model_alias, api_key=key, base_url=url)
    agent = Agent(model=model, system_prompt="Reply concisely.")

    print("  > List three primary colors, comma-separated.\n  < ", end="", flush=True)
    full: list[str] = []
    async for ev in agent.run("List three primary colors, comma-separated."):
        if isinstance(ev, ModelChunkEvent) and ev.content:
            print(ev.content, end="", flush=True)
            full.append(ev.content)
    print(f"\n    [{len(full)} streamed chunks]\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    url, key, model_alias = _check_prerequisites()

    print(
        f"\n LiteLLM AI Gateway at {url}\n"
        f" Model alias       {model_alias}\n"
        f" Auth              Bearer <virtual-key from $LITELLM_GATEWAY_KEY>\n"
    )

    _print_gateway_health(url, key)
    asyncio.run(_run_agent(url, key, model_alias))
    asyncio.run(_run_streaming(url, key, model_alias))

    print("=" * 72)
    print(" Done. The gateway handled OCI signing, vendor adaptation, and any")
    print(" configured fallback / cache / callbacks transparently. Locus saw")
    print(" only an OpenAI-shaped HTTP contract.")
    print("=" * 72)


if __name__ == "__main__":
    main()
