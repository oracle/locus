# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 71: Routing SOC agent traffic through the LiteLLM AI Gateway.

This notebook is the runnable companion to
``docs/how-to/litellm-gateway.md``. It does **not** add a new Tulip
model class — it uses Tulip's existing :class:`OpenAIModel` pointed at
a LiteLLM AI Gateway URL. For an always-on SOC fleet this is a security
posture win, not just plumbing: the gateway fronts your upstream
providers (OpenAI, Anthropic, and any other LiteLLM-supported backend),
so triage agents never hold provider credentials and every model call
crosses one auditable choke point.

Key concepts:

- The LiteLLM Proxy Server (a.k.a. **LiteLLM AI Gateway**) is the
  product. The Python ``litellm.acompletion()`` function is internal
  scaffolding. The gateway is what carries the platform-grade pieces:
  virtual keys, per-team budgets, fallback chains, centralised
  observability, cost tracking, caching, guardrails.
- Tulip consumes the gateway through ``OpenAIModel(base_url=...)``.
  No new Tulip class is needed; the gateway is OpenAI-shaped by design.
- The gateway holds the upstream provider credentials. Tulip only holds
  the gateway-issued virtual key, so provider API keys never land on
  the SOC agent host at all — one less secret to leak when an agent
  box is compromised.

Run it::

    # 1. Start the gateway. The sample ships at examples/litellm-gateway/.
    cd examples/litellm-gateway/
    export OPENAI_API_KEY="sk-..."          # upstream provider key
    export ANTHROPIC_API_KEY="sk-ant-..."   # optional second provider
    export LITELLM_MASTER_KEY="$(openssl rand -hex 32)"
    docker compose up -d

    # 2. Point this notebook at the gateway:
    export LITELLM_GATEWAY_URL="http://localhost:4000"
    export LITELLM_GATEWAY_KEY="$LITELLM_MASTER_KEY"   # or any virtual key issued by /key/generate
    export LITELLM_GATEWAY_MODEL="gpt-4o"  # alias from config.yaml

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
    print(" LiteLLM AI Gateway not configured — skipping the live SOC demo.")
    print("=" * 72)
    print(
        f"\n Missing environment variables: {', '.join(missing)}\n\n"
        " This notebook expects a LiteLLM AI Gateway running in front of\n"
        " your upstream providers. Start the sample gateway in another\n"
        " terminal:\n\n"
        "     cd examples/litellm-gateway/\n"
        '     export OPENAI_API_KEY="sk-..."\n'
        '     export ANTHROPIC_API_KEY="sk-ant-..."\n'
        '     export LITELLM_MASTER_KEY="$(openssl rand -hex 32)"\n'
        "     docker compose up -d\n\n"
        " Then export the gateway URL and key in this shell:\n\n"
        '     export LITELLM_GATEWAY_URL="http://localhost:4000"\n'
        '     export LITELLM_GATEWAY_KEY="$LITELLM_MASTER_KEY"\n'
        '     export LITELLM_GATEWAY_MODEL="gpt-4o"\n\n'
        " Full how-to: docs/how-to/litellm-gateway.md\n"
    )


def _check_prerequisites() -> tuple[str, str, str]:
    missing = [v for v in _REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        _print_skip_banner(missing)
        sys.exit(0)

    url = os.environ["LITELLM_GATEWAY_URL"].rstrip("/")
    key = os.environ["LITELLM_GATEWAY_KEY"]
    # Default to a gpt-4o alias because the sample config.yaml ships it.
    model = os.environ.get("LITELLM_GATEWAY_MODEL", "gpt-4o")
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
# Part 2 — A SOC triage agent talking to the gateway via OpenAIModel
# ---------------------------------------------------------------------------


async def _run_agent(url: str, key: str, model_alias: str) -> None:
    """Build a Tulip triage agent pointed at the gateway and run basic prompts."""
    from tulip.agent import Agent
    from tulip.models.native.openai import OpenAIModel

    print(f"=== Triage agent vs. {model_alias} (via gateway) ===\n")

    # The OPENAI-COMPATIBLE client. ``base_url`` is the gateway endpoint;
    # ``api_key`` is a gateway-issued virtual key. Tulip carries NO provider
    # credentials here — the gateway holds the upstream provider keys.
    model = OpenAIModel(
        model=model_alias,
        api_key=key,
        base_url=url,
    )

    agent = Agent(model=model, system_prompt="You are a concise SOC triage assistant.")

    prompts = [
        "In incident response, what does IOC stand for? One sentence.",
        "A user reports a phishing email. Name the first triage step. One sentence.",
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
    from tulip.agent import Agent
    from tulip.core.events import ModelChunkEvent
    from tulip.models.native.openai import OpenAIModel

    print(f"=== Streaming vs. {model_alias} (via gateway) ===\n")

    model = OpenAIModel(model=model_alias, api_key=key, base_url=url)
    agent = Agent(model=model, system_prompt="Reply concisely.")

    print(
        "  > List three common phishing red flags, comma-separated.\n  < ",
        end="",
        flush=True,
    )
    full: list[str] = []
    async for ev in agent.run("List three common phishing red flags, comma-separated."):
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
    print(" Done. The gateway handled provider auth, vendor adaptation, and any")
    print(" configured fallback / cache / callbacks transparently. The SOC agent")
    print(" saw only an OpenAI-shaped HTTP contract — and no upstream provider")
    print(" key ever touched the agent host.")
    print("=" * 72)


if __name__ == "__main__":
    main()
