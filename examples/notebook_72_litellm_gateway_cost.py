# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""Notebook 72: Per-team cost tracking on the LiteLLM AI Gateway.

Follows notebook 71 (the gateway happy path) with the part enterprise
operators actually care about: **who spent what on which model**.
Issues virtual keys for two pretend teams, drives traffic on each,
then walks the spend surface — per-request rows, per-key rollups,
per-model rollups, and per-team filtering via metadata.

Run it::

    # 1. Bring the gateway + Postgres up (see notebook 71 for the
    #    provider env vars and master key setup).
    cd examples/litellm-gateway/
    docker compose up -d

    # 2. Wire this notebook at the gateway.
    export LITELLM_GATEWAY_URL="http://localhost:4000"
    export LITELLM_MASTER_KEY="<master key from docker-compose env>"

    python examples/notebook_72_litellm_gateway_cost.py

Without ``LITELLM_GATEWAY_URL`` and ``LITELLM_MASTER_KEY`` set the
notebook prints the wiring snippet and exits cleanly — same self-skip
pattern as notebook 71.

Difficulty: Beginner
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


_REQUIRED_ENV = (
    "LITELLM_GATEWAY_URL",
    "LITELLM_MASTER_KEY",
)


def _print_skip_banner(missing: list[str]) -> None:
    print("=" * 72)
    print(" LiteLLM AI Gateway not configured — skipping the cost demo.")
    print("=" * 72)
    print(
        f"\n Missing environment variables: {', '.join(missing)}\n\n"
        " Bring up the gateway (with the Postgres sidecar so /spend/* works):\n\n"
        "     cd examples/litellm-gateway/\n"
        "     export OCI_REGION=... OCI_USER=... OCI_FINGERPRINT=...\n"
        "     export OCI_TENANCY=... OCI_KEY_FILE=... OCI_COMPARTMENT_ID=...\n"
        '     export LITELLM_MASTER_KEY="sk-master-$(openssl rand -hex 16)"\n'
        '     export LITELLM_DB_PASSWORD="$(openssl rand -hex 16)"\n'
        "     docker compose up -d\n\n"
        " Then wire this notebook:\n\n"
        '     export LITELLM_GATEWAY_URL="http://localhost:4000"\n'
        '     export LITELLM_MASTER_KEY="$LITELLM_MASTER_KEY"\n\n'
        " Full how-to: docs/how-to/litellm-gateway.md\n"
    )


def _check_prerequisites() -> tuple[str, str]:
    missing = [v for v in _REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        _print_skip_banner(missing)
        sys.exit(0)
    return (
        os.environ["LITELLM_GATEWAY_URL"].rstrip("/"),
        os.environ["LITELLM_MASTER_KEY"],
    )


# ---------------------------------------------------------------------------
# Gateway helpers
# ---------------------------------------------------------------------------


def _admin(master_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {master_key}", "Content-Type": "application/json"}


def issue_virtual_key(
    url: str,
    master_key: str,
    *,
    team: str,
    models: list[str],
    max_budget_usd: float = 5.0,
) -> str:
    """Issue a per-team virtual key. Returns the raw token."""
    resp = httpx.post(
        f"{url}/key/generate",
        headers=_admin(master_key),
        json={
            "models": models,
            "max_budget": max_budget_usd,
            "duration": "1h",
            "metadata": {"team": team, "owner": "notebook-72", "run": uuid.uuid4().hex[:8]},
        },
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.json()["key"]


def chat(url: str, virtual_key: str, model_alias: str, prompt: str) -> dict[str, Any]:
    """One chat completion under a virtual key. Returns the parsed body."""
    resp = httpx.post(
        f"{url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {virtual_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_alias,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 30,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_spend_logs(
    url: str, master_key: str, *, virtual_key: str | None = None
) -> list[dict[str, Any]]:
    params = {"api_key": virtual_key} if virtual_key else {}
    resp = httpx.get(
        f"{url}/spend/logs",
        headers={"Authorization": f"Bearer {master_key}"},
        params=params,
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_spend_by_key(url: str, master_key: str) -> list[dict[str, Any]]:
    resp = httpx.get(
        f"{url}/global/spend/keys",
        headers={"Authorization": f"Bearer {master_key}"},
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_spend_by_model(url: str, master_key: str) -> list[dict[str, Any]]:
    resp = httpx.get(
        f"{url}/global/spend/models",
        headers={"Authorization": f"Bearer {master_key}"},
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Demo flow
# ---------------------------------------------------------------------------


def main() -> None:
    url, master_key = _check_prerequisites()

    print()
    print("=" * 72)
    print(" Per-team cost tracking on the LiteLLM AI Gateway")
    print("=" * 72)
    print(f" Gateway: {url}")
    print()

    # ----- Step 1: issue two virtual keys, one per pretend team -----------
    team_a_key = issue_virtual_key(url, master_key, team="team-alpha", models=["gpt-4o"])
    team_b_key = issue_virtual_key(
        url, master_key, team="team-beta", models=["gpt-4o", "claude-sonnet-4-6"]
    )
    print(" Virtual keys issued:")
    print(f"   team-alpha (gpt-4o only):  {team_a_key[:24]}...")
    print(f"   team-beta  (gpt-4o, claude): {team_b_key[:24]}...")
    print()

    # ----- Step 2: drive different traffic on each team --------------------
    print(" Driving traffic:")
    for prompt in ("Capital of France?", "Capital of Spain?", "Capital of Italy?"):
        out = chat(url, team_a_key, "gpt-4o", prompt)
        content = out["choices"][0]["message"]["content"].strip()
        toks = out["usage"]["total_tokens"]
        print(f"   [team-alpha] {prompt} → {content!r}  ({toks} tokens)")

    for prompt in ("Capital of Norway?", "Capital of Sweden?"):
        out = chat(url, team_b_key, "gpt-4o", prompt)
        content = out["choices"][0]["message"]["content"].strip()
        toks = out["usage"]["total_tokens"]
        print(f"   [team-beta]  {prompt} → {content!r}  ({toks} tokens)")

    # ----- Step 3: wait for the gateway's async spend flusher --------------
    print()
    print(" Waiting 15s for the gateway's async spend logger to flush ...")
    time.sleep(15)

    # ----- Step 4: walk the spend surface ---------------------------------
    print()
    print("=" * 72)
    print(" /spend/logs — per-request rows for team-alpha")
    print("=" * 72)
    for row in fetch_spend_logs(url, master_key, virtual_key=team_a_key):
        team = (row.get("metadata") or {}).get("team", "?")
        print(
            f"   model={row.get('model', '?'):<32} "
            f"team={team:<12} "
            f"tokens={row.get('total_tokens', 0):<4} "
            f"cost=${row.get('spend', 0):.6f}"
        )

    print()
    print("=" * 72)
    print(" /global/spend/keys — aggregate spend per virtual key")
    print("=" * 72)
    for k in fetch_spend_by_key(url, master_key)[:8]:
        masked = (k.get("api_key") or k.get("token") or "?")[:16] + "..."
        team = (k.get("metadata") or {}).get("team", "?")
        print(f"   key={masked:<22} team={team:<12} total_spend=${k.get('total_spend', 0):.6f}")

    print()
    print("=" * 72)
    print(" /global/spend/models — aggregate spend per upstream model")
    print("=" * 72)
    for m in fetch_spend_by_model(url, master_key)[:8]:
        print(f"   model={m.get('model', '?'):<40} total_spend=${m.get('total_spend', 0):.6f}")

    print()
    print("=" * 72)
    print(" Done. Finance / platform teams can answer:")
    print("   · 'What did team-alpha spend last month?'  → /spend/logs + metadata.team")
    print("   · 'What did Cohere Command cost across all teams?' → /global/spend/models")
    print("   · 'Who is over budget right now?' → /global/spend/keys + max_budget")
    print(" — all from one SQL-backed surface. No Tulip integration glue.")
    print("=" * 72)


if __name__ == "__main__":
    main()
