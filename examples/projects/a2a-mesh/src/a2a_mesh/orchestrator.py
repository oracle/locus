# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""CLI orchestrator — discovers A2A peers, picks one by skill, delegates.

Usage::

    python -m a2a_mesh.orchestrator "Is alert A-101 a true positive?"
    python -m a2a_mesh.orchestrator --stream "Enrich 198.51.100.7."
    python -m a2a_mesh.orchestrator --skill ioc_enrichment "Enrich this domain..."

The orchestrator does **not** itself wrap an agent — it's a pure A2A
client. The decision logic ("alert id → triage, indicator → threat-intel")
is deliberately rule-based so the demo stays inspectable; in real life
you'd put a small Tulip ``Orchestrator`` here whose specialists are
``A2AClient.as_tool()``-wrapped peers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys

import httpx

from tulip.a2a import A2AClient


DEFAULT_PEERS = (
    "http://127.0.0.1:8001",  # threat-intel
    "http://127.0.0.1:8002",  # soc-triage
)

ALERT_RE = re.compile(r"\bA-\d+\b", re.IGNORECASE)


async def discover(urls: tuple[str, ...]) -> list[tuple[str, list[str]]]:
    """Hit /agent-card on each url; return [(url, skills)]."""
    out: list[tuple[str, list[str]]] = []
    for url in urls:
        try:
            card = await A2AClient(url).get_agent_card()
        except (httpx.ConnectError, httpx.HTTPError) as exc:
            print(f"  ✗ {url} unreachable: {exc}", file=sys.stderr)
            continue
        print(f"  ✓ {url} → {card.name} skills={card.skills}")
        out.append((url, list(card.skills)))
    return out


def pick(query: str, peers: list[tuple[str, list[str]]], force: str | None) -> str:
    """Pick a peer URL based on a skill match against the query."""
    wanted: str
    if force:
        wanted = force
    elif ALERT_RE.search(query) or any(
        w in query.lower()
        for w in (
            "alert",
            "triage",
            "severity",
            "escalate",
            "true positive",
            "false positive",
            "incident",
        )
    ):
        wanted = "alert_triage"
    else:
        wanted = "threat_intel"

    for url, skills in peers:
        if wanted in skills:
            print(f"  → routing to {url} (matched skill: {wanted})")
            return url

    msg = f"No peer advertises skill {wanted!r}; available: {peers}"
    raise SystemExit(msg)


async def run_invoke(url: str, prompt: str) -> str:
    return await A2AClient(url).invoke(prompt)


async def run_stream(url: str, prompt: str) -> None:
    """Stream events from /a2a/stream (raw httpx — A2AClient has no stream method)."""
    body = {"messages": [{"role": "user", "content": prompt, "metadata": {}}], "metadata": {}}
    async with (
        httpx.AsyncClient(timeout=120.0) as http,
        http.stream("POST", f"{url}/a2a/stream", json=body) as resp,
    ):
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if not payload:
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                print(payload)
                continue
            kind = event.get("type", "?")
            if kind == "ToolStartEvent":
                print(f"  🔧 {event.get('tool_name')}")
            elif kind == "TerminateEvent":
                print(f"  ✅ {event.get('final_message', '')[:200]}")
            else:
                print(f"  · {kind}")


async def main_async(args: argparse.Namespace) -> int:
    print("Discovering peers…")
    peers = await discover(tuple(args.peer))
    if not peers:
        print("No peers reachable. Did you `make intel` and `make triage`?")
        return 2

    target = pick(args.query, peers, args.skill)

    if args.stream:
        print(f"\nStreaming from {target}…")
        await run_stream(target, args.query)
    else:
        print(f"\nInvoking {target}…")
        reply = await run_invoke(target, args.query)
        print(f"\n  ← {reply}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query", help="prompt to send to the matched agent")
    p.add_argument(
        "--peer",
        action="append",
        default=list(DEFAULT_PEERS),
        help="repeat to add A2A peer URLs (default: threat-intel:8001 + soc-triage:8002)",
    )
    p.add_argument("--skill", help="force route to a specific skill tag")
    p.add_argument("--stream", action="store_true", help="stream via /a2a/stream (SSE)")
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
