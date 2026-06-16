# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""CLI orchestrator — discovers A2A peers, picks one by skill, delegates.

Usage::

    python -m a2a_mesh.orchestrator "Should I buy TSLA?"
    python -m a2a_mesh.orchestrator --stream "Summarise quantum computing."
    python -m a2a_mesh.orchestrator --skill summarize "Summarise this..."

The orchestrator does **not** itself wrap an agent — it's a pure A2A
client. The decision logic ("ticker → finance, summary → research") is
deliberately rule-based so the demo stays inspectable; in real life
you'd put a small Locus ``Orchestrator`` here whose specialists are
``A2AClient.as_tool()``-wrapped peers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys

import httpx

from locus.a2a import A2AClient


DEFAULT_PEERS = (
    "http://127.0.0.1:8001",  # research
    "http://127.0.0.1:8002",  # finance
)

TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")


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
    elif TICKER_RE.search(query) or any(
        w in query.lower() for w in ("buy", "sell", "valuation", "price", "stock")
    ):
        wanted = "valuation"
    else:
        wanted = "research"

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
        print("No peers reachable. Did you `make research` and `make finance`?")
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
        help="repeat to add A2A peer URLs (default: research:8001 + finance:8002)",
    )
    p.add_argument("--skill", help="force route to a specific skill tag")
    p.add_argument("--stream", action="store_true", help="stream via /a2a/stream (SSE)")
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
