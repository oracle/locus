# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""CLI orchestrator — discovers A2A peers, picks one by skill, delegates.

Usage::

    python -m a2a_mesh.orchestrator "Is alert A-101 a true positive?"
    python -m a2a_mesh.orchestrator --stream "Enrich 198.51.100.7."
    python -m a2a_mesh.orchestrator --skill ioc_enrichment "Enrich this domain..."
    python -m a2a_mesh.orchestrator --legacy --stream "Enrich 198.51.100.7."

The orchestrator does **not** itself wrap an agent — it's a pure A2A
client. The decision logic ("alert id → triage, indicator → threat-intel")
is deliberately rule-based so the demo stays inspectable; in real life
you'd put a small Locus ``Orchestrator`` here whose specialists are
``A2AClient.as_tool()``-wrapped peers.

It speaks A2A v1.0 JSON-RPC by default: ``SendMessage`` for a one-shot
answer and ``SendStreamingMessage`` for ``--stream``, reading the reply
from the task's artifacts. ``--legacy`` uses the pre-v1.0 flat endpoints
(``/a2a/invoke``, ``/a2a/stream``) instead, which the peers still serve,
so the demo also shows backwards compatibility.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from typing import Any

import httpx

from locus.a2a import A2AClient, AgentCard, Message, Task, TextPart


DEFAULT_PEERS = (
    "http://127.0.0.1:8001",  # threat-intel
    "http://127.0.0.1:8002",  # soc-triage
)

ALERT_RE = re.compile(r"\bA-\d+\b", re.IGNORECASE)


def skill_ids(card: AgentCard) -> list[str]:
    """The skill ids a card advertises.

    A v1.0 card lists skills as objects; routing matches on their ``id``.
    """
    return [skill.id for skill in card.skills]


async def discover(urls: tuple[str, ...]) -> list[tuple[str, list[str]]]:
    """Fetch the Agent Card of each url; return [(url, skill ids)]."""
    out: list[tuple[str, list[str]]] = []
    for url in urls:
        try:
            card = await A2AClient(url).get_agent_card()
        except (httpx.HTTPError, RuntimeError) as exc:
            print(f"  ✗ {url} unreachable: {exc}", file=sys.stderr)
            continue
        skills = skill_ids(card)
        interfaces = ", ".join(
            f"{i.protocolBinding} v{i.protocolVersion}" for i in card.supportedInterfaces
        )
        print(f"  ✓ {url} → {card.name} skills={skills}")
        print(f"      A2A v{card.protocolVersion} · interfaces: {interfaces or 'none advertised'}")
        out.append((url, skills))
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


def user_message(prompt: str) -> Message:
    """A user message carrying ``prompt`` as its single text part."""
    return Message(role="user", parts=[TextPart(text=prompt)], messageId=uuid.uuid4().hex)


def _parts_text(parts: list[Any]) -> str:
    """The text of a list of parts, whether models or wire dicts."""
    texts = [p.get("text") if isinstance(p, dict) else getattr(p, "text", None) for p in parts]
    return "".join(t for t in texts if t)


def reply_text(task: Task) -> str:
    """The agent's reply: the text of the task's last artifact."""
    if not task.artifacts:
        return f"(no reply; task is {task.status.state.value})"
    return _parts_text(list(task.artifacts[-1].parts))


async def run_send(url: str, prompt: str) -> str:
    """A2A v1.0 ``SendMessage``: one round trip, the reply from the task's artifacts."""
    task = await A2AClient(url).send_message(user_message(prompt))
    return reply_text(task)


async def run_stream(url: str, prompt: str) -> None:
    """A2A v1.0 ``SendStreamingMessage``: render task lifecycle events as they arrive."""
    async for event in A2AClient(url).send_message_streaming(user_message(prompt)):
        if "error" in event:
            error = event["error"]
            print(f"  ✗ {error.get('code')}: {error.get('message')}")
            return
        kind = event.get("kind", "?")
        if kind == "task":
            print(f"  · task {event.get('id')} {event.get('status', {}).get('state')}")
        elif kind == "status-update":
            status = event.get("status", {})
            note = _parts_text((status.get("message") or {}).get("parts", []))
            print(f"  · {status.get('state')}" + (f": {note[:200]}" if note else ""))
        elif kind == "artifact-update":
            text = _parts_text(event.get("artifact", {}).get("parts", []))
            print(f"  ✅ {text[:200]}")
        else:
            print(f"  · {kind}")


async def run_legacy_invoke(url: str, prompt: str) -> str:
    """Pre-v1.0 flat ``/a2a/invoke``, kept to show backwards compatibility."""
    return await A2AClient(url).invoke(prompt)


async def run_legacy_stream(url: str, prompt: str) -> None:
    """Pre-v1.0 ``/a2a/stream``: the agent's events over raw SSE.

    Each event is ``{"type": "text" | "done" | "error" | <event type>, ...}``;
    the stream ends with ``[DONE]``.
    """
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
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                print(payload)
                continue
            kind = event.get("type", "?")
            if kind == "text":
                print(f"  · thinking: {event.get('content', '')[:200]}")
            elif kind == "done":
                print(f"  ✅ {event.get('content', '')[:200]}")
            elif kind == "error":
                print(f"  ✗ {event.get('error')} (correlation id {event.get('correlation_id')})")
            else:
                print(f"  · {kind}")


async def main_async(args: argparse.Namespace) -> int:
    print("Discovering peers…")
    peers = await discover(tuple(args.peer))
    if not peers:
        print("No peers reachable. Did you `make intel` and `make triage`?")
        return 2

    target = pick(args.query, peers, args.skill)

    if args.legacy and args.stream:
        print(f"\nStreaming from {target} (legacy /a2a/stream)…")
        await run_legacy_stream(target, args.query)
    elif args.legacy:
        print(f"\nInvoking {target} (legacy /a2a/invoke)…")
        print(f"\n  ← {await run_legacy_invoke(target, args.query)}")
    elif args.stream:
        print(f"\nStreaming from {target} (SendStreamingMessage)…")
        await run_stream(target, args.query)
    else:
        print(f"\nSending to {target} (SendMessage)…")
        print(f"\n  ← {await run_send(target, args.query)}")
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
    p.add_argument(
        "--stream", action="store_true", help="stream task events (SendStreamingMessage, SSE)"
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help="use the pre-v1.0 flat endpoints (/a2a/invoke, /a2a/stream)",
    )
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
