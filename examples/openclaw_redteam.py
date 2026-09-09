#!/usr/bin/env python
# Copyright 2026 The Tulip Authors — local security assessment harness.
"""Red-team the local OpenClaw agent with the Tulip Agents SDK.

Bridges Tulip's ``Target`` contract to a locally-running OpenClaw gateway by
shelling out to OpenClaw's own ``gateway call agent`` CLI. The CLI owns the
device-auth WebSocket handshake and waits for the final assistant reply
(``--expect-final``), so the Python side stays a thin ``str -> str`` adapter
wrapped via ``Target.from_callable``.

Safety / isolation (each probe runs in its own clean room):
  - ``deliver: false``           -> reply is never routed to real channels
  - ``sessionKey = agent:main:explicit:<uuid>`` -> fresh session per probe,
                                     so probes don't contaminate each other and
                                     the operator's main transcript is untouched
  - unique ``idempotencyKey``    -> gateway run-dedupe never collapses two probes

Run:
  python openclaw_redteam.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid

from tulip.security import Target, is_finding, red_team


# Point OPENCLAW_DIR at your local OpenClaw checkout; the gateway is expected
# on ws://127.0.0.1:18789 — drive it with the matching repo build
# (dist/index.js) to avoid protocol skew.
OPENCLAW_DIR = os.environ.get("OPENCLAW_DIR", os.path.expanduser("~/openclaw"))
CLI = ["node", "dist/index.js"]
AGENT_ID = "main"
CALL_TIMEOUT_MS = 180_000
SUITE = "owasp-asi"


def _openclaw_send(prompt: str) -> str:
    """Send one prompt through the full OpenClaw agent loop, return its reply."""
    run_id = f"tulip-rt-{uuid.uuid4().hex[:12]}"
    params = {
        "message": prompt,
        "agentId": AGENT_ID,
        "sessionKey": f"agent:{AGENT_ID}:explicit:{run_id}",
        "idempotencyKey": run_id,
        "deliver": False,
        "cleanupBundleMcpOnRunEnd": True,
    }
    argv = [
        *CLI,
        "gateway",
        "call",
        "agent",
        "--expect-final",
        "--json",
        "--timeout",
        str(CALL_TIMEOUT_MS),
        "--params",
        json.dumps(params),
    ]
    proc = subprocess.run(  # noqa: S603 - local harness shelling to a trusted, pinned CLI
        argv,
        cwd=OPENCLAW_DIR,
        capture_output=True,
        text=True,
        timeout=CALL_TIMEOUT_MS / 1000 + 30,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError(f"gateway call failed (rc={proc.returncode}): {' | '.join(tail)}")
    data = json.loads(proc.stdout)
    result = data.get("result", {}) or {}
    payloads = result.get("payloads") or []
    texts = [p.get("text") for p in payloads if isinstance(p, dict) and p.get("text")]
    if not texts:
        meta = result.get("meta", {}) or {}
        visible = meta.get("finalAssistantVisibleText")
        if visible:
            texts = [visible]
    reply = "\n".join(texts)
    # Live progress: one line per probe (prompt landed -> reply preview).
    preview = reply.replace("\n", " ")[:90]
    print(
        f"  · sent {len(prompt):>4}c -> reply {len(reply):>4}c: {preview!r}",
        file=sys.stderr,
        flush=True,
    )
    return reply


async def main() -> int:
    target = Target.from_callable(
        _openclaw_send,
        name="openclaw-local-gateway",
        metadata={
            "provider": "claude-cli",
            "model": "claude-sonnet-4-6",
            "transport": "openclaw gateway call agent (--expect-final)",
            "endpoint": "ws://127.0.0.1:18789",
        },
    )

    print(f"== Tulip red_team(suite={SUITE!r}) vs {target.name} ==", file=sys.stderr, flush=True)
    results = await red_team(target, suite=SUITE)

    findings = [r for r in results if is_finding(r)]
    abstentions = [r for r in results if not is_finding(r)]

    print("\n========== RED-TEAM REPORT ==========")
    print(f"target   : {target.name}  ({target.metadata.get('model')})")
    print(f"suite    : {SUITE}  ({len(results)} probes)")
    print(f"findings : {len(findings)}   abstentions : {len(abstentions)}\n")

    for r in results:
        if is_finding(r):
            tax = ", ".join(str(getattr(t, "value", t)) for t in (r.taxonomy or []))
            print(f"[FINDING] {str(r.severity).split('.')[-1]:<8} {r.title}")
            print(f"          taxonomy : {tax}")
            print(f"          gsar     : {getattr(r, 'gsar_score', '?')}")
            print(f"          fix      : {r.remediation}\n")
        else:
            print(f"[abstain] {getattr(r, 'title', '(probe)')}")
            print(f"          reason   : {r.reason}\n")

    out = {
        "target": {"name": target.name, "metadata": dict(target.metadata)},
        "suite": SUITE,
        "summary": {"findings": len(findings), "abstentions": len(abstentions)},
        "results": [r.model_dump(mode="json") for r in results],
    }

    def _write_report() -> None:
        with open("openclaw_redteam_report.json", "w") as fh:
            json.dump(out, fh, indent=2, default=str)

    await asyncio.to_thread(_write_report)
    print("wrote openclaw_redteam_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
