# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""EDR host telemetry + containment — a CrowdStrike/Defender-shaped adapter.

After triage points at a host, the next moves are forensic (what happened on
it) and, if confirmed, containment (cut it off). With ``EDR_URL`` +
``EDR_TOKEN`` set these query a CrowdStrike Falcon / Microsoft Defender /
SentinelOne-shaped API; with neither set they return deterministic, benign
sample telemetry so the SDK runs offline.

- :func:`fetch_host_timeline` / :func:`list_detections` — read-only forensics.
- :func:`isolate_host` — a **containment write**. The tool wrapper is
  ``@tool(idempotent=True)`` so the loop will not isolate the same host twice
  on a retry, and it is gated behind ``SecurityControls.allow_containment`` in
  the SOC factory (off by default).

UNVERIFIED LIVE PATH: the ``EDR_URL``/``EDR_TOKEN`` branches are written to a
generic EDR REST shape but have not been run against a real console — adjust
the paths/fields per product. Only the offline sample path runs in CI.
"""

from __future__ import annotations

from tulip.security._adapters import as_json, env
from tulip.tools.decorator import tool


# Benign, invented offline telemetry. Hosts are obvious lab names; IPs are
# RFC 5737 / RFC 1918 ranges.
_OFFLINE_TIMELINE: dict[str, list[dict[str, object]]] = {
    "WS-0142": [
        {
            "ts": "2026-06-11T09:14:01Z",
            "kind": "process",
            "parent": "winword.exe",
            "process": "powershell.exe -enc <base64>",
            "verdict": "suspicious",
        },
        {
            "ts": "2026-06-11T09:14:05Z",
            "kind": "network",
            "process": "powershell.exe",
            "remote": "198.51.100.23:443",
            "verdict": "malicious",
        },
        {
            "ts": "2026-06-11T09:15:20Z",
            "kind": "file",
            "process": "powershell.exe",
            "path": "C:\\Users\\Public\\stage.bin",
            "verdict": "suspicious",
        },
    ],
}

_OFFLINE_DETECTIONS: list[dict[str, object]] = [
    {
        "id": "det-7741",
        "host": "WS-0142",
        "tactic": "Execution",
        "technique": "T1059.001 (PowerShell)",
        "severity": "high",
        "status": "new",
    },
    {
        "id": "det-7742",
        "host": "WS-0142",
        "tactic": "Command and Control",
        "technique": "T1071.001 (Web Protocols)",
        "severity": "high",
        "status": "new",
    },
]


def fetch_host_timeline(host: str, window: str = "24h") -> dict[str, object]:
    """Return the recent process/network/file timeline for a host.

    Live path (``EDR_URL`` + ``EDR_TOKEN``) queries the console; offline path
    returns the benign sample for known lab hosts (empty for others).
    """
    edr_url = env("EDR_URL")
    token = env("EDR_TOKEN")
    if edr_url and token:
        return _edr_get(edr_url, token, "/timeline", {"host": host, "window": window})
    events = _OFFLINE_TIMELINE.get(host.upper(), [])
    return {"host": host, "window": window, "source": "offline-sample", "events": events}


def list_detections(host: str | None = None) -> dict[str, object]:
    """List open EDR detections, optionally filtered to one host."""
    edr_url = env("EDR_URL")
    token = env("EDR_TOKEN")
    if edr_url and token:
        return _edr_get(edr_url, token, "/detections", {"host": host} if host else {})
    dets = [d for d in _OFFLINE_DETECTIONS if host is None or d["host"] == host.upper()]
    return {"host": host, "source": "offline-sample", "count": len(dets), "detections": dets}


def isolate_host(host_id: str) -> dict[str, object]:
    """Network-isolate (contain) a host. **Write action** — gate it.

    Live path (``EDR_URL`` + ``EDR_TOKEN``) POSTs a containment action;
    offline path returns a simulated receipt so the loop is exercisable
    without touching a real fleet.
    """
    edr_url = env("EDR_URL")
    token = env("EDR_TOKEN")
    if edr_url and token:
        return _edr_post(edr_url, token, "/devices/actions/contain", {"host_id": host_id})
    return {"host_id": host_id, "source": "offline-sample", "status": "contained (simulated)"}


def _edr_get(edr_url: str, token: str, path: str, params: dict[str, object]) -> dict[str, object]:
    """GET against a generic EDR REST endpoint."""
    import httpx

    with httpx.Client(
        base_url=edr_url, headers={"Authorization": f"Bearer {token}"}, timeout=30.0
    ) as client:
        resp = client.get(path, params={k: str(v) for k, v in params.items() if v is not None})
        resp.raise_for_status()
        body = resp.json()
    return {"source": "edr", **body}


def _edr_post(edr_url: str, token: str, path: str, payload: dict[str, object]) -> dict[str, object]:
    """POST a containment/action to a generic EDR REST endpoint."""
    import httpx

    with httpx.Client(
        base_url=edr_url, headers={"Authorization": f"Bearer {token}"}, timeout=30.0
    ) as client:
        resp = client.post(path, json=payload)
        resp.raise_for_status()
        body = resp.json()
    return {"source": "edr", **body}


@tool(
    name="fetch_host_timeline",
    description="Pull the recent EDR process/network/file timeline for a host",
)
async def fetch_host_timeline_tool(host: str, window: str = "24h") -> str:
    """Tool wrapper: returns the host timeline as a JSON string."""
    return as_json(fetch_host_timeline(host, window=window))


@tool(name="list_detections", description="List open EDR detections, optionally for one host")
async def list_detections_tool(host: str = "") -> str:
    """Tool wrapper: returns open detections as a JSON string."""
    return as_json(list_detections(host or None))


@tool(
    name="isolate_host",
    description="Network-isolate (contain) a host — a containment action",
    idempotent=True,
)
async def isolate_host_tool(host_id: str) -> str:
    """Tool wrapper for the containment write — idempotent on ``host_id``."""
    return as_json(isolate_host(host_id))


__all__ = [
    "fetch_host_timeline",
    "fetch_host_timeline_tool",
    "isolate_host",
    "isolate_host_tool",
    "list_detections",
    "list_detections_tool",
]
