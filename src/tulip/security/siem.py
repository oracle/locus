# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""SIEM log/alert search — a Splunk/Elastic-shaped reference adapter.

Reference template — offline-by-default; the live path's shape and field names
are illustrative and unverified. For a proven, maintained vendor adapter see
``tulip-integrations`` (``tulip_integrations.security.splunk``).

Pulling the events behind an alert is the second move of triage, right
after IOC enrichment. With ``SIEM_URL`` + ``SIEM_TOKEN`` set,
:func:`query_siem` POSTs a search to an Elastic/Splunk-shaped endpoint;
with neither set it filters deterministic, benign sample events so the SDK
runs offline. Events are evidence for downstream reasoning, so this adapter
exposes a search (no ``*_to_finding`` — the agent grounds findings from what
the events show).

Hand :data:`siem_query_tool` to an agent or call :func:`query_siem` directly.

UNVERIFIED LIVE PATH: the ``SIEM_URL``/``SIEM_TOKEN`` branch posts to a
Splunk/Elastic-shaped endpoint that has not been run against a real SIEM —
the path and field names are illustrative and will need adjusting per
product. Only the offline sample path is exercised in CI.
"""

from __future__ import annotations

import json

from tulip.security._adapters import as_json, env
from tulip.tools.decorator import tool


# Benign, invented offline events. Hosts/IPs are RFC 5737 / RFC 1918
# documentation ranges; users are obvious test accounts.
_OFFLINE_EVENTS: list[dict[str, object]] = [
    {
        "ts": "2026-06-11T09:14:02Z",
        "host": "WS-0142",
        "event": "process_create",
        "detail": "winword.exe spawned powershell.exe -enc <base64>",
        "severity": "high",
    },
    {
        "ts": "2026-06-11T09:14:05Z",
        "host": "WS-0142",
        "event": "network_connect",
        "detail": "powershell.exe -> 198.51.100.23:443",
        "severity": "high",
    },
    {
        "ts": "2026-06-11T09:11:40Z",
        "host": "WS-0142",
        "event": "auth_failed",
        "detail": "4 failed logons for user svc-backup from 192.0.2.44",
        "severity": "medium",
    },
]


def query_siem(query: str, window: str = "24h", limit: int = 50) -> dict[str, object]:
    """Search the SIEM for events matching ``query`` over a time window.

    Live path (``SIEM_URL`` + ``SIEM_TOKEN`` set) POSTs a search; offline
    path filters the benign sample events by substring. The return shape is
    identical so an agent's downstream reasoning doesn't change.
    """
    siem_url = env("SIEM_URL")
    token = env("SIEM_TOKEN")
    if siem_url and token:
        return _siem_search(siem_url, token, query, window, limit)
    needle = query.lower()
    matched = [e for e in _OFFLINE_EVENTS if needle in json.dumps(e).lower() or needle in ("", "*")]
    return {
        "query": query,
        "window": window,
        "source": "offline-sample",
        "count": len(matched[:limit]),
        "events": matched[:limit],
    }


def _siem_search(
    siem_url: str, token: str, query: str, window: str, limit: int
) -> dict[str, object]:
    """POST a search to an Elastic/Splunk-shaped SIEM endpoint."""
    import httpx

    with httpx.Client(
        base_url=siem_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    ) as client:
        resp = client.post(
            "/services/search/jobs/export",
            json={"search": query, "earliest_time": f"-{window}", "count": limit},
        )
        resp.raise_for_status()
        events = resp.json().get("results", [])
    return {
        "query": query,
        "window": window,
        "source": "siem",
        "count": len(events),
        "events": events[:limit],
    }


@tool(name="query_siem", description="Search SIEM logs/alerts for events matching a query")
async def siem_query_tool(query: str, window: str = "24h") -> str:
    """Tool wrapper: returns matching events as a JSON string."""
    return as_json(query_siem(query, window=window))


__all__ = ["query_siem", "siem_query_tool"]
