# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""SIEM log/alert query — a Splunk/Elastic-shaped vendor tool.

Pulling the events behind an alert is the second move of triage, right
after IOC enrichment. This is a worked vendor integration: with
``SIEM_URL`` + ``SIEM_TOKEN`` set it POSTs a search to an Elastic/Splunk-
shaped endpoint; with neither set it returns deterministic, benign sample
events so the cookbook runs offline.

Use it as a Tulip ``@tool`` (``siem_query_tool``) handed to an agent, or
call :func:`query_siem` directly.
"""

from __future__ import annotations

import json
import os

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
    siem_url = os.environ.get("SIEM_URL")
    token = os.environ.get("SIEM_TOKEN")
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
    return json.dumps(query_siem(query, window=window))


if __name__ == "__main__":
    print(json.dumps(query_siem("powershell", window="6h"), indent=2))
