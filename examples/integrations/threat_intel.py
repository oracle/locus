# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Threat-intel IOC enrichment — a VirusTotal/GreyNoise-shaped vendor tool.

Looking up the reputation and context of an indicator (file hash, IP,
domain) is the bread-and-butter first move of SOC triage. This is a
worked vendor integration: with ``VT_API_KEY`` set it queries a
VirusTotal-shaped API; with none set it returns deterministic, benign
sample reputation (RFC 5737 documentation addresses, ``*.example``
domains, the well-known EICAR test hash) so the cookbook runs offline.

Use it as a Tulip ``@tool`` (``enrich_indicator_tool``) handed to an
agent, or call :func:`enrich_indicator` directly.
"""

from __future__ import annotations

import json
import os
import re

from tulip.tools.decorator import tool


# The standard EICAR test file's well-known SHA-256 — safe by design.
_EICAR_SHA256 = "275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f"

# Benign, invented offline reputation. All addresses are RFC 5737
# documentation ranges and all domains are reserved example domains.
_OFFLINE_REPUTATION: dict[str, dict[str, object]] = {
    _EICAR_SHA256: {"verdict": "known-test-file", "malicious": 0, "note": "EICAR test signature"},
    "198.51.100.23": {"verdict": "malicious", "malicious": 41, "note": "brute-force, spam relay"},
    "192.0.2.44": {"verdict": "suspicious", "malicious": 7, "note": "port scanning"},
    "phish.example.net": {"verdict": "malicious", "malicious": 33, "note": "credential phishing"},
}

_HASH_RE = re.compile(r"^[a-fA-F0-9]{32,64}$")
_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def classify_indicator(indicator: str) -> str:
    """Infer the indicator kind from its shape: hash / ip / domain."""
    if _HASH_RE.match(indicator):
        return "hash"
    if _IP_RE.match(indicator):
        return "ip"
    return "domain"


def enrich_indicator(indicator: str) -> dict[str, object]:
    """Return reputation/context for one IOC.

    Live path (``VT_API_KEY`` set) queries a VirusTotal-shaped endpoint;
    offline path returns the benign sample table. The shape is the same so
    an agent's downstream reasoning is identical either way.
    """
    kind = classify_indicator(indicator)
    api_key = os.environ.get("VT_API_KEY")
    if api_key:
        return _vt_lookup(indicator, kind, api_key)
    entry = _OFFLINE_REPUTATION.get(
        indicator.lower() if kind == "hash" else indicator,
        {"verdict": "no-reports", "malicious": 0, "note": "not in offline sample feed"},
    )
    return {"indicator": indicator, "kind": kind, "source": "offline-sample", **entry}


def _vt_lookup(indicator: str, kind: str, api_key: str) -> dict[str, object]:
    """Query a VirusTotal v3-shaped API for one indicator."""
    import httpx

    path = {"hash": "files", "ip": "ip_addresses", "domain": "domains"}[kind]
    with httpx.Client(
        base_url="https://www.virustotal.com/api/v3",
        headers={"x-apikey": api_key},
        timeout=30.0,
    ) as client:
        resp = client.get(f"/{path}/{indicator}")
        resp.raise_for_status()
        stats = resp.json()["data"]["attributes"]["last_analysis_stats"]
    malicious = int(stats.get("malicious", 0))
    verdict = "malicious" if malicious >= 3 else "suspicious" if malicious else "no-reports"
    return {
        "indicator": indicator,
        "kind": kind,
        "source": "virustotal",
        "verdict": verdict,
        "malicious": malicious,
    }


@tool(
    name="enrich_indicator",
    description="Look up reputation/context for an IOC (hash, IP, or domain)",
)
async def enrich_indicator_tool(indicator: str) -> str:
    """Tool wrapper: returns the enrichment as a JSON string."""
    return json.dumps(enrich_indicator(indicator))


if __name__ == "__main__":
    for ioc in (_EICAR_SHA256, "198.51.100.23", "phish.example.net", "203.0.113.99"):
        print(json.dumps(enrich_indicator(ioc)))
