# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Threat-intel IOC enrichment — a VirusTotal/GreyNoise-shaped adapter.

Looking up the reputation and context of an indicator (file hash, IP,
domain) is the first move of SOC triage. With ``VT_API_KEY`` set,
:func:`enrich_indicator` queries a VirusTotal v3-shaped API; with none set
it returns deterministic, benign sample reputation (RFC 5737 documentation
addresses, ``*.example`` domains, the well-known EICAR test hash) so the
SDK runs offline.

Hand :data:`enrich_indicator_tool` to an agent, call :func:`enrich_indicator`
directly, or use :func:`enrich_to_finding` to turn an enrichment into a
GSAR-grounded :class:`~tulip.security.Finding` (a clean indicator abstains
rather than shipping a non-finding).

UNVERIFIED LIVE PATH: the ``VT_API_KEY`` branch is written to VirusTotal's
documented v3 shape but has not been run against the live API — adjust the
field paths before relying on it against a real tenant. Only the offline
sample path is exercised in CI.
"""

from __future__ import annotations

import re

from tulip.reasoning.gsar import Partition
from tulip.security._adapters import as_json, env, indicator_type, inference_claim, tool_match
from tulip.security.findings import Indicator
from tulip.security.grounded import GroundedFinding, ground_finding
from tulip.security.taxonomy import Severity, TaxonomyTag
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
    offline path returns the benign sample table. The return shape is the
    same so an agent's downstream reasoning is identical either way.
    """
    kind = classify_indicator(indicator)
    api_key = env("VT_API_KEY")
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


def enrich_to_finding(
    indicator: str,
    *,
    severity: Severity | None = None,
    taxonomy: list[TaxonomyTag] | None = None,
) -> GroundedFinding:
    """Enrich an indicator and ground the verdict into a :class:`Finding`.

    A vendor detection count is tool-backed evidence, so a flagged indicator
    ships a finding (severity scaling with the detection count). A clean
    indicator carries no grounded support for a "malicious" claim, so the
    candidate finding abstains — the caller gets an
    :class:`~tulip.security.Abstention`, never an empty finding.
    """
    rep = enrich_indicator(indicator)
    kind = str(rep.get("kind", classify_indicator(indicator)))
    malicious = int(str(rep.get("malicious", 0)))
    verdict = str(rep.get("verdict", "no-reports"))
    source = str(rep.get("source", "offline-sample"))
    note = str(rep.get("note", ""))
    ref = f"tool:enrich_indicator:{source}:malicious={malicious}"
    ind = Indicator(type=indicator_type(kind, indicator), value=indicator)

    statement = f"{indicator} flagged by {malicious} vendor(s) ({verdict}): {note}".strip()
    if malicious >= 3:
        partition = Partition(grounded=[tool_match(statement, ref)])
        sev = severity or Severity.HIGH
    elif malicious >= 1:
        partition = Partition(grounded=[tool_match(statement, ref)])
        sev = severity or Severity.MEDIUM
    else:
        # No detections — the "malicious" claim is unsupported; abstain.
        partition = Partition(
            ungrounded=[inference_claim(f"{indicator} may be malicious", ref)],
        )
        sev = severity or Severity.INFO

    return ground_finding(
        title=f"Malicious indicator: {indicator}",
        description=f"Threat-intel enrichment for {indicator}. {note}".strip(),
        severity=sev,
        asset=indicator,
        remediation="Block the indicator, hunt for related activity, and open a case.",
        partition=partition,
        indicators=[ind],
        taxonomy=taxonomy or [],
    )


@tool(
    name="enrich_indicator",
    description="Look up reputation/context for an IOC (hash, IP, or domain)",
)
async def enrich_indicator_tool(indicator: str) -> str:
    """Tool wrapper: returns the enrichment as a JSON string."""
    return as_json(enrich_indicator(indicator))


__all__ = [
    "classify_indicator",
    "enrich_indicator",
    "enrich_indicator_tool",
    "enrich_to_finding",
]
