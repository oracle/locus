# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Mock security-tools FastMCP server for integration testing.

Deterministic, offline stand-ins for the tool surface a SOC agent
expects from an intel provider: hash lookup, IP reputation, and
indicator search. All data is invented and clearly fake (RFC 5737
addresses, EICAR-style test entries) — no network calls.
"""

import json
from datetime import datetime

from fastmcp import FastMCP


mcp = FastMCP("tulip-security-tools")

# The standard EICAR test file's well-known SHA-256 — safe by design.
_EICAR_SHA256 = "275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f"

# Invented reputation data for RFC 5737 documentation addresses.
_IP_REPUTATION = {
    "192.0.2.44": {"verdict": "suspicious", "reports": ["port scanning"]},
    "198.51.100.23": {"verdict": "malicious", "reports": ["brute-force attempts", "spam relay"]},
}

# Tiny invented indicator database.
_INDICATORS = [
    {"id": 1, "indicator": "phish.example.net", "type": "domain", "confidence": 92},
    {"id": 2, "indicator": "evil.example", "type": "domain", "confidence": 87},
    {"id": 3, "indicator": "198.51.100.23", "type": "ip", "confidence": 95},
]


@mcp.tool()
def get_current_time() -> str:
    """Get the current time for incident timeline entries."""
    return datetime.now().isoformat()


@mcp.tool()
def lookup_hash(sha256: str) -> str:
    """Look up a file hash in the mock malware database."""
    if sha256.lower() == _EICAR_SHA256:
        return json.dumps(
            {"sha256": sha256, "verdict": "known-test-file", "family": "EICAR test signature"}
        )
    return json.dumps({"sha256": sha256, "verdict": "unknown", "family": None})


@mcp.tool()
def ip_reputation(ip: str) -> str:
    """Check an IP address against the mock reputation feed."""
    entry = _IP_REPUTATION.get(ip, {"verdict": "no-reports", "reports": []})
    return json.dumps({"ip": ip, **entry})


@mcp.tool()
def search_indicators(query: str, limit: int = 10) -> str:
    """Search the mock indicator database by substring."""
    filtered = [r for r in _INDICATORS if query.lower() in r["indicator"].lower()]
    return json.dumps(filtered[:limit])


if __name__ == "__main__":
    mcp.run()
