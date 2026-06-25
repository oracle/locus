# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Vulnerability + posture scanning — dependency malware and endpoint checks.

Two read-only scanners that feed the SOC triage loop:

- :func:`scan_dependencies` wraps the SDK's existing OSV malware check
  (:func:`tulip.integrations.osv.check_package_for_malware`) — confirmed
  supply-chain (``MAL-*``) advisories on an ``npx`` / ``uvx`` / ``pipx``
  package, the agentic supply-chain surface (OWASP ASI04).
- :func:`scan_endpoint` checks a network endpoint's TLS-certificate expiry
  and whether the port is open. It is **offline-by-default**: it returns a
  deterministic sample unless ``SCANNER_LIVE=1`` is set, so CI never reaches
  the network. :func:`scan_endpoint_to_finding` grounds an expired-cert
  result into a :class:`~tulip.security.Evidence` (a healthy endpoint
  abstains).
"""

from __future__ import annotations

import socket
import ssl
import time

from tulip.integrations.osv import check_package_for_malware
from tulip.reasoning.gsar import Partition
from tulip.security._adapters import as_json, env, inference_claim, tool_match
from tulip.security.findings import Indicator
from tulip.security.grounded import GroundedFinding, ground_finding
from tulip.security.taxonomy import IndicatorType, OwaspLLM, Severity
from tulip.tools.decorator import tool


# Deterministic offline posture samples, keyed by host. RFC 5737 ranges only.
_OFFLINE_POSTURE: dict[str, dict[str, object]] = {
    "192.0.2.10": {"open": True, "tls_not_after": "2026-01-02", "tls_expired": True},
    "198.51.100.5": {"open": True, "tls_not_after": "2027-09-30", "tls_expired": False},
}


def scan_dependencies(command: str, args: list[str]) -> dict[str, object]:
    """Check a package launch command for known malware advisories (OSV).

    Returns ``{"clean": bool, "advisory": str | None, ...}``. ``clean`` is
    ``True`` when OSV reports no ``MAL-*`` advisory (the check is fail-open —
    network errors and unknown ecosystems read as clean).
    """
    advisory = check_package_for_malware(command, args)
    return {
        "command": command,
        "args": args,
        "source": "osv.dev",
        "clean": advisory is None,
        "advisory": advisory,
    }


def scan_endpoint(host: str, port: int = 443) -> dict[str, object]:
    """Check a network endpoint's port reachability + TLS-cert expiry.

    Offline-by-default: returns a deterministic sample unless ``SCANNER_LIVE``
    is set, so CI never touches the network. The live path uses only the
    stdlib (``socket`` + ``ssl``) — no extra dependency.
    """
    if env("SCANNER_LIVE"):
        return _scan_live(host, port)
    sample = _OFFLINE_POSTURE.get(
        host, {"open": False, "tls_not_after": None, "tls_expired": False}
    )
    return {"host": host, "port": port, "source": "offline-sample", **sample}


def _scan_live(host: str, port: int) -> dict[str, object]:
    """Real port + TLS-expiry probe via the stdlib (gated by ``SCANNER_LIVE``)."""
    result: dict[str, object] = {
        "host": host,
        "port": port,
        "source": "live-scan",
        "open": False,
        "tls_not_after": None,
        "tls_expired": False,
    }
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((host, port), timeout=10.0) as sock:
            result["open"] = True
            with ctx.wrap_socket(sock, server_hostname=host) as tls:
                cert = tls.getpeercert()
        not_after = cert.get("notAfter") if cert else None
        if not_after:
            expiry = ssl.cert_time_to_seconds(str(not_after))
            result["tls_not_after"] = not_after
            result["tls_expired"] = expiry < time.time()
    except (OSError, ssl.SSLError, ValueError):
        # Unreachable / TLS error — leave defaults (closed, no cert).
        return result
    return result


def scan_endpoint_to_finding(host: str, port: int = 443) -> GroundedFinding:
    """Scan an endpoint and ground an expired-certificate result into a finding.

    An expired certificate read off the handshake is tool-backed evidence and
    ships a HIGH finding; a healthy or unreachable endpoint carries no grounded
    defect, so the candidate finding abstains.
    """
    posture = scan_endpoint(host, port=port)
    asset = f"{host}:{port}"
    ref = f"tool:scan_endpoint:{asset}:tls_not_after={posture.get('tls_not_after')}"
    if posture.get("tls_expired"):
        partition = Partition(
            grounded=[
                tool_match(
                    f"TLS certificate on {asset} expired ({posture.get('tls_not_after')})", ref
                )
            ],
        )
    else:
        # A valid (or unreachable) endpoint gives no grounded support for an
        # "expired" finding, so the candidate abstains.
        partition = Partition(
            ungrounded=[inference_claim(f"TLS certificate on {asset} may be expired", ref)],
        )
    return ground_finding(
        title=f"Expired TLS certificate on {asset}",
        description=f"Endpoint {asset} presents an expired certificate.",
        severity=Severity.HIGH,
        asset=asset,
        remediation="Rotate the certificate and enforce automated renewal.",
        partition=partition,
        indicators=[Indicator(type=IndicatorType.ENDPOINT, value=asset)],
        taxonomy=[OwaspLLM.SENSITIVE_INFORMATION_DISCLOSURE],
    )


@tool(
    name="scan_dependencies",
    description="Check a package launch command (npx/uvx/pipx) for known malware advisories",
)
async def scan_dependencies_tool(command: str, args: list[str] | None = None) -> str:
    """Tool wrapper: returns the OSV malware verdict as a JSON string."""
    return as_json(scan_dependencies(command, args or []))


@tool(
    name="scan_endpoint",
    description="Check a network endpoint's port reachability and TLS-certificate expiry",
)
async def scan_endpoint_tool(host: str, port: int = 443) -> str:
    """Tool wrapper: returns the endpoint posture as a JSON string."""
    return as_json(scan_endpoint(host, port=port))


__all__ = [
    "scan_dependencies",
    "scan_dependencies_tool",
    "scan_endpoint",
    "scan_endpoint_to_finding",
    "scan_endpoint_tool",
]
