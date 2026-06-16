# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""URL-safety / SSRF pre-flight guard.

Prevents Server-Side Request Forgery by rejecting HTTP(S) targets that
resolve to loopback, link-local, private, reserved, multicast, or cloud
metadata addresses. Intended for any Locus component that dispatches a
model-supplied URL — MCP over HTTP, user-authored fetch tools, RAG
document fetchers, etc.

Two calling styles are supported:

* :func:`is_safe_url` — returns ``bool`` for silent filtering.
* :func:`validate_url` — raises :class:`locus.core.errors.ValidationError`
  for call sites that want to short-circuit.

Behaviour:

* Cloud metadata hostnames (``metadata.google.internal``) and addresses
  (AWS / GCP / Azure / Alibaba IMDS, including the IPv6 variants) are
  **always** blocked, regardless of ``allow_private`` or the
  ``LOCUS_ALLOW_PRIVATE_URLS`` env var.
* Private / loopback / link-local / CGNAT ranges are blocked by default
  but can be opened up by passing ``allow_private=True`` at the call
  site, or globally via ``LOCUS_ALLOW_PRIVATE_URLS=true``.
* DNS-resolution failures fail **closed** — if the name cannot be
  resolved here, the HTTP client would fail anyway, so blocking early
  loses nothing and avoids leaking DNS probes.

Documented limitations (not addressable at the pre-flight layer):

* **DNS rebinding (TOCTOU):** an attacker-controlled nameserver with
  a very low TTL can return a public IP for this check and a private
  IP for the actual connection. Mitigating this requires
  connection-level IP validation (egress proxy, or checking the socket
  peer after ``connect``). Out of scope here.
* **Redirect following:** a public URL that redirects to a private
  one bypasses the pre-flight check. Callers using ``httpx`` should
  register an event hook that re-runs :func:`validate_url` on each
  ``Location`` target; callers that follow redirects inside a third-
  party SDK must rely on that SDK's egress policy.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from urllib.parse import urlparse

from locus.core.errors import ValidationError


logger = logging.getLogger(__name__)

__all__ = [
    "is_safe_url",
    "validate_url",
]

# Hostnames whose resolution is never safe — cloud metadata endpoints
# with stable names. IP-only endpoints are covered by _ALWAYS_BLOCKED_IPS
# below. Match is exact, case-insensitive, trailing dot stripped.
_BLOCKED_HOSTNAMES: frozenset[str] = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
    }
)

# Specific addresses that must never be reached, even when callers opt
# into private-network access. These are cloud instance-metadata services
# (IMDS) — the single most common SSRF target.
_ALWAYS_BLOCKED_IPS: frozenset[ipaddress.IPv4Address | ipaddress.IPv6Address] = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),  # AWS / GCP / Azure / OCI / DO
        ipaddress.ip_address("169.254.170.2"),  # AWS ECS task-role metadata
        ipaddress.ip_address("169.254.169.253"),  # Azure IMDS wire server
        ipaddress.ip_address("fd00:ec2::254"),  # AWS metadata (IPv6)
        ipaddress.ip_address("100.100.100.200"),  # Alibaba Cloud metadata
    }
)

# The full link-local range — blocked unconditionally because every
# cloud vendor's metadata endpoint lives somewhere inside it, and there
# is no legitimate reason for an agent to target 169.254.x.y directly.
_ALWAYS_BLOCKED_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("169.254.0.0/16"),
)

# Carrier-grade NAT / shared address space (RFC 6598). ``ipaddress``
# considers this neither private nor global, so it must be checked
# explicitly. Used by Tailscale / WireGuard overlays and some ISPs.
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

# Name of the env var that opens up private-network targets globally.
# Values "true" / "1" / "yes" (case-insensitive) enable; anything else
# leaves the default (deny) in place.
_ENV_VAR = "LOCUS_ALLOW_PRIVATE_URLS"


def _env_allow_private() -> bool:
    """Return True when the user has globally opted into private URLs."""
    return os.getenv(_ENV_VAR, "").strip().lower() in ("true", "1", "yes")


def _is_private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if ``ip`` lies in any range that requires opt-in."""
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return True
    # RFC 6598 CGNAT is not covered by ``is_private``.
    return ip in _CGNAT_NETWORK


def _is_always_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if ``ip`` is a metadata endpoint (blocked unconditionally)."""
    if ip in _ALWAYS_BLOCKED_IPS:
        return True
    return any(ip in net for net in _ALWAYS_BLOCKED_NETWORKS)


def is_safe_url(url: str, *, allow_private: bool | None = None) -> bool:
    """Return True when fetching ``url`` is safe per the SSRF guard.

    Args:
        url: Absolute URL to validate.
        allow_private: Override the process-level default. ``None``
            (the default) consults ``LOCUS_ALLOW_PRIVATE_URLS``;
            pass ``True`` explicitly for in-cluster / loopback
            traffic that the caller knows is legitimate.

    Returns:
        ``True`` if the URL is safe, ``False`` otherwise. DNS failures
        and parsing edge cases return ``False`` (fail-closed).
    """
    if allow_private is None:
        allow_private = _env_allow_private()

    # ``urlparse`` does not raise on malformed input — it returns
    # an empty ``hostname`` instead, which we then reject below.
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        return False

    if hostname in _BLOCKED_HOSTNAMES:
        logger.warning("Blocked URL: metadata hostname %s", hostname)
        return False

    try:
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        logger.warning("Blocked URL: DNS resolution failed for %s", hostname)
        return False

    for _family, _stype, _proto, _canon, sockaddr in addr_info:
        # IPv6 ``sockaddr`` is (host, port, flowinfo, scopeid); strip any
        # "%scope" suffix that might appear on IPv6 link-local strings.
        ip_str = str(sockaddr[0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue

        if _is_always_blocked_ip(ip):
            logger.warning("Blocked URL: cloud-metadata address %s -> %s", hostname, ip_str)
            return False

        if not allow_private and _is_private_ip(ip):
            logger.warning("Blocked URL: private/internal address %s -> %s", hostname, ip_str)
            return False

    return True


def validate_url(url: str, *, allow_private: bool | None = None) -> None:
    """Raise :class:`ValidationError` when ``url`` fails the SSRF guard.

    Thin wrapper over :func:`is_safe_url` for call sites that prefer
    to propagate a typed error rather than branch on a boolean.
    """
    if not is_safe_url(url, allow_private=allow_private):
        raise ValidationError(f"URL rejected by SSRF guard: {url!r}")
