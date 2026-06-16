# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OSV malware check for MCP supply-chain entry points.

Before an MCP server is spawned via ``npx`` / ``uvx`` / ``pipx`` /
``bunx`` / ``pnpm dlx``, this module queries the public `OSV
<https://osv.dev>`_ database for **malware** advisories (``MAL-*`` IDs)
on the requested package. Regular CVEs are intentionally ignored — the
goal is to block confirmed supply-chain attacks, not to gate on every
known vulnerability (too noisy for a pre-launch check).

The API is free, public, maintained by Google, and typically responds
in ~300 ms. The check is fail-open: timeouts, HTTP errors, JSON parse
failures, and unrecognised commands all allow the spawn to proceed.
This is a deliberate trade-off — an on-by-default gate that hard-fails
on network blips would be worse than no gate at all.

Global opt-out: set ``LOCUS_MCP_SKIP_OSV=1`` to disable the check
entirely. Useful for air-gapped / CI environments with no network egress.

Inspired by Block/goose's MCP extension malware check.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

__all__ = ["check_package_for_malware"]

_OSV_ENDPOINT = os.getenv("OSV_ENDPOINT", "https://api.osv.dev/v1/query")
_TIMEOUT = 10  # seconds
_USER_AGENT = "locus-osv-check/1.0"
_SKIP_ENV = "LOCUS_MCP_SKIP_OSV"


def check_package_for_malware(command: str, args: list[str]) -> str | None:
    """Look up OSV malware advisories for an MCP launch command.

    Inspects *command* (e.g. ``npx``, ``uvx``) and *args* to infer the
    package name and ecosystem, then POSTs to the OSV API for
    ``MAL-*`` advisories.

    Args:
        command: The executable being launched. Non-supply-chain
            commands (``python``, ``node``, explicit paths, etc.)
            return ``None`` immediately.
        args: The argument list that would be passed to *command*.

    Returns:
        A human-readable reason string when malware is found, suitable
        for propagation via an exception. ``None`` when the package
        is clean, the ecosystem is unknown, the user opted out, or the
        lookup failed for any reason (fail-open).
    """
    if os.getenv(_SKIP_ENV, "").strip().lower() in ("1", "true", "yes"):
        return None

    ecosystem = _infer_ecosystem(command)
    if not ecosystem:
        return None

    package, version = _parse_package_from_args(args, ecosystem)
    if not package:
        return None

    try:
        malware = _query_osv(package, ecosystem, version)
    except Exception as exc:  # noqa: BLE001 — fail-open by design
        logger.debug("OSV check failed for %s/%s (allowing): %s", ecosystem, package, exc)
        return None

    if not malware:
        return None

    ids = ", ".join(m["id"] for m in malware[:3])
    summaries = "; ".join(m.get("summary", m["id"])[:100] for m in malware[:3])
    return (
        f"Package {package!r} ({ecosystem}) has known malware advisories: "
        f"{ids}. Details: {summaries}"
    )


# ---------------------------------------------------------------------------
# Helpers (private — tested via the public entry point only)
# ---------------------------------------------------------------------------


# npm-like launchers (strip the executable extension before matching).
_NPM_COMMANDS: frozenset[str] = frozenset({"npx", "bunx", "pnpx"})
_PYPI_COMMANDS: frozenset[str] = frozenset({"uvx", "pipx"})


def _infer_ecosystem(command: str) -> str | None:
    """Map a launcher executable to its OSV ecosystem name.

    Handles both POSIX (``/usr/local/bin/npx``) and Windows
    (``C:\\npm\\npx.cmd``) absolute paths, stripping common launcher
    extensions.
    """
    # Normalise Windows separators so ``Path.name`` works on any OS.
    base = Path(command.replace("\\", "/")).name.lower()
    base = re.sub(r"\.(cmd|exe|bat)$", "", base)
    if base in _NPM_COMMANDS:
        return "npm"
    if base in _PYPI_COMMANDS:
        return "PyPI"
    return None


def _parse_package_from_args(args: list[str], ecosystem: str) -> tuple[str | None, str | None]:
    """Pull the first non-flag token out of *args* and parse it."""
    package_token: str | None = None
    skip_next = False
    # Flags that take a value: we must skip both the flag *and* its arg.
    _value_flags = {"-p", "--package"}
    for arg in args:
        if not isinstance(arg, str):
            continue
        if skip_next:
            skip_next = False
            continue
        if arg in _value_flags:
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        package_token = arg
        break

    if not package_token:
        return None, None

    if ecosystem == "npm":
        return _parse_npm_package(package_token)
    if ecosystem == "PyPI":
        return _parse_pypi_package(package_token)
    return package_token, None


def _parse_npm_package(token: str) -> tuple[str | None, str | None]:
    """Parse ``@scope/name@version`` or ``name@version``."""
    if token.startswith("@"):
        match = re.match(r"^(@[^/]+/[^@]+)(?:@(.+))?$", token)
        if match:
            version = match.group(2)
            return match.group(1), None if version == "latest" else version
        return token, None
    if "@" in token:
        name, _, version = token.rpartition("@")
        return name, None if version == "latest" else version
    return token, None


def _parse_pypi_package(token: str) -> tuple[str | None, str | None]:
    """Parse ``name==version`` or ``name[extras]==version``."""
    match = re.match(r"^([a-zA-Z0-9._-]+)(?:\[[^\]]*\])?(?:==(.+))?$", token)
    if match:
        return match.group(1), match.group(2)
    return token, None


def _query_osv(package: str, ecosystem: str, version: str | None = None) -> list[dict[str, Any]]:
    """POST to OSV and return just the ``MAL-*`` advisories."""
    payload: dict[str, Any] = {"package": {"name": package, "ecosystem": ecosystem}}
    if version:
        payload["version"] = version

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 — fixed upstream endpoint
        _OSV_ENDPOINT,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
        body = json.loads(resp.read())

    vulns: list[dict[str, Any]] = body.get("vulns", []) or []
    return [v for v in vulns if str(v.get("id", "")).startswith("MAL-")]
