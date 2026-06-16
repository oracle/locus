# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end integration test for the SSRF guard (A.2).

Wires :func:`~locus.tools.url_safety.validate_url` into a real
``httpx.AsyncClient`` event hook so the guard runs at the actual
request-dispatch boundary, exactly as a user-authored fetch tool
would integrate it. Hostname resolution is monkey-patched so we
don't depend on internet access.
"""

from __future__ import annotations

import socket
from typing import Any

import httpx
import pytest
import respx
from httpx import Request, Response

from locus.core.errors import ValidationError
from locus.tools.url_safety import validate_url


def _force_resolution(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
    """Pin DNS to ``ip`` for any host."""

    def _fake(host: str, port: int | None, *_a: Any, **_kw: Any) -> Any:
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 0, "", (ip, port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake)


def _make_client() -> httpx.AsyncClient:
    """Build an httpx client whose request event hook runs the SSRF guard."""

    async def _validate(request: Request) -> None:
        validate_url(str(request.url))

    return httpx.AsyncClient(event_hooks={"request": [_validate]})


# ---------------------------------------------------------------------------
# Public address — guard allows the dispatch.
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_public_url_passes_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_resolution(monkeypatch, "8.8.8.8")
    respx.get("https://api.example.com/v1/data").mock(return_value=Response(200, json={"ok": True}))

    async with _make_client() as client:
        r = await client.get("https://api.example.com/v1/data")
        assert r.status_code == 200
        assert r.json() == {"ok": True}


# ---------------------------------------------------------------------------
# Cloud metadata — always blocked, even on opt-in.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_url_blocked_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_resolution(monkeypatch, "169.254.169.254")
    async with _make_client() as client:
        with pytest.raises(ValidationError, match="SSRF guard"):
            await client.get("https://imds.example/latest/meta-data/")


@pytest.mark.asyncio
async def test_metadata_hostname_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force resolution to a public IP so we know the block fires on
    # hostname alone, not IP class.
    _force_resolution(monkeypatch, "8.8.8.8")
    async with _make_client() as client:
        with pytest.raises(ValidationError, match="SSRF guard"):
            await client.get("https://metadata.google.internal/computeMetadata/")


# ---------------------------------------------------------------------------
# Private IP ranges — blocked by default.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "private_ip",
    ["127.0.0.1", "192.168.1.5", "10.0.0.1", "172.16.0.1", "100.64.0.5", "::1", "fe80::1"],
)
@pytest.mark.asyncio
async def test_private_ip_blocked(monkeypatch: pytest.MonkeyPatch, private_ip: str) -> None:
    _force_resolution(monkeypatch, private_ip)
    async with _make_client() as client:
        with pytest.raises(ValidationError, match="SSRF guard"):
            await client.get("https://internal.example/api")


# ---------------------------------------------------------------------------
# Opt-in env var allows private IPs but never metadata.
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_env_opt_in_unblocks_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCUS_ALLOW_PRIVATE_URLS", "true")
    _force_resolution(monkeypatch, "10.0.0.5")
    respx.get("https://internal.example/health").mock(return_value=Response(200, text="ok"))

    async with _make_client() as client:
        r = await client.get("https://internal.example/health")
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_env_opt_in_does_not_unblock_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCUS_ALLOW_PRIVATE_URLS", "true")
    _force_resolution(monkeypatch, "169.254.169.254")
    async with _make_client() as client:
        with pytest.raises(ValidationError, match="SSRF guard"):
            await client.get("https://imds.example/")
