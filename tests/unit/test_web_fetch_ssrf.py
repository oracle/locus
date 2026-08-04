# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit: the web_fetch SSRF guard (DESIGN-appsec-threat-model.md A3).

``web_fetch`` runs in the credential-holding harness, not the sandbox — so a
model-emitted URL pointing at 169.254.169.254 or an internal host must be
refused before connect, and on every redirect hop. These tests pin that.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket

import httpx
import pytest
import respx

from tulip.providers.web_fetch import (
    HTTPXWebFetcher,
    WebFetchError,
    _assert_public_destination,
    _ip_is_blocked,
)


_A_PUBLIC_IP = "93.184.216.34"  # example.com — a literal, so no DNS is needed


class TestIpClassification:
    @pytest.mark.parametrize(
        "addr",
        [
            "169.254.169.254",  # cloud metadata (link-local)
            "127.0.0.1",  # loopback
            "10.0.0.5",  # RFC-1918
            "192.168.1.1",  # RFC-1918
            "172.16.0.1",  # RFC-1918
            "100.64.0.1",  # CGNAT (extra net)
            "198.18.0.1",  # benchmarking (extra net)
            "0.0.0.0",  # noqa: S104 — unspecified (test data, not a bind)
            "::1",  # IPv6 loopback
            "fe80::1",  # IPv6 link-local
            "fd00::1",  # IPv6 ULA (private)
            "::ffff:169.254.169.254",  # IPv4-mapped metadata — must be unwrapped
        ],
    )
    def test_internal_addresses_are_blocked(self, addr: str) -> None:
        assert _ip_is_blocked(ipaddress.ip_address(addr)) is True

    @pytest.mark.parametrize("addr", [_A_PUBLIC_IP, "8.8.8.8", "1.1.1.1", "2606:4700::1111"])
    def test_public_addresses_are_allowed(self, addr: str) -> None:
        assert _ip_is_blocked(ipaddress.ip_address(addr)) is False


class TestDestinationValidation:
    async def test_non_http_scheme_is_refused(self) -> None:
        for url in ("file:///etc/passwd", "gopher://x/", "ftp://x/"):
            with pytest.raises(WebFetchError, match="http/https only"):
                await _assert_public_destination(url)

    async def test_url_without_host_is_refused(self) -> None:
        with pytest.raises(WebFetchError, match="no host"):
            await _assert_public_destination("http:///nohost")

    async def test_metadata_ip_literal_is_refused(self) -> None:
        with pytest.raises(WebFetchError, match="blocked address"):
            await _assert_public_destination("http://169.254.169.254/latest/meta-data/")

    async def test_public_ip_literal_passes_without_dns(self) -> None:
        await _assert_public_destination(f"http://{_A_PUBLIC_IP}/")  # no raise

    async def test_hostname_resolving_to_internal_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_getaddrinfo(host: str, port: int, *a: object, **k: object) -> list:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))]

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", fake_getaddrinfo)
        with pytest.raises(WebFetchError, match="blocked address"):
            await _assert_public_destination("http://metadata.example.test/")

    async def test_unresolvable_host_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(host: str, port: int, *a: object, **k: object) -> list:
            raise socket.gaierror("nope")

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", boom)
        with pytest.raises(WebFetchError, match="cannot resolve"):
            await _assert_public_destination("http://nx.example.test/")


class TestFetchGuard:
    async def test_fetch_refuses_a_blocked_url_before_any_request(self) -> None:
        # No respx route registered — if a request escaped, respx would raise a
        # different error. The guard must stop it first.
        with pytest.raises(WebFetchError, match="blocked address"):
            await HTTPXWebFetcher().fetch("http://169.254.169.254/latest/meta-data/")

    @respx.mock
    async def test_redirect_to_metadata_is_re_validated_and_refused(self) -> None:
        # Initial hop is a public literal (passes), but it 302s to the metadata
        # endpoint — the manual redirect loop must re-check and refuse it.
        respx.get(f"http://{_A_PUBLIC_IP}/").mock(
            return_value=httpx.Response(
                302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
            )
        )
        with pytest.raises(WebFetchError, match="blocked address"):
            await HTTPXWebFetcher().fetch(f"http://{_A_PUBLIC_IP}/")

    @respx.mock
    async def test_public_fetch_succeeds_and_extracts_text(self) -> None:
        respx.get(f"http://{_A_PUBLIC_IP}/").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<title>Hi</title><p>hello world</p>",
            )
        )
        page = await HTTPXWebFetcher().fetch(f"http://{_A_PUBLIC_IP}/")
        assert page.status == 200
        assert page.title == "Hi"
        assert "hello world" in page.text

    @respx.mock
    async def test_block_private_false_opts_out_of_the_guard(self) -> None:
        # The documented escape hatch for a trusted internal network.
        respx.get("http://10.0.0.5/").mock(return_value=httpx.Response(200, text="internal ok"))
        page = await HTTPXWebFetcher(block_private=False).fetch("http://10.0.0.5/")
        assert page.status == 200
        assert "internal ok" in page.text
