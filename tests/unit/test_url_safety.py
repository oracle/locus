# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for the SSRF pre-flight guard in ``locus.tools.url_safety``."""

from __future__ import annotations

import socket
from typing import Any

import pytest

from locus.core.errors import ValidationError
from locus.tools import url_safety
from locus.tools.url_safety import is_safe_url, validate_url


def _mock_resolution(ip: str) -> Any:
    """Build a monkeypatch function that maps any host to ``ip``."""

    def _fake_getaddrinfo(host: str, port: int | None, *args: Any, **kwargs: Any) -> Any:
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 0, "", (ip, port or 0))]

    return _fake_getaddrinfo


def _mock_dns_failure(host: str, *args: Any, **kwargs: Any) -> Any:
    raise socket.gaierror(-2, "Name or service not known")


# ---------------------------------------------------------------------------
# Happy path — public addresses.
# ---------------------------------------------------------------------------


class TestPublicAddresses:
    def test_public_ipv4_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution("93.184.216.34"))
        assert is_safe_url("https://example.com/") is True

    def test_public_ipv6_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution("2606:2800:220:1::248:1893"))
        assert is_safe_url("https://example.com/") is True

    def test_https_with_path_and_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution("8.8.8.8"))
        assert is_safe_url("https://dns.google/resolve?name=foo.com") is True


# ---------------------------------------------------------------------------
# Private / reserved ranges — blocked unless opt-in.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",  # loopback
        "10.0.0.1",  # RFC1918
        "192.168.1.1",  # RFC1918
        "172.16.5.3",  # RFC1918
        "100.64.0.5",  # CGNAT
        "0.0.0.0",  # noqa: S104 — unspecified (test vector, not a bind target)
        "224.0.0.1",  # multicast
        "::1",  # IPv6 loopback
        "fe80::1",  # IPv6 link-local
        "fc00::1",  # IPv6 ULA (private)
    ],
)
class TestPrivateAddressesBlocked:
    def test_default_blocks(self, ip: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOCUS_ALLOW_PRIVATE_URLS", raising=False)
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution(ip))
        assert is_safe_url("https://internal.example/") is False

    def test_opt_in_allows(self, ip: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOCUS_ALLOW_PRIVATE_URLS", raising=False)
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution(ip))
        assert is_safe_url("https://internal.example/", allow_private=True) is True

    def test_env_var_allows(self, ip: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCUS_ALLOW_PRIVATE_URLS", "true")
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution(ip))
        assert is_safe_url("https://internal.example/") is True


# ---------------------------------------------------------------------------
# Cloud metadata — blocked unconditionally, even with allow_private=True.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "169.254.169.254",  # AWS / GCP / Azure / OCI / DO IMDS
        "169.254.170.2",  # AWS ECS task role metadata
        "169.254.169.253",  # Azure IMDS wire server
        "169.254.99.99",  # arbitrary link-local — whole /16 blocked
        "100.100.100.200",  # Alibaba Cloud metadata
        "fd00:ec2::254",  # AWS IPv6 metadata
    ],
)
class TestMetadataAlwaysBlocked:
    def test_default_blocks(self, ip: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution(ip))
        assert is_safe_url("https://imds.example/") is False

    def test_allow_private_still_blocks(self, ip: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution(ip))
        assert is_safe_url("https://imds.example/", allow_private=True) is False

    def test_env_var_still_blocks(self, ip: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCUS_ALLOW_PRIVATE_URLS", "true")
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution(ip))
        assert is_safe_url("https://imds.example/") is False


class TestMetadataHostnames:
    @pytest.mark.parametrize(
        "host",
        [
            "metadata.google.internal",
            "metadata.goog",
            "METADATA.GOOGLE.INTERNAL",  # case-insensitive
            "metadata.google.internal.",  # trailing dot stripped
        ],
    )
    def test_metadata_hostname_blocked(self, host: str, monkeypatch: pytest.MonkeyPatch) -> None:
        # No DNS required — blocked on hostname alone. Patch getaddrinfo
        # to a public IP just to prove the short-circuit runs first.
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution("8.8.8.8"))
        assert is_safe_url(f"https://{host}/") is False

    def test_metadata_hostname_blocks_with_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution("8.8.8.8"))
        assert is_safe_url("https://metadata.google.internal/", allow_private=True) is False


# ---------------------------------------------------------------------------
# Degenerate / malformed input.
# ---------------------------------------------------------------------------


class TestDegenerateInput:
    def test_empty_url(self) -> None:
        assert is_safe_url("") is False

    def test_garbage_url(self) -> None:
        assert is_safe_url("not a url at all") is False

    def test_scheme_only(self) -> None:
        assert is_safe_url("https://") is False

    def test_dns_failure_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "getaddrinfo", _mock_dns_failure)
        assert is_safe_url("https://nonexistent.invalid/") is False


# ---------------------------------------------------------------------------
# Env-var parsing.
# ---------------------------------------------------------------------------


class TestEnvVarParsing:
    @pytest.mark.parametrize("val", ["true", "TRUE", "True", "1", "yes", "YES"])
    def test_truthy_values(self, val: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCUS_ALLOW_PRIVATE_URLS", val)
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution("127.0.0.1"))
        assert is_safe_url("https://h/") is True

    @pytest.mark.parametrize("val", ["false", "0", "no", "", "maybe", " "])
    def test_falsy_values(self, val: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCUS_ALLOW_PRIVATE_URLS", val)
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution("127.0.0.1"))
        assert is_safe_url("https://h/") is False

    def test_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOCUS_ALLOW_PRIVATE_URLS", raising=False)
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution("127.0.0.1"))
        assert is_safe_url("https://h/") is False


# ---------------------------------------------------------------------------
# validate_url — raises on unsafe, returns None on safe.
# ---------------------------------------------------------------------------


class TestValidateUrl:
    def test_safe_url_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution("8.8.8.8"))
        assert validate_url("https://example.com/") is None

    def test_unsafe_url_raises_validation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution("127.0.0.1"))
        with pytest.raises(ValidationError, match="SSRF guard"):
            validate_url("https://looped.example/")

    def test_metadata_url_raises_despite_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "getaddrinfo", _mock_resolution("169.254.169.254"))
        with pytest.raises(ValidationError, match="SSRF guard"):
            validate_url("https://imds.example/", allow_private=True)


# ---------------------------------------------------------------------------
# Multi-address resolution — any unsafe entry rejects the whole name.
# ---------------------------------------------------------------------------


class TestMultiAddressResolution:
    def test_rejects_if_any_address_unsafe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _multi(host: str, port: int | None, *a: Any, **kw: Any) -> Any:
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.169.254", 0)),
            ]

        monkeypatch.setattr(socket, "getaddrinfo", _multi)
        assert is_safe_url("https://dual.example/") is False


# ---------------------------------------------------------------------------
# __all__ surface stays stable.
# ---------------------------------------------------------------------------


def test_public_exports() -> None:
    assert set(url_safety.__all__) == {"is_safe_url", "validate_url"}
