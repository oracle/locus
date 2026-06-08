# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Tests for secret redaction patterns in ``tulip.tools.executor``.

Covers both the existing ``_sanitize_error`` surface (tool-execution errors)
and the broader ``redact_sensitive_text`` helper introduced alongside the
vendor-prefix / JWT / bearer / URL-query patterns.
"""

from __future__ import annotations

import pytest

from tulip.tools.executor import (
    _sanitize_error,
    redact_sensitive_text,
)


# ---------------------------------------------------------------------------
# Existing patterns still work (regression guard).
# ---------------------------------------------------------------------------


class TestExistingPatterns:
    def test_postgres_url_redacted(self) -> None:
        text = "connect failed: postgresql://user:pw@db.internal/prod"
        assert "postgresql://" not in redact_sensitive_text(text)
        assert "[REDACTED]" in redact_sensitive_text(text)

    def test_redis_url_redacted(self) -> None:
        assert "[REDACTED]" in redact_sensitive_text("redis://u:p@r.example:6379/0")

    def test_password_assignment_redacted(self) -> None:
        assert "[REDACTED]" in redact_sensitive_text("Connection failed (password='hunter2')")

    def test_home_path_redacted(self) -> None:
        assert "[REDACTED]" in redact_sensitive_text("no such file: /Users/alice/secrets.txt")


# ---------------------------------------------------------------------------
# Vendor-prefix API keys.
# ---------------------------------------------------------------------------


class TestVendorPrefixPatterns:
    @pytest.mark.parametrize(
        "token",
        [
            "sk-ant-api03-" + "FAKEFIXTUREabcdefghijklmnopqrstuvwxyz0123456",
            "sk-proj-" + "FAKEFIXTUREabcdefghijklmnopqrstuvwxyz0123456789ABCDEF",
            "sk-"
            "FAKEFIXTUREabcdefghijklmnopqrstuvwxyz0123",  # ≥32 chars; split to defeat secret scanners
            "AKIAIOSFODNN7EXAMPLE",
            "AIza" + "SyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY",  # gitleaks:allow (test fixture)
            "ghp_abcdefghijklmnop0123456789",
            "github_pat_11ABCDEFG_abcdefghijklmnopqrstuvwxyz",
        ],
    )
    def test_vendor_key_is_masked(self, token: str) -> None:
        text = f"call failed with key {token}"
        out = redact_sensitive_text(text)
        assert token not in out
        # Long tokens keep first 6 + last 4 for debuggability.
        assert token[:6] in out
        assert token[-4:] in out
        assert "..." in out

    def test_short_token_does_not_match_vendor_prefix(self) -> None:
        # Too short to be a real key — must not falsely trigger the prefix
        # alternation (the regex minimums guard this).
        text = "trivial string sk-abc"
        assert redact_sensitive_text(text) == text

    def test_prefix_not_matched_inside_larger_token(self) -> None:
        # A random token that happens to contain 'sk-' somewhere in the
        # middle should not be redacted by the vendor-prefix rule. The
        # boundary lookarounds guard this.
        text = "the value XYZsk-ant-abcdefghijklmnopqrstuvwxyz was rejected"
        # The embedded 'sk-ant-...' is not at a word boundary, so the
        # vendor pattern shouldn't fire. Note: other patterns may still
        # redact, but we're asserting the boundary behaviour holds.
        out = redact_sensitive_text(text)
        # If the prefix rule fires, mask substitution produces "..." —
        # but we expect the raw substring to survive here.
        assert "XYZsk-ant-" in out


# ---------------------------------------------------------------------------
# JWT tokens.
# ---------------------------------------------------------------------------


class TestJwtPattern:
    def test_three_part_jwt_masked(self) -> None:
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
            ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        # Phrasing that doesn't trigger the bare-assignment ``token=`` rule
        # so the JWT-specific mask is exercised.
        out = redact_sensitive_text(f"issued jwt {jwt} to caller")
        assert jwt not in out
        assert "eyJhbG" in out  # long-token debuggable prefix preserved
        assert "..." in out

    def test_two_part_jwt_masked(self) -> None:
        # Header + payload only — still begins with eyJ.
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjMifQ"
        out = redact_sensitive_text(jwt)
        assert jwt not in out


# ---------------------------------------------------------------------------
# HTTP Authorization: Bearer ...
# ---------------------------------------------------------------------------


class TestBearerPattern:
    def test_bearer_header_token_masked(self) -> None:
        fake = "abcdefghijklmnopqrstuvwxyz0123456789"  # gitleaks:allow (test fixture)
        text = f"curl -H 'Authorization: Bearer {fake}' ..."
        out = redact_sensitive_text(text)
        assert "abcdefghijklmnopqrstuvwxyz0123456789" not in out
        # Header name must be preserved.
        assert "Authorization: Bearer" in out

    def test_bearer_header_case_insensitive(self) -> None:
        text = "authorization: bearer 0123456789abcdef0123456789abcdef"
        assert "0123456789abcdef0123456789abcdef" not in redact_sensitive_text(text)


# ---------------------------------------------------------------------------
# URL query-string tokens.
# ---------------------------------------------------------------------------


class TestUrlQueryPattern:
    def test_access_token_redacted_url_preserved(self) -> None:
        url = "https://api.example.com/v1/cb?code=abc123&access_token=OPAQUE_TOKEN_VALUE&state=xyz"
        out = redact_sensitive_text(url)
        # Path + host must survive.
        assert "https://api.example.com/v1/cb?" in out
        # Non-sensitive params (state) must survive.
        assert "state=xyz" in out
        # Sensitive values (code, access_token) must be gone.
        assert "OPAQUE_TOKEN_VALUE" not in out
        assert "abc123" not in out

    def test_fragment_preserved(self) -> None:
        url = "https://x.example/p?token=SECRETVAL#section-3"
        out = redact_sensitive_text(url)
        assert "#section-3" in out
        assert "SECRETVAL" not in out

    def test_non_sensitive_query_passes_through(self) -> None:
        url = "https://docs.example/page?q=hello&lang=en"
        assert redact_sensitive_text(url) == url


# ---------------------------------------------------------------------------
# _mask_token short-vs-long behaviour.
# ---------------------------------------------------------------------------


class TestMaskingBoundary:
    def test_short_token_fully_redacted(self) -> None:
        # 17-char Anthropic-style token — below the 18-char debuggability
        # threshold. Note: this test calls the mask helper indirectly via a
        # JWT-like shape that happens to be short; in practice no real secret
        # is this short, so "short" falls through to [REDACTED].
        text = "Authorization: Bearer abc123xyz"  # 9 chars of token
        out = redact_sensitive_text(text)
        assert "[REDACTED]" in out


# ---------------------------------------------------------------------------
# _sanitize_error (caller-visible behaviour: first-line only).
# ---------------------------------------------------------------------------


class TestSanitizeError:
    def test_first_line_only_preserved(self) -> None:
        err = "boom\nsensitive detail: password='hunter2'"
        out = _sanitize_error(err)
        # Multi-line content is dropped entirely — and so are its secrets.
        assert "password" not in out
        assert "hunter2" not in out
        assert out.startswith("boom")

    def test_first_line_secret_redacted(self) -> None:
        err = "HTTP 401: Authorization: Bearer sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        out = _sanitize_error(err)
        assert "sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaaaaaa" not in out

    def test_combined_url_preserved_only_token_redacted(self) -> None:
        err = "fetch failed: https://api.example.com/data?access_token=ZZZSECRETZZZ&user=42"
        out = _sanitize_error(err)
        assert "https://api.example.com/data?" in out
        assert "user=42" in out
        assert "ZZZSECRETZZZ" not in out
