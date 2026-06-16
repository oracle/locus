# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for ``locus.models.rate_limits``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from locus.models.rate_limits import (
    RateLimitBucket,
    RateLimitState,
    parse_rate_limit_headers,
)


_FIXED_NOW = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# RateLimitBucket derived properties.
# ---------------------------------------------------------------------------


class TestBucketDerived:
    def test_used_and_pct(self) -> None:
        b = RateLimitBucket(limit=100, remaining=40, reset_seconds=60, captured_at=_FIXED_NOW)
        assert b.used == 60
        assert b.usage_pct == 60.0

    def test_used_clamps_to_zero_when_remaining_exceeds_limit(self) -> None:
        # Providers sometimes return remaining > limit mid-reset.
        b = RateLimitBucket(limit=50, remaining=55, reset_seconds=30, captured_at=_FIXED_NOW)
        assert b.used == 0

    def test_pct_with_zero_limit(self) -> None:
        b = RateLimitBucket(limit=0, remaining=0, reset_seconds=0, captured_at=_FIXED_NOW)
        assert b.usage_pct == 0.0

    def test_seconds_until_reset_shrinks_with_elapsed_time(self) -> None:
        b = RateLimitBucket(limit=10, remaining=5, reset_seconds=60, captured_at=_FIXED_NOW)
        # 20 seconds later — expect ~40s remaining.
        later = _FIXED_NOW + timedelta(seconds=20)
        assert b.seconds_until_reset(now=later) == pytest.approx(40.0, abs=0.001)

    def test_seconds_until_reset_clamps_to_zero(self) -> None:
        b = RateLimitBucket(limit=10, remaining=5, reset_seconds=10, captured_at=_FIXED_NOW)
        later = _FIXED_NOW + timedelta(seconds=300)
        assert b.seconds_until_reset(now=later) == 0.0

    def test_reset_at_is_capture_plus_reset(self) -> None:
        b = RateLimitBucket(limit=10, remaining=5, reset_seconds=90, captured_at=_FIXED_NOW)
        assert b.reset_at() == _FIXED_NOW + timedelta(seconds=90)


# ---------------------------------------------------------------------------
# Frozen guarantees.
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_bucket_frozen(self) -> None:
        b = RateLimitBucket(captured_at=_FIXED_NOW)
        with pytest.raises(ValidationError, match="frozen"):
            b.limit = 99

    def test_state_frozen(self) -> None:
        s = RateLimitState(captured_at=_FIXED_NOW)
        with pytest.raises(ValidationError, match="frozen"):
            s.provider = "other"


# ---------------------------------------------------------------------------
# Header parsing — happy path.
# ---------------------------------------------------------------------------


class TestHeaderParsing:
    def test_all_four_buckets(self) -> None:
        state = parse_rate_limit_headers(
            {
                "x-ratelimit-limit-requests": "5000",
                "x-ratelimit-remaining-requests": "4999",
                "x-ratelimit-reset-requests": "30",
                "x-ratelimit-limit-requests-1h": "100000",
                "x-ratelimit-remaining-requests-1h": "99500",
                "x-ratelimit-reset-requests-1h": "3500",
                "x-ratelimit-limit-tokens": "150000",
                "x-ratelimit-remaining-tokens": "149900",
                "x-ratelimit-reset-tokens": "45",
                "x-ratelimit-limit-tokens-1h": "5000000",
                "x-ratelimit-remaining-tokens-1h": "4999000",
                "x-ratelimit-reset-tokens-1h": "3200",
            },
            provider="openrouter",
            now=_FIXED_NOW,
        )
        assert state is not None
        assert state.provider == "openrouter"
        assert state.captured_at == _FIXED_NOW
        assert state.requests_min is not None
        assert state.requests_min.limit == 5000
        assert state.requests_min.remaining == 4999
        assert state.requests_min.reset_seconds == 30.0
        assert state.tokens_hour is not None
        assert state.tokens_hour.limit == 5_000_000
        assert state.tokens_hour.reset_seconds == 3200.0

    def test_partial_headers_yields_partial_state(self) -> None:
        state = parse_rate_limit_headers(
            {
                "x-ratelimit-limit-requests": "60",
                "x-ratelimit-remaining-requests": "59",
                "x-ratelimit-reset-requests": "1",
            },
            now=_FIXED_NOW,
        )
        assert state is not None
        assert state.requests_min is not None
        assert state.requests_hour is None
        assert state.tokens_min is None
        assert state.tokens_hour is None

    def test_case_insensitive_header_lookup(self) -> None:
        state = parse_rate_limit_headers(
            {
                "X-RateLimit-Limit-Requests": "60",
                "X-RATELIMIT-REMAINING-REQUESTS": "59",
                "x-Ratelimit-Reset-Requests": "1",
            },
            now=_FIXED_NOW,
        )
        assert state is not None
        assert state.requests_min is not None
        assert state.requests_min.limit == 60

    def test_no_ratelimit_headers_returns_none(self) -> None:
        assert parse_rate_limit_headers({"content-type": "application/json"}) is None
        assert parse_rate_limit_headers({}) is None


# ---------------------------------------------------------------------------
# Reset-value duration parsing (OpenAI's "1m60s" / "200ms" convention).
# ---------------------------------------------------------------------------


class TestDurationParsing:
    @pytest.mark.parametrize(
        ("raw", "seconds"),
        [
            ("60", 60.0),
            ("60s", 60.0),
            ("1m", 60.0),
            ("1m60s", 120.0),
            ("1m30s", 90.0),
            ("200ms", 0.2),
            ("1m200ms", 60.2),
            ("1.5s", 1.5),
            ("0", 0.0),
            ("", 0.0),
        ],
    )
    def test_parses_reset_durations(self, raw: str, seconds: float) -> None:
        state = parse_rate_limit_headers(
            {
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-remaining-requests": "50",
                "x-ratelimit-reset-requests": raw,
            },
            now=_FIXED_NOW,
        )
        assert state is not None
        assert state.requests_min is not None
        assert state.requests_min.reset_seconds == pytest.approx(seconds, abs=0.001)

    def test_unparseable_reset_falls_back_to_zero(self) -> None:
        state = parse_rate_limit_headers(
            {
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-remaining-requests": "50",
                "x-ratelimit-reset-requests": "whenever",
            },
            now=_FIXED_NOW,
        )
        assert state is not None
        assert state.requests_min is not None
        assert state.requests_min.reset_seconds == 0.0


# ---------------------------------------------------------------------------
# Malformed numeric values should not crash.
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_non_numeric_limit_becomes_zero(self) -> None:
        state = parse_rate_limit_headers(
            {
                "x-ratelimit-limit-requests": "garbage",
                "x-ratelimit-remaining-requests": "10",
                "x-ratelimit-reset-requests": "5",
            },
            now=_FIXED_NOW,
        )
        assert state is not None
        assert state.requests_min is not None
        assert state.requests_min.limit == 0

    def test_negative_limit_clamped(self) -> None:
        state = parse_rate_limit_headers(
            {
                "x-ratelimit-limit-requests": "-50",
                "x-ratelimit-remaining-requests": "-10",
                "x-ratelimit-reset-requests": "-5",
            },
            now=_FIXED_NOW,
        )
        assert state is not None
        assert state.requests_min is not None
        assert state.requests_min.limit == 0
        assert state.requests_min.remaining == 0
        assert state.requests_min.reset_seconds == 0.0


# ---------------------------------------------------------------------------
# has_any_bucket / age_seconds.
# ---------------------------------------------------------------------------


class TestStateAccessors:
    def test_has_any_bucket_false_when_all_none(self) -> None:
        s = RateLimitState(captured_at=_FIXED_NOW)
        assert s.has_any_bucket is False

    def test_has_any_bucket_true_with_one_bucket(self) -> None:
        state = parse_rate_limit_headers(
            {
                "x-ratelimit-limit-requests": "1",
                "x-ratelimit-remaining-requests": "1",
                "x-ratelimit-reset-requests": "1",
            },
            now=_FIXED_NOW,
        )
        assert state is not None
        assert state.has_any_bucket is True

    def test_age_seconds(self) -> None:
        s = RateLimitState(captured_at=_FIXED_NOW)
        later = _FIXED_NOW + timedelta(seconds=42)
        assert s.age_seconds(now=later) == pytest.approx(42.0, abs=0.001)


# ---------------------------------------------------------------------------
# JSON round-trip stays stable (captured_at serialises as ISO-8601 UTC).
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_state_roundtrip(self) -> None:
        state = parse_rate_limit_headers(
            {
                "x-ratelimit-limit-requests": "60",
                "x-ratelimit-remaining-requests": "59",
                "x-ratelimit-reset-requests": "1",
            },
            now=_FIXED_NOW,
        )
        assert state is not None
        blob = state.model_dump_json()
        restored = RateLimitState.model_validate_json(blob)
        assert restored == state
