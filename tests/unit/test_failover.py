# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for ``locus.models.failover.classify`` and its taxonomy."""

from __future__ import annotations

import ssl
from typing import Any

import pytest

from locus.models.failover import FailoverDecision, FailoverReason, classify


class _HTTPError(Exception):
    """Stand-in for an SDK exception that carries an HTTP status code + body."""

    def __init__(
        self,
        message: str = "",
        *,
        status_code: int | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        if body is not None:
            self.body = body


# ---------------------------------------------------------------------------
# Status-code fast path.
# ---------------------------------------------------------------------------


class TestStatusCodeRouting:
    @pytest.mark.parametrize(
        ("status", "reason"),
        [
            (401, FailoverReason.AUTH_TRANSIENT),
            (403, FailoverReason.AUTH_PERMANENT),
            (429, FailoverReason.RATE_LIMIT),
            (500, FailoverReason.SERVER_ERROR),
            (502, FailoverReason.SERVER_ERROR),
            (503, FailoverReason.OVERLOADED),
            (529, FailoverReason.OVERLOADED),
            (413, FailoverReason.PAYLOAD_TOO_LARGE),
        ],
    )
    def test_status_maps_to_reason(self, status: int, reason: FailoverReason) -> None:
        decision = classify(_HTTPError("x", status_code=status))
        assert decision.reason is reason
        assert decision.status_code == status

    def test_403_key_limit_is_billing(self) -> None:
        decision = classify(
            _HTTPError("key limit exceeded on this OpenRouter key", status_code=403)
        )
        assert decision.reason is FailoverReason.BILLING
        assert decision.should_rotate_credential is True

    def test_404_with_model_not_found(self) -> None:
        decision = classify(_HTTPError("The model 'gpt-999' is not a valid model", status_code=404))
        assert decision.reason is FailoverReason.MODEL_NOT_FOUND
        assert decision.retryable is False
        assert decision.should_fallback is True

    def test_404_without_model_signal_is_unknown(self) -> None:
        # Could be misconfigured endpoint path — don't falsely claim the
        # model is missing.
        decision = classify(_HTTPError("not found", status_code=404))
        assert decision.reason is FailoverReason.UNKNOWN
        assert decision.retryable is True

    def test_other_4xx_is_format_error(self) -> None:
        decision = classify(_HTTPError("Teapot", status_code=418))
        assert decision.reason is FailoverReason.FORMAT_ERROR
        assert decision.retryable is False
        assert decision.should_fallback is True


# ---------------------------------------------------------------------------
# 402 disambiguation.
# ---------------------------------------------------------------------------


class TestPayment402:
    def test_plain_billing(self) -> None:
        decision = classify(_HTTPError("Insufficient credits", status_code=402))
        assert decision.reason is FailoverReason.BILLING
        assert decision.retryable is False
        assert decision.should_rotate_credential is True

    def test_transient_usage_limit(self) -> None:
        decision = classify(
            _HTTPError(
                "Usage limit exceeded, try again in 5 minutes",
                status_code=402,
            )
        )
        assert decision.reason is FailoverReason.RATE_LIMIT
        assert decision.retryable is True


# ---------------------------------------------------------------------------
# 400 disambiguation: context overflow vs format error.
# ---------------------------------------------------------------------------


class TestBadRequest400:
    def test_explicit_context_overflow(self) -> None:
        decision = classify(
            _HTTPError(
                "This model's maximum context length is 4097 tokens",
                status_code=400,
            )
        )
        assert decision.reason is FailoverReason.CONTEXT_TOO_LONG
        assert decision.should_compress is True

    def test_generic_400_with_small_context_is_format(self) -> None:
        decision = classify(
            _HTTPError("Bad request", status_code=400),
            approx_tokens=500,
            num_messages=4,
        )
        assert decision.reason is FailoverReason.FORMAT_ERROR
        assert decision.retryable is False

    def test_generic_400_with_large_session_is_context_overflow(self) -> None:
        decision = classify(
            _HTTPError("", status_code=400, body={"error": {"message": "Error"}}),
            approx_tokens=150_000,
            context_length=200_000,
            num_messages=120,
        )
        assert decision.reason is FailoverReason.CONTEXT_TOO_LONG
        assert decision.should_compress is True

    def test_400_rate_limit_masquerading(self) -> None:
        decision = classify(
            _HTTPError(
                "Too many requests, please retry after 30s",
                status_code=400,
            )
        )
        assert decision.reason is FailoverReason.RATE_LIMIT

    def test_400_billing_masquerading(self) -> None:
        decision = classify(
            _HTTPError(
                "Your credit balance is too low",
                status_code=400,
            )
        )
        assert decision.reason is FailoverReason.BILLING


# ---------------------------------------------------------------------------
# Structured error codes.
# ---------------------------------------------------------------------------


class TestStructuredErrorCode:
    @pytest.mark.parametrize(
        ("code", "reason"),
        [
            ("resource_exhausted", FailoverReason.RATE_LIMIT),
            ("throttled", FailoverReason.RATE_LIMIT),
            ("insufficient_quota", FailoverReason.BILLING),
            ("billing_not_active", FailoverReason.BILLING),
            ("model_not_found", FailoverReason.MODEL_NOT_FOUND),
            ("context_length_exceeded", FailoverReason.CONTEXT_TOO_LONG),
            ("max_tokens_exceeded", FailoverReason.CONTEXT_TOO_LONG),
        ],
    )
    def test_error_code_routing(self, code: str, reason: FailoverReason) -> None:
        exc = _HTTPError("generic", body={"error": {"code": code}})
        assert classify(exc).reason is reason


# ---------------------------------------------------------------------------
# No status code — message-pattern classification.
# ---------------------------------------------------------------------------


class TestMessagePatterns:
    def test_bare_rate_limit_message(self) -> None:
        decision = classify(Exception("You have been rate limited"))
        assert decision.reason is FailoverReason.RATE_LIMIT

    def test_bare_auth_message(self) -> None:
        decision = classify(Exception("invalid api key"))
        assert decision.reason is FailoverReason.AUTH_TRANSIENT
        assert decision.should_rotate_credential is True

    def test_context_overflow_no_status(self) -> None:
        decision = classify(Exception("The prompt is too long"))
        assert decision.reason is FailoverReason.CONTEXT_TOO_LONG
        assert decision.should_compress is True

    def test_payload_too_large_no_status(self) -> None:
        decision = classify(Exception("Request entity too large"))
        assert decision.reason is FailoverReason.PAYLOAD_TOO_LARGE
        assert decision.should_compress is True

    def test_billing_message(self) -> None:
        decision = classify(Exception("Credits have been exhausted"))
        assert decision.reason is FailoverReason.BILLING


# ---------------------------------------------------------------------------
# Transport / timeout fallbacks.
# ---------------------------------------------------------------------------


class TestTransportErrors:
    def test_python_timeout(self) -> None:
        decision = classify(TimeoutError("read timed out"))
        assert decision.reason is FailoverReason.TIMEOUT

    def test_connection_error(self) -> None:
        decision = classify(ConnectionError("connection refused"))
        assert decision.reason is FailoverReason.TIMEOUT

    def test_ssl_alert_not_treated_as_disconnect(self) -> None:
        decision = classify(ssl.SSLError("SSLV3_ALERT_BAD_RECORD_MAC"))
        assert decision.reason is FailoverReason.TIMEOUT
        # Key point: no compression triggered on flaky TLS.
        assert decision.should_compress is False

    def test_disconnect_small_session_is_timeout(self) -> None:
        decision = classify(
            Exception("Server disconnected without response"),
            approx_tokens=1_000,
            num_messages=3,
        )
        assert decision.reason is FailoverReason.TIMEOUT

    def test_disconnect_large_session_is_context_overflow(self) -> None:
        decision = classify(
            Exception("Server disconnected"),
            approx_tokens=180_000,
            context_length=200_000,
            num_messages=50,
        )
        assert decision.reason is FailoverReason.CONTEXT_TOO_LONG
        assert decision.should_compress is True


# ---------------------------------------------------------------------------
# Cause chain + extraction.
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_status_from_cause_chain(self) -> None:
        inner = _HTTPError("inner", status_code=429)
        outer = RuntimeError("wrapped")
        outer.__cause__ = inner
        decision = classify(outer)
        assert decision.reason is FailoverReason.RATE_LIMIT
        assert decision.status_code == 429

    def test_body_from_response_json(self) -> None:
        class _Response:
            @staticmethod
            def json() -> dict[str, Any]:
                return {"error": {"code": "resource_exhausted"}}

        class _ExcError(Exception):
            response = _Response()

        decision = classify(_ExcError("boom"))
        assert decision.reason is FailoverReason.RATE_LIMIT

    def test_metadata_raw_unwrapping(self) -> None:
        # OpenRouter-style wrapper: the real error is inside metadata.raw.
        body = {
            "error": {
                "message": "Provider returned error",
                "metadata": {"raw": '{"error": {"message": "context length exceeded"}}'},
            }
        }
        decision = classify(_HTTPError("generic", body=body))
        assert decision.reason is FailoverReason.CONTEXT_TOO_LONG
        assert decision.should_compress is True

    def test_rate_limit_sdk_without_status(self) -> None:
        # Some SDKs raise RateLimitError without .status_code attribute.
        class RateLimitError(Exception):
            pass

        decision = classify(RateLimitError("throttle"))
        assert decision.reason is FailoverReason.RATE_LIMIT
        assert decision.status_code == 429


# ---------------------------------------------------------------------------
# Override knobs.
# ---------------------------------------------------------------------------


class TestOverrides:
    def test_status_override_wins_over_attr(self) -> None:
        exc = _HTTPError("", status_code=500)
        decision = classify(exc, status=429)
        assert decision.reason is FailoverReason.RATE_LIMIT

    def test_body_override_wins_over_attr(self) -> None:
        # When no status code is present, structured error codes drive
        # the decision and the ``body=`` kwarg overrides the one on the
        # exception.
        exc = _HTTPError("", body={"error": {"code": "format_bad"}})
        decision = classify(
            exc,
            body={"error": {"code": "context_length_exceeded"}},
        )
        assert decision.reason is FailoverReason.CONTEXT_TOO_LONG


# ---------------------------------------------------------------------------
# Decision object is frozen + serializable.
# ---------------------------------------------------------------------------


class TestDecisionObject:
    def test_frozen(self) -> None:
        from pydantic import ValidationError

        decision = classify(TimeoutError("x"))
        with pytest.raises(ValidationError, match="frozen"):
            decision.reason = FailoverReason.UNKNOWN

    def test_reason_is_str_mixin(self) -> None:
        # FailoverReason values must be usable as JSON / log keys without
        # explicit .value access.
        assert FailoverReason.RATE_LIMIT == "rate_limit"
        assert str(FailoverReason.BILLING.value) == "billing"

    def test_roundtrip_json(self) -> None:
        decision = classify(_HTTPError("", status_code=429))
        blob = decision.model_dump_json()
        restored = FailoverDecision.model_validate_json(blob)
        assert restored == decision


# ---------------------------------------------------------------------------
# Unknown + retryable default.
# ---------------------------------------------------------------------------


class TestFallback:
    def test_completely_unknown_is_retryable(self) -> None:
        decision = classify(Exception("weird unclassifiable message"))
        assert decision.reason is FailoverReason.UNKNOWN
        assert decision.retryable is True
