# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Extra coverage for ``locus.models.failover``.

The base ``test_failover.py`` covers the headline status-code paths.
This file fills in:

- 402 disambiguation (transient usage limit vs hard billing)
- 403 → BILLING when the message indicates spending limit
- 400 generic-error / large-session heuristic → CONTEXT_TOO_LONG
- 4xx generic format error
- Rate-limit detection without a status code (RateLimitError class
  name)
- Body-extraction via ``response.json()`` fallback
- Error-code extraction precedence (``error.code`` >
  ``error.type`` > top-level ``code`` > ``error_code``)
- Message pipeline patterns (payload-too-large, billing, context
  overflow, auth, model-not-found)
- SSL transient → TIMEOUT (precedence over disconnect heuristic)
- Server-disconnect + large session → CONTEXT_TOO_LONG
- Transport-error fallback (OSError)
- Final UNKNOWN bucket
"""

from __future__ import annotations

from typing import Any

from locus.models.failover import FailoverReason, classify


# ---------------------------------------------------------------------------
# Synthetic exceptions
# ---------------------------------------------------------------------------


class _StatusError(Exception):
    def __init__(self, msg: str, status_code: int) -> None:
        super().__init__(msg)
        self.status_code = status_code


class _BodyError(Exception):
    def __init__(self, msg: str, body: dict[str, Any]) -> None:
        super().__init__(msg)
        self.body = body


class RateLimitError(Exception):
    """The class name alone is enough to imply 429 inside ``classify``."""


class _RespBodyError(Exception):
    """Carries a ``response.json()`` payload."""

    def __init__(self, msg: str, body: dict[str, Any]) -> None:
        super().__init__(msg)

        class _R:
            def __init__(self, b: dict[str, Any]) -> None:
                self._b = b

            def json(self) -> dict[str, Any]:
                return self._b

        self.response = _R(body)


# ---------------------------------------------------------------------------
# 402 disambiguation
# ---------------------------------------------------------------------------


class TestStatus402:
    def test_transient_usage_limit_routes_to_rate_limit(self) -> None:
        decision = classify(
            _StatusError("monthly quota — try again later", 402),
            status=402,
        )
        assert decision.reason == FailoverReason.RATE_LIMIT
        assert decision.retryable is True

    def test_hard_billing_lock_stays_billing(self) -> None:
        decision = classify(
            _StatusError("billing not active for this account", 402),
            status=402,
        )
        assert decision.reason == FailoverReason.BILLING


# ---------------------------------------------------------------------------
# 403 spending-limit branch
# ---------------------------------------------------------------------------


class TestStatus403:
    def test_spending_limit_routes_to_billing(self) -> None:
        decision = classify(
            _StatusError("spending limit exceeded", 403),
            status=403,
        )
        assert decision.reason == FailoverReason.BILLING

    def test_key_limit_routes_to_billing(self) -> None:
        decision = classify(
            _StatusError("key limit exceeded", 403),
            status=403,
        )
        assert decision.reason == FailoverReason.BILLING

    def test_plain_403_routes_to_auth_permanent(self) -> None:
        decision = classify(
            _StatusError("forbidden", 403),
            status=403,
        )
        assert decision.reason == FailoverReason.AUTH_PERMANENT


# ---------------------------------------------------------------------------
# 400 routing
# ---------------------------------------------------------------------------


class TestStatus400:
    def test_context_overflow_pattern(self) -> None:
        decision = classify(
            _StatusError("context length exceeded for the model", 400),
            status=400,
        )
        assert decision.reason == FailoverReason.CONTEXT_TOO_LONG
        assert decision.should_compress is True

    def test_model_not_found_pattern(self) -> None:
        decision = classify(
            _StatusError("invalid model identifier", 400),
            status=400,
        )
        assert decision.reason == FailoverReason.MODEL_NOT_FOUND

    def test_rate_limit_pattern(self) -> None:
        decision = classify(
            _StatusError("rate limit reached for this minute", 400),
            status=400,
        )
        assert decision.reason == FailoverReason.RATE_LIMIT

    def test_billing_pattern(self) -> None:
        decision = classify(
            _StatusError("insufficient credits remaining", 400),
            status=400,
        )
        assert decision.reason == FailoverReason.BILLING

    def test_generic_large_session_routes_to_context(self) -> None:
        decision = classify(
            _StatusError("error", 400),
            status=400,
            approx_tokens=100_000,
            context_length=200_000,
            num_messages=50,
        )
        assert decision.reason == FailoverReason.CONTEXT_TOO_LONG

    def test_generic_small_session_falls_to_format_error(self) -> None:
        decision = classify(
            _StatusError("error", 400),
            status=400,
            approx_tokens=100,
            num_messages=5,
        )
        assert decision.reason == FailoverReason.FORMAT_ERROR


class TestStatus4xx:
    def test_generic_4xx_format_error(self) -> None:
        decision = classify(
            _StatusError("teapot", 418),
            status=418,
        )
        assert decision.reason == FailoverReason.FORMAT_ERROR


class TestStatus5xx:
    def test_503_routes_to_overloaded(self) -> None:
        decision = classify(
            _StatusError("service unavailable", 503),
            status=503,
        )
        assert decision.reason == FailoverReason.OVERLOADED

    def test_529_routes_to_overloaded(self) -> None:
        decision = classify(
            _StatusError("anthropic overloaded", 529),
            status=529,
        )
        assert decision.reason == FailoverReason.OVERLOADED

    def test_generic_5xx_server_error(self) -> None:
        decision = classify(
            _StatusError("internal server error", 599),
            status=599,
        )
        assert decision.reason == FailoverReason.SERVER_ERROR


class TestStatus404ModelNotFound:
    def test_404_model_pattern(self) -> None:
        decision = classify(
            _StatusError("model 'gpt-99' does not exist", 404),
            status=404,
        )
        assert decision.reason == FailoverReason.MODEL_NOT_FOUND

    def test_404_other_routes_to_unknown(self) -> None:
        decision = classify(
            _StatusError("not found", 404),
            status=404,
        )
        assert decision.reason == FailoverReason.UNKNOWN


# ---------------------------------------------------------------------------
# RateLimitError without status
# ---------------------------------------------------------------------------


class TestImpliedStatus429:
    def test_rate_limit_error_class_implies_429(self) -> None:
        # The exception class name alone routes to 429 — message stays
        # opaque so it doesn't hit a different pattern.
        decision = classify(RateLimitError("upstream blocked the call"))
        assert decision.reason == FailoverReason.RATE_LIMIT


# ---------------------------------------------------------------------------
# Body extraction
# ---------------------------------------------------------------------------


class TestBodyExtraction:
    def test_body_attr_used_directly(self) -> None:
        decision = classify(
            _BodyError("upstream", {"error": {"code": "rate_limit_exceeded"}}),
        )
        assert decision.reason == FailoverReason.RATE_LIMIT

    def test_response_json_fallback(self) -> None:
        decision = classify(
            _RespBodyError("upstream", {"error": {"code": "model_not_found"}}),
        )
        assert decision.reason == FailoverReason.MODEL_NOT_FOUND

    def test_response_json_raises_falls_through(self) -> None:
        # ``.response.json()`` raises → body stays empty, classification
        # falls through to the message pipeline (no patterns match → UNKNOWN).
        class _BadResp:
            def json(self) -> dict[str, Any]:
                raise RuntimeError("not json")

        class _BadRespError(Exception):
            response = _BadResp()

        decision = classify(_BadRespError("opaque"))
        assert decision.reason == FailoverReason.UNKNOWN


# ---------------------------------------------------------------------------
# Error-code precedence
# ---------------------------------------------------------------------------


class TestErrorCodes:
    def test_throttled_routes_to_rate_limit(self) -> None:
        decision = classify(
            _BodyError("upstream", {"error": {"code": "throttled"}}),
        )
        assert decision.reason == FailoverReason.RATE_LIMIT

    def test_resource_exhausted_routes_to_rate_limit(self) -> None:
        decision = classify(
            _BodyError("upstream", {"error": {"code": "resource_exhausted"}}),
        )
        assert decision.reason == FailoverReason.RATE_LIMIT

    def test_invalid_model_routes_to_model_not_found(self) -> None:
        decision = classify(
            _BodyError("upstream", {"error": {"code": "invalid_model"}}),
        )
        assert decision.reason == FailoverReason.MODEL_NOT_FOUND

    def test_max_tokens_exceeded_routes_to_context(self) -> None:
        decision = classify(
            _BodyError("upstream", {"error": {"code": "max_tokens_exceeded"}}),
        )
        assert decision.reason == FailoverReason.CONTEXT_TOO_LONG

    def test_top_level_code_field_extracted(self) -> None:
        decision = classify(
            _BodyError("upstream", {"code": "throttled"}),
        )
        assert decision.reason == FailoverReason.RATE_LIMIT

    def test_top_level_error_code_field_extracted(self) -> None:
        decision = classify(
            _BodyError("upstream", {"error_code": "billing_not_active"}),
        )
        assert decision.reason == FailoverReason.BILLING


# ---------------------------------------------------------------------------
# Message pipeline (no status / no error code)
# ---------------------------------------------------------------------------


class TestMessagePipeline:
    def test_payload_too_large_pattern(self) -> None:
        decision = classify(Exception("payload too large for upstream"))
        assert decision.reason == FailoverReason.PAYLOAD_TOO_LARGE
        assert decision.should_compress is True

    def test_usage_limit_transient(self) -> None:
        decision = classify(Exception("usage limit reached, try again in 1 hour"))
        assert decision.reason == FailoverReason.RATE_LIMIT

    def test_usage_limit_hard_billing(self) -> None:
        decision = classify(Exception("usage limit reached for billing cycle"))
        assert decision.reason == FailoverReason.BILLING

    def test_context_overflow_message(self) -> None:
        decision = classify(Exception("context length exceeded for the request"))
        assert decision.reason == FailoverReason.CONTEXT_TOO_LONG

    def test_auth_pattern_message(self) -> None:
        decision = classify(Exception("invalid api key"))
        assert decision.reason == FailoverReason.AUTH_TRANSIENT

    def test_model_not_found_message(self) -> None:
        decision = classify(Exception("invalid model name"))
        assert decision.reason == FailoverReason.MODEL_NOT_FOUND


# ---------------------------------------------------------------------------
# SSL / disconnect heuristics
# ---------------------------------------------------------------------------


class TestSslAndDisconnect:
    def test_ssl_transient_routes_to_timeout(self) -> None:
        decision = classify(Exception("[SSL: bad record mac]"))
        assert decision.reason == FailoverReason.TIMEOUT

    def test_disconnect_with_large_session_routes_to_context(self) -> None:
        decision = classify(
            Exception("server disconnected without sending a response"),
            approx_tokens=200_000,
            context_length=200_000,
        )
        assert decision.reason == FailoverReason.CONTEXT_TOO_LONG

    def test_disconnect_with_small_session_routes_to_timeout(self) -> None:
        decision = classify(
            Exception("server disconnected without sending a response"),
            approx_tokens=100,
            num_messages=2,
        )
        assert decision.reason == FailoverReason.TIMEOUT


# ---------------------------------------------------------------------------
# Transport + UNKNOWN tail
# ---------------------------------------------------------------------------


class TestTransportTail:
    def test_oserror_routes_to_timeout(self) -> None:
        decision = classify(OSError("connection refused"))
        assert decision.reason == FailoverReason.TIMEOUT

    def test_timeouterror_routes_to_timeout(self) -> None:
        decision = classify(TimeoutError("read timed out"))
        assert decision.reason == FailoverReason.TIMEOUT

    def test_unknown_exception_with_no_signals(self) -> None:
        decision = classify(ValueError("opaque"))
        assert decision.reason == FailoverReason.UNKNOWN
        assert decision.retryable is True


# ---------------------------------------------------------------------------
# Status-code chain extraction
# ---------------------------------------------------------------------------


class TestStatusChainExtraction:
    def test_walks_cause_chain(self) -> None:
        inner = _StatusError("rate limited", 429)
        outer = Exception("wrapping")
        outer.__cause__ = inner
        decision = classify(outer)
        assert decision.reason == FailoverReason.RATE_LIMIT

    def test_falls_through_when_chain_is_self_referencing(self) -> None:
        # Build a cycle: outer.__cause__ = outer. The walker should not
        # loop forever — it ``break``s when ``current is current``.
        outer = Exception("forever")
        outer.__cause__ = outer
        decision = classify(outer)
        # No status, no message signals → UNKNOWN, but no infinite loop.
        assert decision.reason == FailoverReason.UNKNOWN
