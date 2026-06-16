# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Structured API-error classification and recovery policy.

Locus's exception hierarchy in :mod:`locus.core.errors` tells callers
*what went wrong*. This module tells them *what to do about it*.

Given an arbitrary exception raised by any model provider (OpenAI,
Anthropic, OCI GenAI, Ollama, …) the :func:`classify` entry point
returns a frozen :class:`FailoverDecision` with:

- a :class:`FailoverReason` (a stable, loggable enum value) and
- a set of boolean recovery hints (``retryable``,
  ``should_rotate_credential``, ``should_compress``,
  ``should_fallback``) that downstream retry / pool / compaction
  layers can consult instead of re-matching strings.

Priority pipeline (highest-confidence first):

1. HTTP status-code classification, with message disambiguation for
   402 (billing vs transient usage limit) and 400 (context overflow
   vs format error).
2. Structured error-code classification (``code`` / ``type`` fields in
   the response body).
3. Message pattern matching (no status code available).
4. Server-disconnect heuristic — on a large session, a bare disconnect
   is more often context overflow than transport hiccup.
5. Transport / timeout exception-type heuristics.
6. Fallback: :attr:`FailoverReason.UNKNOWN` (retryable with backoff).

The classifier is intentionally provider-neutral: it does not know
about aggregator-specific quirks (OpenRouter policy gates, Anthropic
thinking-block signature errors, …). Providers that need those
refinements can post-process the result or extend the module.
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

__all__ = [
    "FailoverDecision",
    "FailoverReason",
    "classify",
]


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class FailoverReason(StrEnum):
    """Why an API call failed — determines recovery strategy.

    ``StrEnum`` (Python 3.11+) makes values usable as log keys and
    JSON values without explicit ``.value`` access.
    """

    AUTH_TRANSIENT = "auth_transient"
    """401 / 403 that may succeed after credential refresh or rotation."""

    AUTH_PERMANENT = "auth_permanent"
    """Authentication failed definitively; don't retry without user action."""

    BILLING = "billing"
    """402 or confirmed credit / quota exhaustion — rotate credentials."""

    RATE_LIMIT = "rate_limit"
    """429 or periodic throttling — backoff and/or rotate."""

    OVERLOADED = "overloaded"
    """503 / 529 — provider overloaded, backoff."""

    SERVER_ERROR = "server_error"
    """500 / 502 / other 5xx — internal server error, retry."""

    TIMEOUT = "timeout"
    """Connection / read timeout — rebuild client, retry."""

    CONTEXT_TOO_LONG = "context_too_long"
    """Prompt exceeds model context — compress history before retry."""

    PAYLOAD_TOO_LARGE = "payload_too_large"
    """413 — request body too large, compress or trim."""

    MODEL_NOT_FOUND = "model_not_found"
    """404 / invalid model slug — fall back to a different model."""

    FORMAT_ERROR = "format_error"
    """400 with no compression signal — malformed request, don't retry."""

    UNKNOWN = "unknown"
    """Unclassified; safe default is one retry with backoff."""


# ---------------------------------------------------------------------------
# Decision object
# ---------------------------------------------------------------------------


class FailoverDecision(BaseModel):
    """Classifier output — a reason plus recovery hints."""

    model_config = {"frozen": True}

    reason: FailoverReason
    status_code: int | None = None
    retryable: bool = True
    should_rotate_credential: bool = False
    should_compress: bool = False
    should_fallback: bool = False
    message: str = Field(default="", max_length=1000)


class _Decide(Protocol):
    """Callable signature for the internal ``decide`` closure.

    Defined as a :class:`~typing.Protocol` so mypy can infer the
    ``FailoverDecision`` return type through the helper functions.
    """

    def __call__(
        self,
        reason: FailoverReason,
        *,
        retryable: bool = ...,
        should_rotate_credential: bool = ...,
        should_compress: bool = ...,
        should_fallback: bool = ...,
    ) -> FailoverDecision: ...


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------


_BILLING_PATTERNS: tuple[str, ...] = (
    "insufficient credits",
    "insufficient_quota",
    "credit balance",
    "credits have been exhausted",
    "payment required",
    "billing hard limit",
    "exceeded your current quota",
    "account is deactivated",
    "plan does not include",
)

_RATE_LIMIT_PATTERNS: tuple[str, ...] = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "throttled",
    "throttlingexception",
    "requests per minute",
    "tokens per minute",
    "requests per day",
    "try again in",
    "please retry after",
    "resource_exhausted",
    "too many concurrent requests",
)

_USAGE_LIMIT_PATTERNS: tuple[str, ...] = (
    "usage limit",
    "quota",
    "limit exceeded",
)

_USAGE_LIMIT_TRANSIENT_SIGNALS: tuple[str, ...] = (
    "try again",
    "retry",
    "resets at",
    "reset in",
    "wait",
    "requests remaining",
)

_PAYLOAD_TOO_LARGE_PATTERNS: tuple[str, ...] = (
    "request entity too large",
    "payload too large",
    "error code: 413",
)

_CONTEXT_OVERFLOW_PATTERNS: tuple[str, ...] = (
    "context length",
    "context size",
    "maximum context",
    "token limit",
    "too many tokens",
    "reduce the length",
    "exceeds the limit",
    "context window",
    "prompt is too long",
    "prompt exceeds max length",
    "maximum number of tokens",
    "context length exceeded",
    "input is too long",
    "max input token",
    "max_model_len",
)

_MODEL_NOT_FOUND_PATTERNS: tuple[str, ...] = (
    "is not a valid model",
    "invalid model",
    "model not found",
    "model_not_found",
    "does not exist",
    "no such model",
    "unknown model",
    "unsupported model",
)

_AUTH_PATTERNS: tuple[str, ...] = (
    "invalid api key",
    "invalid_api_key",
    "authentication",
    "unauthorized",
    "forbidden",
    "invalid token",
    "token expired",
    "token revoked",
    "access denied",
)

_SERVER_DISCONNECT_PATTERNS: tuple[str, ...] = (
    "server disconnected",
    "peer closed connection",
    "connection reset by peer",
    "connection was closed",
    "network connection lost",
    "unexpected eof",
    "incomplete chunked read",
)

_SSL_TRANSIENT_PATTERNS: tuple[str, ...] = (
    "bad record mac",
    "ssl alert",
    "tls alert",
    "ssl handshake failure",
    "bad_record_mac",
    "ssl_alert",
    "tls_alert",
    "[ssl:",
)

_TRANSPORT_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "ReadTimeout",
        "ConnectTimeout",
        "PoolTimeout",
        "ConnectError",
        "RemoteProtocolError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "BrokenPipeError",
        "TimeoutError",
        "ReadError",
        "ServerDisconnectedError",
        "SSLError",
        "SSLZeroReturnError",
        "SSLWantReadError",
        "SSLWantWriteError",
        "SSLEOFError",
        "SSLSyscallError",
        "APIConnectionError",
        "APITimeoutError",
    }
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def classify(
    exc: BaseException,
    *,
    status: int | None = None,
    body: dict[str, Any] | None = None,
    approx_tokens: int = 0,
    context_length: int = 200_000,
    num_messages: int = 0,
) -> FailoverDecision:
    """Return a :class:`FailoverDecision` for ``exc``.

    Callers who have already extracted the HTTP status or body from a
    provider-specific SDK can pass them directly; otherwise the
    classifier walks the exception and its ``__cause__`` / ``__context__``
    chain looking for the usual attributes (``.status_code``, ``.status``,
    ``.body``, ``.response.json()``).

    Args:
        exc: The exception raised by the model call.
        status: Override for the HTTP status code. When the SDK does
            not surface one (rate-limit SDKs sometimes don't), pass a
            value here to force-route classification.
        body: Parsed response body. Used for structured error-code
            extraction and for disambiguating 400 generic errors.
        approx_tokens: Approximate token count of the prompt. Drives
            the "large session + bare disconnect → context overflow"
            heuristic.
        context_length: Model's advertised context window. Used
            alongside ``approx_tokens`` for the same heuristic.
        num_messages: Message count in the conversation. A second
            large-session signal for providers that fail without an
            HTTP status.

    Returns:
        A frozen :class:`FailoverDecision`.
    """
    status_code = status if status is not None else _extract_status_code(exc)
    error_type = type(exc).__name__
    # Some SDKs raise RateLimitError without setting .status_code.
    if status_code is None and error_type == "RateLimitError":
        status_code = 429

    resolved_body: dict[str, Any] = body if body is not None else _extract_body(exc)
    error_code = _extract_error_code(resolved_body)
    error_msg = _build_error_msg(exc, resolved_body)

    message_excerpt = _extract_message(exc, resolved_body)

    def _decide(
        reason: FailoverReason,
        *,
        retryable: bool = True,
        should_rotate_credential: bool = False,
        should_compress: bool = False,
        should_fallback: bool = False,
    ) -> FailoverDecision:
        return FailoverDecision(
            reason=reason,
            status_code=status_code,
            retryable=retryable,
            should_rotate_credential=should_rotate_credential,
            should_compress=should_compress,
            should_fallback=should_fallback,
            message=message_excerpt,
        )

    # 1. HTTP status-code classification.
    if status_code is not None:
        by_status = _classify_by_status(
            status_code,
            error_msg,
            resolved_body,
            approx_tokens=approx_tokens,
            context_length=context_length,
            num_messages=num_messages,
            decide=_decide,
        )
        if by_status is not None:
            return by_status

    # 2. Structured error-code classification.
    if error_code:
        by_code = _classify_by_error_code(error_code, _decide)
        if by_code is not None:
            return by_code

    # 3. Message pattern matching.
    by_msg = _classify_by_message(error_msg, _decide)
    if by_msg is not None:
        return by_msg

    # 4. SSL/TLS transient → retry as timeout (must precede the disconnect
    #    heuristic so a flaky TLS record doesn't trigger compression).
    if any(p in error_msg for p in _SSL_TRANSIENT_PATTERNS):
        return _decide(FailoverReason.TIMEOUT, retryable=True)

    # 5. Server disconnect + large session → likely context overflow.
    is_disconnect = any(p in error_msg for p in _SERVER_DISCONNECT_PATTERNS)
    if is_disconnect and status_code is None:
        is_large = (
            approx_tokens > context_length * 0.6 or approx_tokens > 120_000 or num_messages > 200
        )
        if is_large:
            return _decide(
                FailoverReason.CONTEXT_TOO_LONG,
                retryable=True,
                should_compress=True,
            )
        return _decide(FailoverReason.TIMEOUT, retryable=True)

    # 6. Transport / timeout exceptions.
    if error_type in _TRANSPORT_ERROR_TYPES or isinstance(
        exc, (TimeoutError, ConnectionError, OSError)
    ):
        return _decide(FailoverReason.TIMEOUT, retryable=True)

    # 7. Fallback.
    return _decide(FailoverReason.UNKNOWN, retryable=True)


# ---------------------------------------------------------------------------
# Status-code pipeline
# ---------------------------------------------------------------------------


def _classify_by_status(
    status_code: int,
    error_msg: str,
    body: dict[str, Any],
    *,
    approx_tokens: int,
    context_length: int,
    num_messages: int,
    decide: _Decide,
) -> FailoverDecision | None:
    if status_code == 401:
        return decide(
            FailoverReason.AUTH_TRANSIENT,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if status_code == 403:
        if "key limit exceeded" in error_msg or "spending limit" in error_msg:
            return decide(
                FailoverReason.BILLING,
                retryable=False,
                should_rotate_credential=True,
                should_fallback=True,
            )
        return decide(
            FailoverReason.AUTH_PERMANENT,
            retryable=False,
            should_fallback=True,
        )

    if status_code == 402:
        return _classify_402(error_msg, decide)

    if status_code == 404:
        if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
            return decide(
                FailoverReason.MODEL_NOT_FOUND,
                retryable=False,
                should_fallback=True,
            )
        return decide(FailoverReason.UNKNOWN, retryable=True)

    if status_code == 413:
        return decide(
            FailoverReason.PAYLOAD_TOO_LARGE,
            retryable=True,
            should_compress=True,
        )

    if status_code == 429:
        return decide(
            FailoverReason.RATE_LIMIT,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if status_code == 400:
        return _classify_400(
            error_msg,
            body,
            approx_tokens=approx_tokens,
            context_length=context_length,
            num_messages=num_messages,
            decide=decide,
        )

    if status_code in (500, 502):
        return decide(FailoverReason.SERVER_ERROR, retryable=True)

    if status_code in (503, 529):
        return decide(FailoverReason.OVERLOADED, retryable=True)

    if 400 <= status_code < 500:
        return decide(
            FailoverReason.FORMAT_ERROR,
            retryable=False,
            should_fallback=True,
        )

    if 500 <= status_code < 600:
        return decide(FailoverReason.SERVER_ERROR, retryable=True)

    return None


def _classify_402(error_msg: str, decide: _Decide) -> FailoverDecision:
    """Disambiguate 402: some 402s are transient usage limits, not billing."""
    has_usage_limit = any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
    has_transient = any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS)
    if has_usage_limit and has_transient:
        return decide(
            FailoverReason.RATE_LIMIT,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )
    return decide(
        FailoverReason.BILLING,
        retryable=False,
        should_rotate_credential=True,
        should_fallback=True,
    )


def _classify_400(
    error_msg: str,
    body: dict[str, Any],
    *,
    approx_tokens: int,
    context_length: int,
    num_messages: int,
    decide: _Decide,
) -> FailoverDecision:
    if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return decide(
            FailoverReason.CONTEXT_TOO_LONG,
            retryable=True,
            should_compress=True,
        )

    if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return decide(
            FailoverReason.MODEL_NOT_FOUND,
            retryable=False,
            should_fallback=True,
        )

    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return decide(
            FailoverReason.RATE_LIMIT,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if any(p in error_msg for p in _BILLING_PATTERNS):
        return decide(
            FailoverReason.BILLING,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Generic 400 + large session → probable context overflow.
    body_msg = ""
    if isinstance(body, dict):
        err_obj = body.get("error", {})
        if isinstance(err_obj, dict):
            body_msg = str(err_obj.get("message") or "").strip().lower()
        if not body_msg:
            body_msg = str(body.get("message") or "").strip().lower()

    is_generic = len(body_msg) < 30 or body_msg in ("error", "")
    is_large = approx_tokens > context_length * 0.4 or approx_tokens > 80_000 or num_messages > 80
    if is_generic and is_large:
        return decide(
            FailoverReason.CONTEXT_TOO_LONG,
            retryable=True,
            should_compress=True,
        )

    return decide(
        FailoverReason.FORMAT_ERROR,
        retryable=False,
        should_fallback=True,
    )


# ---------------------------------------------------------------------------
# Error-code pipeline
# ---------------------------------------------------------------------------


def _classify_by_error_code(error_code: str, decide: _Decide) -> FailoverDecision | None:
    code = error_code.lower()
    if code in ("resource_exhausted", "throttled", "rate_limit_exceeded"):
        return decide(
            FailoverReason.RATE_LIMIT,
            retryable=True,
            should_rotate_credential=True,
        )
    if code in ("insufficient_quota", "billing_not_active", "payment_required"):
        return decide(
            FailoverReason.BILLING,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )
    if code in ("model_not_found", "model_not_available", "invalid_model"):
        return decide(
            FailoverReason.MODEL_NOT_FOUND,
            retryable=False,
            should_fallback=True,
        )
    if code in ("context_length_exceeded", "max_tokens_exceeded"):
        return decide(
            FailoverReason.CONTEXT_TOO_LONG,
            retryable=True,
            should_compress=True,
        )
    return None


# ---------------------------------------------------------------------------
# Message pipeline
# ---------------------------------------------------------------------------


def _classify_by_message(error_msg: str, decide: _Decide) -> FailoverDecision | None:
    if any(p in error_msg for p in _PAYLOAD_TOO_LARGE_PATTERNS):
        return decide(
            FailoverReason.PAYLOAD_TOO_LARGE,
            retryable=True,
            should_compress=True,
        )

    has_usage_limit = any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
    if has_usage_limit:
        has_transient = any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS)
        if has_transient:
            return decide(
                FailoverReason.RATE_LIMIT,
                retryable=True,
                should_rotate_credential=True,
                should_fallback=True,
            )
        return decide(
            FailoverReason.BILLING,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if any(p in error_msg for p in _BILLING_PATTERNS):
        return decide(
            FailoverReason.BILLING,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return decide(
            FailoverReason.RATE_LIMIT,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return decide(
            FailoverReason.CONTEXT_TOO_LONG,
            retryable=True,
            should_compress=True,
        )

    if any(p in error_msg for p in _AUTH_PATTERNS):
        return decide(
            FailoverReason.AUTH_TRANSIENT,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return decide(
            FailoverReason.MODEL_NOT_FOUND,
            retryable=False,
            should_fallback=True,
        )

    return None


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_status_code(exc: BaseException) -> int | None:
    """Walk the exception / cause chain looking for ``.status_code`` / ``.status``."""
    current: BaseException | None = exc
    for _ in range(5):
        if current is None:
            break
        code = getattr(current, "status_code", None)
        if isinstance(code, int):
            return code
        code = getattr(current, "status", None)
        if isinstance(code, int) and 100 <= code < 600:
            return code
        nxt = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if nxt is None or nxt is current:
            break
        current = nxt
    return None


def _extract_body(exc: BaseException) -> dict[str, Any]:
    """Extract the structured error body from an SDK exception."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            candidate = response.json()
        except Exception:  # noqa: BLE001 — SDKs throw many shapes
            candidate = None
        if isinstance(candidate, dict):
            return candidate
    return {}


def _extract_error_code(body: dict[str, Any]) -> str:
    if not body:
        return ""
    err = body.get("error", {})
    if isinstance(err, dict):
        code = err.get("code") or err.get("type") or ""
        if isinstance(code, str) and code.strip():
            return code.strip()
    code = body.get("code") or body.get("error_code") or ""
    if isinstance(code, (str, int)):
        return str(code).strip()
    return ""


def _extract_message(exc: BaseException, body: dict[str, Any]) -> str:
    if body:
        err = body.get("error", {})
        if isinstance(err, dict):
            msg = err.get("message", "")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()[:500]
        msg = body.get("message", "")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()[:500]
    return str(exc)[:500]


def _build_error_msg(exc: BaseException, body: dict[str, Any]) -> str:
    """Combine exception, top-level body, and wrapped ``metadata.raw`` messages."""
    raw = str(exc).lower()
    body_msg = ""
    metadata_msg = ""
    if isinstance(body, dict):
        err = body.get("error", {})
        if isinstance(err, dict):
            body_msg = str(err.get("message") or "").lower()
            metadata = err.get("metadata", {})
            if isinstance(metadata, dict):
                raw_inner = metadata.get("raw") or ""
                if isinstance(raw_inner, str) and raw_inner.strip():
                    try:
                        inner = json.loads(raw_inner)
                    except (json.JSONDecodeError, TypeError):
                        inner = None
                    if isinstance(inner, dict):
                        inner_err = inner.get("error", {})
                        if isinstance(inner_err, dict):
                            metadata_msg = str(inner_err.get("message") or "").lower()
        if not body_msg:
            body_msg = str(body.get("message") or "").lower()
    parts = [raw]
    if body_msg and body_msg not in raw:
        parts.append(body_msg)
    if metadata_msg and metadata_msg not in raw and metadata_msg not in body_msg:
        parts.append(metadata_msg)
    return " ".join(parts)
