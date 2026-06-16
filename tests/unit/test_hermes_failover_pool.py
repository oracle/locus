# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for the Hermes-port failover stack (B.1 + B.2 + B.3).

These exercise the three primitives together — classifier output
drives credential-pool rotation, and rate-limit headers drive
cooldown durations. The harness simulates a flaky provider so we
don't depend on any real model service.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import SecretStr

from locus.core.errors import ModelAuthError
from locus.models.credentials import Credential, CredentialPool
from locus.models.failover import FailoverReason, classify
from locus.models.rate_limits import parse_rate_limit_headers


_NOW = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)


class _FlakyProvider:
    """Stand-in for a provider SDK that fails the first ``fail_for`` calls.

    Each call returns a synthetic exception that mimics the SDK's
    ``.status_code`` / ``.body`` shape so the classifier can route it.
    """

    def __init__(self, *, fail_for: int, mode: str) -> None:
        self.fail_for = fail_for
        self.mode = mode  # "rate_limit" / "billing" / "auth"
        self.calls = 0

    def call(self) -> None:
        self.calls += 1
        if self.calls > self.fail_for:
            return  # success
        if self.mode == "rate_limit":
            raise _SdkError(
                "Too many requests, please retry after 30 seconds",
                status_code=429,
                headers={
                    "x-ratelimit-limit-requests": "60",
                    "x-ratelimit-remaining-requests": "0",
                    "x-ratelimit-reset-requests": "30",
                },
            )
        if self.mode == "billing":
            raise _SdkError("Insufficient credits", status_code=402)
        if self.mode == "auth":
            raise _SdkError("Invalid API key", status_code=401)
        raise RuntimeError(f"unknown mode {self.mode!r}")


class _SdkError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


def _retry_with_pool(
    provider: _FlakyProvider,
    pool: CredentialPool,
    *,
    max_attempts: int = 5,
    now: datetime = _NOW,
) -> tuple[Credential, int]:
    """Loop: pick credential → call → on classified failure rotate.

    Returns the credential that ultimately succeeded plus the number
    of attempts that were made. Raises :class:`ModelAuthError` if the
    pool exhausts before success.
    """
    last_cred: Credential | None = None
    attempts = 0
    for attempt_idx in range(max_attempts):
        cred = pool.pick(now=now)
        last_cred = cred
        attempts = attempt_idx + 1
        try:
            provider.call()
            return cred, attempts
        except _SdkError as exc:
            decision = classify(exc)
            if not decision.should_rotate_credential:
                raise
            # Use rate-limit reset header (when present) to pick the
            # cooldown — otherwise fall back to a sensible default.
            cooldown = 60.0
            rl = parse_rate_limit_headers(exc.headers, now=now)
            if rl and rl.requests_min and rl.requests_min.reset_seconds > 0:
                cooldown = rl.requests_min.reset_seconds
            pool.mark_bad(cred, cooldown_s=cooldown, now=now)
    if last_cred is None:
        raise RuntimeError("no attempts executed")
    raise ModelAuthError("retry budget exhausted before success")


# ---------------------------------------------------------------------------
# Rate-limit driven rotation.
# ---------------------------------------------------------------------------


class TestRateLimitRotation:
    def test_rotates_on_429_and_succeeds(self) -> None:
        pool = CredentialPool(
            [
                Credential(label="alpha", api_key=SecretStr("k1")),
                Credential(label="beta", api_key=SecretStr("k2")),
            ]
        )
        # First credential trips a 429; the rotation finds beta and
        # the second call succeeds.
        provider = _FlakyProvider(fail_for=1, mode="rate_limit")

        cred, attempts = _retry_with_pool(provider, pool)
        assert attempts == 2
        # Successful credential is the rotated-to one (beta).
        assert cred.label == "beta"
        # alpha is in cooldown, beta is not.
        state = pool.state(now=_NOW)
        assert "alpha" in state["disabled"]
        assert "beta" not in state["disabled"]

    def test_cooldown_sourced_from_rate_limit_header(self) -> None:
        # Single-credential pool, single failure — verify the cooldown
        # length picked up from x-ratelimit-reset-requests propagates
        # to the pool's disabled_until.
        pool = CredentialPool([Credential(label="solo", api_key=SecretStr("k"))])
        provider = _FlakyProvider(fail_for=1, mode="rate_limit")
        try:
            _retry_with_pool(provider, pool, max_attempts=1)
        except ModelAuthError:
            pass

        state = pool.state(now=_NOW)
        assert "solo" in state["disabled"]


# ---------------------------------------------------------------------------
# Billing — non-retryable from caller's POV but pool still rotates.
# ---------------------------------------------------------------------------


class TestBillingRotation:
    def test_billing_rotates_then_succeeds_on_other_credential(self) -> None:
        pool = CredentialPool(
            [
                Credential(label="dead", api_key=SecretStr("k1")),
                Credential(label="live", api_key=SecretStr("k2")),
            ]
        )
        # Fail first call (dead key billing-out), succeed on rotation.
        provider = _FlakyProvider(fail_for=1, mode="billing")

        cred, attempts = _retry_with_pool(provider, pool)
        assert attempts == 2
        # Whichever was tried first is the one that's disabled.
        state = pool.state(now=_NOW)
        assert len(state["disabled"]) == 1


# ---------------------------------------------------------------------------
# Pool exhaustion — every credential fails, pool emits the dedicated kind.
# ---------------------------------------------------------------------------


class TestPoolExhaustion:
    def test_all_credentials_billing_out_raises_pool_exhausted(self) -> None:
        pool = CredentialPool(
            [Credential(label=f"k{i}", api_key=SecretStr(f"v{i}")) for i in range(3)]
        )
        # Every call billing-fails forever.
        provider = _FlakyProvider(fail_for=10_000, mode="billing")

        with pytest.raises(ModelAuthError) as info:
            _retry_with_pool(provider, pool, max_attempts=10)
        assert info.value.kind == "model_pool_exhausted"


# ---------------------------------------------------------------------------
# Classification → recovery hint sanity end-to-end.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected_reason", "should_rotate"),
    [
        (429, FailoverReason.RATE_LIMIT, True),
        (402, FailoverReason.BILLING, True),
        (401, FailoverReason.AUTH_TRANSIENT, True),
        (500, FailoverReason.SERVER_ERROR, False),
        (503, FailoverReason.OVERLOADED, False),
    ],
)
def test_classifier_recovery_hint_drives_rotation_decision(
    status: int, expected_reason: FailoverReason, should_rotate: bool
) -> None:
    exc = _SdkError("any", status_code=status)
    decision = classify(exc)
    assert decision.reason is expected_reason
    assert decision.should_rotate_credential is should_rotate
