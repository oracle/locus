# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for ``locus.models.credentials``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr, ValidationError

from locus.core.errors import ModelAuthError
from locus.models.credentials import Credential, CredentialPool


_FIXED_NOW = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)


def _make(label: str, key: str = "secret") -> Credential:
    return Credential(label=label, api_key=SecretStr(key))


# ---------------------------------------------------------------------------
# Credential model.
# ---------------------------------------------------------------------------


class TestCredential:
    def test_frozen(self) -> None:
        cred = _make("primary")
        with pytest.raises(ValidationError, match="frozen"):
            cred.label = "other"

    def test_empty_label_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Credential(label="", api_key=SecretStr("k"))

    def test_whitespace_label_rejected(self) -> None:
        with pytest.raises(ValidationError, match="whitespace"):
            Credential(label="   ", api_key=SecretStr("k"))

    def test_label_stripped(self) -> None:
        cred = Credential(label="  primary  ", api_key=SecretStr("k"))
        assert cred.label == "primary"

    def test_secret_str_repr_hides_key(self) -> None:
        cred = _make("primary", "super-secret-value")
        assert "super-secret-value" not in repr(cred)
        assert cred.api_key.get_secret_value() == "super-secret-value"


# ---------------------------------------------------------------------------
# Pool construction.
# ---------------------------------------------------------------------------


class TestPoolConstruction:
    def test_empty_pool_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CredentialPool([])

    def test_duplicate_labels_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            CredentialPool([_make("same"), _make("same", "different-key")])

    def test_size_and_labels(self) -> None:
        pool = CredentialPool([_make("a"), _make("b"), _make("c")])
        assert pool.size() == 3
        assert pool.labels() == ["a", "b", "c"]
        assert pool.available() == 3


# ---------------------------------------------------------------------------
# Round-robin behaviour.
# ---------------------------------------------------------------------------


class TestRoundRobin:
    def test_rotates_through_all(self) -> None:
        pool = CredentialPool([_make("a"), _make("b"), _make("c")])
        picks = [pool.pick().label for _ in range(6)]
        assert picks == ["a", "b", "c", "a", "b", "c"]

    def test_single_credential_pool_returns_same(self) -> None:
        pool = CredentialPool([_make("only")])
        assert pool.pick().label == "only"
        assert pool.pick().label == "only"


# ---------------------------------------------------------------------------
# mark_bad / cooldown.
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_disabled_credential_is_skipped(self) -> None:
        pool = CredentialPool([_make("a"), _make("b"), _make("c")])
        cred_b = next(c for c in [_make("b")])
        # Must mark_bad on a cred whose label matches; the key contents
        # are not compared.
        pool.mark_bad(
            Credential(label="b", api_key=SecretStr("whatever")),
            cooldown_s=60.0,
            now=_FIXED_NOW,
        )
        picks = [pool.pick(now=_FIXED_NOW).label for _ in range(4)]
        assert "b" not in picks
        assert set(picks) == {"a", "c"}

    def test_cooldown_expires(self) -> None:
        pool = CredentialPool([_make("a"), _make("b")])
        pool.mark_bad(_make("a"), cooldown_s=30.0, now=_FIXED_NOW)
        later = _FIXED_NOW + timedelta(seconds=31)
        picks = [pool.pick(now=later).label for _ in range(4)]
        assert "a" in picks

    def test_mark_bad_extends_but_never_shortens(self) -> None:
        pool = CredentialPool([_make("a"), _make("b")])
        pool.mark_bad(_make("a"), cooldown_s=120.0, now=_FIXED_NOW)
        # A shorter subsequent cooldown should be ignored.
        pool.mark_bad(_make("a"), cooldown_s=10.0, now=_FIXED_NOW)
        at_60s = _FIXED_NOW + timedelta(seconds=60)
        # Still disabled at t+60s (original cooldown was 120s).
        picks = [pool.pick(now=at_60s).label for _ in range(4)]
        assert "a" not in picks

    def test_negative_cooldown_rejected(self) -> None:
        pool = CredentialPool([_make("a")])
        with pytest.raises(ValueError, match="non-negative"):
            pool.mark_bad(_make("a"), cooldown_s=-1.0)

    def test_mark_bad_unknown_label_is_noop(self) -> None:
        pool = CredentialPool([_make("a")])
        # Should not raise, should not affect anything.
        pool.mark_bad(_make("ghost"), cooldown_s=60.0)
        assert pool.available() == 1

    def test_clear_cooldowns(self) -> None:
        pool = CredentialPool([_make("a"), _make("b")])
        pool.mark_bad(_make("a"), cooldown_s=300.0, now=_FIXED_NOW)
        pool.mark_bad(_make("b"), cooldown_s=300.0, now=_FIXED_NOW)
        assert pool.available(now=_FIXED_NOW) == 0
        pool.clear_cooldowns()
        assert pool.available(now=_FIXED_NOW) == 2


# ---------------------------------------------------------------------------
# Exhaustion.
# ---------------------------------------------------------------------------


class TestExhaustion:
    def test_all_disabled_raises_pool_exhausted(self) -> None:
        pool = CredentialPool([_make("a"), _make("b")])
        pool.mark_bad(_make("a"), cooldown_s=60.0, now=_FIXED_NOW)
        pool.mark_bad(_make("b"), cooldown_s=60.0, now=_FIXED_NOW)
        with pytest.raises(ModelAuthError) as exc_info:
            pool.pick(now=_FIXED_NOW)
        assert exc_info.value.kind == "model_pool_exhausted"
        assert "soonest reset" in str(exc_info.value)

    def test_exhausted_recovers_once_cooldown_ends(self) -> None:
        pool = CredentialPool([_make("a")])
        pool.mark_bad(_make("a"), cooldown_s=30.0, now=_FIXED_NOW)
        with pytest.raises(ModelAuthError):
            pool.pick(now=_FIXED_NOW)
        # 31s later — back online.
        assert pool.pick(now=_FIXED_NOW + timedelta(seconds=31)).label == "a"


# ---------------------------------------------------------------------------
# state() summary is log-safe.
# ---------------------------------------------------------------------------


class TestStateSummary:
    def test_state_does_not_include_secrets(self) -> None:
        pool = CredentialPool([_make("a", "topsecret-1"), _make("b", "topsecret-2")])
        pool.mark_bad(_make("a"), cooldown_s=60.0, now=_FIXED_NOW)
        summary = pool.state(now=_FIXED_NOW)
        blob = str(summary)
        assert "topsecret" not in blob
        assert summary["size"] == 2
        assert summary["available"] == 1
        assert "a" in summary["disabled"]

    def test_state_omits_expired_entries(self) -> None:
        pool = CredentialPool([_make("a")])
        pool.mark_bad(_make("a"), cooldown_s=30.0, now=_FIXED_NOW)
        later = _FIXED_NOW + timedelta(seconds=31)
        summary = pool.state(now=later)
        assert summary["disabled"] == {}
        assert summary["available"] == 1


# ---------------------------------------------------------------------------
# Concurrency sanity check — many threads picking simultaneously.
# ---------------------------------------------------------------------------


class TestThreadsafety:
    def test_concurrent_pick_no_data_race(self) -> None:
        import threading as th

        pool = CredentialPool([_make("a"), _make("b"), _make("c")])
        results: list[str] = []
        lock = th.Lock()

        def worker() -> None:
            for _ in range(100):
                label = pool.pick().label
                with lock:
                    results.append(label)

        threads = [th.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 8 threads times 100 picks each.
        assert len(results) == 800
        # All labels must be valid pool labels.
        assert set(results) <= {"a", "b", "c"}
