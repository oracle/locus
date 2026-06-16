# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for ``locus.models.pooled.CredentialPoolModel``."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from locus.core.errors import ModelAuthError
from locus.core.messages import Message
from locus.models import ModelResponse
from locus.models.credentials import Credential, CredentialPool
from locus.models.pooled import CredentialPoolModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ScriptedModel:
    """Concrete model that returns scripted responses or raises scripted errors."""

    def __init__(
        self,
        cred: Credential,
        *,
        events: list[Any],
    ) -> None:
        self.cred = cred
        self._events = list(events)
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        if not self._events:
            raise RuntimeError(f"{self.cred.label}: out of scripted events")
        self.calls += 1
        evt = self._events.pop(0)
        if isinstance(evt, BaseException):
            raise evt
        return evt


class _RateLimit429Error(Exception):
    def __init__(
        self,
        msg: str = "Too many requests",
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(msg)
        self.status_code = 429
        if headers is not None:
            self.headers = headers


def _pool(*labels: str) -> CredentialPool:
    return CredentialPool(
        [Credential(label=name, api_key=SecretStr(f"k-{name}")) for name in labels]
    )


def _ok_response(text: str) -> ModelResponse:
    return ModelResponse(message=Message.assistant(text))


# ---------------------------------------------------------------------------
# Construction guards.
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_max_attempts_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            CredentialPoolModel(pool=_pool("a"), build_model=lambda c: None, max_attempts=0)

    def test_negative_cooldown_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CredentialPoolModel(
                pool=_pool("a"),
                build_model=lambda c: None,
                default_cooldown_s=-1.0,
            )


# ---------------------------------------------------------------------------
# Happy path — first credential works.
# ---------------------------------------------------------------------------


class TestComplete:
    @pytest.mark.asyncio
    async def test_first_call_succeeds(self) -> None:
        pool = _pool("alpha", "beta")
        built: list[str] = []

        def _build(cred: Credential) -> _ScriptedModel:
            built.append(cred.label)
            return _ScriptedModel(cred, events=[_ok_response("hi from " + cred.label)])

        wrapped = CredentialPoolModel(pool=pool, build_model=_build)
        resp = await wrapped.complete([Message.user("hello")])
        assert resp.message.content == "hi from alpha"
        assert built == ["alpha"]
        assert wrapped.attempts == 1
        assert wrapped.last_credential is not None
        assert wrapped.last_credential.label == "alpha"

    @pytest.mark.asyncio
    async def test_rotates_on_classified_rate_limit(self) -> None:
        pool = _pool("alpha", "beta")

        scripts = {
            "alpha": [_RateLimit429Error()],
            "beta": [_ok_response("from beta")],
        }

        def _build(cred: Credential) -> _ScriptedModel:
            return _ScriptedModel(cred, events=scripts[cred.label])

        wrapped = CredentialPoolModel(pool=pool, build_model=_build)
        resp = await wrapped.complete([Message.user("hello")])
        assert resp.message.content == "from beta"
        assert wrapped.attempts == 2
        assert wrapped.last_credential is not None
        assert wrapped.last_credential.label == "beta"
        # alpha now in cooldown.
        assert "alpha" in pool.state()["disabled"]

    @pytest.mark.asyncio
    async def test_non_rotation_error_propagates(self) -> None:
        # 4xx that isn't auth/rate/billing — should not rotate.
        class _BadFormatError(Exception):
            status_code = 418  # Teapot — format_error in classifier

        pool = _pool("alpha", "beta")

        def _build(cred: Credential) -> _ScriptedModel:
            return _ScriptedModel(cred, events=[_BadFormatError("teapot")])

        wrapped = CredentialPoolModel(pool=pool, build_model=_build)
        with pytest.raises(_BadFormatError):
            await wrapped.complete([Message.user("hello")])
        # Never marked alpha bad — pool should still have 2 available.
        assert pool.available() == 2

    @pytest.mark.asyncio
    async def test_max_attempts_exhausted_raises_last_error(self) -> None:
        pool = _pool("alpha", "beta", "gamma")

        def _build(cred: Credential) -> _ScriptedModel:
            return _ScriptedModel(cred, events=[_RateLimit429Error()])

        wrapped = CredentialPoolModel(pool=pool, build_model=_build, max_attempts=2)
        with pytest.raises(_RateLimit429Error):
            await wrapped.complete([Message.user("hello")])
        assert wrapped.attempts == 2
        # The third credential (gamma) was never touched because we
        # capped at 2 attempts.
        assert "gamma" not in pool.state()["disabled"]

    @pytest.mark.asyncio
    async def test_pool_exhaustion_surfaces_pool_error(self) -> None:
        pool = _pool("only")

        def _build(cred: Credential) -> _ScriptedModel:
            return _ScriptedModel(cred, events=[_RateLimit429Error(), _RateLimit429Error()])

        wrapped = CredentialPoolModel(pool=pool, build_model=_build, max_attempts=5)
        with pytest.raises(ModelAuthError) as info:
            await wrapped.complete([Message.user("hello")])
        # Pool's exhausted error wins, with its dedicated kind.
        assert info.value.kind == "model_pool_exhausted"


# ---------------------------------------------------------------------------
# Header-driven cooldowns.
# ---------------------------------------------------------------------------


class TestCooldownFromHeaders:
    @pytest.mark.asyncio
    async def test_uses_x_ratelimit_reset_header(self) -> None:
        pool = _pool("alpha", "beta")

        scripts = {
            "alpha": [
                _RateLimit429Error(
                    headers={
                        "x-ratelimit-limit-requests": "60",
                        "x-ratelimit-remaining-requests": "0",
                        "x-ratelimit-reset-requests": "30",
                    }
                )
            ],
            "beta": [_ok_response("ok")],
        }

        def _build(cred: Credential) -> _ScriptedModel:
            return _ScriptedModel(cred, events=scripts[cred.label])

        wrapped = CredentialPoolModel(pool=pool, build_model=_build, default_cooldown_s=1.0)
        await wrapped.complete([Message.user("x")])
        # Cooldown should reflect the 30s header value, not the 1s default.
        # (Indirect assertion: pool.state should still show alpha disabled
        # after 1.5s, which would be past the default but well under 30s.)
        from datetime import UTC, datetime, timedelta

        future = datetime.now(UTC) + timedelta(seconds=1.5)
        state = pool.state(now=future)
        assert "alpha" in state["disabled"]


# ---------------------------------------------------------------------------
# Model cache — same credential reuses the built instance.
# ---------------------------------------------------------------------------


class TestBuildCache:
    @pytest.mark.asyncio
    async def test_build_called_once_per_credential(self) -> None:
        pool = _pool("solo")
        builds: list[Credential] = []

        def _build(cred: Credential) -> _ScriptedModel:
            builds.append(cred)
            return _ScriptedModel(cred, events=[_ok_response("a"), _ok_response("b")])

        wrapped = CredentialPoolModel(pool=pool, build_model=_build)
        await wrapped.complete([Message.user("1")])
        await wrapped.complete([Message.user("2")])
        # Single credential, two calls — but only one build.
        assert len(builds) == 1


# ---------------------------------------------------------------------------
# Streaming surface — opening errors rotate, mid-stream errors propagate.
# ---------------------------------------------------------------------------


class TestStream:
    @pytest.mark.asyncio
    async def test_opening_error_rotates(self) -> None:
        pool = _pool("alpha", "beta")

        class _StreamingModel:
            def __init__(self, cred: Credential, *, fail_open: bool) -> None:
                self.cred = cred
                self._fail_open = fail_open

            def stream(
                self,
                messages: list[Message],
                tools: list[dict[str, Any]] | None = None,
                **kwargs: Any,
            ) -> Any:
                if self._fail_open:
                    raise _RateLimit429Error("opening 429")

                async def _gen():
                    yield f"chunk-{self.cred.label}"

                return _gen()

        def _build(cred: Credential) -> _StreamingModel:
            return _StreamingModel(cred, fail_open=(cred.label == "alpha"))

        wrapped = CredentialPoolModel(pool=pool, build_model=_build)
        chunks = [c async for c in wrapped.stream([Message.user("hi")])]
        assert chunks == ["chunk-beta"]
        assert "alpha" in pool.state()["disabled"]
