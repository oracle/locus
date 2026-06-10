# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Credential-pool-aware model wrapper.

Composes :class:`~tulip.models.credentials.CredentialPool`,
:func:`~tulip.models.failover.classify`, and any concrete
``ModelProtocol`` implementation into a single drop-in model that
rotates credentials when the classifier says rotation should help.

This is the "Hermes port glue" that turns the three primitives shipped
in milestone B (classifier / rate-limit tracker / credential pool) into
something callers can hand to ``Agent`` without writing the retry loop
themselves.

Typical wiring::

    from tulip.models.pooled import CredentialPoolModel
    from tulip.models.native.anthropic import AnthropicModel

    pool = CredentialPool([
        Credential(label="primary", api_key=SecretStr(os.environ["KEY_A"])),
        Credential(label="backup",  api_key=SecretStr(os.environ["KEY_B"])),
    ])

    def _build(cred: Credential) -> AnthropicModel:
        return AnthropicModel(
            model="claude-sonnet-4-6",
            api_key=cred.api_key,
            ...
        )

    model = CredentialPoolModel(pool=pool, build_model=_build)
    agent = Agent(config=AgentConfig(model=model, ...))

The wrapper is provider-agnostic — the user supplies ``build_model``,
which receives a :class:`Credential` and returns a freshly-configured
model instance. Models are cached by credential label so successive
calls don't re-instantiate clients.

Errors raised by the underlying model are classified via
:func:`tulip.models.failover.classify`. If the decision says
``should_rotate_credential = True`` the active credential is marked
bad in the pool and the next ``pick()`` is tried. Other errors propagate
unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from tulip.models.credentials import Credential, CredentialPool
from tulip.models.failover import classify
from tulip.models.rate_limits import parse_rate_limit_headers


if TYPE_CHECKING:
    from tulip.core.messages import Message
    from tulip.models import ModelResponse


logger = logging.getLogger(__name__)


__all__ = ["CredentialPoolModel"]


#: Default cooldown when an exception doesn't carry rate-limit headers
#: pointing at a more specific value.
_DEFAULT_COOLDOWN_S = 60.0


BuildModelFn = Callable[[Credential], Any]


class CredentialPoolModel:
    """A ModelProtocol-shaped wrapper that rotates a CredentialPool.

    Args:
        pool: The :class:`CredentialPool` to rotate through.
        build_model: Factory called as ``build_model(credential)`` to
            produce a concrete model instance. Called at most once
            per credential — the result is cached by label so client
            objects (and any TLS / SDK state) are reused.
        max_attempts: Maximum credential rotations per call. Default 3
            — beyond this the most recent error is re-raised so the
            caller's surrounding retry / failover logic can decide.
        default_cooldown_s: Cooldown applied when ``mark_bad`` runs
            and the exception doesn't carry an
            ``x-ratelimit-reset-requests`` header to source a more
            specific value from.
    """

    name = "CredentialPoolModel"

    def __init__(
        self,
        *,
        pool: CredentialPool,
        build_model: BuildModelFn,
        max_attempts: int = 3,
        default_cooldown_s: float = _DEFAULT_COOLDOWN_S,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if default_cooldown_s < 0:
            raise ValueError("default_cooldown_s must be non-negative")

        self._pool = pool
        self._build = build_model
        self._max_attempts = max_attempts
        self._default_cooldown_s = default_cooldown_s
        self._cache: dict[str, Any] = {}
        # Bookkeeping for tests / observability.
        self.attempts = 0
        self.last_credential: Credential | None = None

    # ------------------------------------------------------------------
    # ModelProtocol surface
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Forward the call to the active credential's model, rotating on failure."""
        last_exc: BaseException | None = None
        for _ in range(self._max_attempts):
            cred = self._pool.pick()
            model = self._get_model(cred)
            self.attempts += 1
            self.last_credential = cred
            try:
                return await model.complete(messages, tools, **kwargs)  # type: ignore[no-any-return]
            except BaseException as exc:  # noqa: BLE001
                last_exc = exc
                decision = classify(exc)
                if not decision.should_rotate_credential:
                    raise
                self._mark_bad(cred, exc)
        # Pool may still have available credentials but we've burned the
        # attempt budget. Surface the most recent error.
        assert last_exc is not None
        raise last_exc

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream from the active credential's model, rotating only on the
        opening exception.

        Mid-stream errors propagate to the caller because a partial
        stream cannot safely be retried on a different credential —
        the model has already started emitting tokens that the agent
        may have surfaced to the user.
        """
        last_exc: BaseException | None = None
        for _ in range(self._max_attempts):
            cred = self._pool.pick()
            model = self._get_model(cred)
            self.attempts += 1
            self.last_credential = cred
            try:
                stream = model.stream(messages, tools, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                last_exc = exc
                decision = classify(exc)
                if not decision.should_rotate_credential:
                    raise
                self._mark_bad(cred, exc)
                continue
            # Got past the opening — yield through. If the underlying
            # iterator raises mid-stream, that propagates.
            async for chunk in stream:
                yield chunk
            return
        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_model(self, cred: Credential) -> Any:
        if cred.label not in self._cache:
            self._cache[cred.label] = self._build(cred)
        return self._cache[cred.label]

    def _mark_bad(self, cred: Credential, exc: BaseException) -> None:
        cooldown = self._extract_cooldown(exc)
        self._pool.mark_bad(
            cred,
            cooldown_s=cooldown,
            reason=f"{type(exc).__name__}: {exc}"[:200],
        )

    def _extract_cooldown(self, exc: BaseException) -> float:
        """Return cooldown seconds for ``exc``, sourced from headers when present."""
        headers = getattr(exc, "headers", None)
        if isinstance(headers, dict) and headers:
            rl = parse_rate_limit_headers(headers)
            if rl and rl.requests_min and rl.requests_min.reset_seconds > 0:
                return float(rl.requests_min.reset_seconds)
        return self._default_cooldown_s
