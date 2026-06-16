# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Multi-credential pool with rotation and cooldown.

For providers that allow multiple API keys under a single account
(OpenAI projects, OpenRouter sibling keys, Anthropic org tokens, OCI
API keys, …) a :class:`CredentialPool` lets the model wrapper rotate
through keys when any one of them trips rate-limit or billing errors.

The pool is deliberately tiny and provider-neutral:

* :class:`Credential` is a frozen Pydantic model that wraps a
  :class:`~pydantic.SecretStr`.  Labels are required so logs /
  telemetry can distinguish which key is active without revealing the
  secret.
* :class:`CredentialPool` rotates round-robin over the available
  credentials and can temporarily disable one via :meth:`mark_bad`.

Design notes:

* **Locking:** the pool uses a short-lived ``threading.Lock``; the
  critical section is a few field reads and a timestamp comparison,
  well below the cost of acquiring an ``asyncio.Lock`` from within
  sync code.  Async callers can safely invoke the sync methods — the
  lock never spans an ``await``.
* **Observability:** :meth:`pick` returns the next available
  :class:`Credential` and never the same instance twice in a row
  while others are available.  :meth:`mark_bad` accepts a ``reason``
  purely for logging; classification is the caller's job (see
  :mod:`locus.models.failover`).
* **Exhaustion:** when every credential is in cooldown,
  :meth:`pick` raises
  :class:`~locus.core.errors.ModelAuthError` with
  ``kind = "model_pool_exhausted"`` so downstream retry logic can
  distinguish "no more keys to try" from "current key failed".
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_validator

from locus.core.errors import ModelAuthError


logger = logging.getLogger(__name__)

__all__ = [
    "Credential",
    "CredentialPool",
]


class Credential(BaseModel):
    """One named API key."""

    model_config = {"frozen": True}

    label: str = Field(
        min_length=1,
        max_length=64,
        description=(
            "Human-readable identifier for logs / telemetry (e.g. "
            "'primary', 'openrouter-backup'). Never contains the secret."
        ),
    )
    api_key: SecretStr

    @field_validator("label")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("label must not be whitespace-only")
        return stripped


class CredentialPool:
    """Rotating pool of :class:`Credential` objects with per-entry cooldown."""

    def __init__(self, credentials: Sequence[Credential]) -> None:
        if not credentials:
            raise ValueError("CredentialPool requires at least one credential")
        labels: set[str] = set()
        for cred in credentials:
            if cred.label in labels:
                raise ValueError(f"duplicate credential label in pool: {cred.label!r}")
            labels.add(cred.label)
        self._credentials: list[Credential] = list(credentials)
        self._index: int = 0
        self._disabled_until: dict[str, datetime] = {}
        self._lock = threading.Lock()

    # ---- Introspection --------------------------------------------------

    def size(self) -> int:
        """Total credentials in the pool, including ones currently disabled."""
        return len(self._credentials)

    def available(self, *, now: datetime | None = None) -> int:
        """Count of credentials currently eligible for :meth:`pick`."""
        at = now if now is not None else datetime.now(UTC)
        with self._lock:
            return sum(1 for cred in self._credentials if self._is_available(cred, at))

    def labels(self) -> list[str]:
        """All labels in insertion order (disabled or not)."""
        return [cred.label for cred in self._credentials]

    # ---- Mutation -------------------------------------------------------

    def pick(self, *, now: datetime | None = None) -> Credential:
        """Return the next available credential, round-robin.

        Raises:
            ModelAuthError: When every credential is in cooldown. The
                error carries ``kind = "model_pool_exhausted"`` so
                retry logic can distinguish it from a per-key failure.
        """
        at = now if now is not None else datetime.now(UTC)
        with self._lock:
            size = len(self._credentials)
            for offset in range(size):
                idx = (self._index + offset) % size
                cred = self._credentials[idx]
                if self._is_available(cred, at):
                    self._index = (idx + 1) % size
                    return cred
            # All disabled — compute soonest reset for the error message.
            soonest = min(
                self._disabled_until.values(),
                default=None,
            )
        earliest = soonest.isoformat() if soonest is not None else "<unknown>"
        exc = ModelAuthError(
            f"credential pool exhausted — soonest reset at {earliest}",
        )
        # Override the shared ModelError.kind so callers can route on it.
        exc.kind = "model_pool_exhausted"
        raise exc

    def mark_bad(
        self,
        cred: Credential,
        *,
        cooldown_s: float = 60.0,
        reason: str = "",
        now: datetime | None = None,
    ) -> None:
        """Temporarily disable ``cred`` for ``cooldown_s`` seconds.

        Silently ignores credentials that are not in the pool — the
        classifier may have seen a rotated key that has since been
        removed.
        """
        if cooldown_s < 0:
            raise ValueError("cooldown_s must be non-negative")
        at = now if now is not None else datetime.now(UTC)
        with self._lock:
            if not any(c.label == cred.label for c in self._credentials):
                logger.debug(
                    "mark_bad called with unknown credential %r — ignoring",
                    cred.label,
                )
                return
            until = at + timedelta(seconds=cooldown_s)
            current = self._disabled_until.get(cred.label)
            # Extend an existing cooldown but never shorten it.
            if current is None or until > current:
                self._disabled_until[cred.label] = until
        if reason:
            logger.info(
                "credential %r disabled for %.1fs: %s",
                cred.label,
                cooldown_s,
                reason,
            )
        else:
            logger.info("credential %r disabled for %.1fs", cred.label, cooldown_s)

    def clear_cooldowns(self) -> None:
        """Re-enable every credential immediately (test / admin helper)."""
        with self._lock:
            self._disabled_until.clear()

    def state(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Return a dict summary (labels plus cooldown expiries).

        Safe to log — never includes the secret itself.
        """
        at = now if now is not None else datetime.now(UTC)
        with self._lock:
            return {
                "size": len(self._credentials),
                "available": sum(1 for cred in self._credentials if self._is_available(cred, at)),
                "disabled": {
                    label: ts.isoformat() for label, ts in self._disabled_until.items() if ts > at
                },
            }

    # ---- Internals ------------------------------------------------------

    def _is_available(self, cred: Credential, at: datetime) -> bool:
        until = self._disabled_until.get(cred.label)
        return until is None or until <= at
