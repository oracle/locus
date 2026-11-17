# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A view of a checkpointer pinned to one scope.

:class:`ScopedCheckpointer` is what ``agent.run(..., scope=...)`` writes
through: the runtime resolves the run's scope once and every checkpointer
call the run makes — its saves, loads and deletes, the tool-effect
ledger's records, a fork — goes through the view, which hands the backend
each thread id already in its stored form
(:class:`~locus.memory.scope.StoredThreadId`). Nothing is carried in a
context variable, so two runs whose events are consumed alternately in
one task, or a run whose consumer sits in another context, each keep
their own scope.

The view shares the backend's store and connections; it is not a
checkpointer of its own, and ``close()`` closes the one it views.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from locus.memory.checkpointer import BaseCheckpointer
from locus.memory.scope import CheckpointScope


if TYPE_CHECKING:
    from datetime import timedelta

    from locus.core.protocols import CheckpointerCapabilities
    from locus.core.state import AgentState
    from locus.memory.encryption import EncryptedSerializer
    from locus.memory.time_travel import CheckpointInfo


__all__ = ["ScopedCheckpointer", "pinned_checkpointer", "pinned_scope", "scoped_checkpointer"]


def _given(*args: Any) -> tuple[Any, ...]:
    """``args`` without the trailing ``None`` defaults: the call reaches ``inner`` as it was made."""
    trimmed = list(args)
    while trimmed and trimmed[-1] is None:
        trimmed.pop()
    return tuple(trimmed)


def pinned_scope(checkpointer: Any, scope: CheckpointScope | None) -> CheckpointScope:
    """The scope a run on ``checkpointer`` is pinned to: ``scope``, else the checkpointer's own.

    A checkpointer without a scope pins to the unscoped namespace
    (``CheckpointScope()``), so the result is never ``None``: the run
    remembers it, and a ``resume()`` continues under it.
    """
    if scope is not None:
        return scope
    own = getattr(checkpointer, "scope", None)
    return own if isinstance(own, CheckpointScope) else CheckpointScope()


def pinned_checkpointer(checkpointer: Any, scope: CheckpointScope | None) -> Any:
    """The checkpointer a run writes through: a view pinned to its scope, always.

    ``scope`` is the run's, resolved once when the run starts; ``None``
    pins the view to the checkpointer's own scope (:func:`pinned_scope`),
    so no call the run makes — however late, and in whatever context its
    events are consumed — ever consults a :meth:`CheckpointScope.active`
    block. ``None`` for no checkpointer.
    """
    if checkpointer is None:
        return None
    return scoped_checkpointer(checkpointer, pinned_scope(checkpointer, scope))


def scoped_checkpointer(checkpointer: Any, scope: CheckpointScope | None) -> Any:
    """``checkpointer`` as a run under ``scope`` uses it.

    ``None`` is the checkpointer itself (the run inherits its scope); a
    scope — the empty one included, which is "explicitly unscoped" — is a
    :class:`ScopedCheckpointer` over it, through
    :meth:`~locus.memory.checkpointer.BaseCheckpointer.with_scope` when
    the checkpointer has it.
    """
    if scope is None:
        return checkpointer
    if isinstance(checkpointer, BaseCheckpointer):
        return checkpointer.with_scope(scope)
    # A duck-typed checkpointer (a test double among them) is wrapped as it is.
    return ScopedCheckpointer(checkpointer, scope)


class ScopedCheckpointer(BaseCheckpointer):
    """``inner`` with every call under ``scope``, whatever scope ``inner`` has or is active.

    Built by :meth:`BaseCheckpointer.with_scope`; the runtime builds one
    per scoped run. Thread ids are keyed here and handed to ``inner`` as
    :class:`~locus.memory.scope.StoredThreadId`, which its ``_scoped``
    passes through. A checkpointer that is not a
    :class:`~locus.memory.checkpointer.BaseCheckpointer` (a duck-typed
    one) receives the stored id as a plain thread id and stores it as it
    is, which is the same key.

    Args:
        inner: The checkpointer to view.
        scope: The scope of every call; the empty scope is the unscoped
            namespace, whatever ``inner`` was built with.
    """

    def __init__(self, inner: Any, scope: CheckpointScope) -> None:
        if not isinstance(scope, CheckpointScope):
            raise TypeError("scope must be a CheckpointScope")
        self.inner = inner
        self._view_scope = scope

    # -- the settings are the inner's ------------------------------------------

    @property
    def scope(self) -> CheckpointScope | None:  # type: ignore[override]
        """The scope of every call through this view; ``None`` for the unscoped namespace."""
        return None if self._view_scope.is_empty else self._view_scope

    @property
    def ttl(self) -> timedelta | None:  # type: ignore[override]
        return getattr(self.inner, "ttl", None)

    @property
    def encryption(self) -> EncryptedSerializer | None:  # type: ignore[override]
        return getattr(self.inner, "encryption", None)

    @property
    def capabilities(self) -> CheckpointerCapabilities:
        capabilities: CheckpointerCapabilities = self.inner.capabilities
        return capabilities

    @property
    def deletes_single_checkpoints(self) -> bool:
        # Strictly ``True``: a duck-typed inner without the property keeps every save.
        return getattr(self.inner, "deletes_single_checkpoints", False) is True

    def _effective_scope(self) -> CheckpointScope | None:
        return self.scope

    def _stored(self, thread_id: str) -> str:
        return self._view_scope.key(thread_id)

    def with_scope(self, scope: CheckpointScope | None) -> BaseCheckpointer:
        """A view of the same checkpointer under another scope (``None``: the inner's own)."""
        if scope is None:
            inner: BaseCheckpointer = self.inner
            return inner
        return ScopedCheckpointer(self.inner, scope)

    # -- core ------------------------------------------------------------------

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        saved: str = await self.inner.save(
            state, *_given(self._stored(thread_id), checkpoint_id, metadata)
        )
        return saved

    async def load(self, thread_id: str, checkpoint_id: str | None = None) -> AgentState | None:
        loaded: AgentState | None = await self.inner.load(
            *_given(self._stored(thread_id), checkpoint_id)
        )
        return loaded

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
        ids: list[str] = await self.inner.list_checkpoints(self._stored(thread_id), limit)
        return ids

    async def delete(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        deleted: bool = await self.inner.delete(*_given(self._stored(thread_id), checkpoint_id))
        return deleted

    async def exists(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        exists = getattr(self.inner, "exists", None)
        if exists is None:
            return await super().exists(thread_id, checkpoint_id)
        present: bool = await exists(*_given(self._stored(thread_id), checkpoint_id))
        return present

    async def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if close is not None:
            await close()

    # -- extended --------------------------------------------------------------

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = await self.inner.search(query, limit=limit)
        return results

    async def query_by_metadata(
        self, key: str, value: Any, limit: int = 100
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = await self.inner.query_by_metadata(key, value, limit=limit)
        return results

    async def get_metadata(
        self, thread_id: str, checkpoint_id: str | None = None
    ) -> dict[str, Any] | None:
        meta: dict[str, Any] | None = await self.inner.get_metadata(
            *_given(self._stored(thread_id), checkpoint_id)
        )
        return meta

    async def vacuum(self, older_than_days: int = 30) -> int:
        removed: int = await self.inner.vacuum(older_than_days)
        return removed

    async def copy_thread(self, source_thread_id: str, dest_thread_id: str) -> bool:
        copied: bool = await self.inner.copy_thread(
            self._stored(source_thread_id), self._stored(dest_thread_id)
        )
        return copied

    async def list_threads(
        self, limit: int = 100, pattern: str = "*", *, scope: CheckpointScope | None = None
    ) -> list[str]:
        threads: list[str] = await self.inner.list_threads(
            limit=limit, pattern=pattern, scope=scope or self._view_scope
        )
        return threads

    async def expire(self) -> int:
        removed: int = await self.inner.expire()
        return removed

    async def list_with_metadata(self, limit: int = 100) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = await self.inner.list_with_metadata(limit=limit)
        return items

    # ``fork`` is the base default over this view's load / list / save, so
    # the fork and its ledger land in the view's scope.

    def __repr__(self) -> str:
        return f"ScopedCheckpointer({self.inner!r}, scope={self._view_scope})"

    # Last on purpose: a method named ``list`` shadows the builtin for the
    # annotations of anything defined after it.
    async def list(self, thread_id: str, limit: int = 10) -> list[CheckpointInfo]:
        infos: list[CheckpointInfo] = await self.inner.list(self._stored(thread_id), limit)
        # The backend saw the stored id; the caller named the thread.
        return [replace(info, thread_id=thread_id) for info in infos]
