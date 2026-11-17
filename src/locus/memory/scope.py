# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Per-application and per-user scoping of checkpoint threads.

A multi-tenant deployment keeps one application's threads apart from
another's, and one user's from the next, without the caller folding the
tenancy into every ``thread_id``. A :class:`CheckpointScope` names the
application and the user; the checkpointer stores a thread under
``app=<app_id>/user=<user_id>/<thread_id>`` (a key prefix on a key-value
store, a directory prefix on files, an object-key prefix in a bucket, the
``thread_id`` column's prefix in a table), and reads it back under the
same scope only. From another scope the same ``thread_id`` is a miss, not
another tenant's thread.

Two ways to give a checkpointer its scope:

* fixed for the instance: ``MemoryCheckpointer(scope=CheckpointScope(...))``;
* per run: ``agent.run(prompt, thread_id=..., scope=CheckpointScope(...))``
  (also ``arun``, ``run_sync``, ``resume`` and ``continue_turn``), which
  activates the scope for every checkpointer call the run makes, the
  tool-effect ledger's included. :meth:`CheckpointScope.active` does the
  same around any block of your own.

A run-level scope takes precedence over the instance's. Threads written
with no scope are stored under their bare id and stay reachable from a
checkpointer with no scope, so turning scoping on changes nothing for
existing data. ``list_threads()`` lists one scope at a time: the
checkpointer's own by default, ``scope=`` for another, and the unscoped
threads for a checkpointer without a scope.
"""

from __future__ import annotations

import contextvars
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass


__all__ = ["CheckpointScope", "current_scope"]

#: What an ``app_id`` or ``user_id`` may be made of. The ids become part of a
#: storage key, so the separators (``/``), the glob and SQL ``LIKE``
#: metacharacters and whitespace are kept out of them.
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]*$")
#: A stored thread id that carries a scope.
_SCOPED = re.compile(r"^(app|user)=[^/]+(/|$)")

_current: contextvars.ContextVar[CheckpointScope | None] = contextvars.ContextVar(
    "locus_checkpoint_scope", default=None
)


def _check_id(value: str | None, field: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not _ID.match(value):
        raise ValueError(
            f"{field} must be a non-empty string of letters, digits and '. _ : @ + -' "
            f"(it becomes part of a storage key); got {value!r}"
        )


@dataclass(frozen=True)
class CheckpointScope:
    """The application and user a thread belongs to.

    Attributes:
        app_id: The application (a product, a service, a deployment) the
            thread belongs to; ``None`` for no application boundary.
        user_id: The end user the thread belongs to; ``None`` for no user
            boundary.

    Both may be set, either, or neither; a scope with neither is the
    unscoped namespace. The ids are made of letters, digits and ``. _ : @
    + -``: they become part of a storage key.

    Example:
        >>> scope = CheckpointScope(app_id="billing", user_id="u-42")
        >>> scope.key("case-4821")
        'app=billing/user=u-42/case-4821'
        >>> with scope.active():
        ...     await checkpointer.load("case-4821")  # app=billing/user=u-42/case-4821
    """

    app_id: str | None = None
    user_id: str | None = None

    def __post_init__(self) -> None:
        _check_id(self.app_id, "app_id")
        _check_id(self.user_id, "user_id")

    @property
    def is_empty(self) -> bool:
        """Whether this is the unscoped namespace (no application, no user)."""
        return self.app_id is None and self.user_id is None

    @property
    def prefix(self) -> str:
        """What the scope puts in front of a thread id in storage (``""`` when empty)."""
        parts = []
        if self.app_id is not None:
            parts.append(f"app={self.app_id}/")
        if self.user_id is not None:
            parts.append(f"user={self.user_id}/")
        return "".join(parts)

    def key(self, thread_id: str) -> str:
        """The id ``thread_id`` is stored under in this scope."""
        return f"{self.prefix}{thread_id}"

    def owns(self, stored_id: str) -> bool:
        """Whether a stored id belongs to this scope.

        The unscoped namespace owns the ids that carry no scope at all, so a
        listing without a scope never shows another scope's threads.
        """
        if self.is_empty:
            return _SCOPED.match(stored_id) is None
        return stored_id.startswith(self.prefix)

    def unkey(self, stored_id: str) -> str | None:
        """The thread id a stored id names in this scope; ``None`` when it is not in it."""
        if not self.owns(stored_id):
            return None
        return stored_id[len(self.prefix) :]

    def strip(self, stored_ids: Iterable[str]) -> list[str]:
        """The thread ids of this scope among ``stored_ids``, their prefix removed, in order."""
        found: list[str] = []
        for stored in stored_ids:
            thread_id = self.unkey(stored)
            if thread_id is not None:
                found.append(thread_id)
        return found

    @contextmanager
    def active(self) -> Iterator[None]:
        """Make this the scope of every checkpointer call inside the block.

        Context-local: an agent run started inside the block, and any task
        it spawns, keeps the scope; a concurrent run outside it does not see
        it. Inside the block the scope takes precedence over the one a
        checkpointer was constructed with.
        """
        token = _current.set(self)
        try:
            yield
        finally:
            try:
                _current.reset(token)
            except ValueError:
                # Reset from another context than the one that entered (a
                # generator driven from two tasks): clear instead of leaking.
                _current.set(None)

    def __str__(self) -> str:
        return self.prefix.rstrip("/") or "<unscoped>"


def current_scope() -> CheckpointScope | None:
    """The scope a :meth:`CheckpointScope.active` block (or a scoped run) has set, if any."""
    return _current.get()
