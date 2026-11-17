# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Per-application and per-user scoping of checkpoint threads.

A multi-tenant deployment keeps one application's threads apart from
another's, and one user's from the next, without the caller folding the
tenancy into every ``thread_id``. A :class:`CheckpointScope` names the
application and the user; the checkpointer stores a thread under
``app=<app_id>/user=<user_id>/<thread_id>`` (a key prefix on a key-value
store, a directory on files, an object-key prefix in a bucket, the
``thread_id`` column's prefix in a table), and reads it back under the
same scope only. From another scope the same ``thread_id`` is a miss, not
another tenant's thread.

**The stored id is a grammar, and no input can forge it.** Each segment —
the ``app_id``, the ``user_id`` and the thread id — is percent-encoded
(:func:`urllib.parse.quote` with nothing kept safe) before it is joined,
so a ``/`` or ``=`` inside any of them cannot form a scope marker:
``load("user=victim/t")`` under an app-only scope is stored as
``app=billing/user%3Dvictim%2Ft``, a thread of its own, never the user's.
A scope resolves only the ids that are exactly one encoded segment past
its own prefix: an app-only scope lists and loads the app-level threads,
not any user's under it. The unscoped namespace stores a thread under its
bare id, as every earlier release did, except for an id that could pass
for a scope marker (``app=…``, ``user=…`` or their encoded forms), which
is percent-encoded whole; a stored id that carries a scope is never a
thread of the unscoped namespace.

Two ways to give a checkpointer its scope:

* fixed for the instance: ``MemoryCheckpointer(scope=CheckpointScope(...))``;
* per run: ``agent.run(prompt, thread_id=..., scope=CheckpointScope(...))``
  (also ``arun``, ``run_sync``, ``resume`` and ``continue_turn``). The
  runtime resolves the run's scope once, when the run starts, and hands
  every checkpointer call it makes — the tool-effect ledger's included —
  a view of the checkpointer pinned to that scope
  (:meth:`~locus.memory.checkpointer.BaseCheckpointer.with_scope`), so
  nothing about the run depends on the context its events are consumed
  in. ``scope=None`` inherits the checkpointer's own scope;
  ``scope=CheckpointScope()`` runs explicitly unscoped.

:meth:`CheckpointScope.active` is a convenience for code of your own: it
makes the scope the default of every direct checkpointer call inside the
block (a run started inside the block picks it up as its scope). It is a
plain context variable: enter and leave it in the same task, around a
block, not across a generator's suspension points.

Threads written with no scope stay reachable from a checkpointer with no
scope, so turning scoping on changes nothing for existing data.
``list_threads()`` lists one scope at a time: the checkpointer's own by
default, ``scope=`` for another, and the unscoped threads for a
checkpointer without a scope.
"""

from __future__ import annotations

import contextvars
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import quote, unquote


__all__ = ["CheckpointScope", "StoredThreadId", "current_scope"]

#: What an ``app_id`` or ``user_id`` may be made of. The ids become part of a
#: storage key, so the separators (``/``), the glob and SQL ``LIKE``
#: metacharacters and whitespace are kept out of them.
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]*$")
#: A stored id that carries a scope: it starts with a scope marker.
_SCOPED = re.compile(r"^(app|user)=[^/]+(/|$)")
#: An unscoped thread id that could pass for a scope marker, or decode to
#: one: stored percent-encoded whole, so the stored namespaces never meet.
_LOOKALIKE = re.compile(r"^(app|user)[=%]")
#: One encoded segment: no separator and no marker character left in it.
_SEGMENT = re.compile(r"^[^/=]+$")

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


def _encode(segment: str) -> str:
    """A segment of a stored id: percent-encoded, nothing kept safe but the unreserved set."""
    return quote(segment, safe="")


class StoredThreadId(str):  # noqa: SLOT000 — a str subtype cannot declare a slot for thread_id
    """A thread id already in its stored form (its scope applied).

    :meth:`BaseCheckpointer._scoped <locus.memory.checkpointer.BaseCheckpointer._scoped>`
    returns one and passes one through unchanged, which is how a scoped
    view of a checkpointer hands the backend the id to store under without
    the backend applying its own scope on top. ``thread_id`` is the id the
    caller named, for a backend that keys something else on it (a tenant).
    """

    thread_id: str

    def __new__(cls, stored: str, thread_id: str | None = None) -> StoredThreadId:
        self = super().__new__(cls, stored)
        self.thread_id = thread_id if thread_id is not None else stored
        return self


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
    + -``: they become part of a storage key (percent-encoded, so
    ``ann@example.com`` is stored as ``ann%40example.com``).

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
            parts.append(f"app={_encode(self.app_id)}/")
        if self.user_id is not None:
            parts.append(f"user={_encode(self.user_id)}/")
        return "".join(parts)

    def key(self, thread_id: str) -> StoredThreadId:
        """The id ``thread_id`` is stored under in this scope.

        A scoped thread id is percent-encoded behind the prefix, so no
        character of it can extend the scope grammar. An unscoped id is
        stored bare, unless it could pass for a scope marker.
        """
        if self.is_empty:
            stored = _encode(thread_id) if _LOOKALIKE.match(thread_id) else thread_id
        else:
            stored = f"{self.prefix}{_encode(thread_id)}"
        return StoredThreadId(stored, thread_id)

    def unkey(self, stored_id: str) -> str | None:
        """The thread id a stored id names in this scope; ``None`` when it is not in it.

        A stored id is in the scope when it is the scope's prefix followed
        by exactly one encoded segment: a user's thread under the same
        application is not an app-only scope's, and a scoped id is never
        the unscoped namespace's.
        """
        if self.is_empty:
            if _SCOPED.match(stored_id):
                return None
            return unquote(stored_id) if _LOOKALIKE.match(stored_id) else stored_id
        if not stored_id.startswith(self.prefix):
            return None
        segment = stored_id[len(self.prefix) :]
        if not _SEGMENT.match(segment):
            return None
        return unquote(segment)

    def owns(self, stored_id: str) -> bool:
        """Whether a stored id names a thread of this scope (see :meth:`unkey`)."""
        return self.unkey(stored_id) is not None

    def strip(self, stored_ids: Iterable[str]) -> list[str]:
        """The thread ids of this scope among ``stored_ids``, decoded, in order."""
        found: list[str] = []
        for stored in stored_ids:
            thread_id = self.unkey(stored)
            if thread_id is not None:
                found.append(thread_id)
        return found

    @contextmanager
    def active(self) -> Iterator[None]:
        """Make this the default scope of every direct checkpointer call in the block.

        Context-local: a task spawned inside the block inherits it, a
        concurrent task outside it does not. Inside the block the scope
        takes precedence over the one a checkpointer was constructed with,
        and a run started inside it takes it as its scope. Enter and leave
        the block in the same task; across a generator's suspension points
        the context may differ, and the reset then raises.
        """
        token = _current.set(self)
        try:
            yield
        finally:
            _current.reset(token)

    def __str__(self) -> str:
        return self.prefix.rstrip("/") or "<unscoped>"


def current_scope() -> CheckpointScope | None:
    """The scope a :meth:`CheckpointScope.active` block has set, if any."""
    return _current.get()
