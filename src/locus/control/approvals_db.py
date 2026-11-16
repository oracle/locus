# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""What the database-backed approval stores share: the state machine over rows.

:class:`~locus.control.approvals.FileApprovals` loads and saves the whole store
on every call. That is right for one file and wrong for a database, where a
decision made on one replica must be seen by the next call on another, and two
replicas redeeming the same approval must not both run the side effect. So the
stores here work one record at a time, under the database's own locking:

* ``submit`` looks the call up by digest with the rows locked, and inserts a
  new record only when no live one exists. A concurrent insert of the same
  record loses on the primary key and re-reads.
* ``decide`` and ``consume`` lock the row, apply the same transition the
  in-memory store applies (:func:`~locus.control.approvals._transition`,
  :func:`~locus.control.approvals._consumed`), and write it back. Two consumers
  of one approval serialise on the row, and the second finds it consumed.

The drivers the SDK already uses for checkpoints (``asyncpg``, ``redis.asyncio``,
``oracledb``'s async API) are asynchronous, and :class:`ApprovalStore` is a
synchronous protocol: the gate calls ``store.submit(...)`` from inside a tool,
the command line calls it with no event loop at all. Each store therefore owns
one event loop on a background thread, where its connection pool lives for the
life of the store. The synchronous methods block on that loop; the ``a``-prefixed
variants (``asubmit``, ``adecide``, ...) await it, so an application that is
already asynchronous never blocks its own loop on a database round trip.

A subclass provides the connection and a :class:`Transaction` over it; this
module provides everything else.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Protocol, TypeVar
from urllib.parse import urlsplit, urlunsplit

from locus.control.approvals import (
    _LIVE,
    ApprovalAuthority,
    ApprovalAuthorityError,
    ApprovalRecord,
    Verdict,
    _check_decision,
    _consumed,
    _edited_arguments,
    _new_record,
    _transition,
    call_digest,
)


if TYPE_CHECKING:
    from concurrent.futures import Future


__all__ = ["DatabaseApprovals", "Transaction", "record_from_json", "record_to_json"]

T = TypeVar("T")

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

#: How many times ``submit`` re-reads after losing an insert race.
_SUBMIT_ATTEMPTS = 3


def sql_identifier(value: str, what: str) -> str:
    """``value`` if it is safe to splice into DDL as a table or schema name."""
    if not _IDENTIFIER.match(value):
        raise ValueError(
            f"invalid {what}: {value!r}. Must start with a letter or underscore and contain "
            "only letters, digits and underscores (max 63 chars)."
        )
    return value


def record_to_json(record: ApprovalRecord) -> str:
    """The record as the JSON document a row stores."""
    return json.dumps(asdict(record), sort_keys=True, default=str)


def record_from_json(document: str | bytes | Mapping[str, Any]) -> ApprovalRecord:
    """The record a row stores, from its JSON document (or an already-parsed one)."""
    data = document if isinstance(document, Mapping) else json.loads(document)
    return ApprovalRecord(**data)


def without_password(url: str) -> str:
    """``url`` with any password removed, for logs and checkpoints."""
    parts = urlsplit(url)
    if parts.password is None:
        return url
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{parts.username}@{host}" if parts.username else host
    if parts.port is not None:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


class Transaction(Protocol):
    """One unit of work against the rows, with the database's locks held.

    A store yields one from :meth:`DatabaseApprovals._transaction`; the state
    machine calls these and nothing else. ``lock=True`` asks for the rows to be
    locked until the unit of work ends, so a concurrent decision on the same
    record waits rather than interleaves.
    """

    async def get(self, approval_id: str, *, lock: bool = False) -> ApprovalRecord | None:
        """The record, or ``None``."""
        ...

    async def same_call(self, digest: str, *, lock: bool = False) -> list[ApprovalRecord]:
        """Every record of a call with this digest."""
        ...

    async def insert(self, record: ApprovalRecord) -> bool:
        """Add a new record; ``False`` when one with its id already exists."""
        ...

    async def update(self, record: ApprovalRecord) -> None:
        """Write a changed record over the one with its id."""
        ...

    async def waiting(self) -> list[ApprovalRecord]:
        """Every record whose status is ``pending``."""
        ...


class _Loop:
    """An event loop on a daemon thread, started on first use."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._guard = threading.Lock()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        with self._guard:
            if self._loop is None:
                ready = threading.Event()
                loop = asyncio.new_event_loop()

                def run() -> None:
                    asyncio.set_event_loop(loop)
                    ready.set()
                    loop.run_forever()

                self._thread = threading.Thread(target=run, name=self._name, daemon=True)
                self._thread.start()
                ready.wait()
                self._loop = loop
            return self._loop

    def submit(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self) -> None:
        with self._guard:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
        loop.close()


class DatabaseApprovals:
    """An :class:`~locus.control.approvals.ApprovalStore` over database rows.

    Subclasses implement :meth:`_transaction` (and :meth:`_disconnect`); the
    submit / decide / consume state machine, the approver authority, and the
    sync-over-async plumbing are here. ``authority`` is as for
    :class:`~locus.control.approvals.InMemoryApprovals`.
    """

    def __init__(
        self,
        authority: ApprovalAuthority
        | Callable[[ApprovalRecord], ApprovalAuthority | None]
        | None = None,
    ) -> None:
        self.authority = authority
        self._loop = _Loop(f"{type(self).__name__}-io")

    # -- what a subclass provides -------------------------------------------

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[Transaction]:
        """A unit of work; committed when the block ends, rolled back on error."""
        raise NotImplementedError
        yield  # pragma: no cover - makes this an async generator for the type checker

    async def _disconnect(self) -> None:
        """Release the connection pool. Called once, on the store's loop."""

    @property
    def spec(self) -> str:
        """The store spec, as :func:`locus.control.approval_store` reads it; no secrets."""
        raise NotImplementedError

    # -- authority -------------------------------------------------------------

    def authority_for(self, record: ApprovalRecord) -> ApprovalAuthority | None:
        """The authority ``record``'s decisions are checked against, if any."""
        if self.authority is None or isinstance(self.authority, ApprovalAuthority):
            return self.authority
        return self.authority(record)

    # -- the state machine, on the store's loop ------------------------------

    async def _do_submit(
        self,
        principal: str,
        tool: str,
        args: Mapping[str, Any],
        *,
        reason: str = "",
        labels: Iterable[str] = (),
        context: Mapping[str, str] | None = None,
    ) -> str:
        """:meth:`submit`, on the store's loop."""
        digest = call_digest(principal, tool, args, context)
        for _ in range(_SUBMIT_ATTEMPTS):
            async with self._transaction() as txn:
                same_call = await txn.same_call(digest, lock=True)
                live = [r for r in same_call if r.status in _LIVE]
                if live:
                    return max(live, key=lambda r: r.created_at).approval_id
                record = _new_record(
                    digest,
                    len(same_call),
                    principal,
                    tool,
                    args,
                    reason=reason,
                    labels=labels,
                    context=context,
                )
                if await txn.insert(record):
                    return record.approval_id
            # Another submitter inserted the same record first: read theirs.
        raise RuntimeError(f"could not submit approval for {tool!r}: the store kept changing")

    async def _do_get(self, approval_id: str) -> ApprovalRecord | None:
        """:meth:`get`, on the store's loop."""
        async with self._transaction() as txn:
            return await txn.get(approval_id)

    async def _do_state(self, approval_id: str) -> str | None:
        """:meth:`state`, on the store's loop."""
        record = await self._do_get(approval_id)
        return record.status if record is not None else None

    async def _do_decide(
        self,
        approval_id: str,
        verdict: Verdict,
        *,
        by: str,
        arguments: Mapping[str, Any] | None = None,
        reason: str = "",
    ) -> ApprovalRecord:
        """:meth:`decide`, on the store's loop."""
        _check_decision(verdict, by)
        refused: ApprovalAuthorityError | None = None
        async with self._transaction() as txn:
            record = await txn.get(approval_id, lock=True)
            if record is None:
                raise KeyError(approval_id)
            edited = _edited_arguments(record, verdict, arguments)
            try:
                decided = _transition(
                    record, verdict, by, edited, self.authority_for(record), reason
                )
            except ApprovalAuthorityError as error:
                # The refusal is kept on the record, so it must commit: raising
                # through the transaction would roll it back.
                if error.record is not None:
                    await txn.update(error.record)
                refused = error
            else:
                await txn.update(decided)
        if refused is not None:
            raise refused
        return decided

    async def _do_consume(self, approval_id: str) -> ApprovalRecord:
        """:meth:`consume`, on the store's loop."""
        async with self._transaction() as txn:
            record = await txn.get(approval_id, lock=True)
            if record is None:
                raise KeyError(approval_id)
            consumed = _consumed(record)
            await txn.update(consumed)
            return consumed

    async def _do_pending(self) -> list[ApprovalRecord]:
        """:meth:`pending`, on the store's loop."""
        async with self._transaction() as txn:
            waiting = await txn.waiting()
        return sorted(waiting, key=lambda r: r.created_at)

    # -- the same, awaited from any loop --------------------------------------

    async def _await(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run ``coro`` on the store's loop and await its result here."""
        return await asyncio.wrap_future(self._loop.submit(coro))

    async def asubmit(
        self,
        principal: str,
        tool: str,
        args: Mapping[str, Any],
        *,
        reason: str = "",
        labels: Iterable[str] = (),
        context: Mapping[str, str] | None = None,
    ) -> str:
        """:meth:`submit`, awaited."""
        return await self._await(
            self._do_submit(principal, tool, args, reason=reason, labels=labels, context=context)
        )

    async def aget(self, approval_id: str) -> ApprovalRecord | None:
        """:meth:`get`, awaited."""
        return await self._await(self._do_get(approval_id))

    async def astate(self, approval_id: str) -> str | None:
        """:meth:`state`, awaited."""
        return await self._await(self._do_state(approval_id))

    async def adecide(
        self,
        approval_id: str,
        verdict: Verdict,
        *,
        by: str,
        arguments: Mapping[str, Any] | None = None,
        reason: str = "",
    ) -> ApprovalRecord:
        """:meth:`decide`, awaited."""
        return await self._await(
            self._do_decide(approval_id, verdict, by=by, arguments=arguments, reason=reason)
        )

    async def aconsume(self, approval_id: str) -> ApprovalRecord:
        """:meth:`consume`, awaited."""
        return await self._await(self._do_consume(approval_id))

    async def apending(self) -> list[ApprovalRecord]:
        """:meth:`pending`, awaited."""
        return await self._await(self._do_pending())

    async def aclose(self) -> None:
        """Release the connection pool and stop the store's loop."""
        await self._await(self._disconnect())
        self._loop.stop()

    # -- the ApprovalStore protocol, synchronous ------------------------------

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        return self._loop.submit(coro).result()

    def submit(
        self,
        principal: str,
        tool: str,
        args: Mapping[str, Any],
        *,
        reason: str = "",
        labels: Iterable[str] = (),
        context: Mapping[str, str] | None = None,
    ) -> str:
        """Record a held call, or return the live record for the same call."""
        return self._run(
            self._do_submit(principal, tool, args, reason=reason, labels=labels, context=context)
        )

    def state(self, approval_id: str) -> str | None:
        """The record's status, or ``None`` for an unknown id."""
        return self._run(self._do_state(approval_id))

    def get(self, approval_id: str) -> ApprovalRecord | None:
        """The full record, or ``None`` for an unknown id."""
        return self._run(self._do_get(approval_id))

    def decide(
        self,
        approval_id: str,
        verdict: Verdict,
        *,
        by: str,
        arguments: Mapping[str, Any] | None = None,
        reason: str = "",
    ) -> ApprovalRecord:
        """Approve or deny a pending record, naming who decided and why."""
        return self._run(
            self._do_decide(approval_id, verdict, by=by, arguments=arguments, reason=reason)
        )

    def consume(self, approval_id: str) -> ApprovalRecord:
        """Mark a decided record as acted on, so it cannot be used again."""
        return self._run(self._do_consume(approval_id))

    def pending(self) -> list[ApprovalRecord]:
        """Records still waiting for a decision, oldest first."""
        return self._run(self._do_pending())

    def close(self) -> None:
        """Release the connection pool and stop the store's loop."""
        self._run(self._disconnect())
        self._loop.stop()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.spec!r})"
