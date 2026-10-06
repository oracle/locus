# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""What the agent has read, so it only changes what it has seen.

Each file the agent reads is recorded as it was then: its content hash, or
its size and modification time when the backend did not hash it. A change
is refused unless the file is still as the agent last saw it — the rule
opencode and Claude Code both enforce. Without it, an edit written against
a file the agent never read, or one that a formatter, a ``git checkout`` or
a person changed since, lands on text the model has not seen.

The hash is what is compared when there is one. A ``touch`` or a checkout
that rewrites identical bytes changes the modification time and nothing the
agent could have got wrong, so it does not make the file stale.
"""

from __future__ import annotations

import threading

from tulip.deepagent.backends.protocol import BackendError
from tulip.harness.backend import FileStat, WorkspaceBackend


__all__ = ["ReadLedger"]


def _same(seen: FileStat, now: FileStat) -> bool:
    if seen.sha256 and now.sha256:
        return seen.sha256 == now.sha256
    return (seen.size, seen.mtime) == (now.size, now.mtime)


class ReadLedger:
    """The files read in one session, keyed by resolved path.

    Args:
        require_read: Whether a change needs a read first. ``False`` turns
            the rule off; the ledger still records reads.
    """

    def __init__(self, *, require_read: bool = True) -> None:
        self.require_read = require_read
        self._seen: dict[str, FileStat] = {}
        self._lock = threading.Lock()

    def note(self, backend: WorkspaceBackend, path: str) -> None:
        """Record that the agent now knows ``path`` as it is."""
        try:
            stat = backend.stat(path)
        except BackendError:
            return
        with self._lock:
            self._seen[backend.resolve(path)] = stat

    def forget(self) -> None:
        """Forget every read — a new session starts knowing nothing."""
        with self._lock:
            self._seen.clear()

    def unseen(self, backend: WorkspaceBackend, path: str, shown: str) -> str | None:
        """Why the agent may not change ``path`` yet, or ``None`` when it may.

        ``shown`` is the path as the model wrote it, for the message.
        """
        if not self.require_read:
            return None
        try:
            now = backend.stat(path)
        except BackendError:
            # Not there (any more): a create, which needs no read.
            return None
        with self._lock:
            seen = self._seen.get(backend.resolve(path))
        if seen is None:
            return (
                f"{shown} has not been read in this session — read it first, so the change "
                "is made against what is there. Nothing was written."
            )
        if not _same(seen, now):
            return (
                f"{shown} has changed since you last read it — read it again before "
                "changing it. Nothing was written."
            )
        return None
