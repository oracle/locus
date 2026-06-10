# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Backends for the deepagent filesystem-as-memory toolset.

- :class:`StateBackend` — in-memory ephemeral. The default when
  ``create_deepagent(enable_filesystem=True)`` is set without an
  explicit backend.
- :class:`FilesystemBackend` — rooted on-disk, opt-in. Persists the
  agent's scratchspace beyond the run (or shares it between runs
  with the same root).
- :class:`BackendProtocol` — the typed surface tools call into.
- :class:`BackendError`, :class:`FileInfo`, :class:`Match` — error
  contract and result types.
"""

from tulip.deepagent.backends.filesystem import FilesystemBackend
from tulip.deepagent.backends.protocol import (
    BackendError,
    BackendProtocol,
    ErrorCode,
    FileInfo,
    Match,
)
from tulip.deepagent.backends.state import StateBackend


__all__ = [
    "BackendError",
    "BackendProtocol",
    "ErrorCode",
    "FileInfo",
    "FilesystemBackend",
    "Match",
    "StateBackend",
]
