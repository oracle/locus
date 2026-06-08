"""Specialised memory managers.

Where :mod:`tulip.memory.manager` ships the framework-level memory
contracts (:class:`BaseMemoryManager`, :class:`LLMMemoryManager`) that
work against any :class:`~tulip.memory.store.BaseStore`, this
subpackage holds higher-fidelity memory managers built on top of
purpose-built external libraries.

:class:`Mem0MemoryManager` delegates to the open-source
`mem0 <https://pypi.org/project/mem0ai/>`_ memory layer, bringing in
fact extraction, scoped retrieval by user/agent/thread, and a
self-hostable vector store. The portable ``LLMMemoryManager`` stays
first-class for plain backends (InMemory / Redis / Postgres /
OpenSearch) and for test environments where pulling in ``mem0ai`` would
be overkill.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from tulip.memory.managers.mem0 import Mem0MemoryManager

__all__ = ["Mem0MemoryManager"]


def __getattr__(name: str) -> Any:  # pragma: no cover - lazy export wiring
    # Lazy-load the Mem0-backed manager so importing
    # ``tulip.memory.managers`` (or even ``tulip.memory``) does not
    # require the optional ``mem0ai`` dependency to be installed. Only
    # consumers that actually construct the class pay the import cost.
    if name == "Mem0MemoryManager":
        from tulip.memory.managers.mem0 import Mem0MemoryManager  # noqa: PLC0415

        return Mem0MemoryManager
    raise AttributeError(f"module 'tulip.memory.managers' has no attribute {name!r}")
