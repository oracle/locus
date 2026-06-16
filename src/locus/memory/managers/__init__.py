# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Specialised memory managers.

Where :mod:`locus.memory.manager` ships the framework-level memory
contracts (:class:`BaseMemoryManager`, :class:`LLMMemoryManager`) that
work against any :class:`~locus.memory.store.BaseStore`, this
subpackage holds higher-fidelity memory managers built on top of
purpose-built external libraries.

For Oracle 26ai deployments, prefer
:class:`OracleAgentMemoryManager` — it delegates to Oracle's official
``oracleagentmemory`` client (the same one published at
https://pypi.org/project/oracleagentmemory/) and brings in
context-card summaries, scoped retrieval by user/agent/thread, and
LongMemEval-benchmarked extraction quality. The portable
``LLMMemoryManager`` stays first-class for non-Oracle backends
(InMemory / Redis / Postgres / OpenSearch) and for test
environments where pulling in `oracleagentmemory` would be overkill.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from locus.memory.managers.oracle_agent_memory import OracleAgentMemoryManager

__all__ = ["OracleAgentMemoryManager"]


def __getattr__(name: str) -> Any:  # pragma: no cover - lazy export wiring
    # Lazy-load the Oracle-backed manager so importing
    # ``locus.memory.managers`` (or even ``locus.memory``) does not
    # require the optional ``oracleagentmemory`` dependency to be
    # installed. Only consumers that actually construct the class pay
    # the import cost.
    if name == "OracleAgentMemoryManager":
        from locus.memory.managers.oracle_agent_memory import (  # noqa: PLC0415
            OracleAgentMemoryManager,
        )

        return OracleAgentMemoryManager
    raise AttributeError(f"module 'locus.memory.managers' has no attribute {name!r}")
