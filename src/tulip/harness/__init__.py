# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A coding harness: workspace tools, written once, run anywhere.

The tools every coding agent converges on — read, write, edit, glob, grep,
ls, bash and its background handles, apply_patch, notebook_edit, a todo
list — written once against :class:`WorkspaceBackend` and run unchanged on
the host (:class:`LocalBackend`), in memory (:class:`MemoryBackend`), in any
sandbox that can run a command and move a file (:class:`SessionBackend`), or
in an NVIDIA OpenShell sandbox (:class:`OpenShellBackend`, the
``openshell`` extra).

The tool bodies never gate. :func:`build_harness` pairs each tool with the
action spec that labels its calls (:mod:`tulip.harness.labels`) and applies
one ``wrap`` to all of them — the single place a gate goes. Without one, the
tools are ungated. Every command the harness runs produces an
:class:`ExecRecord` (:mod:`tulip.harness.evidence`).

    from tulip.harness import LocalBackend, build_harness

    harness = build_harness(LocalBackend("."), wrap=my_gate)
    agent = Agent(model=model, tools=harness.tools, system_prompt=harness.prompt_fragment)

See ``docs/harness.md``.
"""

from tulip.harness.backend import (
    SKIP_DIRS,
    BackendCapabilities,
    BackendError,
    ExecResult,
    ExecUnsupportedError,
    FileStat,
    JobError,
    JobOutput,
    JobStatus,
    LineWindow,
    WorkspaceBackend,
)
from tulip.harness.evidence import ExecRecord
from tulip.harness.labels import (
    KIND_EXEC,
    KIND_NETWORK,
    KIND_READ,
    KIND_WRITE,
    CommandClass,
    action_spec,
    classify_command,
)
from tulip.harness.ledger import ReadLedger
from tulip.harness.local import LocalBackend
from tulip.harness.openshell import OpenShellBackend, OpenShellSession
from tulip.harness.session import SessionBackend, SessionLike
from tulip.harness.state import MemoryBackend
from tulip.harness.tools.common import FileChange, HarnessConfig, HarnessContext
from tulip.harness.toolset import (
    ALL_TOOLS,
    DEFAULT_TOOLS,
    PATCH_TOOLS,
    Harness,
    build_harness,
)


__all__ = [
    "ALL_TOOLS",
    "DEFAULT_TOOLS",
    "KIND_EXEC",
    "KIND_NETWORK",
    "KIND_READ",
    "KIND_WRITE",
    "PATCH_TOOLS",
    "SKIP_DIRS",
    "BackendCapabilities",
    "BackendError",
    "CommandClass",
    "ExecRecord",
    "ExecResult",
    "ExecUnsupportedError",
    "FileChange",
    "FileStat",
    "Harness",
    "HarnessConfig",
    "HarnessContext",
    "JobError",
    "JobOutput",
    "JobStatus",
    "LineWindow",
    "LocalBackend",
    "MemoryBackend",
    "OpenShellBackend",
    "OpenShellSession",
    "ReadLedger",
    "SessionBackend",
    "SessionLike",
    "WorkspaceBackend",
    "action_spec",
    "build_harness",
    "classify_command",
]
