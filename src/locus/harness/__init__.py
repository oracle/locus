# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A coding harness: workspace tools, written once, run anywhere.

The tools every coding agent converges on — read, write, edit, glob, grep,
ls, bash and its background handles, apply_patch, notebook_edit, a todo
list — written once against :class:`WorkspaceBackend` and run unchanged on
the host (:class:`LocalBackend`), in memory (:class:`MemoryBackend`), in any
sandbox that can run a command and move a file (:class:`SessionBackend`), or
in an NVIDIA OpenShell sandbox (:class:`OpenShellBackend`, the
``openshell`` extra), or in an OCI GenAI Sandbox (:class:`OCISandboxBackend`,
the ``oci-sandbox`` extra).

The tool bodies never gate. :func:`build_harness` pairs each tool with the
action spec that labels its calls (:mod:`locus.harness.labels`) and applies
one ``wrap`` to all of them — the single place a gate goes. Without one, the
tools are ungated. Every command the harness runs produces an
:class:`ExecRecord` (:mod:`locus.harness.evidence`).

    from locus.harness import LocalBackend, build_harness

    harness = build_harness(LocalBackend("."), wrap=my_gate)
    agent = Agent(model=model, tools=harness.tools, system_prompt=harness.prompt_fragment)

See ``docs/harness.md``.
"""

from locus.harness.backend import (
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
from locus.harness.evidence import ExecRecord
from locus.harness.labels import (
    KIND_EXEC,
    KIND_NETWORK,
    KIND_READ,
    KIND_WRITE,
    CommandClass,
    action_spec,
    classify_command,
)
from locus.harness.ledger import ReadLedger
from locus.harness.local import LocalBackend
from locus.harness.oci_sandbox import OCISandboxBackend, OCISandboxSession
from locus.harness.openshell import OpenShellBackend, OpenShellSession
from locus.harness.session import SessionBackend, SessionLike
from locus.harness.state import MemoryBackend
from locus.harness.tools.common import FileChange, HarnessConfig, HarnessContext
from locus.harness.toolset import (
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
    "OCISandboxBackend",
    "OCISandboxSession",
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
