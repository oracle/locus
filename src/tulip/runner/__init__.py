# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The box runner's protocol with the Tulip gateway.

A *box runner* is one agent's loop running inside a sandbox box (an NVIDIA
OpenShell sandbox), started by a Tulip gateway for one run. The gateway stays
the control plane: it decides every call, performs the tools a box must not,
keeps the run's checkpoints and writes every record. This package is the
runner's side of that contract:

- :class:`RunManifest` — what to run, pinned, with a canonical digest.
- :class:`RemoteGate` — the admission hook: every tool call is decided by the
  gateway (``POST /v1/admit``), failing closed.
- :class:`GatewayEvents` — progress events to the run's stream, batched,
  numbered and spooled through outages.
- :class:`GatewayCheckpointer` — the agent's checkpoints, kept by the gateway.
- :class:`RemoteTool` — a ``runs: gateway`` tool, performed server-side.
- :func:`next_op` / :func:`fetch_manifest` — the start-up handshake.

All of them share one :class:`GatewayClient`, configured from the box's
environment (:meth:`RunnerConfig.from_env`). Building the agent from a
manifest comes in a later release.
"""

from __future__ import annotations

from tulip.runner.checkpoint import GatewayCheckpointer
from tulip.runner.client import (
    ADMIT_TOKEN_VAR,
    ADMIT_URL_VAR,
    RUN_ID_VAR,
    GatewayClient,
    GatewayError,
    GatewayUnavailable,
    RunnerConfig,
)
from tulip.runner.events import ALLOWED_TYPES, GatewayEvents
from tulip.runner.gate import AdmitResult, Hold, RemoteGate
from tulip.runner.handshake import NextOp, fetch_manifest, next_op
from tulip.runner.manifest import (
    MANIFEST_VERSION,
    Budgets,
    McpMount,
    ModelRoute,
    RunManifest,
    ToolEntry,
    canonical_digest,
)
from tulip.runner.tools import RemoteTool, RemoteToolError


__all__ = [
    "ADMIT_TOKEN_VAR",
    "ADMIT_URL_VAR",
    "ALLOWED_TYPES",
    "MANIFEST_VERSION",
    "RUN_ID_VAR",
    "AdmitResult",
    "Budgets",
    "GatewayCheckpointer",
    "GatewayClient",
    "GatewayError",
    "GatewayEvents",
    "GatewayUnavailable",
    "Hold",
    "McpMount",
    "ModelRoute",
    "NextOp",
    "RemoteGate",
    "RemoteTool",
    "RemoteToolError",
    "RunManifest",
    "RunnerConfig",
    "ToolEntry",
    "canonical_digest",
    "fetch_manifest",
    "next_op",
]
