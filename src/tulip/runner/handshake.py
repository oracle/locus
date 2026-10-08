# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""What a box runner does when it starts: ask the gateway.

A runner process starts in two situations — the run's first start, and coming
back after it parked on a hold — and only the gateway knows which. So the
runner's first call is always::

    GET /internal/v1/runner/next
    -> {"op": "start"}
     | {"op": "resume", "resume": {"approval_id", "decision", "justification"?}}
     | {"op": "resume", "resume": {"answer": ...}}       (a question the run asked)
     | {"op": "stop", "reason": ...}

and then, unless told to stop, ``GET /internal/v1/runner/manifest`` for the
:class:`~tulip.runner.manifest.RunManifest`. The workload token says which run
the box is serving, so neither route takes a run id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from tulip.runner.manifest import RunManifest


if TYPE_CHECKING:
    from tulip.runner.client import GatewayClient


__all__ = ["NextOp", "fetch_manifest", "next_op"]


@dataclass(frozen=True)
class NextOp:
    """What the gateway told the runner to do."""

    op: Literal["start", "resume", "stop"]
    resume: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def approval_id(self) -> str | None:
        value = self.resume.get("approval_id")
        return str(value) if value else None

    @property
    def decision(self) -> str | None:
        value = self.resume.get("decision")
        return str(value) if value else None


async def next_op(client: GatewayClient) -> NextOp:
    """Ask the gateway what to do. An answer this runner does not understand is a stop."""
    body = await client.request("GET", "/internal/v1/runner/next") or {}
    op = str(body.get("op") or "")
    if op not in ("start", "resume", "stop"):
        return NextOp(op="stop", reason=f"the gateway answered an unknown op {op!r}")
    if op == "resume" and not isinstance(body.get("resume"), dict):
        return NextOp(op="stop", reason="the gateway said resume but sent nothing to resume with")
    return NextOp(
        op=op,  # type: ignore[arg-type]  # checked just above
        resume=dict(body.get("resume") or {}),
        reason=str(body.get("reason") or ""),
    )


async def fetch_manifest(client: GatewayClient) -> RunManifest:
    """Fetch and validate the run's manifest; refuse one for another run.

    Raises:
        ValueError: the manifest names a different run than the box serves.
        pydantic.ValidationError: the manifest is malformed.
    """
    body = await client.request("GET", "/internal/v1/runner/manifest")
    manifest = RunManifest.model_validate(body)
    if manifest.run_id != client.run_id:
        raise ValueError(
            f"the gateway sent the manifest of run {manifest.run_id!r} to the box of run "
            f"{client.run_id!r}"
        )
    return manifest
