# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""How a box runner tells the gateway how its process ended, and starts children.

The gateway alone writes a run's ``done``, ``error`` or held state, from what
it decided and observed. What it cannot observe is the agent's own answer and
why the loop stopped, so before the runner exits it reports them once::

    POST /internal/v1/runs/{run_id}/result
    {"status": "done" | "parked" | "refused" | "error",
     "final_message"?, "stop_reason"?, "error"?,
     "usage_reported"?: {...}, "cost_usd_reported"?,
     "waiting"?: {"kind": "approval", "call_id", "tool", "approval_id"}
               | {"kind": "question", "call_id", "question"}}

``usage_reported`` is what the runner counted; the gateway records it as such
and keeps the box guard's metered count as the authoritative one.
``waiting`` says what a parked run waits for, so the gateway can show it and
answer the next ``GET /internal/v1/runner/next`` with ``resume`` when it comes.

A subagent runs in the same box, in-process, but is its own run on the
gateway (its own admission, its own record, linked to its parent)::

    POST /internal/v1/runs/{run_id}/children
    {"call_id", "agent", "prompt"}  ->  {"run_id", "admit_token"}

The child's ``admit_token`` is a workload token bound to the child run; it is
never logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from tulip.runner.client import GatewayClient


__all__ = ["ChildRun", "RunResult", "mint_child", "report_result"]

Status = Literal["done", "parked", "refused", "error"]


@dataclass(frozen=True)
class RunResult:
    """How the runner's process ended, as reported to the gateway."""

    status: Status
    final_message: str | None = None
    stop_reason: str | None = None
    error: str | None = None
    usage_reported: dict[str, int] | None = None
    cost_usd_reported: float | None = None
    waiting: dict[str, Any] | None = None

    def body(self) -> dict[str, Any]:
        """The JSON body, without the fields that are unset."""
        values: dict[str, Any] = {
            "status": self.status,
            "final_message": self.final_message,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "usage_reported": self.usage_reported,
            "cost_usd_reported": self.cost_usd_reported,
            "waiting": self.waiting,
        }
        return {key: value for key, value in values.items() if value is not None}


async def report_result(client: GatewayClient, result: RunResult) -> None:
    """Report how this run's process ended. Raises the client's errors."""
    await client.request("POST", f"/internal/v1/runs/{client.run_id}/result", json=result.body())


@dataclass(frozen=True)
class ChildRun:
    """A subagent's run, minted by the gateway. ``admit_token`` is never repr'd."""

    run_id: str
    admit_token: str = field(repr=False)


async def mint_child(client: GatewayClient, *, call_id: str, agent: str, prompt: str) -> ChildRun:
    """Have the gateway open a child run for a subagent.

    Raises:
        ValueError: the gateway's answer named no run or no token.
        GatewayError / GatewayUnavailable: the request failed.
    """
    answer = await client.request(
        "POST",
        f"/internal/v1/runs/{client.run_id}/children",
        json={"call_id": call_id, "agent": agent, "prompt": prompt},
    )
    answer = answer or {}
    run_id = str(answer.get("run_id") or "")
    token = str(answer.get("admit_token") or "")
    if not run_id or not token:
        raise ValueError("the gateway opened no child run (no run_id or admit_token)")
    return ChildRun(run_id=run_id, admit_token=token)
