# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A box runner's checkpoints, kept by the gateway.

A runner parks when a call waits on a person: it saves the agent's state and
exits, and comes back when the decision is made. The state is kept by the
gateway (which stores it with the run), not in the box, so a box that is
restored or replaced resumes from the same place.

:class:`GatewayCheckpointer` is an ordinary
:class:`~tulip.memory.checkpointer.BaseCheckpointer`; hand it to the agent like
any other. It talks to three routes, all scoped to the box's own run::

    PUT /internal/v1/runs/{run_id}/checkpoint
        {"thread_id", "checkpoint_id"?, "metadata"?, "state"}  -> {"checkpoint_id"}
    GET /internal/v1/runs/{run_id}/checkpoint?thread_id=&checkpoint_id=
        -> {"checkpoint_id", "state"}   (404: none)
    GET /internal/v1/runs/{run_id}/checkpoints?thread_id=&limit=
        -> {"checkpoint_ids": [...]}    (newest first)

``state`` is :meth:`AgentState.to_checkpoint`'s dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tulip.memory.checkpointer import BaseCheckpointer


if TYPE_CHECKING:
    from tulip.core.state import AgentState
    from tulip.runner.client import GatewayClient


__all__ = ["GatewayCheckpointer"]


class GatewayCheckpointer(BaseCheckpointer):
    """Checkpoints stored by the gateway, against the box's run."""

    def __init__(self, client: GatewayClient) -> None:
        self._client = client

    @property
    def _base(self) -> str:
        return f"/internal/v1/runs/{self._client.run_id}"

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = {"thread_id": thread_id, "state": state.to_checkpoint()}
        if checkpoint_id is not None:
            body["checkpoint_id"] = checkpoint_id
        if metadata:
            body["metadata"] = metadata
        answer = await self._client.request("PUT", f"{self._base}/checkpoint", json=body)
        saved = (answer or {}).get("checkpoint_id") or checkpoint_id
        if not saved:
            raise ValueError("the gateway stored the checkpoint but did not name it")
        return str(saved)

    async def load(self, thread_id: str, checkpoint_id: str | None = None) -> AgentState | None:
        from tulip.core.state import AgentState

        params = {"thread_id": thread_id}
        if checkpoint_id is not None:
            params["checkpoint_id"] = checkpoint_id
        answer = await self._client.request(
            "GET", f"{self._base}/checkpoint", params=params, allow_404=True
        )
        if not answer or not answer.get("state"):
            return None
        return AgentState.from_checkpoint(answer["state"])

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
        answer = await self._client.request(
            "GET",
            f"{self._base}/checkpoints",
            params={"thread_id": thread_id, "limit": limit},
            allow_404=True,
        )
        return [str(cid) for cid in (answer or {}).get("checkpoint_ids") or []][:limit]
