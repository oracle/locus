# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Tools a box runner offers but the gateway performs.

Some tools cannot run in a box: they hand out a secret, open a connection,
widen the box's network, start another agent, schedule a task, or move a
playbook forward. The manifest marks them ``runs: gateway``. The runner offers
them to the model like any other tool, and the call's body is one request::

    POST /internal/v1/runs/{run_id}/tools/{name}
    {"call_id": ..., "arguments": {...}}  ->  {"content": ..., "error": ...?}

The gateway admits the call itself (it is not routed through
:class:`~tulip.runner.gate.RemoteGate` first: the gateway would only weigh it
twice) and performs it server-side. ``content`` is what the model reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote


if TYPE_CHECKING:
    from tulip.runner.client import GatewayClient
    from tulip.runner.manifest import ToolEntry


__all__ = ["RemoteTool", "RemoteToolError"]


class RemoteToolError(Exception):
    """The gateway performed the call and reported that it failed."""


class RemoteTool:
    """A ``runs: gateway`` tool: described locally, performed by the gateway."""

    def __init__(self, client: GatewayClient, entry: ToolEntry) -> None:
        if entry.runs != "gateway":
            raise ValueError(f"tool {entry.name!r} runs on {entry.runs}, not on the gateway")
        self._client = client
        self.entry = entry

    @property
    def name(self) -> str:
        return self.entry.name

    @property
    def path(self) -> str:
        return f"/internal/v1/runs/{self._client.run_id}/tools/{quote(self.name, safe='')}"

    async def call(self, call_id: str, arguments: dict[str, Any]) -> Any:
        """Have the gateway perform the call; return what the model should read.

        Raises:
            RemoteToolError: the gateway reports the call failed.
            GatewayError / GatewayUnavailable: the request itself failed.
        """
        answer = await self._client.request(
            "POST", self.path, json={"call_id": call_id, "arguments": arguments}
        )
        answer = answer or {}
        if answer.get("error"):
            raise RemoteToolError(str(answer["error"]))
        return answer.get("content")
