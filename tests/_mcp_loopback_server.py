# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A FastMCP server for the MCP client tests (``unit/test_mcp_fidelity.py``).

Imported by the tests to serve on a thread, and run as a script
(``python _mcp_loopback_server.py <port>``) when a test needs a server it can
kill outright: stopping uvicorn in-thread leaves sockets open, while a killed
process has the kernel reset them — which is what a crashed server does.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import sys

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic import BaseModel


PNG_1PX = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
    )
).decode()


class Hotel(BaseModel):
    id: str
    name: str
    price: float


def build_server(port: int) -> FastMCP:
    mcp = FastMCP("fidelity", host="127.0.0.1", port=port)

    @mcp.tool()
    def search_hotels(city: str) -> list[Hotel]:
        """Typed return: text + structuredContent + outputSchema."""
        return [Hotel(id="h1", name=f"Grand {city}", price=420.0)]

    @mcp.tool()
    def explicit_both(city: str) -> CallToolResult:
        """Distinct text and structuredContent."""
        return CallToolResult(
            content=[TextContent(type="text", text=f"Found 1 hotel in {city}")],
            structuredContent={"widget": "hotel_list", "items": [{"id": "h9"}]},
        )

    @mcp.tool()
    def explicit_error() -> CallToolResult:
        """isError=True, returned rather than raised."""
        return CallToolResult(
            content=[TextContent(type="text", text="rate not available")],
            structuredContent={"code": "NO_RATE"},
            isError=True,
        )

    @mcp.tool()
    def photo() -> CallToolResult:
        """An image block next to a caption."""
        return CallToolResult(
            content=[
                TextContent(type="text", text="the lobby"),
                ImageContent(type="image", data=PNG_1PX, mimeType="image/png"),
            ]
        )

    @mcp.tool()
    async def slow_progress(ctx: Context) -> str:  # type: ignore[type-arg]
        """Three progress notifications, then a result."""
        for i in range(3):
            await ctx.report_progress(i + 1, 3, message=f"step {i + 1}")
            await asyncio.sleep(0.05)
        return "done"

    @mcp.tool()
    async def sleepy(seconds: float) -> str:
        """Blocks long enough to be cancelled or orphaned."""
        await asyncio.sleep(seconds)
        return "woke"

    @mcp.tool()
    def whoami(ctx: Context) -> str:  # type: ignore[type-arg]
        """Echo the auth and identity headers the request carried."""
        request = ctx.request_context.request
        headers = request.headers if request is not None else {}
        return (
            f"auth={headers.get('authorization', '<none>')} "
            f"user={headers.get('x-user-id', '<none>')} "
            f"tenant={headers.get('x-tenant', '<none>')}"
        )

    @mcp.tool()
    def auth_digest(ctx: Context) -> str:  # type: ignore[type-arg]
        """A digest of the Authorization header — proves which token arrived
        without echoing it into the tool result (and so into state)."""
        request = ctx.request_context.request
        auth = request.headers.get("authorization", "") if request is not None else ""
        return hashlib.sha256(auth.encode()).hexdigest()[:16]

    return mcp


if __name__ == "__main__":
    build_server(int(sys.argv[1])).run(transport="streamable-http")
