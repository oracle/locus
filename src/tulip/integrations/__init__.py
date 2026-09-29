# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Integrations with external frameworks."""

from tulip.integrations.fastmcp import (
    MCPClient,
    MCPConnectionError,
    MCPRequestContext,
    MCPToolNotAllowedError,
    MCPToolResult,
    TulipMCPServer,
    create_mcp_server,
    mcp_tool_to_tulip,
)


__all__ = [
    # Consume MCP servers.
    "MCPClient",
    "MCPConnectionError",
    "MCPRequestContext",
    "MCPToolNotAllowedError",
    "MCPToolResult",
    # Expose Tulip's own tools and agent as an MCP server.
    "TulipMCPServer",
    "create_mcp_server",
    "mcp_tool_to_tulip",
]
