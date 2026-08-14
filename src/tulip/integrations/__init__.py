# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Integrations with external frameworks."""

from tulip.integrations.fastmcp import (
    MCPClient,
    TulipMCPServer,
    create_mcp_server,
    mcp_tool_to_tulip,
)


__all__ = [
    # Consume MCP servers.
    "MCPClient",
    # Expose Tulip's own tools and agent as an MCP server.
    "TulipMCPServer",
    "create_mcp_server",
    "mcp_tool_to_tulip",
]
