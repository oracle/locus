# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integrations with external frameworks."""

from locus.integrations.fastmcp import (
    ElicitationAnswer,
    LocusMCPServer,
    MCPClient,
    MCPConnectionError,
    MCPElicitation,
    MCPRequestContext,
    MCPToolNotAllowedError,
    MCPToolResult,
    create_mcp_server,
    create_tool_server,
    mcp_tool_to_locus,
)
from locus.integrations.mcp_auth import (
    FileTokenStore,
    InMemoryTokenStore,
    KeyringTokenStore,
    MCPOAuth,
    MCPTokenStore,
)


__all__ = [
    # Consume MCP servers.
    "MCPClient",
    "MCPConnectionError",
    "MCPRequestContext",
    "MCPToolNotAllowedError",
    "MCPToolResult",
    # Sign in to remote MCP servers.
    "MCPOAuth",
    "MCPTokenStore",
    "InMemoryTokenStore",
    "FileTokenStore",
    "KeyringTokenStore",
    # A server's questions.
    "MCPElicitation",
    "ElicitationAnswer",
    # Expose Locus's own tools and agent as an MCP server.
    "LocusMCPServer",
    "create_mcp_server",
    "create_tool_server",
    "mcp_tool_to_locus",
]
