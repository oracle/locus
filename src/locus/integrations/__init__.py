# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integrations with external frameworks."""

from locus.integrations.fastmcp import (
    LocusMCPServer,
    create_mcp_server,
    mcp_tool_to_locus,
)


__all__ = [
    "LocusMCPServer",
    "create_mcp_server",
    "mcp_tool_to_locus",
]
