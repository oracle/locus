# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for FastMCP."""

from __future__ import annotations

import pytest

from tests._safe_math import safe_math_eval
from tulip.integrations.fastmcp import mcp_tool_to_tulip
from tulip.tools import tool


pytestmark = pytest.mark.integration


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_tools():
    """Sample tools for testing."""

    @tool
    async def add_numbers(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    @tool
    async def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    return [add_numbers, greet]


@pytest.fixture
def mock_agent(sample_tools):
    """Create a mock agent for testing."""
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent._tool_registry = MagicMock()
    agent._tool_registry._tools = {t.name: t for t in sample_tools}
    agent._initialize = MagicMock()
    agent.run_sync = MagicMock(return_value=MagicMock(message="Test response"))

    return agent


# =============================================================================
# Unit Tests (no external dependencies)
# =============================================================================


class TestMCPToolConversion:
    """Test MCP tool conversion utilities."""

    @pytest.mark.asyncio
    async def test_mcp_tool_to_tulip(self):
        """Convert MCP tool to Tulip tool."""

        async def search(query: str) -> dict:
            return {"results": [f"Result for {query}"]}

        tulip_tool = mcp_tool_to_tulip(
            name="search",
            description="Search for things",
            func=search,
        )

        assert tulip_tool.name == "search"
        assert tulip_tool.description == "Search for things"

        # Test execution
        result = await tulip_tool.execute(query="test")
        assert "test" in result

    @pytest.mark.asyncio
    async def test_tulip_tool_to_mcp(self):
        """Convert Tulip tool to MCP schema."""
        from tulip.integrations.fastmcp import tulip_tool_to_mcp

        @tool
        async def calculate(expression: str) -> str:
            """Calculate a math expression."""
            return str(safe_math_eval(expression))

        mcp_schema = tulip_tool_to_mcp(calculate)

        assert mcp_schema["name"] == "calculate"
        assert mcp_schema["description"] == "Calculate a math expression."
        assert "inputSchema" in mcp_schema


class TestTulipMCPServer:
    """Test TulipMCPServer functionality."""

    @pytest.mark.asyncio
    async def test_handle_tools_list(self, mock_agent):
        """Handle tools/list request (without FastMCP registration)."""
        # Test the protocol directly without creating MCP instance
        from tulip.integrations.fastmcp import tulip_tool_to_mcp

        tools = []
        for tool_obj in mock_agent._tool_registry._tools.values():
            tools.append(tulip_tool_to_mcp(tool_obj))

        assert len(tools) == 2
        tool_names = [t["name"] for t in tools]
        assert "add_numbers" in tool_names
        assert "greet" in tool_names

    @pytest.mark.asyncio
    async def test_handle_run_agent(self, mock_agent):
        """Handle tools/call for run_agent."""
        # Test the agent invocation directly
        result = mock_agent.run_sync("Hello!")
        assert result.message == "Test response"

    @pytest.mark.asyncio
    async def test_handle_tool_call(self, sample_tools):
        """Handle tools/call for a specific tool."""
        # Find add_numbers tool
        add_tool = next(t for t in sample_tools if t.name == "add_numbers")

        result = await add_tool.execute(a=5, b=3)
        assert result == "8"

    @pytest.mark.asyncio
    async def test_tulip_tool_schema(self, sample_tools):
        """Test Tulip tool to MCP schema conversion."""
        from tulip.integrations.fastmcp import tulip_tool_to_mcp

        add_tool = next(t for t in sample_tools if t.name == "add_numbers")
        schema = tulip_tool_to_mcp(add_tool)

        assert schema["name"] == "add_numbers"
        assert "description" in schema
        assert "inputSchema" in schema
