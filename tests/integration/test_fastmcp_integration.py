# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for FastMCP."""

from __future__ import annotations

import os

import pytest

from locus import Agent
from locus.integrations.fastmcp import LocusMCPServer, mcp_tool_to_locus
from locus.tools import tool
from tests._safe_math import safe_math_eval


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
    async def test_mcp_tool_to_locus(self):
        """Convert MCP tool to Locus tool."""

        async def search(query: str) -> dict:
            return {"results": [f"Result for {query}"]}

        locus_tool = mcp_tool_to_locus(
            name="search",
            description="Search for things",
            func=search,
        )

        assert locus_tool.name == "search"
        assert locus_tool.description == "Search for things"

        # Test execution
        result = await locus_tool.execute(query="test")
        assert "test" in result

    @pytest.mark.asyncio
    async def test_locus_tool_to_mcp(self):
        """Convert Locus tool to MCP schema."""
        from locus.integrations.fastmcp import locus_tool_to_mcp

        @tool
        async def calculate(expression: str) -> str:
            """Calculate a math expression."""
            return str(safe_math_eval(expression))

        mcp_schema = locus_tool_to_mcp(calculate)

        assert mcp_schema["name"] == "calculate"
        assert mcp_schema["description"] == "Calculate a math expression."
        assert "inputSchema" in mcp_schema


class TestLocusMCPServer:
    """Test LocusMCPServer functionality."""

    @pytest.mark.asyncio
    async def test_handle_tools_list(self, mock_agent):
        """Handle tools/list request (without FastMCP registration)."""
        # Test the protocol directly without creating MCP instance
        from locus.integrations.fastmcp import locus_tool_to_mcp

        tools = []
        for tool_obj in mock_agent._tool_registry._tools.values():
            tools.append(locus_tool_to_mcp(tool_obj))

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
    async def test_locus_tool_schema(self, sample_tools):
        """Test Locus tool to MCP schema conversion."""
        from locus.integrations.fastmcp import locus_tool_to_mcp

        add_tool = next(t for t in sample_tools if t.name == "add_numbers")
        schema = locus_tool_to_mcp(add_tool)

        assert schema["name"] == "add_numbers"
        assert "description" in schema
        assert "inputSchema" in schema


# =============================================================================
# Live Integration Tests (require OCI)
# =============================================================================


def load_local_config() -> dict:
    """Load local config if available."""
    from pathlib import Path

    import yaml

    config_path = Path(__file__).parent.parent.parent / "config.local.yaml"
    if config_path.exists():
        with config_path.open() as f:
            return yaml.safe_load(f) or {}
    return {}


@pytest.mark.requires_oci
class TestLocusMCPServerLive:
    """Live integration tests with real model."""

    @pytest.fixture
    def live_agent(self):
        """Create a live agent with OCI."""
        from locus.models import OCIModel

        config = load_local_config().get("oci", {})
        model = OCIModel(
            model_id=config.get("models", {}).get(
                "gpt", os.getenv("OCI_MODEL_ID", "openai.gpt-5.4")
            ),
            profile_name=config.get("profile_name", "DEFAULT"),
            auth_type=config.get("auth_type", "api_key"),
            region=config.get("region", "eu-frankfurt-1"),
            compartment_id=config.get("compartment_id"),
            service_endpoint=config.get("service_endpoint"),
            max_tokens=256,
        )

        @tool
        async def get_time() -> str:
            """Get current time."""
            from datetime import datetime

            return datetime.now().isoformat()

        return Agent(model=model, tools=[get_time])

    @pytest.mark.asyncio
    async def test_live_mcp_server_tools_list(self, live_agent):
        """List tools from live agent."""
        server = LocusMCPServer(agent=live_agent, name="live-test")

        response = await server.handle_request(
            {
                "method": "tools/list",
                "params": {},
            }
        )

        assert "tools" in response
        tool_names = [t["name"] for t in response["tools"]]
        assert "get_time" in tool_names

    @pytest.mark.asyncio
    async def test_live_mcp_server_run_agent(self, live_agent):
        """Run agent via MCP."""
        server = LocusMCPServer(agent=live_agent, name="live-test")

        response = await server.handle_request(
            {
                "method": "tools/call",
                "params": {
                    "name": "run_agent",
                    "arguments": {"prompt": "What is 2+2? Just say the number."},
                },
            }
        )

        assert "content" in response
        assert "4" in response["content"][0]["text"]
