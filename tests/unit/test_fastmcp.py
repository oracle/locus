# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for MCP integration utilities."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from locus.integrations.fastmcp import (
    _create_tool_wrapper,
    _json_schema_type_to_python,
    build_args_model,
    locus_tool_to_mcp,
    mcp_tool_to_locus,
)
from locus.tools.decorator import tool


class TestJsonSchemaTypeToPython:
    """Tests for _json_schema_type_to_python function."""

    def test_string_type(self):
        """Test string type conversion."""
        result = _json_schema_type_to_python({"type": "string"})
        assert result is str

    def test_integer_type(self):
        """Test integer type conversion."""
        result = _json_schema_type_to_python({"type": "integer"})
        assert result is int

    def test_number_type(self):
        """Test number type conversion."""
        result = _json_schema_type_to_python({"type": "number"})
        assert result is float

    def test_boolean_type(self):
        """Test boolean type conversion."""
        result = _json_schema_type_to_python({"type": "boolean"})
        assert result is bool

    def test_object_type(self):
        """Test object type conversion."""
        from typing import Any

        result = _json_schema_type_to_python({"type": "object"})
        assert result == dict[str, Any]

    def test_array_type_with_items(self):
        """Test array type with items schema."""
        result = _json_schema_type_to_python({"type": "array", "items": {"type": "string"}})
        # Should return list[str]
        assert result.__origin__ is list

    def test_array_type_without_items(self):
        """Test array type without items schema."""
        result = _json_schema_type_to_python({"type": "array"})
        assert result.__origin__ is list

    def test_nullable_type(self):
        """Test nullable type (list with null)."""
        result = _json_schema_type_to_python({"type": ["string", "null"]})
        assert result is str

    def test_unknown_type(self):
        """Test unknown type returns Any."""
        from typing import Any

        result = _json_schema_type_to_python({"type": "unknown_type"})
        assert result is Any

    def test_no_type_property(self):
        """Test schema without type property."""
        from typing import Any

        result = _json_schema_type_to_python({})
        assert result is Any


class TestBuildArgsModel:
    """Tests for build_args_model function."""

    def test_build_simple_model(self):
        """Test building a simple model."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        model = build_args_model("TestTool", schema)

        assert model is not None
        assert "name" in model.model_fields
        assert "age" in model.model_fields

    def test_build_model_with_defaults(self):
        """Test building model with default values."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "default": "default_query"},
            },
        }
        model = build_args_model("SearchTool", schema)

        assert model is not None
        # Create instance with defaults
        instance = model()
        assert instance.query == "default_query"

    def test_build_model_none_schema(self):
        """Test with None schema returns None."""
        result = build_args_model("Test", None)
        assert result is None

    def test_build_model_no_properties(self):
        """Test with no properties returns None."""
        result = build_args_model("Test", {"type": "object"})
        assert result is None

    def test_build_model_empty_properties(self):
        """Test with empty properties returns None."""
        result = build_args_model("Test", {"type": "object", "properties": {}})
        assert result is None

    def test_build_model_invalid_property(self):
        """Test skips invalid properties."""
        schema = {
            "type": "object",
            "properties": {
                "valid": {"type": "string"},
                "invalid": "not a dict",  # Invalid
            },
        }
        model = build_args_model("Test", schema)

        assert model is not None
        assert "valid" in model.model_fields
        assert "invalid" not in model.model_fields

    def test_model_name_formatting(self):
        """Test model name formatting with special chars."""
        schema = {"type": "object", "properties": {"param": {"type": "string"}}}
        model = build_args_model("my-tool with spaces", schema)

        assert model is not None
        assert "my_tool_with_spaces" in model.__name__


class TestMcpToolToLocus:
    """Tests for mcp_tool_to_locus function."""

    @pytest.mark.asyncio
    async def test_convert_simple_tool(self):
        """Test converting a simple MCP tool."""

        async def mcp_func(query: str) -> str:
            return f"Result: {query}"

        locus_tool = mcp_tool_to_locus(
            name="search",
            description="Search for items",
            func=mcp_func,
        )

        assert locus_tool.name == "search"
        assert locus_tool.description == "Search for items"

    @pytest.mark.asyncio
    async def test_convert_tool_returns_string(self):
        """Test tool that returns string."""

        async def mcp_func() -> str:
            return "string result"

        locus_tool = mcp_tool_to_locus(
            name="test",
            description="Test",
            func=mcp_func,
        )

        result = await locus_tool.execute()
        assert result == "string result"

    @pytest.mark.asyncio
    async def test_convert_tool_returns_dict(self):
        """Test tool that returns dict (converted to JSON)."""

        async def mcp_func() -> dict:
            return {"key": "value"}

        locus_tool = mcp_tool_to_locus(
            name="test",
            description="Test",
            func=mcp_func,
        )

        result = await locus_tool.execute()
        assert result == '{"key": "value"}'


class TestLocusToolToMcp:
    """Tests for locus_tool_to_mcp function."""

    def test_convert_tool_with_parameters(self):
        """Test converting tool with parameters."""

        @tool
        def search(query: str, limit: int = 10) -> str:
            """Search for items."""
            return query

        mcp_schema = locus_tool_to_mcp(search)

        assert mcp_schema["name"] == "search"
        assert mcp_schema["description"] == "Search for items."
        assert "inputSchema" in mcp_schema

    def test_convert_tool_without_description(self):
        """Test converting tool without description."""

        @tool
        def simple_tool() -> str:
            return "result"

        mcp_schema = locus_tool_to_mcp(simple_tool)

        assert mcp_schema["name"] == "simple_tool"
        # Description defaults to empty string if None
        assert mcp_schema["description"] == "" or mcp_schema["description"]

    def test_convert_tool_schema_structure(self):
        """Test MCP schema has correct structure."""

        @tool
        def my_tool(x: int) -> str:
            """A tool."""
            return str(x)

        mcp_schema = locus_tool_to_mcp(my_tool)

        assert "name" in mcp_schema
        assert "description" in mcp_schema
        assert "inputSchema" in mcp_schema


class TestCreateToolWrapper:
    """Tests for _create_tool_wrapper function."""

    def test_wrapper_for_tool_without_params(self):
        """Test creating wrapper for tool without parameters."""

        @tool
        async def no_params() -> str:
            """No params tool."""
            return "result"

        wrapper = _create_tool_wrapper(no_params)

        assert callable(wrapper)
        assert wrapper.__name__ == "no_params"
        assert wrapper.__doc__ == "No params tool."

    def test_wrapper_for_tool_with_params(self):
        """Test creating wrapper for tool with parameters."""

        @tool
        async def with_params(query: str, limit: int = 10) -> str:
            """With params tool."""
            return f"{query}:{limit}"

        wrapper = _create_tool_wrapper(with_params)

        assert callable(wrapper)

    @pytest.mark.asyncio
    async def test_wrapper_executes_no_params(self):
        """Test wrapper execution for tool without params."""

        @tool
        async def simple() -> str:
            """Simple tool."""
            return "executed"

        wrapper = _create_tool_wrapper(simple)
        result = await wrapper()

        assert result == "executed"

    @pytest.mark.asyncio
    async def test_wrapper_handles_dict_result(self):
        """Test wrapper converts dict to JSON."""
        import json

        @tool
        async def dict_tool() -> dict:
            """Returns dict."""
            return {"status": "ok"}

        wrapper = _create_tool_wrapper(dict_tool)
        result = await wrapper()

        # Result is JSON string
        parsed = json.loads(result)
        assert parsed == {"status": "ok"}


class TestCreateToolWrapperWithParams:
    """Tests for _create_tool_wrapper with parameters."""

    def test_wrapper_has_correct_name(self):
        """Test wrapper has tool name."""

        @tool
        async def my_tool(query: str) -> str:
            """My tool."""
            return query

        wrapper = _create_tool_wrapper(my_tool)
        assert wrapper.__name__ == "my_tool"

    @pytest.mark.asyncio
    async def test_wrapper_with_required_param(self):
        """Test wrapper executes with required param."""

        @tool
        async def search_tool(query: str) -> str:
            """Search tool."""
            return f"searched: {query}"

        wrapper = _create_tool_wrapper(search_tool)
        result = await wrapper(query="hello")

        assert result == "searched: hello"

    @pytest.mark.asyncio
    async def test_wrapper_with_optional_param(self):
        """Test wrapper executes with optional param."""

        @tool
        async def search_tool(query: str, limit: int = 10) -> str:
            """Search tool."""
            return f"searched: {query}, limit: {limit}"

        wrapper = _create_tool_wrapper(search_tool)
        result = await wrapper(query="hello")

        assert "searched: hello" in result

    @pytest.mark.asyncio
    async def test_wrapper_filters_none_values(self):
        """Test wrapper filters None values from kwargs."""

        @tool
        async def search_tool(query: str) -> str:
            """Search tool."""
            return f"searched: {query}"

        wrapper = _create_tool_wrapper(search_tool)
        result = await wrapper(query="hello")

        assert result == "searched: hello"


class TestLocusMCPServer:
    """Tests for LocusMCPServer class."""

    def test_create_server(self):
        """Test creating MCP server."""
        mock_agent = MagicMock()
        server = mcp_tool_to_locus.__module__  # Get module

        from locus.integrations.fastmcp import LocusMCPServer

        server = LocusMCPServer(agent=mock_agent, name="test-server")

        assert server.name == "test-server"
        assert server.version == "1.0.0"
        assert server.agent is mock_agent

    def test_create_server_with_version(self):
        """Test creating server with custom version."""
        mock_agent = MagicMock()

        from locus.integrations.fastmcp import LocusMCPServer

        server = LocusMCPServer(agent=mock_agent, name="test-server", version="2.0.0")

        assert server.version == "2.0.0"

    @pytest.mark.asyncio
    async def test_handle_request_tools_list(self):
        """Test handle_request for tools/list."""
        mock_agent = MagicMock()
        mock_agent._tool_registry = MagicMock()
        mock_agent._tool_registry.tools = {}
        mock_agent._initialize = MagicMock()

        from locus.integrations.fastmcp import LocusMCPServer

        server = LocusMCPServer(agent=mock_agent)

        result = await server.handle_request({"method": "tools/list"})

        assert "tools" in result
        assert isinstance(result["tools"], list)

    @pytest.mark.asyncio
    async def test_handle_request_tools_list_with_tools(self):
        """Test handle_request for tools/list with tools."""

        @tool
        async def my_tool(x: int) -> str:
            """A test tool."""
            return str(x)

        mock_agent = MagicMock()
        mock_agent._tool_registry = MagicMock()
        mock_agent._tool_registry.tools = {"my_tool": my_tool}
        mock_agent._initialize = MagicMock()

        from locus.integrations.fastmcp import LocusMCPServer

        server = LocusMCPServer(agent=mock_agent)

        result = await server.handle_request({"method": "tools/list"})

        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_handle_request_run_agent(self):
        """Test handle_request for run_agent tool."""
        mock_agent = MagicMock()
        mock_agent._tool_registry = MagicMock()
        mock_agent._tool_registry.tools = {}
        mock_agent._initialize = MagicMock()
        mock_agent.run_sync = MagicMock()
        mock_agent.run_sync.return_value = MagicMock(message="Hello world")

        from locus.integrations.fastmcp import LocusMCPServer

        server = LocusMCPServer(agent=mock_agent)

        result = await server.handle_request(
            {"method": "tools/call", "params": {"name": "run_agent", "arguments": {"prompt": "Hi"}}}
        )

        assert "content" in result
        assert result["content"][0]["text"] == "Hello world"

    @pytest.mark.asyncio
    async def test_handle_request_call_tool(self):
        """Test handle_request for calling a tool."""

        @tool
        async def my_tool(x: int) -> str:
            """A test tool."""
            return f"result: {x}"

        mock_agent = MagicMock()
        mock_agent._tool_registry = MagicMock()
        mock_agent._tool_registry.tools = {"my_tool": my_tool}
        mock_agent._tool_registry.get = MagicMock(return_value=my_tool)
        mock_agent._initialize = MagicMock()

        from locus.integrations.fastmcp import LocusMCPServer

        server = LocusMCPServer(agent=mock_agent)

        result = await server.handle_request(
            {"method": "tools/call", "params": {"name": "my_tool", "arguments": {"x": 42}}}
        )

        assert "content" in result
        assert "result: 42" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_request_call_tool_dict_result(self):
        """Test handle_request for tool returning dict."""

        @tool
        async def dict_tool() -> dict:
            """A dict tool."""
            return {"status": "ok"}

        mock_agent = MagicMock()
        mock_agent._tool_registry = MagicMock()
        mock_agent._tool_registry.tools = {"dict_tool": dict_tool}
        mock_agent._tool_registry.get = MagicMock(return_value=dict_tool)
        mock_agent._initialize = MagicMock()

        from locus.integrations.fastmcp import LocusMCPServer

        server = LocusMCPServer(agent=mock_agent)

        result = await server.handle_request(
            {"method": "tools/call", "params": {"name": "dict_tool", "arguments": {}}}
        )

        import json

        assert "content" in result
        parsed = json.loads(result["content"][0]["text"])
        assert parsed == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_handle_request_unknown_tool(self):
        """Test handle_request for unknown tool."""
        mock_agent = MagicMock()
        mock_agent._tool_registry = MagicMock()
        mock_agent._tool_registry.tools = {}
        mock_agent._tool_registry.get = MagicMock(return_value=None)
        mock_agent._initialize = MagicMock()

        from locus.integrations.fastmcp import LocusMCPServer

        server = LocusMCPServer(agent=mock_agent)

        result = await server.handle_request(
            {"method": "tools/call", "params": {"name": "unknown_tool", "arguments": {}}}
        )

        assert "error" in result
        assert result["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_handle_request_unknown_method(self):
        """Test handle_request for unknown method."""
        mock_agent = MagicMock()

        from locus.integrations.fastmcp import LocusMCPServer

        server = LocusMCPServer(agent=mock_agent)

        result = await server.handle_request({"method": "unknown/method"})

        assert "error" in result
        assert result["error"]["code"] == -32601


class TestMCPClient:
    """Tests for MCPClient class."""

    def test_create_client_with_base_url(self):
        """Test creating client with base_url."""
        from locus.integrations.fastmcp import MCPClient

        client = MCPClient(base_url="https://example.com/mcp")

        assert client.base_url == "https://example.com/mcp"
        assert client.server_command is None

    def test_create_client_with_server_command(self):
        """Test creating client with server_command."""
        from locus.integrations.fastmcp import MCPClient

        client = MCPClient(server_command=["python", "server.py"])

        assert client.server_command == ["python", "server.py"]
        assert client.base_url is None

    def test_create_client_with_auth(self):
        """Test creating client with access_token."""
        from locus.integrations.fastmcp import MCPClient

        test_token = "test-token-value"  # noqa: S105
        client = MCPClient(base_url="https://example.com", access_token=test_token)

        assert client.access_token == test_token

    def test_create_client_with_verify_ssl(self):
        """Test creating client with verify_ssl option."""
        from locus.integrations.fastmcp import MCPClient

        client = MCPClient(base_url="https://example.com", verify_ssl=False)

        assert client.verify_ssl is False

    def test_verify_url_defaults_on(self):
        """New SSRF guard is opt-out, not opt-in."""
        from locus.integrations.fastmcp import MCPClient

        client = MCPClient(base_url="https://example.com")
        assert client.verify_url is True
        assert client.allow_private_url is False

    @pytest.mark.asyncio
    async def test_connect_http_rejects_metadata_url(self, monkeypatch):
        """SSRF guard rejects cloud-metadata base_url before any HTTP dispatch."""
        import socket

        from locus.core.errors import ValidationError
        from locus.integrations.fastmcp import MCPClient

        def _fake(host, port, *a, **kw):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.169.254", port or 0)),
            ]

        monkeypatch.setattr(socket, "getaddrinfo", _fake)

        client = MCPClient(base_url="https://imds.example/")
        with pytest.raises(ValidationError, match="SSRF guard"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_connect_http_verify_url_false_skips_guard(self, monkeypatch):
        """verify_url=False disables the pre-flight (stdio / in-cluster case)."""
        import socket
        import sys
        import types

        from locus.integrations.fastmcp import MCPClient

        # getaddrinfo would flag this as private, but the guard is off.
        def _fake(host, port, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", port or 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _fake)

        # Stub the mcp module so the import inside _connect_http succeeds
        # and raises a distinguishable error after the guard would have run.
        mcp_pkg = types.ModuleType("mcp")
        mcp_client = types.ModuleType("mcp.client")
        mcp_session = types.ModuleType("mcp.client.session")
        mcp_http = types.ModuleType("mcp.client.streamable_http")

        class _FakeSession:
            pass

        def _fake_http(*a, **kw):
            raise RuntimeError("would have connected — guard skipped")

        mcp_session.ClientSession = _FakeSession
        mcp_http.streamablehttp_client = _fake_http
        monkeypatch.setitem(sys.modules, "mcp", mcp_pkg)
        monkeypatch.setitem(sys.modules, "mcp.client", mcp_client)
        monkeypatch.setitem(sys.modules, "mcp.client.session", mcp_session)
        monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", mcp_http)

        client = MCPClient(base_url="https://internal.example/", verify_url=False)
        # If the guard had fired we'd see ValidationError with "SSRF guard".
        # Instead we must see the stub's distinguishable RuntimeError.
        with pytest.raises(RuntimeError, match="would have connected"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_connect_stdio_blocks_malware_package(self, monkeypatch):
        """OSV pre-check refuses to spawn a malware-flagged MCP package."""
        import sys
        import types

        from locus.core.errors import ValidationError
        from locus.integrations import osv
        from locus.integrations.fastmcp import MCPClient

        # Stub the mcp stdio surface so the connect path reaches our check.
        mcp_pkg = types.ModuleType("mcp")

        class _StdioParams:
            def __init__(self, command: str, args: list[str]) -> None:
                self.command = command
                self.args = args

        mcp_pkg.StdioServerParameters = _StdioParams
        mcp_client = types.ModuleType("mcp.client")
        mcp_session = types.ModuleType("mcp.client.session")
        mcp_stdio = types.ModuleType("mcp.client.stdio")
        mcp_session.ClientSession = type("ClientSession", (), {})
        mcp_stdio.stdio_client = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("should not reach stdio_client — guard should fire first")
        )
        monkeypatch.setitem(sys.modules, "mcp", mcp_pkg)
        monkeypatch.setitem(sys.modules, "mcp.client", mcp_client)
        monkeypatch.setitem(sys.modules, "mcp.client.session", mcp_session)
        monkeypatch.setitem(sys.modules, "mcp.client.stdio", mcp_stdio)

        # Force OSV to return a malware verdict.
        monkeypatch.setattr(
            osv, "_query_osv", lambda *a, **kw: [{"id": "MAL-TEST", "summary": "hi"}]
        )
        monkeypatch.delenv("LOCUS_MCP_SKIP_OSV", raising=False)

        client = MCPClient(server_command=["npx", "some-evil-package"])
        with pytest.raises(ValidationError, match="MAL-TEST"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_connect_stdio_verify_packages_false_skips_osv(self, monkeypatch):
        """verify_packages=False bypasses the OSV pre-check."""
        import sys
        import types

        from locus.integrations import osv
        from locus.integrations.fastmcp import MCPClient

        mcp_pkg = types.ModuleType("mcp")

        class _StdioParams:
            def __init__(self, command: str, args: list[str]) -> None:
                pass

        mcp_pkg.StdioServerParameters = _StdioParams
        mcp_client = types.ModuleType("mcp.client")
        mcp_session = types.ModuleType("mcp.client.session")
        mcp_stdio = types.ModuleType("mcp.client.stdio")
        mcp_session.ClientSession = type("ClientSession", (), {})

        def _raise_distinguishable(*a, **kw):
            raise RuntimeError("guard skipped — reached stdio_client")

        mcp_stdio.stdio_client = _raise_distinguishable
        monkeypatch.setitem(sys.modules, "mcp", mcp_pkg)
        monkeypatch.setitem(sys.modules, "mcp.client", mcp_client)
        monkeypatch.setitem(sys.modules, "mcp.client.session", mcp_session)
        monkeypatch.setitem(sys.modules, "mcp.client.stdio", mcp_stdio)

        # Even with malware in OSV, the guard should not run.
        monkeypatch.setattr(osv, "_query_osv", lambda *a, **kw: [{"id": "MAL-X", "summary": "x"}])

        client = MCPClient(server_command=["npx", "some-package"], verify_packages=False)
        with pytest.raises(RuntimeError, match="guard skipped"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_connect_no_config_raises(self):
        """Test connect without base_url or server_command raises."""
        from locus.integrations.fastmcp import MCPClient

        client = MCPClient()

        with pytest.raises(ValueError, match="Must provide"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_connect_already_connected(self):
        """Test connect when already connected returns early."""
        from locus.integrations.fastmcp import MCPClient

        client = MCPClient(base_url="https://example.com")
        client._connected = True

        # Should not raise
        await client.connect()

    @pytest.mark.asyncio
    async def test_list_tools_not_connected_raises(self):
        """Test list_tools when not connected raises."""
        from locus.integrations.fastmcp import MCPClient

        client = MCPClient(base_url="https://example.com")

        with pytest.raises(RuntimeError, match="Not connected"):
            await client.list_tools()

    @pytest.mark.asyncio
    async def test_call_tool_not_connected_raises(self):
        """Test call_tool when not connected raises."""
        from locus.integrations.fastmcp import MCPClient

        client = MCPClient(base_url="https://example.com")

        with pytest.raises(RuntimeError, match="Not connected"):
            await client.call_tool("test", {})

    @pytest.mark.asyncio
    async def test_close_not_connected(self):
        """Test close when not connected."""
        from locus.integrations.fastmcp import MCPClient

        client = MCPClient(base_url="https://example.com")

        # Should not raise
        await client.close()
        assert client._connected is False

    @pytest.mark.asyncio
    async def test_close_with_session(self):
        """Test close with active session."""
        from locus.integrations.fastmcp import MCPClient

        client = MCPClient(base_url="https://example.com")
        client._connected = True
        client._session = AsyncMock()
        client._client_context = AsyncMock()

        await client.close()

        assert client._connected is False
        assert client._session is None
        assert client._client_context is None

    @pytest.mark.asyncio
    async def test_close_handles_session_error(self):
        """Test close handles session error gracefully."""
        from locus.integrations.fastmcp import MCPClient

        client = MCPClient(base_url="https://example.com")
        client._connected = True
        client._session = MagicMock()
        client._session.__aexit__ = AsyncMock(side_effect=Exception("error"))
        client._client_context = MagicMock()
        client._client_context.__aexit__ = AsyncMock(side_effect=Exception("error"))

        # Should not raise
        await client.close()

        assert client._connected is False

    @pytest.mark.asyncio
    async def test_context_manager_enter(self):
        """Test async context manager enter."""
        from locus.integrations.fastmcp import MCPClient

        client = MCPClient(base_url="https://example.com")
        client._connected = True  # Pretend already connected

        result = await client.__aenter__()

        assert result is client

    @pytest.mark.asyncio
    async def test_context_manager_exit(self):
        """Test async context manager exit."""
        from locus.integrations.fastmcp import MCPClient

        client = MCPClient(base_url="https://example.com")
        client._connected = True

        await client.__aexit__(None, None, None)

        assert client._connected is False


class TestCreateMCPServer:
    """Tests for create_mcp_server function."""

    def test_create_mcp_server(self):
        """Test creating MCP server from agent."""
        from locus.integrations.fastmcp import LocusMCPServer, create_mcp_server

        mock_agent = MagicMock()
        server = create_mcp_server(mock_agent)

        assert isinstance(server, LocusMCPServer)
        assert server.name == "locus-agent"
        assert server.version == "1.0.0"

    def test_create_mcp_server_custom_params(self):
        """Test creating MCP server with custom params."""
        from locus.integrations.fastmcp import create_mcp_server

        mock_agent = MagicMock()
        server = create_mcp_server(mock_agent, name="custom-server", version="2.0.0")

        assert server.name == "custom-server"
        assert server.version == "2.0.0"


class TestSchemaConversionEdgeCases:
    """Edge case tests for schema conversion."""

    def test_nested_array_type(self):
        """Test nested array type."""
        schema = {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
        result = _json_schema_type_to_python(schema)
        assert result.__origin__ is list

    def test_array_with_object_items(self):
        """Test array of objects."""
        schema = {"type": "array", "items": {"type": "object"}}
        result = _json_schema_type_to_python(schema)
        assert result.__origin__ is list

    def test_nullable_integer(self):
        """Test nullable integer type."""
        result = _json_schema_type_to_python({"type": ["integer", "null"]})
        assert result is int

    def test_all_null_type(self):
        """Test type that's just null."""
        from typing import Any

        result = _json_schema_type_to_python({"type": ["null"]})
        # No non-null type, returns Any
        assert result is Any


class TestBuildArgsModelValidation:
    """Validation tests for build_args_model."""

    def test_required_field_validation(self):
        """Test that required fields are properly marked."""
        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"},
                "optional_field": {"type": "string", "default": "default"},
            },
            "required": ["required_field"],
        }
        model = build_args_model("Test", schema)

        # Required field should not have default
        assert model.model_fields["required_field"].is_required()

        # Optional field should have default
        instance = model(required_field="value")
        assert instance.optional_field == "default"

    def test_description_preserved(self):
        """Test that field descriptions are preserved."""
        schema = {
            "type": "object",
            "properties": {"field": {"type": "string", "description": "My description"}},
        }
        model = build_args_model("Test", schema)

        field_info = model.model_fields["field"]
        assert field_info.description == "My description"

    def test_non_dict_schema(self):
        """Test with non-dict schema returns None."""
        result = build_args_model("Test", "not a dict")
        assert result is None

    def test_non_dict_properties(self):
        """Test with non-dict properties returns None."""
        result = build_args_model("Test", {"type": "object", "properties": "not a dict"})
        assert result is None
