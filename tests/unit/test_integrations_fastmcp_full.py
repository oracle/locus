# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage tests for ``locus.integrations.fastmcp``.

The existing ``test_fastmcp.py`` covers the helper-function paths.
This file targets the remaining gaps:

- ``_create_tool_wrapper`` for required vs optional vs zero-arg
  branches plus the unsafe-name guards
- ``LocusMCPServer.handle_request`` for tools/list, tools/call
  (run_agent, registry tool, unknown tool), unknown method
- ``LocusMCPServer.run`` invokes FastMCP with the chosen transport
- ``MCPClient.connect`` rejects when neither URL nor command is set
- ``MCPClient.list_tools`` / ``call_tool`` raise when not connected
- ``MCPClient.close`` is idempotent and silently swallows teardown
  exceptions per its docstring
- ``MCPClient.__aenter__`` / ``__aexit__`` wrap connect/close
- ``MCPClient.list_tools`` and ``call_tool`` SDK-shape extraction
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from locus.integrations.fastmcp import (
    LocusMCPServer,
    MCPClient,
    _create_tool_wrapper,
    create_mcp_server,
)
from locus.tools.decorator import tool


# ---------------------------------------------------------------------------
# _create_tool_wrapper
# ---------------------------------------------------------------------------


def _make_tool(*, name: str = "echo", parameters: dict | None = None) -> Any:
    """Build a minimal Tool with the given name + parameter schema."""

    @tool(name=name, description="echo")
    async def fn(**kwargs: Any) -> str:
        return str(kwargs)

    if parameters is not None:
        fn.parameters = parameters  # type: ignore[attr-defined]
    return fn


class TestCreateToolWrapper:
    @pytest.mark.asyncio
    async def test_no_args_wrapper(self) -> None:
        t = _make_tool(name="ping", parameters={"type": "object", "properties": {}})
        wrapper = _create_tool_wrapper(t)
        assert wrapper.__name__ == "ping"
        result = await wrapper()
        # The actual user fn returns ``str(kwargs)``; with no args this is ``"{}"``.
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_required_param_wrapper(self) -> None:
        t = _make_tool(
            name="search",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
        wrapper = _create_tool_wrapper(t)
        result = await wrapper(q="hi")
        assert "q" in result
        assert "hi" in result

    @pytest.mark.asyncio
    async def test_optional_param_default_dropped(self) -> None:
        t = _make_tool(
            name="opt",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": [],
            },
        )
        wrapper = _create_tool_wrapper(t)
        # Caller omits ``q`` — wrapper should not pass None through.
        result = await wrapper()
        assert "None" not in result

    @pytest.mark.asyncio
    async def test_filters_stray_kwargs(self) -> None:
        t = _make_tool(
            name="filtered",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
        wrapper = _create_tool_wrapper(t)
        result = await wrapper(q="hi", stray="x")
        assert "stray" not in result

    def test_unsafe_tool_name_raises(self) -> None:
        t = _make_tool(name="bad", parameters={"type": "object", "properties": {}})
        t.name = "bad name with spaces"
        with pytest.raises(ValueError, match="Unsafe tool name"):
            _create_tool_wrapper(t)

    def test_unsafe_param_name_raises(self) -> None:
        t = _make_tool(
            name="ok",
            parameters={
                "type": "object",
                "properties": {"with space": {"type": "string"}},
            },
        )
        with pytest.raises(ValueError, match="Unsafe parameter name"):
            _create_tool_wrapper(t)


# ---------------------------------------------------------------------------
# LocusMCPServer.handle_request
# ---------------------------------------------------------------------------


class _Tool:
    name = "search"
    description = "search the index"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return "search-result"


class _Registry:
    def __init__(self, tool_obj: _Tool) -> None:
        self.tools = {tool_obj.name: tool_obj}

    def get(self, name: str) -> Any:
        return self.tools.get(name)


class _StubAgent:
    def __init__(self) -> None:
        self._tool_registry = _Registry(_Tool())

    def _initialize(self) -> None:
        return None

    def run_sync(self, prompt: str) -> Any:
        return MagicMock(message=f"answered: {prompt}")


class TestServerHandleRequest:
    @pytest.mark.asyncio
    async def test_tools_list(self) -> None:
        server = LocusMCPServer(agent=_StubAgent())
        server._mcp = MagicMock()  # avoid FastMCP construction
        result = await server.handle_request({"method": "tools/list"})
        names = {t["name"] for t in result["tools"]}
        assert "search" in names

    @pytest.mark.asyncio
    async def test_tools_call_run_agent(self) -> None:
        server = LocusMCPServer(agent=_StubAgent())
        server._mcp = MagicMock()
        result = await server.handle_request(
            {
                "method": "tools/call",
                "params": {"name": "run_agent", "arguments": {"prompt": "hi"}},
            }
        )
        assert result["content"][0]["text"] == "answered: hi"

    @pytest.mark.asyncio
    async def test_tools_call_registry_tool(self) -> None:
        server = LocusMCPServer(agent=_StubAgent())
        server._mcp = MagicMock()
        result = await server.handle_request(
            {
                "method": "tools/call",
                "params": {"name": "search", "arguments": {}},
            }
        )
        assert result["content"][0]["text"] == "search-result"

    @pytest.mark.asyncio
    async def test_tools_call_unknown_tool(self) -> None:
        server = LocusMCPServer(agent=_StubAgent())
        server._mcp = MagicMock()
        result = await server.handle_request(
            {
                "method": "tools/call",
                "params": {"name": "missing", "arguments": {}},
            }
        )
        assert "Unknown tool" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_unknown_method(self) -> None:
        server = LocusMCPServer(agent=_StubAgent())
        server._mcp = MagicMock()
        result = await server.handle_request({"method": "weird/op"})
        assert "Unknown method" in result["error"]["message"]


class TestServerRun:
    def test_run_calls_fastmcp_with_transport(self) -> None:
        server = LocusMCPServer(agent=_StubAgent())
        fake_mcp = MagicMock()
        server._mcp = fake_mcp
        server.run(transport="stdio")
        fake_mcp.run.assert_called_once_with(transport="stdio")


class TestCreateMcpServer:
    def test_returns_server_instance(self) -> None:
        agent = _StubAgent()
        server = create_mcp_server(agent, name="my-agent", version="2.0.0")
        assert isinstance(server, LocusMCPServer)
        assert server.name == "my-agent"
        assert server.version == "2.0.0"


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------


class TestMCPClientConnect:
    @pytest.mark.asyncio
    async def test_neither_url_nor_command_raises(self) -> None:
        client = MCPClient()
        with pytest.raises(ValueError, match="Must provide either base_url or server_command"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_already_connected_short_circuits(self) -> None:
        client = MCPClient(base_url="https://example.com")
        client._connected = True
        # Should NOT actually try to connect — no HTTP call, no exception.
        await client.connect()


class TestMCPClientPreConnectGuards:
    @pytest.mark.asyncio
    async def test_list_tools_without_session_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Not connected"):
            await MCPClient().list_tools()

    @pytest.mark.asyncio
    async def test_call_tool_without_session_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Not connected"):
            await MCPClient().call_tool("x", {})


class TestMCPClientCalls:
    @pytest.mark.asyncio
    async def test_list_tools_extracts_fields(self) -> None:
        client = MCPClient()
        mcp_tool = MagicMock()
        mcp_tool.name = "search"
        mcp_tool.description = "desc"
        mcp_tool.inputSchema = {"type": "object"}
        result = MagicMock(tools=[mcp_tool])
        client._session = AsyncMock()
        client._session.list_tools.return_value = result
        tools = await client.list_tools()
        assert tools[0]["name"] == "search"
        assert tools[0]["description"] == "desc"

    @pytest.mark.asyncio
    async def test_list_tools_handles_no_input_schema(self) -> None:
        client = MCPClient()

        class _T:
            name = "search"
            description = None
            # No ``inputSchema`` attr — the client's ``hasattr`` branch
            # falls through to ``{}``.

        result = MagicMock(tools=[_T()])
        client._session = AsyncMock()
        client._session.list_tools.return_value = result
        tools = await client.list_tools()
        assert tools[0]["description"] == ""
        assert tools[0]["inputSchema"] == {}

    @pytest.mark.asyncio
    async def test_call_tool_extracts_text_content(self) -> None:
        client = MCPClient()
        item = MagicMock()
        item.text = "tool output"
        result = MagicMock(content=[item])
        client._session = AsyncMock()
        client._session.call_tool.return_value = result
        out = await client.call_tool("search", {"q": "hi"})
        assert out == "tool output"

    @pytest.mark.asyncio
    async def test_call_tool_no_text_attr_falls_back_to_str(self) -> None:
        client = MCPClient()
        item = MagicMock(spec=[])
        result = MagicMock(content=[item])
        client._session = AsyncMock()
        client._session.call_tool.return_value = result
        out = await client.call_tool("search", {})
        assert isinstance(out, str)

    @pytest.mark.asyncio
    async def test_call_tool_no_content_attr(self) -> None:
        client = MCPClient()
        result = MagicMock(spec=[])
        client._session = AsyncMock()
        client._session.call_tool.return_value = result
        out = await client.call_tool("search", {})
        assert isinstance(out, str)


class TestMCPClientClose:
    @pytest.mark.asyncio
    async def test_close_with_open_session_and_context(self) -> None:
        client = MCPClient()
        session = AsyncMock()
        ctx = AsyncMock()
        client._session = session
        client._client_context = ctx
        client._connected = True
        await client.close()
        assert client._session is None
        assert client._client_context is None
        assert client._connected is False

    @pytest.mark.asyncio
    async def test_close_swallows_session_exit_exception(self) -> None:
        client = MCPClient()
        session = AsyncMock()
        session.__aexit__.side_effect = RuntimeError("session close fail")
        client._session = session
        client._client_context = None
        client._connected = True
        await client.close()  # Must not raise.
        assert client._session is None

    @pytest.mark.asyncio
    async def test_close_swallows_context_exit_exception(self) -> None:
        client = MCPClient()
        ctx = AsyncMock()
        ctx.__aexit__.side_effect = RuntimeError("ctx close fail")
        client._session = None
        client._client_context = ctx
        client._connected = True
        await client.close()
        assert client._client_context is None


class TestMCPClientAsyncContext:
    @pytest.mark.asyncio
    async def test_aenter_calls_connect(self) -> None:
        client = MCPClient(base_url="https://example.com")
        client._connected = True  # short-circuit connect()
        entered = await client.__aenter__()
        assert entered is client

    @pytest.mark.asyncio
    async def test_aexit_calls_close(self) -> None:
        client = MCPClient()
        await client.__aexit__(None, None, None)
        assert client._connected is False
