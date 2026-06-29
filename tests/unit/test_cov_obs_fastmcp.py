# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Extra coverage for tulip.integrations.fastmcp — missing branches.

Note: no module-level pytestmark — asyncio_mode=auto picks up async tests
automatically, and sync tests must stay sync to avoid PytestWarning errors.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from tulip.integrations.fastmcp import MCPClient, TulipMCPServer


# ---------------------------------------------------------------------------
# Shared fixture: inject a fake fastmcp module so _create_mcp() never touches
# the real fastmcp/mcp/pydantic stack (which has a KeyError on 'pydantic.root_model'
# in this environment).
# ---------------------------------------------------------------------------


class _FakeFastMCP:
    """Minimal FastMCP stand-in that records @tool()-decorated functions."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._tools: dict[str, Any] = {}

    def tool(self):
        def decorator(func):
            self._tools[func.__name__] = func
            return func

        return decorator

    async def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        func = self._tools[name]
        return await func(**(arguments or {}))

    def run(self, transport: str = "stdio") -> None:  # noqa: ARG002
        pass


def _inject_fake_fastmcp(monkeypatch) -> type:
    """Inject _FakeFastMCP into sys.modules["fastmcp"] for the duration of a test."""
    fake_mod = types.ModuleType("fastmcp")
    fake_mod.FastMCP = _FakeFastMCP  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastmcp", fake_mod)
    return _FakeFastMCP


# ---------------------------------------------------------------------------
# TulipMCPServer._create_mcp — run_agent closure (lines 349, 350)
# ---------------------------------------------------------------------------


class TestCreateMcpRunAgentClosure:
    async def test_run_agent_closure_called_via_fastmcp(self, monkeypatch):
        """Call the run_agent closure registered inside _create_mcp via the
        fake FastMCP.call_tool — covers lines 349-350."""
        _inject_fake_fastmcp(monkeypatch)

        mock_result = MagicMock()
        mock_result.message = "response from agent"
        mock_agent = MagicMock()
        mock_agent._tool_registry = MagicMock()
        mock_agent._tool_registry.tools = {}
        mock_agent._initialize = MagicMock()
        mock_agent.run_sync = MagicMock(return_value=mock_result)

        server = TulipMCPServer(agent=mock_agent)
        mcp = server._create_mcp()

        # Invoke the registered closure through the fake FastMCP.
        result = await mcp.call_tool("run_agent", arguments={"prompt": "hello test"})

        mock_agent.run_sync.assert_called_once_with("hello test")
        assert result is not None

    async def test_run_agent_stream_closure_no_events(self, monkeypatch):
        """run_agent_stream when agent.run() yields nothing returns the
        fallback string without raising."""
        _inject_fake_fastmcp(monkeypatch)

        mock_agent = MagicMock()
        mock_agent._tool_registry = MagicMock()
        mock_agent._tool_registry.tools = {}
        mock_agent._initialize = MagicMock()

        async def empty_run(prompt):
            return
            yield  # makes this an async generator (unreachable after return)

        mock_agent.run = empty_run

        server = TulipMCPServer(agent=mock_agent)
        mcp = server._create_mcp()

        result = await mcp.call_tool("run_agent_stream", arguments={"prompt": "test"})
        assert result is not None


# ---------------------------------------------------------------------------
# TulipMCPServer.run — creates _mcp when None (line 375)
# ---------------------------------------------------------------------------


class TestTulipMCPServerRun:
    def test_run_creates_mcp_when_none(self):
        """server.run() when _mcp is None calls _create_mcp (line 375)."""
        mock_agent = MagicMock()

        server = TulipMCPServer(agent=mock_agent)
        assert server._mcp is None

        mock_mcp = MagicMock()
        with patch.object(server, "_create_mcp", return_value=mock_mcp) as mock_create:
            server.run()

        mock_create.assert_called_once()
        assert server._mcp is mock_mcp
        mock_mcp.run.assert_called_once()

    def test_run_reuses_existing_mcp(self):
        """server.run() when _mcp already exists does NOT call _create_mcp."""
        mock_agent = MagicMock()
        server = TulipMCPServer(agent=mock_agent)

        mock_mcp = MagicMock()
        server._mcp = mock_mcp  # pre-set

        with patch.object(server, "_create_mcp") as mock_create:
            server.run()

        mock_create.assert_not_called()
        mock_mcp.run.assert_called_once()


# ---------------------------------------------------------------------------
# MCPClient._connect_http — BearerAuth + full connection path (lines 549-592)
# ---------------------------------------------------------------------------


class TestConnectHttpWithToken:
    async def test_connect_http_with_access_token_creates_bearer_auth(self, monkeypatch):
        """_connect_http with access_token exercises the BearerAuth class
        (lines 549, 551-553, 555, 558-559, 561) and the full connection
        path (lines 575, 583-592)."""
        mcp_pkg = types.ModuleType("mcp")
        mcp_client_pkg = types.ModuleType("mcp.client")
        mcp_session_mod = types.ModuleType("mcp.client.session")
        mcp_http_mod = types.ModuleType("mcp.client.streamable_http")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.initialize = AsyncMock()

        client_session_cls = MagicMock(return_value=mock_session)
        mcp_session_mod.ClientSession = client_session_cls

        captured: dict = {}

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        def _fake_http_client(url, auth=None, httpx_client_factory=None):
            captured["auth"] = auth
            if httpx_client_factory is not None:
                httpx_client_factory()  # exercises line 575
            return mock_ctx

        mcp_http_mod.streamablehttp_client = _fake_http_client

        monkeypatch.setitem(sys.modules, "mcp", mcp_pkg)
        monkeypatch.setitem(sys.modules, "mcp.client", mcp_client_pkg)
        monkeypatch.setitem(sys.modules, "mcp.client.session", mcp_session_mod)
        monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", mcp_http_mod)

        client = MCPClient(
            base_url="https://example.com/mcp",
            verify_url=False,
            access_token="my-secret-token",  # noqa: S106
        )
        await client.connect()

        assert client._connected is True
        assert captured.get("auth") is not None

        # Exercise BearerAuth.auth_flow (lines 558-559) by calling it directly.
        bearer = captured["auth"]
        mock_request = MagicMock()
        mock_request.headers = {}
        gen = bearer.auth_flow(mock_request)
        next(gen)  # executes: request.headers["Authorization"] = ...; yield request
        assert "Authorization" in mock_request.headers
        assert "Bearer" in mock_request.headers["Authorization"]

    async def test_connect_http_without_token(self, monkeypatch):
        """_connect_http without access_token (auth=None) covers the
        _httpx_factory and connection lines (575, 583-592)."""
        mcp_pkg = types.ModuleType("mcp")
        mcp_client_pkg = types.ModuleType("mcp.client")
        mcp_session_mod = types.ModuleType("mcp.client.session")
        mcp_http_mod = types.ModuleType("mcp.client.streamable_http")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.initialize = AsyncMock()
        mcp_session_mod.ClientSession = MagicMock(return_value=mock_session)

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        captured_factory: dict = {}

        def _fake_http_client(url, auth=None, httpx_client_factory=None):
            if httpx_client_factory:
                captured_factory["called"] = True
                httpx_client_factory()  # line 575
            return mock_ctx

        mcp_http_mod.streamablehttp_client = _fake_http_client

        monkeypatch.setitem(sys.modules, "mcp", mcp_pkg)
        monkeypatch.setitem(sys.modules, "mcp.client", mcp_client_pkg)
        monkeypatch.setitem(sys.modules, "mcp.client.session", mcp_session_mod)
        monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", mcp_http_mod)

        client = MCPClient(base_url="https://safe.example.com/mcp", verify_url=False)
        await client.connect()

        assert client._connected is True
        assert captured_factory.get("called") is True


# ---------------------------------------------------------------------------
# MCPClient._connect_stdio — full path after OSV check (lines 627-631)
# ---------------------------------------------------------------------------


class TestConnectStdioFullPath:
    async def test_connect_stdio_reaches_session_setup(self, monkeypatch):
        """With verify_packages=False and a stubbed stdio_client, _connect_stdio
        completes the full connection path (lines 626-631)."""
        mcp_pkg = types.ModuleType("mcp")
        mcp_client_pkg = types.ModuleType("mcp.client")
        mcp_session_mod = types.ModuleType("mcp.client.session")
        mcp_stdio_mod = types.ModuleType("mcp.client.stdio")

        class _StdioParams:
            def __init__(self, command: str, args: list) -> None:
                self.command = command
                self.args = args

        mcp_pkg.StdioServerParameters = _StdioParams

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.initialize = AsyncMock()
        mcp_session_mod.ClientSession = MagicMock(return_value=mock_session)

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        def _fake_stdio_client(params):
            return mock_ctx

        mcp_stdio_mod.stdio_client = _fake_stdio_client

        monkeypatch.setitem(sys.modules, "mcp", mcp_pkg)
        monkeypatch.setitem(sys.modules, "mcp.client", mcp_client_pkg)
        monkeypatch.setitem(sys.modules, "mcp.client.session", mcp_session_mod)
        monkeypatch.setitem(sys.modules, "mcp.client.stdio", mcp_stdio_mod)

        client = MCPClient(
            server_command=["python", "server.py"],
            verify_packages=False,
        )
        await client.connect()

        assert client._connected is True
        mock_session.initialize.assert_called_once()


# ---------------------------------------------------------------------------
# MCPClient.to_tulip_tools — lines 701-722 (sync — need to provide event loop)
# Line 710: return await self.call_tool(name, kwargs)
# ---------------------------------------------------------------------------


class TestToTulipTools:
    def test_to_tulip_tools_creates_tool_list_and_covers_line_710(self, monkeypatch):
        """to_tulip_tools creates Tool objects from MCP tool dicts (line 717).
        The func closure (line 710) is exercised by calling the returned fn.

        to_tulip_tools calls asyncio.get_event_loop().run_until_complete() internally,
        which requires a running-but-not-yet-started loop in the calling thread.
        We monkeypatch asyncio.get_event_loop to provide a fresh one.

        MCPClient is a Pydantic BaseModel; instance attr assignment is blocked.
        Patch call_tool at the CLASS level so the func closure can resolve it.
        """
        import asyncio as _asyncio  # noqa: PLC0415

        loop = _asyncio.new_event_loop()
        monkeypatch.setattr(_asyncio, "get_event_loop", lambda: loop)

        # Patch call_tool on the class (Pydantic forbids setting extra instance attrs).
        # When set as a class attribute and accessed via an instance, Python binds
        # self as the first positional argument.
        async def _fake_call_tool(self_arg, name: str, kwargs: dict) -> str:  # noqa: ARG001
            return f"result:{name}"

        monkeypatch.setattr(MCPClient, "call_tool", _fake_call_tool)

        try:
            client = MCPClient(base_url="https://example.com")
            client._connected = True

            mcp_tools = [
                {"name": "search", "description": "Search something", "inputSchema": None},
                {
                    "name": "fetch",
                    "description": "Fetch a URL",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            ]

            tools = client.to_tulip_tools(mcp_tools)  # line 717: run_until_complete
            assert len(tools) == 2
            assert tools[0].name == "search"
            assert tools[1].name == "fetch"

            # Call tools[0].fn() to exercise the func closure → line 710.
            result = loop.run_until_complete(tools[0].fn())
            assert result is not None
        finally:
            loop.close()

    def test_to_tulip_tools_empty_list(self):
        """to_tulip_tools on empty list returns empty list without touching
        asyncio.get_event_loop (no tools to iterate over)."""
        client = MCPClient(base_url="https://example.com")
        tools = client.to_tulip_tools([])
        assert tools == []
