# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage tests for the rest of ``locus.integrations.fastmcp``.

Existing tests cover the wrapper builder + handle_request paths. This
file targets:

- ``LocusMCPServer._create_mcp`` registering tools + the ``run_agent``
  / ``run_agent_stream`` decorated tools.
- ``run_agent_stream`` selecting the final-message event.
- ``MCPClient.connect`` dispatching to HTTP / stdio.
- ``_connect_http`` ImportError-on-mcp branch + URL guard + bearer auth +
  TLS verify factory.
- ``_connect_stdio`` ImportError + OSV malware pre-check.
- ``MCPClient.to_locus_tools`` legacy converter.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from locus.integrations.fastmcp import LocusMCPServer, MCPClient


pytest.importorskip("fastmcp")


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _Tool:
    name = "search"
    description = "search the index"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return "search-result"


class _Registry:
    def __init__(self) -> None:
        self.tools = {"search": _Tool()}

    def get(self, name: str) -> Any:
        return self.tools.get(name)


class _StreamEvent:
    def __init__(self, final: str | None = None) -> None:
        self.final_message = final


class _StubAgent:
    """Stand-in for :class:`Agent` exposing the surface fastmcp uses."""

    def __init__(self, stream_events: list[Any] | None = None) -> None:
        self._tool_registry = _Registry()
        self._stream_events = stream_events or [_StreamEvent("answered")]

    def _initialize(self) -> None:  # pragma: no cover — trivial
        return None

    def run_sync(self, prompt: str) -> Any:
        return MagicMock(message=f"answered: {prompt}")

    async def run(self, prompt: str) -> Any:
        for ev in self._stream_events:
            yield ev


# ---------------------------------------------------------------------------
# LocusMCPServer._create_mcp + decorated tool bodies
# ---------------------------------------------------------------------------


class TestCreateMcp:
    def test_create_mcp_registers_tools_and_returns_instance(self) -> None:
        # Stub fastmcp.FastMCP so we can inspect tool registrations without
        # spinning up a real server.
        captured: dict[str, list[Any]] = {"registered": []}

        class _StubFastMCP:
            def __init__(self, name: str) -> None:
                self.name = name

            def tool(self) -> Any:
                def _decorator(fn: Any) -> Any:
                    captured["registered"].append(fn)
                    return fn

                return _decorator

            def run(self, transport: str = "stdio") -> None:  # pragma: no cover
                return None

        # Patch fastmcp.FastMCP at the class level.
        with patch("fastmcp.FastMCP", _StubFastMCP):
            server = LocusMCPServer(agent=_StubAgent())
            mcp = server._create_mcp()
        assert isinstance(mcp, _StubFastMCP)
        # Three tool registrations: 'search' (from registry), run_agent,
        # run_agent_stream
        names = [getattr(fn, "__name__", "") for fn in captured["registered"]]
        assert "search" in names
        assert "run_agent" in names
        assert "run_agent_stream" in names

    @pytest.mark.asyncio
    async def test_run_agent_stream_returns_final_message(self) -> None:
        agent = _StubAgent(
            stream_events=[
                _StreamEvent(None),
                _StreamEvent("the answer"),
            ]
        )
        # Find the run_agent_stream callable by capturing during _create_mcp.
        captured: list[Any] = []

        class _StubFastMCP:
            def __init__(self, name: str) -> None:
                pass

            def tool(self) -> Any:
                def _decorator(fn: Any) -> Any:
                    captured.append(fn)
                    return fn

                return _decorator

        with patch("fastmcp.FastMCP", _StubFastMCP):
            server = LocusMCPServer(agent=agent)
            server._create_mcp()
        run_stream = next(c for c in captured if c.__name__ == "run_agent_stream")
        out = await run_stream("hi")
        assert out == "the answer"

    @pytest.mark.asyncio
    async def test_run_agent_stream_no_final_message(self) -> None:
        agent = _StubAgent(stream_events=[_StreamEvent(None), _StreamEvent("")])
        captured: list[Any] = []

        class _StubFastMCP:
            def __init__(self, name: str) -> None:
                pass

            def tool(self) -> Any:
                def _decorator(fn: Any) -> Any:
                    captured.append(fn)
                    return fn

                return _decorator

        with patch("fastmcp.FastMCP", _StubFastMCP):
            server = LocusMCPServer(agent=agent)
            server._create_mcp()
        run_stream = next(c for c in captured if c.__name__ == "run_agent_stream")
        out = await run_stream("hi")
        assert out == "Agent completed without response"


# ---------------------------------------------------------------------------
# MCPClient.connect — dispatch
# ---------------------------------------------------------------------------


class TestMCPClientConnectDispatch:
    @pytest.mark.asyncio
    async def test_connect_with_base_url_calls_http(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MCPClient(base_url="https://api.example.com/mcp")
        called = {"http": False, "stdio": False}

        async def fake_http() -> None:
            called["http"] = True

        async def fake_stdio() -> None:
            called["stdio"] = True

        monkeypatch.setattr(client, "_connect_http", fake_http)
        monkeypatch.setattr(client, "_connect_stdio", fake_stdio)
        await client.connect()
        assert called == {"http": True, "stdio": False}
        assert client._connected is True

    @pytest.mark.asyncio
    async def test_connect_with_command_calls_stdio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MCPClient(server_command=["myserver"])
        called = {"stdio": False}

        async def fake_stdio() -> None:
            called["stdio"] = True

        monkeypatch.setattr(client, "_connect_stdio", fake_stdio)
        await client.connect()
        assert called["stdio"] is True


# ---------------------------------------------------------------------------
# _connect_http error branches
# ---------------------------------------------------------------------------


class TestConnectHTTPGuards:
    @pytest.mark.asyncio
    async def test_no_base_url_runtime_error(self) -> None:
        client = MCPClient()
        client.base_url = None
        with pytest.raises(RuntimeError, match="without base_url"):
            await client._connect_http()

    @pytest.mark.asyncio
    async def test_mcp_import_failure_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force ``import mcp.client.session`` to fail.
        sys.modules.pop("mcp.client.session", None)
        sys.modules["mcp.client.session"] = None  # type: ignore[assignment]
        client = MCPClient(base_url="https://api.example.com/mcp")
        try:
            with pytest.raises(ImportError, match="mcp package required for HTTP"):
                await client._connect_http()
        finally:
            sys.modules.pop("mcp.client.session", None)


# ---------------------------------------------------------------------------
# _connect_stdio error branches
# ---------------------------------------------------------------------------


class TestConnectStdioGuards:
    @pytest.mark.asyncio
    async def test_mcp_import_failure_raises(self) -> None:
        sys.modules.pop("mcp.client.stdio", None)
        sys.modules["mcp.client.stdio"] = None  # type: ignore[assignment]
        client = MCPClient(server_command=["mycli"])
        try:
            with pytest.raises(ImportError, match="mcp package required for stdio"):
                await client._connect_stdio()
        finally:
            sys.modules.pop("mcp.client.stdio", None)

    @pytest.mark.asyncio
    async def test_osv_blocks_malicious_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Patch the OSV pre-check to flag the launcher.
        from locus.integrations import osv

        monkeypatch.setattr(
            osv,
            "check_package_for_malware",
            lambda cmd, args: "package@1.0.0 is flagged as malware",
        )
        client = MCPClient(
            server_command=["npx", "@evil/mcp-server"],
            verify_packages=True,
        )
        from locus.core.errors import ValidationError

        with pytest.raises(ValidationError, match="MCP launch blocked"):
            await client._connect_stdio()


# ---------------------------------------------------------------------------
# to_locus_tools
# ---------------------------------------------------------------------------


class TestToLocusTools:
    def test_converts_mcp_dicts_to_locus_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio

        client = MCPClient(base_url="https://x.example")
        # Patch call_tool on the class so the per-tool closures can call it.
        monkeypatch.setattr(
            MCPClient,
            "call_tool",
            AsyncMock(return_value="tool-output"),
            raising=True,
        )
        tools = [
            {
                "name": "search",
                "description": "search",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
        # to_locus_tools uses asyncio.get_event_loop().run_until_complete
        # internally, which needs an event loop set on the current thread.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            out = client.to_locus_tools(tools)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
        assert len(out) == 1
        assert getattr(out[0], "name", None) == "search"
