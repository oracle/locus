# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""MCP client internals that a loopback server cannot easily provoke.

``test_mcp_fidelity.py`` covers the behaviour end to end against a real
server; these pin the edges — every content-block kind, a hung connect, a
session that never answers, per-identity eviction — with fakes.
"""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tulip.integrations import fastmcp
from tulip.integrations.fastmcp import (
    MCPClient,
    MCPConnectionError,
    MCPRequestContext,
    MCPToolNotAllowedError,
    MCPToolResult,
    _await_while_alive,
    _convert_call_result,
    _is_connection_loss,
    _SessionHandle,
)


def _block(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**fields)


# ---------------------------------------------------------------------------
# Result conversion
# ---------------------------------------------------------------------------


def test_every_content_kind_is_rendered_for_the_model() -> None:
    png = base64.b64encode(b"\x89PNG fake").decode()
    raw = SimpleNamespace(
        content=[
            _block(type="text", text="caption"),
            _block(type="image", data=png, mimeType="image/png"),
            _block(type="image", data="%%%not-base64%%%", mimeType="image/jpeg"),
            _block(type="audio", data="AAAA", mimeType="audio/wav"),
            _block(type="resource", resource=_block(uri="file:///a.txt", text="file body")),
            _block(
                type="resource", resource=_block(uri="file:///b.bin", mimeType="application/zip")
            ),
            _block(type="resource_link", name="report", uri="https://x/r.pdf"),
            _block(type="mystery"),
        ],
        structuredContent=None,
        isError=False,
        meta={"trace": "t1"},
    )
    result = _convert_call_result(raw)
    lines = result.text.split("\n")
    assert lines[0] == "caption"
    assert "[tulip-image media_type=image/png]" in result.text
    assert "[image: image/jpeg]" in result.text
    assert "[audio: audio/wav]" in result.text
    assert "file body" in result.text
    assert "[resource: file:///b.bin (application/zip)]" in result.text
    assert "[resource link: report https://x/r.pdf]" in result.text
    assert result.meta == {"trace": "t1"}


def test_images_can_be_markers_instead_of_embedded() -> None:
    raw = SimpleNamespace(content=[_block(type="image", data="AAAA", mimeType=None)])
    assert _convert_call_result(raw, embed_images=False).text == "[image: image]"


def test_structured_only_result_falls_back_to_json_text() -> None:
    raw = SimpleNamespace(content=[], structuredContent={"a": 1}, isError=False)
    result = _convert_call_result(raw)
    assert result.text == '{"a": 1}'
    assert result.structured_content == {"a": 1}


def test_empty_result_and_contentless_result() -> None:
    assert _convert_call_result(SimpleNamespace(content=[])).text == ""
    assert _convert_call_result(SimpleNamespace()).text.startswith("namespace(")


def test_a_block_whose_dump_raises_is_skipped() -> None:
    class Bad:
        type = "text"
        text = "still here"

        def model_dump(self, **_: Any) -> dict[str, Any]:
            raise ValueError("boom")

    result = _convert_call_result(SimpleNamespace(content=[Bad()]))
    assert result.text == "still here"
    assert result.content == []


def test_to_tool_output_keeps_non_text_blocks_only() -> None:
    result = MCPToolResult(
        text="t",
        is_error=True,
        structured_content={"k": 1},
        content=[{"type": "text", "text": "t"}, {"type": "image", "data": "x"}],
    )
    out = result.to_tool_output()
    assert out == "t"
    assert out.is_error is True
    assert out.content_blocks == [{"type": "image", "data": "x"}]


def test_connection_loss_classification() -> None:
    class BrokenResourceError(Exception):
        pass

    mcp_closed = Exception("closed")
    mcp_closed.error = SimpleNamespace(code=-32000)  # type: ignore[attr-defined]
    assert _is_connection_loss(BrokenResourceError())
    assert _is_connection_loss(ExceptionGroup("g", [ConnectionResetError()]))
    assert _is_connection_loss(mcp_closed)
    assert not _is_connection_loss(ValueError("tool failed"))


def test_accepts_kwarg_handles_unintrospectable_callables() -> None:
    assert fastmcp._accepts_kwarg(print, "x") in (True, False)
    assert fastmcp._accepts_kwarg(object(), "x") is False


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------


class _HangingTransport:
    async def __aenter__(self) -> Any:
        await asyncio.sleep(3600)

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _Streams:
    async def __aenter__(self) -> tuple[Any, Any]:
        return (object(), object())

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _Session:
    def __init__(self, *_: Any) -> None:
        self.initialized = False

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def initialize(self) -> None:
        self.initialized = True


async def test_a_hung_connect_times_out_as_a_connection_error() -> None:
    handle = _SessionHandle("hung")
    with pytest.raises(MCPConnectionError, match="timed out"):
        await handle.start(_HangingTransport(), _Session, connect_timeout=0.05)
    assert not handle.alive


async def test_a_runner_that_ignores_stop_is_cancelled_on_close() -> None:
    class Stubborn(_Session):
        async def __aexit__(self, *exc: Any) -> None:
            await asyncio.sleep(3600)

    handle = _SessionHandle("stubborn")
    await handle.start(_Streams(), Stubborn, connect_timeout=5)
    assert handle.alive
    await handle.close(grace=0.05)
    assert not handle.alive


async def test_a_session_that_ends_after_setup_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    class Dies(_Session):
        async def __aexit__(self, *exc: Any) -> None:
            raise RuntimeError("transport died")

    handle = _SessionHandle("dies")
    await handle.start(_Streams(), Dies, connect_timeout=5)
    assert handle._stop is not None
    handle._stop.set()
    await asyncio.wait({handle.task})  # type: ignore[arg-type]
    assert "transport died" in caplog.text


async def test_setup_ending_without_a_session_fails_start() -> None:
    class Swallower:
        async def __aenter__(self) -> tuple[Any, Any]:
            return (object(), object())

        async def __aexit__(self, *exc: Any) -> bool:
            return True  # swallows the body's exception

    class NoInit(_Session):
        async def initialize(self) -> None:
            raise asyncio.CancelledError

    handle = _SessionHandle("swallow")
    with pytest.raises(MCPConnectionError, match="during setup"):
        await handle.start(Swallower(), NoInit, connect_timeout=5)


async def test_cancelled_setup_reports_unreachable() -> None:
    class CancelsOnInit(_Session):
        async def initialize(self) -> None:
            raise asyncio.CancelledError

    handle = _SessionHandle("refused")
    with pytest.raises(MCPConnectionError, match="unreachable"):
        await handle.start(_Streams(), CancelsOnInit, connect_timeout=5)


async def test_await_while_alive_pings_and_gives_up_on_a_dead_ping() -> None:
    handle = _SessionHandle("pinged")
    handle.session = SimpleNamespace(send_ping=AsyncMock(side_effect=ConnectionError("gone")))
    handle.task = asyncio.ensure_future(asyncio.sleep(3600))
    try:
        with pytest.raises(MCPConnectionError):
            await _await_while_alive(handle, asyncio.sleep(3600), liveness_interval=0.01)
    finally:
        handle.task.cancel()


async def test_await_while_alive_tolerates_slow_pings_and_missing_ping() -> None:
    async def slow_ping() -> None:
        await asyncio.sleep(3600)

    async def answer() -> str:
        await asyncio.sleep(0.05)
        return "answer"

    handle = _SessionHandle("slow")
    handle.task = asyncio.ensure_future(asyncio.sleep(3600))
    try:
        handle.session = SimpleNamespace(send_ping=slow_ping)
        assert await _await_while_alive(handle, answer(), liveness_interval=0.01) == "answer"
        handle.session = SimpleNamespace()
        assert await _await_while_alive(handle, answer(), liveness_interval=0.01) == "answer"
    finally:
        handle.task.cancel()


async def test_await_while_alive_propagates_cancellation() -> None:
    handle = _SessionHandle("cancel")
    handle.task = asyncio.ensure_future(asyncio.sleep(3600))
    inner = asyncio.Event()

    async def never() -> None:
        inner.set()
        await asyncio.sleep(3600)

    waiter = asyncio.ensure_future(_await_while_alive(handle, never()))
    await inner.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    handle.task.cancel()


# ---------------------------------------------------------------------------
# Client plumbing with fake sessions
# ---------------------------------------------------------------------------


def _session(**calls: Any) -> Any:
    session = MagicMock()
    for name, value in calls.items():
        setattr(session, name, value)
    return session


def test_labels() -> None:
    assert MCPClient(name="crm").label == "crm"
    assert MCPClient(server_command=["npx", "srv"]).label == "npx"
    assert MCPClient().label == "MCPClient"


async def test_list_tools_keeps_title_and_annotations_and_filters() -> None:
    tool = SimpleNamespace(
        name="search",
        description="d",
        inputSchema={"type": "object"},
        outputSchema=None,
        title="Search",
        annotations=SimpleNamespace(model_dump=lambda **_: {"readOnlyHint": True}),
    )
    hidden = SimpleNamespace(name="drop_db", description="", inputSchema={})
    client = MCPClient(tool_filter=lambda s: not s["name"].startswith("drop"))
    client._session = AsyncMock()
    client._session.list_tools.return_value = SimpleNamespace(tools=[tool, hidden])
    [schema] = await client.list_tools()
    assert schema["title"] == "Search"
    assert schema["annotations"] == {"readOnlyHint": True}
    with pytest.raises(MCPToolNotAllowedError):
        await client.call_tool("drop_db", {})


async def test_call_timeout_and_progress_callback_are_forwarded() -> None:
    captured: dict[str, Any] = {}

    async def call_tool(name: str, arguments: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        await kwargs["progress_callback"](1, 2, "half")
        return SimpleNamespace(content=[_block(type="text", text="ok")])

    async def failing_observer(p: float, t: float | None, m: str | None) -> None:
        raise RuntimeError("observer bug")

    client = MCPClient(call_timeout=7)
    client._session = _session(call_tool=call_tool)
    result = await client.call_tool_result("t", {}, progress_callback=failing_observer)
    assert result.text == "ok"
    assert captured["read_timeout_seconds"].total_seconds() == 7


async def test_a_leaked_cancellation_from_a_legacy_session_becomes_a_connection_error() -> None:
    client = MCPClient()
    client._session = _session(call_tool=AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(MCPConnectionError, match="was lost"):
        await client.call_tool("t", {})
    assert client._session is None


async def test_a_protocol_error_on_a_live_session_is_raised_as_is() -> None:
    client = MCPClient()
    client._session = _session(call_tool=AsyncMock(side_effect=ValueError("unknown tool")))
    with pytest.raises(ValueError, match="unknown tool"):
        await client.call_tool("t", {})
    assert client._session is not None


async def test_sync_headers_provider_and_identity_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[dict[str, str]] = []
    closed: list[str] = []

    async def fake_connect_http(
        self: MCPClient, extra_headers: Any = None, *, install: bool = True
    ) -> _SessionHandle:
        opened.append(dict(extra_headers or {}))
        handle = _SessionHandle(str(extra_headers))
        handle.session = _session(
            call_tool=AsyncMock(
                return_value=SimpleNamespace(content=[_block(type="text", text=str(extra_headers))])
            )
        )
        handle.task = asyncio.ensure_future(asyncio.sleep(3600))

        async def _close(grace: float = 5.0) -> None:
            closed.append(handle.label)
            assert handle.task is not None
            handle.task.cancel()

        handle.close = _close  # type: ignore[method-assign]
        return handle

    monkeypatch.setattr(MCPClient, "_connect_http", fake_connect_http)

    def provider(rctx: MCPRequestContext) -> dict[str, str]:
        return {"X-User": str(rctx.metadata["user"])}

    client = MCPClient(
        base_url="http://mcp.test/mcp",
        headers_provider=provider,
        metadata_headers_key=None,
        max_sessions=1,
    )
    token = fastmcp._request_metadata.set({"user": "a"})
    try:
        assert "X-User" in await client.call_tool("t", {})
        assert "X-User" in await client.call_tool("t", {})  # reused
    finally:
        fastmcp._request_metadata.reset(token)
    token = fastmcp._request_metadata.set({"user": "b"})
    try:
        await client.call_tool("t", {})
    finally:
        fastmcp._request_metadata.reset(token)
    assert opened == [{"X-User": "a"}, {"X-User": "b"}]
    assert len(closed) == 1  # "a" evicted when "b" arrived (max_sessions=1)

    # A dead identity session is replaced on next use.
    [(key, handle)] = list(client._identity_sessions.items())
    assert handle.task is not None
    handle.task.cancel()
    await asyncio.sleep(0)
    token = fastmcp._request_metadata.set({"user": "b"})
    try:
        await client.call_tool("t", {})
    finally:
        fastmcp._request_metadata.reset(token)
    assert opened[-1] == {"X-User": "b"}
    assert len(opened) == 3
    await client.close()


async def test_discarding_an_identity_session_forgets_it() -> None:
    client = MCPClient(base_url="http://mcp.test/mcp")
    handle = _SessionHandle("x")
    client._identity_sessions[(("a", "b"),)] = handle
    await client._discard(handle)
    assert client._identity_sessions == {}


async def test_the_default_session_reconnects_when_its_runner_died(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MCPClient(base_url="http://mcp.test/mcp")
    dead = _SessionHandle("dead")
    dead.task = asyncio.ensure_future(asyncio.sleep(0))
    await dead.task
    client._runner = dead
    client._session = object()
    client._ever_connected = True
    reconnected: list[bool] = []

    async def fake_connect(self: MCPClient) -> None:
        reconnected.append(True)
        self._session = _session(
            call_tool=AsyncMock(
                return_value=SimpleNamespace(content=[_block(type="text", text="back")])
            )
        )
        self._connected = True

    monkeypatch.setattr(MCPClient, "connect", fake_connect)
    assert await client.call_tool("t", {}) == "back"
    assert reconnected == [True]


async def test_new_style_streamable_http_client_is_used_when_legacy_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mcp.client.streamable_http as sh

    captured: dict[str, Any] = {}

    def new_client(url: str, *, http_client: Any) -> Any:
        captured["headers"] = dict(http_client.headers)
        return _Streams()

    monkeypatch.delattr(sh, "streamablehttp_client", raising=False)
    monkeypatch.setattr(sh, "streamable_http_client", new_client, raising=False)
    monkeypatch.setattr("mcp.client.session.ClientSession", _Session)
    client = MCPClient(base_url="http://mcp.test/mcp", verify_url=False, headers={"X-T": "1"})
    await client.connect()
    try:
        assert captured["headers"]["x-t"] == "1"
        assert client._session.initialized is True
    finally:
        await client.close()


async def test_load_tools_without_dynamic_headers_connects_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MCPClient(base_url="http://mcp.test/mcp")
    calls: list[str] = []

    async def fake_connect(self: MCPClient) -> None:
        calls.append("connect")
        self._connected = True
        self._ever_connected = True
        self._session = AsyncMock()
        self._session.list_tools.return_value = SimpleNamespace(
            tools=[SimpleNamespace(name="a", description="", inputSchema={"type": "object"})]
        )

    monkeypatch.setattr(MCPClient, "connect", fake_connect)
    tools = await client.load_tools({"user": "x"})
    tools_again = await client.load_tools()
    assert [t.name for t in tools] == ["a"] == [t.name for t in tools_again]
    assert calls == ["connect"]


def test_to_tulip_tools_respects_the_allowlist() -> None:
    client = MCPClient(allowed_tools=["keep"])
    tools = client.to_tulip_tools(
        [{"name": "keep", "inputSchema": None}, {"name": "drop", "inputSchema": None}]
    )
    assert [t.name for t in tools] == ["keep"]
    assert tools[0].emits_progress is True


class TestMcpFieldAcrossSdkMajors:
    """mcp 1.x models expose ``isError``; mcp 2.x renamed it ``is_error``."""

    def test_v1_camel_case_attributes(self) -> None:
        raw = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="boom")],
            isError=True,
            structuredContent={"code": "X"},
        )
        result = fastmcp._convert_call_result(raw)
        assert result.is_error is True
        assert result.structured_content == {"code": "X"}

    def test_v2_snake_case_attributes(self) -> None:
        raw = SimpleNamespace(
            content=[SimpleNamespace(type="image", data="", mime_type="image/gif")],
            is_error=True,
            structured_content={"code": "Y"},
        )
        result = fastmcp._convert_call_result(raw, embed_images=False)
        assert result.is_error is True
        assert result.structured_content == {"code": "Y"}
        assert result.text == "[image: image/gif]"

    def test_missing_on_both_uses_default(self) -> None:
        assert fastmcp._mcp_field(object(), "inputSchema", "input_schema", default={}) == {}
