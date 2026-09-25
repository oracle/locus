# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-identity MCP session pool, without a real server.

The loopback tests (``test_mcp_secrets_and_sessions.py``) need the mcp 1.x
``FastMCP`` server and are skipped where mcp 2.x is installed; these drive
the same pool logic — keying, rotation, idle TTL, the LRU cap, accounting and
typed connect errors — against stub sessions, so they run everywhere.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pytest

from tulip.integrations.fastmcp import (
    MCPClient,
    MCPConnectionError,
    MCPRequestContext,
    _SessionHandle,
    _TypedConnectErrors,
)
from tulip.tools.context import ToolContext, bind_tool_context


class _StubSession:
    def __init__(self, handle: _SessionHandle) -> None:
        self.handle = handle
        self.calls: list[dict[str, str]] = []

    async def call_tool(self, name: str, arguments: Any = None, **kw: Any) -> Any:
        self.calls.append(dict(self.handle.headers))
        from types import SimpleNamespace

        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok", model_dump=lambda **_: {})],
            isError=False,
            structuredContent=None,
            meta=None,
        )


class _Pool(MCPClient):
    """An MCPClient whose sessions are parked stub tasks, not HTTP connections."""

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        object.__setattr__(self, "opened_handles", [])
        object.__setattr__(self, "fail_with", None)

    async def _connect_http(
        self, extra_headers: Any = None, *, install: bool = True
    ) -> _SessionHandle:
        if self.fail_with is not None:  # type: ignore[attr-defined]
            raise self.fail_with  # type: ignore[attr-defined]
        handle = _SessionHandle(self.label, dict(extra_headers or {}))
        stop = asyncio.Event()
        handle._stop = stop
        handle.task = asyncio.get_running_loop().create_task(stop.wait())
        handle.session = _StubSession(handle)
        self.opened_handles.append(handle)  # type: ignore[attr-defined]
        return handle


def _pool(**kwargs: Any) -> _Pool:
    return _Pool(base_url="http://mcp.invalid/mcp", verify_url=False, **kwargs)


def _as(client: MCPClient, headers: dict[str, str]) -> None:
    client.headers_provider = lambda _rctx: headers


async def test_session_key_shares_one_session_across_rotating_tokens() -> None:
    client = _pool(session_key=lambda _r, h: h["X-User"])
    try:
        for i in range(50):
            _as(client, {"Authorization": f"Bearer t{i}", "X-User": "alice"})
            await client.call_tool("t", {})
        [handle] = client.opened_handles  # type: ignore[attr-defined]
        # Each call went out with that turn's headers.
        assert handle.session.calls[0]["Authorization"] == "Bearer t0"
        assert handle.session.calls[-1]["Authorization"] == "Bearer t49"
        assert client.session_stats == {"opened": 1, "closed": 0, "live": 1}
    finally:
        await client.close()
    assert client.session_stats == {"opened": 1, "closed": 1, "live": 0}
    assert not handle.alive


async def test_async_session_key_and_none_falls_back_to_headers() -> None:
    async def key(_rctx: MCPRequestContext, headers: Any) -> str | None:
        return headers.get("X-User")

    client = _pool(session_key=key)
    try:
        _as(client, {"Authorization": "Bearer a"})  # no principal → full headers
        await client.call_tool("t", {})
        _as(client, {"Authorization": "Bearer b"})
        await client.call_tool("t", {})
        assert client.session_stats["opened"] == 2
    finally:
        await client.close()


async def test_idle_sessions_close_after_the_ttl() -> None:
    client = _pool(session_idle_ttl=0.1)
    _as(client, {"Authorization": "Bearer idle"})
    try:
        await client.call_tool("t", {})
        [handle] = client.opened_handles  # type: ignore[attr-defined]
        await asyncio.wait_for(client._sweeper, 5)  # type: ignore[arg-type]
        assert not handle.alive
        assert client.session_stats == {"opened": 1, "closed": 1, "live": 0}
    finally:
        await client.close()


async def test_a_dead_session_is_replaced_and_counted() -> None:
    client = _pool(session_idle_ttl=None)
    _as(client, {"Authorization": "Bearer x"})
    try:
        await client.call_tool("t", {})
        [first] = client.opened_handles  # type: ignore[attr-defined]
        await first.close()  # the connection died
        await client.call_tool("t", {})
        assert len(client.opened_handles) == 2  # type: ignore[attr-defined]
        assert client.session_stats == {"opened": 2, "closed": 1, "live": 1}
    finally:
        await client.close()


async def test_evict_idle_drops_dead_sessions() -> None:
    client = _pool(session_idle_ttl=60)
    _as(client, {"Authorization": "Bearer x"})
    try:
        await client.call_tool("t", {})
        [handle] = client.opened_handles  # type: ignore[attr-defined]
        await handle.close()
        await client._evict_idle()
        assert client.session_stats["live"] == 0
    finally:
        await client.close()


async def test_max_sessions_closes_the_least_recently_used_idle_session() -> None:
    client = _pool(max_sessions=2, session_idle_ttl=None)
    try:
        for token in ("a", "b", "c"):
            _as(client, {"Authorization": f"Bearer {token}"})
            await client.call_tool("t", {})
        a, b, c = client.opened_handles  # type: ignore[attr-defined]
        assert not a.alive
        assert b.alive
        assert c.alive
        assert client.session_stats == {"opened": 3, "closed": 1, "live": 2}

        # A busy LRU session is spared in favour of an idle one.
        b.in_flight = 1
        _as(client, {"Authorization": "Bearer d"})
        await client.call_tool("t", {})
        assert b.alive
        assert not c.alive
        b.in_flight = 0
    finally:
        await client.close()


async def test_open_and_close_are_logged_without_headers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _pool()
    _as(client, {"Authorization": "Bearer s3cr3t-value"})
    with caplog.at_level(logging.INFO, logger="tulip.integrations.fastmcp"):
        await client.call_tool("t", {})
        await client.close()
    assert "session opened" in caplog.text
    assert "session closed" in caplog.text
    assert "s3cr3t-value" not in caplog.text


async def test_opening_an_identity_session_to_a_down_server_is_typed() -> None:
    client = _pool()
    object.__setattr__(client, "fail_with", httpx.ConnectError("All connection attempts failed"))
    _as(client, {"Authorization": "Bearer x"})
    with pytest.raises(MCPConnectionError, match="unreachable"):
        await client.call_tool("t", {})
    assert client.session_stats == {"opened": 0, "closed": 0, "live": 0}


async def test_non_connection_errors_on_open_pass_through() -> None:
    client = _pool()
    object.__setattr__(client, "fail_with", ValueError("bad config"))
    _as(client, {"Authorization": "Bearer x"})
    with pytest.raises(ValueError, match="bad config"):
        await client.call_tool("t", {})


async def test_reconnecting_the_default_session_to_a_down_server_is_typed() -> None:
    client = _pool()
    client._ever_connected = True  # connected once; the session has since gone

    async def down() -> None:
        raise httpx.ConnectError("refused")

    object.__setattr__(client, "connect", down)
    with pytest.raises(MCPConnectionError):
        await client.call_tool("t", {})


async def test_typed_connect_errors_leaves_other_outcomes_alone() -> None:
    guard = _TypedConnectErrors("srv")
    async with guard:
        pass
    with pytest.raises(MCPConnectionError, match="already typed"):
        async with guard:
            raise MCPConnectionError("already typed")
    with pytest.raises(asyncio.CancelledError):
        async with guard:
            raise asyncio.CancelledError
    with pytest.raises(MCPConnectionError, match="ConnectionRefusedError"):
        async with guard:
            raise ConnectionRefusedError("nope")


async def test_ephemeral_run_metadata_reaches_the_headers() -> None:
    client = _pool()
    ctx = ToolContext(
        tool_call_id="c1",
        tool_name="t",
        run_id="r",
        iteration=0,
        invocation_metadata={"user": "u"},
        ephemeral_metadata={"mcp_headers": {"Authorization": "Bearer eph"}},
    )
    try:
        with bind_tool_context(ctx):
            await client.call_tool("t", {})
        [handle] = client.opened_handles  # type: ignore[attr-defined]
        assert handle.session.calls == [{"Authorization": "Bearer eph"}]
        assert ctx.get_metadata("mcp_headers") == {"Authorization": "Bearer eph"}
        assert ctx.get_metadata("user") == "u"
        assert "eph" not in repr(ctx)
    finally:
        await client.close()
