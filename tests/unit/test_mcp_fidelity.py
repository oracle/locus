# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""MCP client fidelity and resilience, against a real MCP server on loopback.

Every test here talks to an actual ``mcp`` server (``MCPServer`` on mcp 2.x,
``FastMCP`` on 1.x) over streamable HTTP
on 127.0.0.1 — no mocks of the SDK — because the defects these pin down lived
in the interaction with the real transport:

* ``call_tool`` kept only text: ``structuredContent``, ``isError``, images and
  the tool's ``outputSchema`` were dropped on the floor.
* progress notifications were never requested, so never surfaced.
* auth was one static bearer fixed at connect; no custom or per-run headers.
* every server tool was attached, with no allowlist.
* a server down at attach killed the whole run with ``CancelledError`` (an
  anyio cancel scope entered in the run's task), a server killed mid-run
  raised ``CancelledError`` out of the tool call, and never-closed clients
  logged "asynchronous generator" errors at loop shutdown.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests._mcp_deps import require_server_deps


uvicorn = require_server_deps()

from tests._mcp_loopback_server import PNG_1PX, build_server  # noqa: E402
from tulip.agent import Agent  # noqa: E402
from tulip.core.events import (  # noqa: E402
    TerminateEvent,
    ToolCompleteEvent,
    ToolProgressEvent,
    ToolStartEvent,
)
from tulip.integrations.fastmcp import (  # noqa: E402
    MCPClient,
    MCPConnectionError,
    MCPRequestContext,
    MCPToolNotAllowedError,
)
from tulip.testing import ScriptedModel, text, tool_call  # noqa: E402


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _LoopbackServer:
    """A FastMCP server on a daemon thread, stoppable mid-test."""

    def __init__(self, port: int | None = None) -> None:
        self.port = port or _free_port()
        self.url = f"http://127.0.0.1:{self.port}/mcp"
        self._server: Any = None
        self._thread: threading.Thread | None = None

    def start(self) -> _LoopbackServer:
        app = build_server(self.port).streamable_http_app()
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="critical",
            lifespan="on",
            timeout_graceful_shutdown=0,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:  # pragma: no cover - environment failure
                raise RuntimeError("loopback MCP server did not start")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.should_exit = True
        self._server.force_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None


class _ProcessServer:
    """The same server in a child process, for tests that kill it."""

    def __init__(self, port: int | None = None) -> None:
        self.port = port or _free_port()
        self.url = f"http://127.0.0.1:{self.port}/mcp"
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self) -> _ProcessServer:
        script = Path(__file__).parents[1] / "_mcp_loopback_server.py"
        self._proc = subprocess.Popen(  # noqa: S603 — our own script, fixed argv
            [sys.executable, str(script), str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 30
        while True:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return self
            except OSError:
                if time.monotonic() > deadline:  # pragma: no cover - environment failure
                    self.stop()
                    raise RuntimeError("MCP server process did not start") from None
                time.sleep(0.05)

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait(timeout=10)
            self._proc = None


@pytest.fixture(scope="module")
def server() -> Iterator[_LoopbackServer]:
    srv = _LoopbackServer().start()
    yield srv
    srv.stop()


@pytest.fixture
def own_server() -> Iterator[_ProcessServer]:
    """A server this test may kill."""
    srv = _ProcessServer().start()
    yield srv
    srv.stop()


def _client(url: str, **kwargs: Any) -> MCPClient:
    return MCPClient(base_url=url, verify_url=False, **kwargs)


def _agent(model: ScriptedModel, *clients: MCPClient) -> Agent:
    return Agent(model=model, mcp_servers=list(clients), reflexion=False, grounding=False)


async def _events(agent: Agent, prompt: str = "go", **kwargs: Any) -> list[Any]:
    return [event async for event in agent.run(prompt, **kwargs)]


def _completes(events: list[Any]) -> dict[str, ToolCompleteEvent]:
    return {e.tool_call_id: e for e in events if isinstance(e, ToolCompleteEvent)}


# ---------------------------------------------------------------------------
# 1. Result fidelity
# ---------------------------------------------------------------------------


async def test_structured_content_reaches_the_complete_event(server: _LoopbackServer) -> None:
    model = ScriptedModel([tool_call("explicit_both", call_id="c1", city="Rome"), text("ok")])
    events = await _events(_agent(model, _client(server.url)))

    done = _completes(events)["c1"]
    assert done.result == "Found 1 hotel in Rome"  # the model's view is unchanged
    assert done.structured_content == {"widget": "hotel_list", "items": [{"id": "h9"}]}
    # ...and the structured payload is not what the model was shown.
    tool_msg = next(m for m in model.received_messages[1] if m.tool_call_id == "c1")
    assert tool_msg.content == "Found 1 hotel in Rome"


async def test_is_error_becomes_a_tool_error_the_model_is_told_about(
    server: _LoopbackServer,
) -> None:
    model = ScriptedModel([tool_call("explicit_error", call_id="c2"), text("sorry")])
    events = await _events(_agent(model, _client(server.url)))

    done = _completes(events)["c2"]
    assert done.error == "rate not available"
    assert done.result is None
    assert not done.success
    assert done.structured_content == {"code": "NO_RATE"}
    tool_msg = next(m for m in model.received_messages[1] if m.tool_call_id == "c2")
    assert tool_msg.content == "Error: rate not available"


async def test_images_are_embedded_and_passed_through_not_dropped(
    server: _LoopbackServer,
) -> None:
    from tulip.core.media import images

    model = ScriptedModel([tool_call("photo", call_id="c3"), text("nice")])
    events = await _events(_agent(model, _client(server.url)))

    done = _completes(events)["c3"]
    assert done.result is not None
    assert done.result.startswith("the lobby")
    [image] = images(done.result)  # adapters that can show images will
    assert image.media_type == "image/png"
    assert done.content_blocks == [{"type": "image", "data": PNG_1PX, "mimeType": "image/png"}]


async def test_output_schema_is_listed_and_kept_on_the_tool(server: _LoopbackServer) -> None:
    client = _client(server.url)
    await client.connect()
    try:
        schemas = {t["name"]: t for t in await client.list_tools()}
        assert schemas["search_hotels"]["outputSchema"]["type"] == "object"
        tools = {t.name: t for t in client.to_tulip_tools(list(schemas.values()))}
        assert tools["search_hotels"].output_schema == schemas["search_hotels"]["outputSchema"]
        assert tools["search_hotels"].emits_progress is True
    finally:
        await client.close()


async def test_call_tool_result_and_back_compat_call_tool(server: _LoopbackServer) -> None:
    client = _client(server.url)
    await client.connect()
    try:
        rich = await client.call_tool_result("search_hotels", {"city": "Paris"})
        assert rich.structured_content == {
            "result": [{"id": "h1", "name": "Grand Paris", "price": 420.0}]
        }
        assert rich.is_error is False
        # The string API is unchanged: text only.
        plain = await client.call_tool("explicit_both", {"city": "Rome"})
        assert plain == "Found 1 hotel in Rome"
        assert type(plain) is str
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# 2. Progress
# ---------------------------------------------------------------------------


async def test_progress_notifications_stream_as_events(server: _LoopbackServer) -> None:
    model = ScriptedModel([tool_call("slow_progress", call_id="p1"), text("ok")])
    events = await _events(_agent(model, _client(server.url)))

    progress = [e for e in events if isinstance(e, ToolProgressEvent)]
    assert [(e.progress, e.total, e.message) for e in progress] == [
        (1.0, 3.0, "step 1"),
        (2.0, 3.0, "step 2"),
        (3.0, 3.0, "step 3"),
    ]
    assert {e.tool_call_id for e in progress} == {"p1"}
    assert {e.tool_name for e in progress} == {"slow_progress"}
    # Live: between the call's start and its completion.
    kinds = [
        type(e).__name__
        for e in events
        if isinstance(e, (ToolStartEvent, ToolProgressEvent, ToolCompleteEvent))
    ]
    assert kinds == ["ToolStartEvent", *["ToolProgressEvent"] * 3, "ToolCompleteEvent"]


async def test_progress_callback_on_the_client(server: _LoopbackServer) -> None:
    seen: list[tuple[float, float | None, str | None]] = []
    client = _client(server.url)
    await client.connect()
    try:
        result = await client.call_tool_result(
            "slow_progress", {}, progress_callback=lambda p, t, m: seen.append((p, t, m))
        )
    finally:
        await client.close()
    assert result.text == "done"
    assert [s[0] for s in seen] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# 3. Headers: static custom, per-run via metadata, per-run via provider
# ---------------------------------------------------------------------------


async def test_custom_static_headers_are_sent(server: _LoopbackServer) -> None:
    client = _client(server.url, headers={"X-Tenant": "acme"}, access_token="static-token")  # noqa: S106
    await client.connect()
    try:
        out = await client.call_tool("whoami", {})
    finally:
        await client.close()
    assert "tenant=acme" in out
    assert "auth=Bearer static-token" in out


async def test_per_run_headers_from_run_metadata(server: _LoopbackServer) -> None:
    """One agent config, two users, two tokens — never crossed."""
    client = _client(server.url)
    model = ScriptedModel(
        [
            tool_call("whoami", call_id="a"),
            text("ok"),
            tool_call("whoami", call_id="b"),
            text("ok"),
        ]
    )
    agent = _agent(model, client)
    alice = await _events(
        agent,
        metadata={"mcp_headers": {"Authorization": "Bearer tok-alice", "X-User-Id": "alice"}},
    )
    bob = await _events(
        agent,
        metadata={"mcp_headers": {"Authorization": "Bearer tok-bob", "X-User-Id": "bob"}},
    )
    try:
        assert _completes(alice)["a"].result == "auth=Bearer tok-alice user=alice tenant=<none>"
        assert _completes(bob)["b"].result == "auth=Bearer tok-bob user=bob tenant=<none>"
    finally:
        await client.close()


async def test_per_run_headers_from_a_provider(server: _LoopbackServer) -> None:
    seen: list[MCPRequestContext] = []

    async def provider(rctx: MCPRequestContext) -> dict[str, str]:
        seen.append(rctx)
        return {"Authorization": f"Bearer {rctx.metadata['user']}-token"}

    client = _client(server.url, headers_provider=provider, access_token="ignored-static")  # noqa: S106
    model = ScriptedModel([tool_call("whoami", call_id="w1"), text("ok")])
    events = await _events(_agent(model, client), metadata={"user": "carol"})
    try:
        assert (
            _completes(events)["w1"].result == "auth=Bearer carol-token user=<none> tenant=<none>"
        )
        # tools/list at attach and the tool call both saw the run metadata.
        assert seen[0].tool_name is None
        call_ctx = next(r for r in seen if r.tool_name == "whoami")
        assert call_ctx.tool_call_id == "w1"
        assert call_ctx.metadata["user"] == "carol"
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# 4. Tool filtering
# ---------------------------------------------------------------------------


async def test_allowed_tools_limits_what_is_attached_and_callable(
    server: _LoopbackServer,
) -> None:
    client = _client(server.url, allowed_tools=["whoami", "explicit_both"])
    model = ScriptedModel(["ok"])
    await _events(_agent(model, client))
    try:
        assert sorted(model.offered_tools[0]) == ["explicit_both", "whoami"]
        with pytest.raises(MCPToolNotAllowedError):
            await client.call_tool("explicit_error", {})
    finally:
        await client.close()


async def test_tool_filter_predicate(server: _LoopbackServer) -> None:
    client = _client(server.url, tool_filter=lambda schema: "outputSchema" in schema)
    await client.connect()
    try:
        names = {t["name"] for t in await client.list_tools()}
    finally:
        await client.close()
    assert "search_hotels" in names  # typed return → declares outputSchema
    assert "explicit_both" not in names  # returns CallToolResult → no outputSchema


# ---------------------------------------------------------------------------
# 5. Resilience
# ---------------------------------------------------------------------------


async def test_a_server_down_at_attach_does_not_cancel_the_run() -> None:
    dead = _client(f"http://127.0.0.1:{_free_port()}/mcp", reconnect_interval=0)
    model = ScriptedModel(["answered without tools"])
    events = await _events(_agent(model, dead))

    terminate = [e for e in events if isinstance(e, TerminateEvent)][-1]
    assert terminate.final_message == "answered without tools"
    assert model.offered_tools[0] == []


async def test_tools_attach_on_a_later_run_once_the_server_is_up() -> None:
    port = _free_port()
    client = _client(f"http://127.0.0.1:{port}/mcp", reconnect_interval=0)
    model = ScriptedModel(["first", "second"])
    agent = _agent(model, client)
    await _events(agent)
    assert model.offered_tools[0] == []

    srv = _LoopbackServer(port).start()
    try:
        await _events(agent)
        assert "whoami" in model.offered_tools[1]
    finally:
        await client.close()
        srv.stop()


async def test_a_server_killed_mid_run_is_a_tool_error_and_the_run_continues(
    own_server: _ProcessServer,
) -> None:
    client = _client(own_server.url)
    agent = _agent(ScriptedModel(["warm"]), client)
    await _events(agent)  # attached and connected

    own_server.stop()
    agent._model = ScriptedModel([tool_call("whoami", call_id="k1"), text("carried on")])
    events = await _events(agent)

    done = _completes(events)["k1"]
    assert done.error is not None
    assert "MCPConnectionError" in done.error
    terminate = [e for e in events if isinstance(e, TerminateEvent)][-1]
    assert terminate.final_message == "carried on"


async def test_a_call_in_flight_when_the_server_dies_fails_fast(
    own_server: _ProcessServer,
) -> None:
    client = _client(own_server.url, liveness_interval=0.5)
    await client.connect()
    call = asyncio.ensure_future(client.call_tool_result("sleepy", {"seconds": 30}))
    await asyncio.sleep(0.3)
    own_server.stop()
    with pytest.raises(MCPConnectionError):
        await asyncio.wait_for(call, timeout=15)
    await client.close()


async def test_the_next_call_reconnects_after_a_restart(own_server: _ProcessServer) -> None:
    client = _client(own_server.url)
    await client.connect()
    port = own_server.port
    own_server.stop()
    with pytest.raises(MCPConnectionError):
        await client.call_tool("whoami", {})
    restarted = _ProcessServer(port).start()
    try:
        assert (await client.call_tool("whoami", {})).startswith("auth=")
    finally:
        await client.close()
        restarted.stop()


async def test_cancelling_the_run_still_cancels_it(server: _LoopbackServer) -> None:
    """Converting leaked cancellations must not swallow a real one."""
    client = _client(server.url)
    model = ScriptedModel([tool_call("sleepy", call_id="s1", seconds=30), text("never")])
    agent = _agent(model, client)
    started = asyncio.Event()

    async def consume() -> None:
        async for event in agent.run("go"):
            if isinstance(event, ToolStartEvent):
                started.set()

    task = asyncio.ensure_future(consume())
    await asyncio.wait_for(started.wait(), timeout=10)
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await client.close()


def test_unclosed_clients_leave_no_asyncgen_errors_at_shutdown(server: _LoopbackServer) -> None:
    """Never-closed clients (live and dead) used to log cross-task cancel-scope
    errors when the loop finalized their transports."""
    errors: list[dict[str, Any]] = []

    async def main() -> None:
        asyncio.get_running_loop().set_exception_handler(lambda _loop, ctx: errors.append(ctx))
        live = _client(server.url)
        dead = _client(f"http://127.0.0.1:{_free_port()}/mcp")
        model = ScriptedModel([tool_call("whoami"), text("ok")])
        await _agent(model, live, dead).arun("hi")

    asyncio.run(main())
    assert [e.get("message") for e in errors] == []
