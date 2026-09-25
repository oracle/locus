# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-run MCP headers: never persisted, typed failures, bounded sessions.

* ``metadata={"mcp_headers": ...}`` carries a bearer token; it must reach the
  MCP server and nothing else — not the checkpoint, not the event stream.
* Opening a per-identity session to a server that is down is an
  :class:`MCPConnectionError` (a typed tool error), not a raw transport error.
* Per-identity sessions can be keyed by principal (``session_key``), are closed
  after ``session_idle_ttl`` and capped by ``max_sessions``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests._mcp_deps import require_server_deps


require_server_deps()

from tests.unit.test_mcp_fidelity import (  # noqa: E402
    _free_port,
    _LoopbackServer,
    _ProcessServer,
)
from tulip.agent import Agent  # noqa: E402
from tulip.core.events import ToolCompleteEvent  # noqa: E402
from tulip.integrations.fastmcp import (  # noqa: E402
    MCPClient,
    MCPConnectionError,
    MCPRequestContext,
)
from tulip.memory.backends.file import FileCheckpointer  # noqa: E402
from tulip.testing import ScriptedModel, text, tool_call  # noqa: E402


SECRET = "tok-7f3a9c-DO-NOT-PERSIST"  # noqa: S105 — a test marker, not a credential


def _digest(token: str) -> str:
    return hashlib.sha256(f"Bearer {token}".encode()).hexdigest()[:16]


@pytest.fixture(scope="module")
def server() -> Iterator[_LoopbackServer]:
    srv = _LoopbackServer().start()
    yield srv
    srv.stop()


def _client(url: str, **kwargs: Any) -> MCPClient:
    return MCPClient(base_url=url, verify_url=False, **kwargs)


def _agent(model: ScriptedModel, client: MCPClient, **kwargs: Any) -> Agent:
    return Agent(model=model, mcp_servers=[client], reflexion=False, grounding=False, **kwargs)


def _completes(events: list[Any]) -> dict[str, ToolCompleteEvent]:
    return {e.tool_call_id: e for e in events if isinstance(e, ToolCompleteEvent)}


def _dir_bytes(root: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file())


# ---------------------------------------------------------------------------
# 1. The per-run bearer token is never persisted
# ---------------------------------------------------------------------------


async def test_mcp_headers_reach_the_server_but_never_the_checkpoint_or_events(
    server: _LoopbackServer, tmp_path: Path
) -> None:
    client = _client(server.url)
    checkpointer = FileCheckpointer(tmp_path / "cp")
    model = ScriptedModel(
        [
            tool_call("auth_digest", call_id="d1"),
            text("first turn"),
            tool_call("auth_digest", call_id="d2"),
            text("second turn"),
        ]
    )
    agent = _agent(model, client, checkpointer=checkpointer)
    metadata = {"mcp_headers": {"Authorization": f"Bearer {SECRET}"}, "user": "alice"}
    events: list[Any] = []
    try:
        for _ in range(2):  # a new turn on the checkpointed thread, too
            async for event in agent.run("go", thread_id="t-1", metadata=dict(metadata)):
                events.append(event)
    finally:
        await client.close()

    # The token did its job: the server saw it on both turns.
    done = _completes(events)
    assert done["d1"].result == _digest(SECRET)
    assert done["d2"].result == _digest(SECRET)

    # ...and it is nowhere at rest.
    saved = _dir_bytes(tmp_path / "cp")
    assert saved, "the checkpointer wrote nothing"
    assert SECRET.encode() not in saved
    assert b"mcp_headers" not in saved
    # Non-secret metadata is still persisted as before.
    loaded = await checkpointer.load("t-1")
    assert loaded is not None
    assert loaded.metadata.get("user") == "alice"

    for event in events:
        assert SECRET not in event.model_dump_json()

    # The caller's dict is not mutated.
    assert metadata["mcp_headers"] == {"Authorization": f"Bearer {SECRET}"}


async def test_a_custom_metadata_headers_key_is_scrubbed_too(
    server: _LoopbackServer, tmp_path: Path
) -> None:
    client = _client(server.url, metadata_headers_key="upstream_auth")
    checkpointer = FileCheckpointer(tmp_path / "cp")
    model = ScriptedModel([tool_call("auth_digest", call_id="d1"), text("ok")])
    agent = _agent(model, client, checkpointer=checkpointer)
    try:
        events = [
            e
            async for e in agent.run(
                "go",
                thread_id="t-2",
                metadata={"upstream_auth": {"Authorization": f"Bearer {SECRET}"}},
            )
        ]
    finally:
        await client.close()
    assert _completes(events)["d1"].result == _digest(SECRET)
    assert SECRET.encode() not in _dir_bytes(tmp_path / "cp")


# ---------------------------------------------------------------------------
# 2. A per-identity session to a down server is a typed error
# ---------------------------------------------------------------------------


async def test_per_identity_session_to_a_down_server_raises_mcp_connection_error() -> None:
    client = _client(
        f"http://127.0.0.1:{_free_port()}/mcp",
        headers_provider=lambda _rctx: {"Authorization": "Bearer someone"},
        connect_timeout=5,
    )
    with pytest.raises(MCPConnectionError):
        await client.call_tool_result("whoami", {})
    with pytest.raises(MCPConnectionError):
        await client.list_tools()
    await client.close()


async def test_a_new_identity_on_a_dead_server_is_a_tool_error_in_the_run() -> None:
    srv = _ProcessServer().start()
    client = _client(srv.url, connect_timeout=5)
    try:
        agent = _agent(ScriptedModel(["warm"]), client)
        await asyncio.wait_for(
            _drain(agent, metadata={"mcp_headers": {"Authorization": "Bearer a"}}), 30
        )  # tools attached
        srv.stop()
        agent._model = ScriptedModel([tool_call("whoami", call_id="w1"), text("carried on")])
        events = await asyncio.wait_for(
            _drain(agent, metadata={"mcp_headers": {"Authorization": "Bearer b"}}), 30
        )
    finally:
        srv.stop()
        await client.close()
    error = _completes(events)["w1"].error
    assert error is not None
    assert "MCPConnectionError" in error


async def _drain(agent: Agent, **kwargs: Any) -> list[Any]:
    return [e async for e in agent.run("go", **kwargs)]


# ---------------------------------------------------------------------------
# 5. Session keying, idle TTL, cap and accounting
# ---------------------------------------------------------------------------


def _principal(_rctx: MCPRequestContext, headers: dict[str, str]) -> str:
    return headers["X-User-Id"]


async def test_rotating_tokens_for_one_principal_share_one_session(
    server: _LoopbackServer,
) -> None:
    client = _client(server.url, session_key=_principal)
    turns = 50
    script: list[Any] = []
    for i in range(turns):
        script += [tool_call("auth_digest", call_id=f"c{i}"), text(f"turn {i}")]
    agent = _agent(ScriptedModel(script), client)
    try:
        for i in range(turns):
            events = await _drain(
                agent,
                metadata={
                    "mcp_headers": {"Authorization": f"Bearer jwt-{i}", "X-User-Id": "alice"}
                },
            )
            # Every call carried THIS turn's token, not the one the session
            # was opened with.
            assert _completes(events)[f"c{i}"].result == _digest(f"jwt-{i}")
            assert len(client._identity_sessions) <= 1
        stats = client.session_stats
        assert stats["opened"] == 1
        assert stats["live"] == 1
    finally:
        await client.close()
    assert client.session_stats["live"] == 0


async def test_default_keying_still_isolates_identities(server: _LoopbackServer) -> None:
    client = _client(server.url)
    try:
        for token in ("a", "b", "a"):
            client.headers_provider = lambda _r, t=token: {"Authorization": f"Bearer {t}"}
            assert (await client.call_tool("auth_digest", {})) == _digest(token)
        assert client.session_stats["opened"] == 2
    finally:
        await client.close()


async def test_idle_sessions_are_closed_after_the_ttl(server: _LoopbackServer) -> None:
    client = _client(server.url, session_idle_ttl=0.3)
    client.headers_provider = lambda _r: {"Authorization": "Bearer idle"}
    try:
        await client.call_tool("auth_digest", {})
        [handle] = client._identity_sessions.values()
        assert handle.alive
        deadline = asyncio.get_running_loop().time() + 5
        while client._identity_sessions and asyncio.get_running_loop().time() < deadline:  # noqa: ASYNC110 — polling a background sweep
            await asyncio.sleep(0.05)
        assert not client._identity_sessions
        assert not handle.alive
        assert client.session_stats["closed"] == 1
        # A later request simply opens a fresh session.
        assert (await client.call_tool("auth_digest", {})) == _digest("idle")
    finally:
        await client.close()


async def test_max_sessions_closes_the_least_recently_used(server: _LoopbackServer) -> None:
    client = _client(server.url, max_sessions=2)
    try:
        handles = []
        for token in ("a", "b", "c"):
            client.headers_provider = lambda _r, t=token: {"Authorization": f"Bearer {t}"}
            await client.call_tool("auth_digest", {})
            handles.append(list(client._identity_sessions.values())[-1])
        assert len(client._identity_sessions) == 2
        assert not handles[0].alive
        assert handles[1].alive
        assert handles[2].alive
        assert client.session_stats == {"opened": 3, "closed": 1, "live": 2}
    finally:
        await client.close()


async def test_session_open_and_close_are_logged(
    server: _LoopbackServer, caplog: pytest.LogCaptureFixture
) -> None:
    client = _client(server.url, session_key=_principal)
    client.headers_provider = lambda _r: {"Authorization": f"Bearer {SECRET}", "X-User-Id": "u"}
    with caplog.at_level("INFO", logger="tulip.integrations.fastmcp"):
        await client.call_tool("auth_digest", {})
        await client.close()
    messages = [r.getMessage() for r in caplog.records]
    assert any("session opened" in m for m in messages)
    assert any("session closed" in m for m in messages)
    assert not any(SECRET in m for m in messages)
    assert json.dumps(client.session_stats)  # plain, exportable numbers
