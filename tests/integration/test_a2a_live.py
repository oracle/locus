# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end integration test for the A2A spec transport.

Spins up an :class:`A2AServer` on a real loopback uvicorn process and
exercises the spec wire surface with :class:`A2AClient` over real HTTP
+ SSE. The unit tests in ``tests/unit/test_a2a_protocol.py`` cover the
same behaviour through FastAPI's in-process test client; this file
rounds it out by proving the **wire encoding** (JSON-RPC envelopes,
SSE framing, well-known URL routing, bearer auth headers) actually
works on a live socket.

Skipped in default CI runs unless ``RUN_A2A_INTEGRATION=1`` is set —
real-port binding is flaky in shared runner environments.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest

from locus.a2a import (
    A2AClient,
    A2AServer,
    AgentSkill,
    Message,
    TaskState,
    TextPart,
)


# Skip this whole module unless the integration flag is set. The flag
# is present on the dedicated integration runner; default unit-test
# runs (CI on every PR) skip it because real port binding is racy.
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_A2A_INTEGRATION") != "1",
    reason="set RUN_A2A_INTEGRATION=1 to run the live A2A integration test",
)


class _StubAgent:
    """Synthetic agent that yields a pre-canned event stream."""

    def __init__(self, reply: str = "answer", final_reason: str = "complete") -> None:
        self._reply = reply
        self._final_reason = final_reason

    async def run(self, prompt: str) -> AsyncIterator[Any]:
        from locus.core.events import TerminateEvent, ThinkEvent

        yield ThinkEvent(iteration=0, reasoning=f"reasoning about {prompt!r}", tool_calls=[])
        yield TerminateEvent(
            reason=self._final_reason,
            iterations_used=1,
            final_confidence=1.0,
            total_tool_calls=0,
            final_message=self._reply,
        )


def _free_port() -> int:
    """Bind an ephemeral port, close it, return the number.

    There's a TOCTOU window — but it's the only portable way to get a
    free port for a child uvicorn. The integration runner is single
    user so the race rarely loses.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def live_server() -> Iterator[tuple[str, str]]:
    """Boot an A2AServer on a real loopback port and yield (url, api_key)."""
    import uvicorn

    api_key = "integration-test-secret"
    port = _free_port()

    server = A2AServer(
        agent=_StubAgent(reply="quantum is fast"),
        name="research",
        description="answers research questions",
        url=f"http://127.0.0.1:{port}",
        skills=[
            AgentSkill(
                id="research",
                name="Research",
                description="Look stuff up.",
                tags=["search"],
            ),
        ],
        api_key=api_key,
    )

    config = uvicorn.Config(
        app=server.app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    uv = uvicorn.Server(config)

    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()

    # Wait until uvicorn is accepting connections.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not uv.started:
        time.sleep(0.05)
    assert uv.started, "uvicorn did not start within the deadline"

    try:
        yield f"http://127.0.0.1:{port}", api_key
    finally:
        uv.should_exit = True
        thread.join(timeout=5.0)


def _msg(text: str, *, mid: str = "m-1") -> Message:
    return Message(role="user", parts=[TextPart(text=text)], messageId=mid)


# ---------------------------------------------------------------------------
# Spec wire surface — well-known card, JSON-RPC, lifecycle, SSE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_well_known_agent_card_over_real_http(live_server: tuple[str, str]) -> None:
    url, api_key = live_server
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{url}/.well-known/agent-card.json",
            headers={"Authorization": f"Bearer {api_key}"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "research"
    assert body["url"] == url
    # Spec capabilities object present.
    assert body["capabilities"]["streaming"] is True
    # Skills are typed objects, not strings.
    assert isinstance(body["skills"][0], dict)
    assert body["skills"][0]["id"] == "research"


@pytest.mark.asyncio
async def test_message_send_round_trip(live_server: tuple[str, str]) -> None:
    url, api_key = live_server
    a2a = A2AClient(url=url, api_key=api_key)
    task = await a2a.send_message(_msg("what is quantum computing?"))
    assert task.status.state == TaskState.completed
    assert task.artifacts, "completed task should ship a reply artifact"
    artifact_text = task.artifacts[-1].parts[0]
    # discriminated union — .text only on TextPart
    assert getattr(artifact_text, "text", None) == "quantum is fast"


@pytest.mark.asyncio
async def test_tasks_get_then_cancel_terminal(live_server: tuple[str, str]) -> None:
    url, api_key = live_server
    a2a = A2AClient(url=url, api_key=api_key)
    task = await a2a.send_message(_msg("hi"))
    refetched = await a2a.get_task(task.id)
    assert refetched.id == task.id
    assert refetched.status.state == TaskState.completed
    # Already terminal — cancel must surface TaskNotCancelable (-32002).
    with pytest.raises(RuntimeError, match="-32002"):
        await a2a.cancel_task(task.id)


@pytest.mark.asyncio
async def test_message_stream_emits_lifecycle(live_server: tuple[str, str]) -> None:
    url, api_key = live_server
    a2a = A2AClient(url=url, api_key=api_key)
    kinds: list[str] = []
    async for event in a2a.send_message_streaming(_msg("stream me")):
        kinds.append(event.get("kind") or event.get("status", {}).get("state") or "?")
        # Cap iterations defensively in case the stream loops.
        if len(kinds) > 20:
            break
    # Initial Task envelope (kind=task) → working status → artifact-update → final status-update.
    assert "task" in kinds
    assert "status-update" in kinds
    assert "artifact-update" in kinds


# ---------------------------------------------------------------------------
# Auth + backwards-compat surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bearer_required_on_well_known_card(live_server: tuple[str, str]) -> None:
    url, _ = live_server
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{url}/.well-known/agent-card.json")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_legacy_invoke_still_works(live_server: tuple[str, str]) -> None:
    url, api_key = live_server
    a2a = A2AClient(url=url, api_key=api_key)
    text = await a2a.invoke("legacy ping")
    assert text == "quantum is fast"


@pytest.mark.asyncio
async def test_get_agent_card_falls_back_to_legacy(live_server: tuple[str, str]) -> None:
    """Sanity-check the well-known → /agent-card fallback path even on the
    spec-compliant server (which serves both)."""
    url, api_key = live_server
    a2a = A2AClient(url=url, api_key=api_key)
    card = await a2a.get_agent_card()
    assert card.name == "research"
    assert card.skills[0].id == "research"


# Smoke: smoothly construct a client without the key (loopback peer).
def test_client_construct_without_key() -> None:
    A2AClient(url="http://127.0.0.1:1")


# Smoke: confirm asyncio event-loop hygiene — multiple sequential
# send_message calls reuse a fresh httpx client each time.
@pytest.mark.asyncio
async def test_two_sequential_sends(live_server: tuple[str, str]) -> None:
    url, api_key = live_server
    a2a = A2AClient(url=url, api_key=api_key)
    t1 = await a2a.send_message(_msg("first", mid="m-1"))
    t2 = await a2a.send_message(_msg("second", mid="m-2"))
    assert t1.id != t2.id
    assert t1.status.state == TaskState.completed
    assert t2.status.state == TaskState.completed


# Smoke: confirm asyncio.run() on the as_tool() wrapper still works.
# Sync test on purpose — ``as_tool`` calls ``asyncio.run`` internally,
# which only works when called outside a running event loop.
def test_as_tool_invokes_remote_real_http(live_server: tuple[str, str]) -> None:
    url, api_key = live_server
    a2a = A2AClient(url=url, api_key=api_key)
    tool = a2a.as_tool(name="research_remote", description="ask the research agent")
    result = tool.fn("what is AI?")
    assert result == "quantum is fast"
