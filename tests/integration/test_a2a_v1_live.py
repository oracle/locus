# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end integration tests for the A2A v1.0 protocol surface.

Companion to ``test_a2a_live.py`` (which covers the legacy / v0.x wire).
Boots an :class:`A2AServer` on a real loopback uvicorn and drives the
v1.0 JSON-RPC methods (``SendMessage``, ``GetTask``, ``ListTasks``,
``CancelTask``, ``SendStreamingMessage``, ``SubscribeToTask``) using
raw ``httpx`` so we assert the actual wire shape — not what an
in-process FastAPI test client would let through.

Skipped in default CI unless ``RUN_A2A_INTEGRATION=1`` is set, matching
the policy of the sibling file (real port binding is racy on shared
runners).
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest

from locus.a2a import (
    A2A_V1_PROTOCOL_VERSION,
    A2AClient,
    A2AServer,
    AgentSkill,
    TaskState,
)
from locus.a2a.spec import (
    CONTENT_TYPE_NOT_SUPPORTED,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PUSH_NOTIFICATION_NOT_SUPPORTED,
    TASK_NOT_CANCELABLE,
    TASK_NOT_FOUND,
    UNSUPPORTED_OPERATION,
    VERSION_NOT_SUPPORTED,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_A2A_INTEGRATION") != "1",
    reason="set RUN_A2A_INTEGRATION=1 to run the live A2A v1 integration tests",
)


# ---------------------------------------------------------------------------
# Server fixture
# ---------------------------------------------------------------------------


class _StubAgent:
    """Deterministic agent that emits a fixed reply for every prompt."""

    def __init__(self, reply: str = "v1-reply") -> None:
        self._reply = reply

    async def run(self, prompt: str) -> AsyncIterator[Any]:
        from locus.core.events import TerminateEvent, ThinkEvent

        yield ThinkEvent(iteration=0, reasoning=f"thinking about {prompt!r}", tool_calls=[])
        yield TerminateEvent(
            reason="complete",
            iterations_used=1,
            final_confidence=1.0,
            total_tool_calls=0,
            final_message=self._reply,
        )


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def live_server() -> Iterator[tuple[str, str]]:
    import uvicorn

    api_key = "integration-test-secret"  # gitleaks:allow — loopback test fixture
    port = _free_port()

    server = A2AServer(
        agent=_StubAgent(reply="v1-reply"),
        name="v1-research",
        description="v1 protocol test agent",
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

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not uv.started:
        time.sleep(0.05)
    assert uv.started, "uvicorn did not start within deadline"

    try:
        yield f"http://127.0.0.1:{port}", api_key
    finally:
        uv.should_exit = True
        thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Helpers — raw JSON-RPC over httpx so we assert the wire, not the client.
# ---------------------------------------------------------------------------


def _v1_message(text: str, *, mid: str | None = None) -> dict[str, Any]:
    return {
        "messageId": mid or uuid.uuid4().hex,
        "role": "ROLE_USER",
        "parts": [{"text": text}],
    }


async def _rpc(
    url: str,
    api_key: str,
    method: str,
    params: dict[str, Any],
    *,
    version: str | None = A2A_V1_PROTOCOL_VERSION,
    request_id: str | None = None,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if version is not None:
        headers["A2A-Version"] = version
    body = {
        "jsonrpc": "2.0",
        "id": request_id or uuid.uuid4().hex,
        "method": method,
        "params": params,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{url}/", json=body, headers=headers)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Version negotiation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v1_version_header_dispatches_v1_path(live_server: tuple[str, str]) -> None:
    url, api_key = live_server
    body = await _rpc(url, api_key, "SendMessage", {"message": _v1_message("hi v1")})
    assert "result" in body, body
    task = body["result"]["task"]
    # v1 task state is the TASK_STATE_* enum on the wire.
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    # v1 role on the wire is ROLE_*.
    assert task["history"][0]["role"] == "ROLE_USER"


@pytest.mark.asyncio
async def test_v1_unsupported_version_yields_version_not_supported(
    live_server: tuple[str, str],
) -> None:
    url, api_key = live_server
    body = await _rpc(
        url,
        api_key,
        "SendMessage",
        {"message": _v1_message("hi")},
        version="9.99",
    )
    assert body.get("error", {}).get("code") == VERSION_NOT_SUPPORTED


@pytest.mark.asyncio
async def test_v1_method_auto_detect_without_version_header(
    live_server: tuple[str, str],
) -> None:
    """A v1 method name should be routed to v1 even when no header is set."""
    url, api_key = live_server
    body = await _rpc(
        url,
        api_key,
        "SendMessage",
        {"message": _v1_message("hi auto")},
        version=None,
    )
    assert "result" in body, body
    assert body["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"


@pytest.mark.asyncio
async def test_v0_methods_still_work_without_version(live_server: tuple[str, str]) -> None:
    """Regression guard: v0 ``message/send`` keeps working when no v1 header is sent."""
    url, api_key = live_server
    body = await _rpc(
        url,
        api_key,
        "message/send",
        {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "hi v0"}],
                "messageId": uuid.uuid4().hex,
            }
        },
        version=None,
    )
    assert "result" in body, body
    # v0 keeps the lowercase enum on the wire.
    assert body["result"]["status"]["state"] == TaskState.completed.value


# ---------------------------------------------------------------------------
# Core JSON-RPC methods
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v1_send_then_get_task(live_server: tuple[str, str]) -> None:
    url, api_key = live_server
    sent = await _rpc(url, api_key, "SendMessage", {"message": _v1_message("hello")})
    # SendMessage returns SendMessageResponse (oneof {task, message}), so the
    # `task` wrapper IS spec-correct here.
    task_id = sent["result"]["task"]["id"]
    # GetTask must return the Task directly per the v1 JSON-RPC binding —
    # the canonical a2a-sdk client does
    # ``json_format.ParseDict(result, Task())`` on the result.
    got = await _rpc(url, api_key, "GetTask", {"id": task_id})
    assert got["result"]["id"] == task_id
    assert got["result"]["status"]["state"] == "TASK_STATE_COMPLETED"


@pytest.mark.asyncio
async def test_v1_get_task_missing_id_yields_task_not_found(
    live_server: tuple[str, str],
) -> None:
    url, api_key = live_server
    body = await _rpc(url, api_key, "GetTask", {"id": "does-not-exist"})
    assert body.get("error", {}).get("code") == TASK_NOT_FOUND


@pytest.mark.asyncio
async def test_v1_cancel_terminal_task_yields_not_cancelable(
    live_server: tuple[str, str],
) -> None:
    url, api_key = live_server
    sent = await _rpc(url, api_key, "SendMessage", {"message": _v1_message("done")})
    task_id = sent["result"]["task"]["id"]
    body = await _rpc(url, api_key, "CancelTask", {"id": task_id})
    assert body.get("error", {}).get("code") == TASK_NOT_CANCELABLE


@pytest.mark.asyncio
async def test_v1_list_tasks_paginates_and_filters(live_server: tuple[str, str]) -> None:
    url, api_key = live_server
    # Seed 3 tasks.
    sent_ids: list[str] = []
    for i in range(3):
        sent = await _rpc(
            url,
            api_key,
            "SendMessage",
            {"message": _v1_message(f"task-{i}", mid=f"m-{i}")},
        )
        sent_ids.append(sent["result"]["task"]["id"])

    # Page 1: size=2 → 2 tasks + non-empty nextPageToken.
    page1 = await _rpc(url, api_key, "ListTasks", {"pageSize": 2})
    res1 = page1["result"]
    assert res1["totalSize"] == 3
    assert res1["pageSize"] == 2
    assert len(res1["tasks"]) == 2
    assert res1["nextPageToken"] != ""

    # Page 2: continuation.
    page2 = await _rpc(
        url, api_key, "ListTasks", {"pageSize": 2, "pageToken": res1["nextPageToken"]}
    )
    res2 = page2["result"]
    assert len(res2["tasks"]) == 1
    assert res2["nextPageToken"] == ""

    # Status filter: all 3 should be in COMPLETED.
    filtered = await _rpc(url, api_key, "ListTasks", {"status": "TASK_STATE_COMPLETED"})
    assert filtered["result"]["totalSize"] == 3

    # No matches for SUBMITTED.
    no_match = await _rpc(url, api_key, "ListTasks", {"status": "TASK_STATE_SUBMITTED"})
    assert no_match["result"]["totalSize"] == 0


@pytest.mark.asyncio
async def test_v1_list_tasks_invalid_page_token_yields_invalid_params(
    live_server: tuple[str, str],
) -> None:
    url, api_key = live_server
    body = await _rpc(url, api_key, "ListTasks", {"pageToken": "not-an-int"})
    assert body.get("error", {}).get("code") == INVALID_PARAMS


# ---------------------------------------------------------------------------
# Streaming (SSE)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v1_send_streaming_message_emits_v1_envelopes(
    live_server: tuple[str, str],
) -> None:
    url, api_key = live_server
    body = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "SendStreamingMessage",
        "params": {"message": _v1_message("stream me")},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "A2A-Version": A2A_V1_PROTOCOL_VERSION,
        "Accept": "text/event-stream",
    }

    envelopes: list[dict[str, Any]] = []
    async with (
        httpx.AsyncClient(timeout=10.0) as client,
        client.stream("POST", f"{url}/", json=body, headers=headers) as resp,
    ):
        assert resp.status_code == 200
        # Spec doesn't pin a media type, but SSE is the binding for HTTP/REST.
        ctype = resp.headers.get("content-type", "")
        assert "event-stream" in ctype, f"expected SSE content-type, got {ctype!r}"
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = json.loads(line[len("data:") :].strip())
            # JSON-RPC envelope wraps a v1 StreamResponse oneof.
            result = payload.get("result", {})
            envelopes.append(result)
            if len(envelopes) > 30:
                break

    # Should see at least one Task envelope and one TaskStatusUpdateEvent
    # with v1 enum (TASK_STATE_*) — never the legacy lowercase form.
    assert envelopes, "stream produced no events"
    saw_v1_state = False
    for env in envelopes:
        if env.get("statusUpdate"):
            state = env["statusUpdate"]["status"]["state"]
            assert state.startswith("TASK_STATE_"), f"non-v1 state on wire: {state!r}"
            saw_v1_state = True
        if env.get("task"):
            assert env["task"]["status"]["state"].startswith("TASK_STATE_")
    assert saw_v1_state, "no TaskStatusUpdateEvent observed in stream"


@pytest.mark.asyncio
async def test_v1_subscribe_to_unknown_task_yields_task_not_found(
    live_server: tuple[str, str],
) -> None:
    url, api_key = live_server
    body = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "SubscribeToTask",
        "params": {"id": "does-not-exist"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "A2A-Version": A2A_V1_PROTOCOL_VERSION,
        "Accept": "text/event-stream",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{url}/", json=body, headers=headers)
    # Preflight should reject before the stream opens — JSON error envelope.
    data = resp.json()
    assert data.get("error", {}).get("code") == TASK_NOT_FOUND


@pytest.mark.asyncio
async def test_v1_subscribe_to_terminal_task_yields_unsupported_operation(
    live_server: tuple[str, str],
) -> None:
    url, api_key = live_server
    sent = await _rpc(url, api_key, "SendMessage", {"message": _v1_message("done")})
    task_id = sent["result"]["task"]["id"]
    body = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "SubscribeToTask",
        "params": {"id": task_id},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "A2A-Version": A2A_V1_PROTOCOL_VERSION,
        "Accept": "text/event-stream",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{url}/", json=body, headers=headers)
    data = resp.json()
    assert data.get("error", {}).get("code") == UNSUPPORTED_OPERATION


# ---------------------------------------------------------------------------
# Error envelopes — invalid params, push-notifications, unknown methods
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v1_send_message_missing_message_yields_invalid_params(
    live_server: tuple[str, str],
) -> None:
    url, api_key = live_server
    body = await _rpc(url, api_key, "SendMessage", {"configuration": {"blocking": True}})
    assert body.get("error", {}).get("code") == INVALID_PARAMS


@pytest.mark.asyncio
async def test_v1_push_notification_methods_signal_unsupported(
    live_server: tuple[str, str],
) -> None:
    url, api_key = live_server
    for method in (
        "CreateTaskPushNotificationConfig",
        "GetTaskPushNotificationConfig",
        "ListTaskPushNotificationConfigs",
        "DeleteTaskPushNotificationConfig",
    ):
        body = await _rpc(url, api_key, method, {})
        assert body.get("error", {}).get("code") == PUSH_NOTIFICATION_NOT_SUPPORTED, (
            f"{method} did not return PUSH_NOTIFICATION_NOT_SUPPORTED: {body!r}"
        )


@pytest.mark.asyncio
async def test_v1_get_extended_agent_card_signals_unsupported(
    live_server: tuple[str, str],
) -> None:
    url, api_key = live_server
    body = await _rpc(url, api_key, "GetExtendedAgentCard", {})
    assert body.get("error", {}).get("code") == UNSUPPORTED_OPERATION


@pytest.mark.asyncio
async def test_v1_unknown_method_yields_method_not_found(
    live_server: tuple[str, str],
) -> None:
    url, api_key = live_server
    body = await _rpc(url, api_key, "NotAMethod", {})
    assert body.get("error", {}).get("code") == METHOD_NOT_FOUND


@pytest.mark.asyncio
async def test_v1_unsupported_input_mode_yields_content_type_error(
    live_server: tuple[str, str],
) -> None:
    """Server card declares text/plain as the only input mode; a video/mp4
    part must be rejected with CONTENT_TYPE_NOT_SUPPORTED."""
    url, api_key = live_server
    body = await _rpc(
        url,
        api_key,
        "SendMessage",
        {
            "message": {
                "messageId": uuid.uuid4().hex,
                "role": "ROLE_USER",
                "parts": [{"url": "https://example.com/clip.mp4", "mediaType": "video/mp4"}],
            }
        },
    )
    assert body.get("error", {}).get("code") == CONTENT_TYPE_NOT_SUPPORTED


# ---------------------------------------------------------------------------
# Client surface — verify A2AClient defaults to v1 and round-trips.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a2a_client_send_then_list_tasks_v1(live_server: tuple[str, str]) -> None:
    """Round-trip via the high-level :class:`A2AClient` (v1 default)."""
    from locus.a2a import Message, TextPart

    url, api_key = live_server
    a2a = A2AClient(url=url, api_key=api_key)
    assert a2a._protocol_version == A2A_V1_PROTOCOL_VERSION  # noqa: SLF001

    task = await a2a.send_message(
        Message(role="user", parts=[TextPart(text="from client")], messageId="m-c-1")
    )
    assert task.status.state == TaskState.completed

    tasks, _next_token = await a2a.list_tasks(page_size=10)
    ids = [t.id for t in tasks]
    assert task.id in ids


# ---------------------------------------------------------------------------
# Real-model end-to-end — proves the v1 wire works against a live LLM,
# not just a deterministic stub agent. Uses OCI GenAI / gpt-4o-mini via
# OCIChatCompletionsModel. Skipped unless OCI_PROFILE is set and ~/.oci/
# config exists.
# ---------------------------------------------------------------------------


_OCI_PROFILE = os.environ.get("OCI_PROFILE")
_OCI_REGION = os.environ.get("OCI_REGION", "us-chicago-1")
_OCI_MODEL = os.environ.get("LOCUS_TEST_MODEL", "openai.gpt-4o-mini")


def _has_oci_config() -> bool:
    return os.path.exists(os.path.expanduser("~/.oci/config"))


@pytest.fixture
def real_model_server() -> Iterator[tuple[str, str]]:
    """Boot an A2AServer backed by a real Locus Agent + OCI GenAI model."""
    if not _has_oci_config() or not _OCI_PROFILE:
        pytest.skip("OCI_PROFILE not set or ~/.oci/config missing — skipping real-model test")

    import uvicorn

    from locus.agent import Agent
    from locus.models.providers.oci import OCIChatCompletionsModel

    model = OCIChatCompletionsModel(
        model=_OCI_MODEL,
        profile=_OCI_PROFILE,
        region=_OCI_REGION,
    )
    agent = Agent(
        model=model,
        tools=[],
        system_prompt=(
            "You are a terse assistant. Reply with one short sentence, no preamble, no markdown."
        ),
        max_iterations=2,
    )

    api_key = "real-model-test-fixture"  # gitleaks:allow — loopback test fixture
    port = _free_port()
    server = A2AServer(
        agent=agent,
        name="v1-real-model",
        description="v1 protocol test agent backed by a real LLM",
        url=f"http://127.0.0.1:{port}",
        skills=[
            AgentSkill(
                id="qa",
                name="QA",
                description="Answer questions.",
                tags=["qa"],
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

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not uv.started:
        time.sleep(0.05)
    assert uv.started, "uvicorn did not start within deadline"

    try:
        yield f"http://127.0.0.1:{port}", api_key
    finally:
        uv.should_exit = True
        thread.join(timeout=5.0)


@pytest.mark.asyncio
async def test_v1_real_model_send_message_round_trip(
    real_model_server: tuple[str, str],
) -> None:
    """SendMessage v1 against a real GPT-backed agent — proves the v1 wire
    survives a real ModelChunk → AgentResponse → Task lifecycle, not just the
    deterministic stub."""
    url, api_key = real_model_server
    body = await _rpc(
        url,
        api_key,
        "SendMessage",
        {"message": _v1_message("What is the capital of France? One word.")},
    )
    assert "result" in body, body
    task = body["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    # Wire enums are v1.0 form.
    assert task["history"][0]["role"] == "ROLE_USER"
    # The model produced *something* in an artifact.
    assert task["artifacts"], "real-model task should ship an artifact"
    artifact_text = ""
    for part in task["artifacts"][-1]["parts"]:
        if part.get("text"):
            artifact_text += part["text"]
    assert artifact_text.strip(), "artifact text should be non-empty"
    # The actual answer should contain "Paris" — a soft conformance check that
    # the request body and response body really survived the round-trip.
    assert "paris" in artifact_text.lower(), (
        f"expected 'Paris' in real-model reply, got: {artifact_text!r}"
    )


@pytest.mark.asyncio
async def test_v1_real_model_streaming_emits_v1_envelopes(
    real_model_server: tuple[str, str],
) -> None:
    """SendStreamingMessage v1 over real LLM — confirms SSE framing carries
    the v1 StreamResponse oneof end-to-end through real model chunks."""
    url, api_key = real_model_server
    body = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "SendStreamingMessage",
        "params": {
            "message": _v1_message("Reply with the single word: OK."),
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "A2A-Version": A2A_V1_PROTOCOL_VERSION,
        "Accept": "text/event-stream",
    }

    envelopes: list[dict[str, Any]] = []
    async with (
        httpx.AsyncClient(timeout=60.0) as client,
        client.stream("POST", f"{url}/", json=body, headers=headers) as resp,
    ):
        assert resp.status_code == 200
        assert "event-stream" in resp.headers.get("content-type", "")
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = json.loads(line[len("data:") :].strip())
            result = payload.get("result", {})
            envelopes.append(result)
            if len(envelopes) > 100:
                break

    assert envelopes, "real-model stream produced no events"
    saw_terminal_state = False
    for env in envelopes:
        if env.get("statusUpdate"):
            state = env["statusUpdate"]["status"]["state"]
            assert state.startswith("TASK_STATE_"), f"non-v1 state on wire: {state!r}"
            if state in {"TASK_STATE_COMPLETED", "TASK_STATE_FAILED"}:
                saw_terminal_state = True
        if env.get("task"):
            assert env["task"]["status"]["state"].startswith("TASK_STATE_")
    assert saw_terminal_state, "stream never reached a terminal status update"
