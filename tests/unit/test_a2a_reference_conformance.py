# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Conformance: the reference A2A v1 Python SDK as the client, Locus as the server.

Runs only where ``a2a-sdk`` is installed (it is a test-time dependency, not
a runtime one). Each binding is exercised with the reference SDK's own
transport: JSON-RPC and HTTP+JSON through ``httpx.ASGITransport`` on the
server's app, gRPC through a loopback ``grpc.aio`` server. The card is read
with the reference card resolver, so the shapes Locus serves are validated
against the reference protobuf models, not against Locus's own.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest


a2a = pytest.importorskip("a2a")
pytest.importorskip("grpc")

from a2a.client.card_resolver import A2ACardResolver  # noqa: E402
from a2a.client.transports.grpc import GrpcTransport  # noqa: E402
from a2a.client.transports.jsonrpc import JsonRpcTransport  # noqa: E402
from a2a.client.transports.rest import RestTransport  # noqa: E402
from a2a.types import a2a_pb2 as pb  # noqa: E402
from a2a.utils import errors as ref_errors  # noqa: E402

from locus.a2a import A2AServer, PushNotificationReceiver, PushNotificationSettings  # noqa: E402
from locus.a2a.spec_v1 import BINDING_GRPC, BINDING_HTTP_JSON, BINDING_JSONRPC  # noqa: E402
from locus.agent import Agent  # noqa: E402
from locus.core.events import TerminateEvent, ThinkEvent  # noqa: E402
from locus.core.messages import Role  # noqa: E402
from locus.testing import FunctionModel, text, tool_call  # noqa: E402
from locus.tools.decorator import tool  # noqa: E402


@tool  # noqa: E402
def ask_region(question: str, options: str | None = None) -> str:
    """Pause the run for a person: the interrupt marker any tool may return."""
    option_list = [o.strip() for o in options.split(",")] if options else None
    return json.dumps({"__interrupt__": True, "question": question, "options": option_list})


class _StubAgent:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def run(self, prompt: str) -> Any:
        for event in self._events:
            yield event


def _terminate(final_message: str) -> TerminateEvent:
    return TerminateEvent(
        reason="complete",
        iterations_used=1,
        final_confidence=1.0,
        total_tool_calls=0,
        final_message=final_message,
    )


def _stub() -> _StubAgent:
    return _StubAgent(
        [ThinkEvent(iteration=0, reasoning="thinking", tool_calls=[]), _terminate("done")]
    )


def _question_agent() -> Agent:
    def handler(messages: list[Any], tools: list[dict[str, Any]]) -> Any:
        answers = [m.content for m in messages if m.role == Role.TOOL]
        if answers:
            return text(f"Deployed to {answers[-1]}.")
        return tool_call("ask_region", question="Which region?")

    return Agent(model=FunctionModel(handler), tools=[ask_region], reflexion=False, grounding=False)


def _user(text_: str, *, task_id: str = "") -> pb.SendMessageRequest:
    return pb.SendMessageRequest(
        message=pb.Message(
            message_id="m-1", role=pb.Role.ROLE_USER, parts=[pb.Part(text=text_)], task_id=task_id
        )
    )


class _Peer:
    """A Locus server plus a reference-SDK transport for one binding."""

    def __init__(
        self, server: A2AServer, transport: Any, http: httpx.AsyncClient, grpc_server: Any
    ) -> None:
        self.server = server
        self.transport = transport
        self.http = http
        self.grpc_server = grpc_server

    async def aclose(self) -> None:
        await self.transport.close()
        if self.grpc_server is not None:
            await self.grpc_server.stop(grace=None)
        await self.http.aclose()
        await self.server.aclose()


async def _peer(binding: str, agent: Any, **server_kwargs: Any) -> _Peer:
    import grpc

    grpc_server = None
    kwargs: dict[str, Any] = {"transports": [BINDING_JSONRPC, BINDING_HTTP_JSON]}
    if binding == BINDING_GRPC:
        kwargs = {"transports": [BINDING_JSONRPC, BINDING_GRPC], "grpc_url": "pending"}
    server = A2AServer(
        agent=agent, allow_unauthenticated=True, url="http://testserver", **kwargs, **server_kwargs
    )
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://testserver"
    )
    if binding == BINDING_GRPC:
        grpc_server, port = server.grpc_server("127.0.0.1", 0)
        server._grpc_url = f"127.0.0.1:{port}"
        await grpc_server.start()
    card = await A2ACardResolver(http, "http://testserver").get_agent_card()
    bindings = {i.protocol_binding: i.url for i in card.supported_interfaces}
    assert binding in bindings
    if binding == BINDING_JSONRPC:
        transport: Any = JsonRpcTransport(http, card, bindings[binding], is_shared_httpx=True)
    elif binding == BINDING_HTTP_JSON:
        transport = RestTransport(http, card, bindings[binding], is_shared_httpx=True)
    else:
        transport = GrpcTransport(grpc.aio.insecure_channel(bindings[binding]), card)
    return _Peer(server, transport, http, grpc_server)


@pytest.fixture(params=[BINDING_JSONRPC, BINDING_HTTP_JSON, BINDING_GRPC])
async def peer(request: pytest.FixtureRequest) -> AsyncIterator[_Peer]:
    built = await _peer(request.param, _stub())
    try:
        yield built
    finally:
        await built.aclose()


class TestReferenceClient:
    async def test_card_parses_with_the_reference_resolver(self, peer: _Peer) -> None:
        card = peer.transport.agent_card
        assert card.name == "Locus Agent"
        assert card.capabilities.streaming is True
        assert {i.protocol_version for i in card.supported_interfaces} == {"1.0"}

    async def test_send_get_list_cancel(self, peer: _Peer) -> None:
        response = await peer.transport.send_message(_user("hi"))
        assert response.HasField("task")
        task = response.task
        assert task.status.state == pb.TaskState.TASK_STATE_COMPLETED
        assert task.artifacts[-1].parts[0].text == "done"
        assert task.history[0].role == pb.Role.ROLE_USER

        fetched = await peer.transport.get_task(pb.GetTaskRequest(id=task.id, history_length=1))
        assert fetched.id == task.id
        assert len(fetched.history) == 1

        listed = await peer.transport.list_tasks(
            pb.ListTasksRequest(status=pb.TaskState.TASK_STATE_COMPLETED)
        )
        assert [t.id for t in listed.tasks] == [task.id]
        assert listed.total_size == 1

        with pytest.raises(ref_errors.TaskNotCancelableError):
            await peer.transport.cancel_task(pb.CancelTaskRequest(id=task.id))
        with pytest.raises(ref_errors.TaskNotFoundError):
            await peer.transport.get_task(pb.GetTaskRequest(id="nope"))
        with pytest.raises(ref_errors.UnsupportedOperationError):
            await peer.transport.send_message(_user("more", task_id=task.id))
        with pytest.raises(ref_errors.PushNotificationNotSupportedError):
            await peer.transport.create_task_push_notification_config(
                pb.TaskPushNotificationConfig(task_id=task.id, url="https://8.8.8.8/x")
            )

    async def test_streaming(self, peer: _Peer) -> None:
        frames = [frame async for frame in peer.transport.send_message_streaming(_user("hi"))]
        assert frames[0].HasField("task")
        assert frames[0].task.status.state == pb.TaskState.TASK_STATE_SUBMITTED
        assert any(f.HasField("artifact_update") for f in frames)
        assert frames[-1].HasField("status_update")
        assert frames[-1].status_update.status.state == pb.TaskState.TASK_STATE_COMPLETED
        with pytest.raises(ref_errors.UnsupportedOperationError):
            async for _ in peer.transport.subscribe(
                pb.SubscribeToTaskRequest(id=frames[0].task.id)
            ):
                pass


@pytest.fixture(params=[BINDING_JSONRPC, BINDING_HTTP_JSON, BINDING_GRPC])
async def pushing_peer(
    request: pytest.FixtureRequest,
) -> AsyncIterator[tuple[_Peer, PushNotificationReceiver]]:
    receiver = PushNotificationReceiver(token="tok")  # noqa: S106
    hooks = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=receiver.app), base_url="http://receiver"
    )
    built = await _peer(
        request.param,
        _stub(),
        push_notifications=PushNotificationSettings(allow_private_urls=True, http_client=hooks),
    )
    try:
        yield built, receiver
    finally:
        await built.aclose()
        await hooks.aclose()


class TestReferencePush:
    async def test_config_lifecycle_and_delivery(
        self, pushing_peer: tuple[_Peer, PushNotificationReceiver]
    ) -> None:
        peer, receiver = pushing_peer
        request = _user("hi")
        request.configuration.task_push_notification_config.url = "http://127.0.0.1:1/a2a/push"
        request.configuration.task_push_notification_config.token = "tok"  # noqa: S105
        request.configuration.return_immediately = True
        response = await peer.transport.send_message(request)
        task_id = response.task.id
        await peer.server.drain()
        states = [n.state for n in receiver.received if n.kind == "statusUpdate"]
        assert states[0] == "TASK_STATE_WORKING"
        assert states[-1] == "TASK_STATE_COMPLETED"
        assert any(n.kind == "artifactUpdate" for n in receiver.received)

        configs = await peer.transport.list_task_push_notification_configs(
            pb.ListTaskPushNotificationConfigsRequest(task_id=task_id)
        )
        assert len(configs.configs) == 1
        assert configs.configs[0].url == "http://127.0.0.1:1/a2a/push"
        config_id = configs.configs[0].id
        got = await peer.transport.get_task_push_notification_config(
            pb.GetTaskPushNotificationConfigRequest(task_id=task_id, id=config_id)
        )
        assert got.token == "tok"  # noqa: S105
        created = await peer.transport.create_task_push_notification_config(
            pb.TaskPushNotificationConfig(id="c2", task_id=task_id, url="http://127.0.0.1:1/other")
        )
        assert created.id == "c2"
        await peer.transport.delete_task_push_notification_config(
            pb.DeleteTaskPushNotificationConfigRequest(task_id=task_id, id="c2")
        )
        remaining = await peer.transport.list_task_push_notification_configs(
            pb.ListTaskPushNotificationConfigsRequest(task_id=task_id)
        )
        assert [c.id for c in remaining.configs] == [config_id]


@pytest.fixture(params=[BINDING_JSONRPC, BINDING_HTTP_JSON, BINDING_GRPC])
async def pausing_peer(request: pytest.FixtureRequest) -> AsyncIterator[_Peer]:
    built = await _peer(request.param, _question_agent())
    try:
        yield built
    finally:
        await built.aclose()


class TestReferenceInterrupt:
    async def test_input_required_then_resume(self, pausing_peer: _Peer) -> None:
        response = await pausing_peer.transport.send_message(_user("deploy"))
        task = response.task
        assert task.status.state == pb.TaskState.TASK_STATE_INPUT_REQUIRED
        assert task.status.message.parts[0].text == "Which region?"
        assert task.status.message.metadata["kind"] == "question"

        resumed = await pausing_peer.transport.send_message(
            _user("eu-frankfurt-1", task_id=task.id)
        )
        assert resumed.task.id == task.id
        assert resumed.task.status.state == pb.TaskState.TASK_STATE_COMPLETED
        assert resumed.task.artifacts[-1].parts[0].text == "Deployed to eu-frankfurt-1."
