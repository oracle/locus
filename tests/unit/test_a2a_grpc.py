# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The A2A v1.0 gRPC binding, server and client, over a loopback ``grpc.aio`` server."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest


grpc = pytest.importorskip("grpc")
pytest.importorskip("grpc_status")

from locus.a2a import (  # noqa: E402
    A2AClient,
    A2AClientError,
    A2AServer,
    A2ATransportUnavailableError,
    AgentCard,
    AgentInterface,
    Message,
    PushNotificationSettings,
    TaskState,
    TextPart,
)
from locus.a2a.grpc_binding import (  # noqa: E402
    RPC_SIGNATURES,
    GrpcClientTransport,
    V1Messages,
    grpc_status_to_error,
    grpc_target,
    v1_messages,
)
from locus.a2a.spec import (  # noqa: E402
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    PUSH_NOTIFICATION_NOT_SUPPORTED,
    TASK_NOT_CANCELABLE,
    TASK_NOT_FOUND,
    UNSUPPORTED_OPERATION,
)
from locus.a2a.spec_v1 import A2A_GRPC_SERVICE, BINDING_GRPC, BINDING_JSONRPC  # noqa: E402
from locus.core.events import TerminateEvent, ThinkEvent  # noqa: E402


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


def _agent() -> _StubAgent:
    return _StubAgent(
        [ThinkEvent(iteration=0, reasoning="thinking", tool_calls=[]), _terminate("done")]
    )


@pytest.fixture
async def served(request: pytest.FixtureRequest) -> AsyncIterator[tuple[A2AServer, str]]:
    kwargs = dict(getattr(request, "param", {}))
    server = A2AServer(
        agent=_agent(),
        allow_unauthenticated=True,
        url="http://127.0.0.1",
        transports=[BINDING_JSONRPC, BINDING_GRPC],
        grpc_url="set-below",
        **kwargs,
    )
    aio_server, port = server.grpc_server("127.0.0.1", 0)
    server._grpc_url = f"127.0.0.1:{port}"
    await aio_server.start()
    try:
        yield server, f"127.0.0.1:{port}"
    finally:
        await aio_server.stop(grace=None)
        await server.aclose()


def _message(text: str, **extra: Any) -> Message:
    return Message(role="user", parts=[TextPart(text=text)], messageId="m", **extra)


class TestDescriptors:
    def test_messages_load_once_and_round_trip_json(self) -> None:
        messages = v1_messages()
        assert messages is v1_messages()
        assert isinstance(messages, V1Messages)
        assert messages.service.full_name == A2A_GRPC_SERVICE
        assert {m.name for m in messages.service.methods} == set(RPC_SIGNATURES)
        task = messages.from_dict(
            {
                "id": "t1",
                "contextId": "c1",
                "status": {"state": "TASK_STATE_COMPLETED", "timestamp": "2026-11-16T12:00:00Z"},
                "artifacts": [{"artifactId": "a", "parts": [{"text": "x"}, {"data": {"k": [1]}}]}],
                "unknownField": 1,
            },
            "Task",
        )
        assert messages.to_dict(task)["artifacts"][0]["parts"][1]["data"] == {"k": [1.0]}
        assert messages["google.protobuf.Empty"]().SerializeToString() == b""

    def test_grpc_target_forms(self) -> None:
        assert grpc_target("peer:50051") == ("peer:50051", False)
        assert grpc_target("grpc://peer:50051") == ("peer:50051", False)
        assert grpc_target("grpcs://peer") == ("peer:443", True)
        assert grpc_target("http://peer") == ("peer:80", False)

    def test_status_translation(self) -> None:
        from google.rpc import error_details_pb2, status_pb2

        status = status_pb2.Status(code=5, message="gone")
        info = error_details_pb2.ErrorInfo(reason="TASK_NOT_FOUND", domain="a2a-protocol.org")
        status.details.add().Pack(info)
        assert grpc_status_to_error(status, "fallback").code == TASK_NOT_FOUND
        assert grpc_status_to_error(None, "fallback").code == INTERNAL_ERROR
        assert grpc_status_to_error(status_pb2.Status(code=2), "fallback").message == "fallback"


class TestGrpcBinding:
    async def test_card_advertises_grpc(self, served: tuple[A2AServer, str]) -> None:
        server, target = served
        card = server._build_card()
        assert [(i.protocolBinding, i.url) for i in card.supportedInterfaces][1] == (
            BINDING_GRPC,
            target,
        )

    async def test_lifecycle_over_grpc(self, served: tuple[A2AServer, str]) -> None:
        _, target = served
        client = A2AClient(target, binding="grpc")
        assert client.binding == BINDING_GRPC
        try:
            task = await client.send_message(_message("hi"))
            assert task.status.state == TaskState.completed
            assert task.artifacts[-1].parts[0].text == "done"  # type: ignore[union-attr]

            fetched = await client.get_task(task.id, history_length=1)
            assert fetched.id == task.id
            assert len(fetched.history) == 1
            listed, next_page = await client.list_tasks(status="completed", page_size=10)
            assert [t.id for t in listed] == [task.id]
            assert next_page == ""

            with pytest.raises(A2AClientError) as err:
                await client.cancel_task(task.id)
            assert err.value.code == TASK_NOT_CANCELABLE
            with pytest.raises(A2AClientError) as missing:
                await client.get_task("nope")
            assert missing.value.code == TASK_NOT_FOUND
            with pytest.raises(A2AClientError) as follow_up:
                await client.send_message(_message("more", taskId=task.id))
            assert follow_up.value.code == UNSUPPORTED_OPERATION
            with pytest.raises(A2AClientError) as push:
                await client.set_push_notification_config(task.id, {"url": "https://h/x"})
            assert push.value.code == PUSH_NOTIFICATION_NOT_SUPPORTED

            kinds = []
            async for event in client.send_message_streaming(_message("again")):
                kinds.append(event.get("kind"))
            assert kinds[0] == "task"
            assert kinds[-1] == "status-update"
            assert "artifact-update" in kinds

            frames = [f async for f in client.subscribe(task.id)]
            assert frames[0]["error"]["code"] == UNSUPPORTED_OPERATION
            frames = [f async for f in client.subscribe("nope")]
            assert frames[0]["error"]["code"] == TASK_NOT_FOUND
        finally:
            await client.aclose()

    @pytest.mark.parametrize(
        "served", [{"push_notifications": PushNotificationSettings()}], indirect=True
    )
    async def test_push_config_methods_over_grpc(self, served: tuple[A2AServer, str]) -> None:
        _, target = served
        client = A2AClient(f"grpc://{target}", binding="grpc")
        try:
            task = await client.send_message(_message("hi"))
            stored = await client.set_push_notification_config(
                task.id, {"id": "c1", "url": "https://8.8.8.8/x", "token": "t"}
            )
            assert stored.taskId == task.id
            assert stored.id == "c1"
            assert (await client.get_push_notification_config(task.id, "c1")).token == "t"  # noqa: S105
            assert [c.id for c in await client.list_push_notification_configs(task.id)] == ["c1"]
            await client.delete_push_notification_config(task.id, "c1")
            assert await client.list_push_notification_configs(task.id) == []
        finally:
            await client.aclose()

    async def test_transport_rejects_unknown_methods(self, served: tuple[A2AServer, str]) -> None:
        _, target = served
        transport = GrpcClientTransport(target)
        with pytest.raises(A2AClientError) as err:
            await transport.call("Nope", {})
        assert err.value.code == METHOD_NOT_FOUND
        with pytest.raises(A2AClientError):
            async for _ in transport.stream("GetTask", {"id": "x"}):
                pass
        with pytest.raises(A2AClientError):
            transport._method("Nope")
        await transport.aclose()
        await transport.aclose()

    async def test_bearer_auth_and_version_check(self) -> None:
        server = A2AServer(agent=_agent(), api_key="s3cret", transports=["grpc"], grpc_port=0)
        aio_server, port = server.grpc_server("127.0.0.1", 0)
        await aio_server.start()
        try:
            anonymous = A2AClient(f"127.0.0.1:{port}", binding="grpc")
            with pytest.raises(A2AClientError, match="bearer"):
                await anonymous.get_task("x")
            await anonymous.aclose()

            authed = A2AClient(f"127.0.0.1:{port}", api_key="s3cret", binding="grpc")
            with pytest.raises(A2AClientError) as err:
                await authed.get_task("x")
            assert err.value.code == TASK_NOT_FOUND
            await authed.aclose()

            messages = v1_messages()
            channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
            method = channel.unary_unary(
                f"/{A2A_GRPC_SERVICE}/GetTask",
                request_serializer=messages["GetTaskRequest"].SerializeToString,
                response_deserializer=messages["Task"].FromString,
            )
            with pytest.raises(grpc.aio.AioRpcError) as rpc_err:
                await method(
                    messages.from_dict({"id": "x"}, "GetTaskRequest"),
                    metadata=[("authorization", "Bearer s3cret"), ("a2a-version", "0.3")],
                )
            assert rpc_err.value.code() == grpc.StatusCode.FAILED_PRECONDITION
            assert "0.3" in rpc_err.value.details()
            await channel.close()
        finally:
            await aio_server.stop(grace=None)

    async def test_unreachable_grpc_falls_back_to_jsonrpc(self) -> None:
        import httpx

        http_server = A2AServer(agent=_agent(), allow_unauthenticated=True)
        card = AgentCard(
            name="peer",
            description="d",
            url="http://testserver",
            supportedInterfaces=[
                AgentInterface(
                    protocolBinding=BINDING_GRPC, url="127.0.0.1:1", protocolVersion="1.0"
                ),
                AgentInterface(
                    protocolBinding=BINDING_JSONRPC, url="http://testserver/", protocolVersion="1.0"
                ),
            ],
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=http_server.app), base_url="http://testserver"
        ) as http:
            client = A2AClient.from_card(card, prefer=["grpc", "jsonrpc"], http_client=http)
            assert client.binding == BINDING_GRPC
            task = await client.send_message(_message("hi"))
            assert task.status.state == TaskState.completed
            assert client.binding == BINDING_JSONRPC

        only = A2AClient("127.0.0.1:1", binding="grpc")
        with pytest.raises(A2ATransportUnavailableError):
            await only.get_task("x")
        with pytest.raises(A2ATransportUnavailableError):
            async for _ in only.subscribe("x"):
                pass
        await only.aclose()

    def test_run_serves_grpc_alongside_http(self, monkeypatch: pytest.MonkeyPatch) -> None:
        served_ports: list[int] = []

        class _FakeHttp:
            def __init__(self, config: Any) -> None:
                self.config = config

            async def serve(self) -> None:
                served_ports.append(self.config.port)

        import uvicorn

        monkeypatch.setattr(uvicorn, "Server", _FakeHttp)
        server = A2AServer(agent=_agent(), allow_unauthenticated=True, grpc_port=0)
        server.run(host="127.0.0.1", port=18099)
        assert served_ports == [18099]

    def test_grpc_server_needs_a_port(self) -> None:
        server = A2AServer(agent=_agent(), allow_unauthenticated=True)
        with pytest.raises(ValueError, match="no gRPC port"):
            server.grpc_server()
        with pytest.raises(ValueError, match="needs grpc_port"):
            A2AServer(agent=_agent(), allow_unauthenticated=True, transports=["grpc"])
