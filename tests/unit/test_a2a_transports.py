# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The A2A v1.0 HTTP+JSON binding and the client's transport selection.

The server is driven in-process through ``httpx.ASGITransport`` so the
client, the server and its background work share one event loop.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from locus.a2a import (
    A2AClient,
    A2AClientError,
    A2AServer,
    A2ATransportUnavailableError,
    AgentCard,
    AgentInterface,
    Message,
    TaskState,
    TextPart,
)
from locus.a2a.protocol_v1 import A2A_ERROR_MAPPING, error_mapping, http_error_body
from locus.a2a.rest import query_to_params
from locus.a2a.spec import (
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    TASK_NOT_CANCELABLE,
    TASK_NOT_FOUND,
    UNSUPPORTED_OPERATION,
    VERSION_NOT_SUPPORTED,
)
from locus.a2a.spec_v1 import A2A_JSON_MEDIA_TYPE, BINDING_GRPC, BINDING_HTTP_JSON, BINDING_JSONRPC
from locus.a2a.transports import (
    HttpJsonClientTransport,
    JsonRpcClientTransport,
    make_transport,
    normalise_binding,
    select_interfaces,
)
from locus.core.events import TerminateEvent, ThinkEvent


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


def _server(**kwargs: Any) -> A2AServer:
    agent = _StubAgent(
        [ThinkEvent(iteration=0, reasoning="thinking", tool_calls=[]), _terminate("done")]
    )
    return A2AServer(agent=agent, allow_unauthenticated=True, url="http://testserver", **kwargs)


def _http(server: A2AServer) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://testserver"
    )


def _user(text: str, **extra: Any) -> dict[str, Any]:
    return {
        "message": {"role": "ROLE_USER", "parts": [{"text": text}], "messageId": "m-1", **extra}
    }


def _sse(body: str) -> list[dict[str, Any]]:
    frames = []
    for block in body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                frames.append(json.loads(line[5:].strip()))
    return frames


# ---------------------------------------------------------------------------
# Server: HTTP+JSON routes
# ---------------------------------------------------------------------------


class TestHttpJsonBinding:
    async def test_card_advertises_both_http_bindings(self) -> None:
        async with _http(_server()) as http:
            card = (await http.get("/.well-known/agent-card.json")).json()
        bindings = [i["protocolBinding"] for i in card["supportedInterfaces"]]
        assert bindings == [BINDING_JSONRPC, BINDING_HTTP_JSON]
        assert all(i["url"] == "http://testserver/" for i in card["supportedInterfaces"])
        assert card["capabilities"]["pushNotifications"] is False

    async def test_send_get_list_cancel(self) -> None:
        async with _http(_server()) as http:
            resp = await http.post(
                "/message:send", json=_user("hi"), headers={"A2A-Version": "1.0"}
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith(A2A_JSON_MEDIA_TYPE)
            task = resp.json()["task"]
            assert task["status"]["state"] == "TASK_STATE_COMPLETED"
            assert task["artifacts"][-1]["parts"][0]["text"] == "done"

            got = (await http.get(f"/tasks/{task['id']}", params={"historyLength": 1})).json()
            assert got["id"] == task["id"]
            assert len(got["history"]) == 1

            listed = (await http.get("/tasks", params={"status": "TASK_STATE_COMPLETED"})).json()
            assert [t["id"] for t in listed["tasks"]] == [task["id"]]
            assert listed["totalSize"] == 1

            cancel = await http.post(f"/tasks/{task['id']}:cancel", json={})
            assert cancel.status_code == 400
            error = cancel.json()["error"]
            assert error["status"] == "FAILED_PRECONDITION"
            assert error["details"][0]["reason"] == "TASK_NOT_CANCELABLE"

    async def test_unknown_task_is_404_with_error_info(self) -> None:
        async with _http(_server()) as http:
            resp = await http.get("/tasks/nope")
        assert resp.status_code == 404
        error = resp.json()["error"]
        assert error["code"] == 5
        assert error["status"] == "NOT_FOUND"
        assert error["details"][0]["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"
        assert error["details"][0]["reason"] == "TASK_NOT_FOUND"
        assert error["details"][0]["domain"] == "a2a-protocol.org"

    async def test_bad_json_and_bad_version(self) -> None:
        async with _http(_server()) as http:
            bad = await http.post(
                "/message:send", content=b"{not json", headers={"Content-Type": "application/json"}
            )
            assert bad.status_code == 400
            assert bad.json()["error"]["details"][0]["reason"] == "JSON_PARSE_ERROR"
            array = await http.post(
                "/message:send", content=b"[]", headers={"Content-Type": "application/json"}
            )
            assert array.json()["error"]["details"][0]["reason"] == "INVALID_REQUEST"
            version = await http.post(
                "/message:send", json=_user("hi"), headers={"A2A-Version": "0.3"}
            )
            assert version.status_code == 400
            assert version.json()["error"]["details"][0]["reason"] == "VERSION_NOT_SUPPORTED"
            empty = await http.post("/message:send", content=b"")
            assert empty.status_code == 400
            assert empty.json()["error"]["details"][0]["reason"] == "INVALID_PARAMS"

    async def test_follow_up_on_finished_task_is_unsupported_operation(self) -> None:
        async with _http(_server()) as http:
            task = (await http.post("/message:send", json=_user("hi"))).json()["task"]
            again = await http.post("/message:send", json=_user("more", taskId=task["id"]))
        assert again.status_code == 400
        assert again.json()["error"]["details"][0]["reason"] == "UNSUPPORTED_OPERATION"

    async def test_stream_and_subscribe(self) -> None:
        async with _http(_server()) as http:
            resp = await http.post("/message:stream", json=_user("hi"))
            assert resp.headers["content-type"].startswith("text/event-stream")
            frames = _sse(resp.text)
            assert "task" in frames[0]
            assert frames[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
            assert any("artifactUpdate" in f for f in frames)
            task_id = frames[0]["task"]["id"]

            done = await http.post(f"/tasks/{task_id}:subscribe")
            error = _sse(done.text)[0]["error"]
            assert error["details"][0]["reason"] == "UNSUPPORTED_OPERATION"

            missing = await http.get("/tasks/nope:subscribe")
            assert _sse(missing.text)[0]["error"]["details"][0]["reason"] == "TASK_NOT_FOUND"

    async def test_push_config_routes_when_push_is_off(self) -> None:
        async with _http(_server()) as http:
            task = (await http.post("/message:send", json=_user("hi"))).json()["task"]
            resp = await http.post(
                f"/tasks/{task['id']}/pushNotificationConfigs",
                json={"url": "https://hooks.example.com/a2a"},
            )
        assert resp.status_code == 400
        assert resp.json()["error"]["details"][0]["reason"] == "PUSH_NOTIFICATION_NOT_SUPPORTED"

    async def test_extended_card_is_unsupported(self) -> None:
        async with _http(_server()) as http:
            resp = await http.get("/extendedAgentCard")
        assert resp.status_code == 400
        assert resp.json()["error"]["details"][0]["reason"] == "UNSUPPORTED_OPERATION"

    async def test_bearer_auth_covers_rest_routes(self) -> None:
        server = A2AServer(agent=_StubAgent([_terminate("ok")]), api_key="s3cret")
        async with _http(server) as http:
            assert (await http.get("/tasks")).status_code == 401
            ok = await http.get("/tasks", headers={"Authorization": "Bearer s3cret"})
            assert ok.status_code == 200

    async def test_jsonrpc_only_server_has_no_rest_routes(self) -> None:
        server = _server(transports=["jsonrpc"])
        async with _http(server) as http:
            assert (await http.get("/tasks")).status_code in (404, 405)
            card = (await http.get("/.well-known/agent-card.json")).json()
        assert [i["protocolBinding"] for i in card["supportedInterfaces"]] == [BINDING_JSONRPC]

    def test_query_params_are_typed(self) -> None:
        params = query_to_params(
            {"historyLength": "3", "includeArtifacts": "false", "contextId": "c"}
        )
        assert params == {"historyLength": 3, "includeArtifacts": False, "contextId": "c"}
        with pytest.raises(Exception, match="integer"):
            query_to_params({"pageSize": "many"})

    def test_error_mapping_covers_every_spec_code(self) -> None:
        for code in (
            TASK_NOT_FOUND,
            TASK_NOT_CANCELABLE,
            UNSUPPORTED_OPERATION,
            VERSION_NOT_SUPPORTED,
        ):
            assert code in A2A_ERROR_MAPPING
        assert error_mapping(-99999) == (500, "INTERNAL", "INTERNAL_ERROR")
        body = http_error_body(PARSE_ERROR, "bad")
        assert body["error"]["code"] == 3
        assert body["error"]["status"] == "INVALID_ARGUMENT"
        assert error_mapping(INVALID_REQUEST)[0] == 400


# ---------------------------------------------------------------------------
# Client: transports and selection
# ---------------------------------------------------------------------------


def _card(*interfaces: tuple[str, str], url: str = "http://testserver") -> AgentCard:
    return AgentCard(
        name="peer",
        description="d",
        url=url,
        supportedInterfaces=[
            AgentInterface(protocolBinding=b, url=u, protocolVersion="1.0") for b, u in interfaces
        ],
    )


class TestSelection:
    def test_preference_order_and_version_filter(self) -> None:
        card = _card(
            (BINDING_GRPC, "peer:50051"),
            (BINDING_HTTP_JSON, "http://peer/"),
            (BINDING_JSONRPC, "http://peer/"),
        )
        card.supportedInterfaces.append(
            AgentInterface(
                protocolBinding=BINDING_JSONRPC, url="http://old/", protocolVersion="0.3"
            )
        )
        assert select_interfaces(card) == [
            (BINDING_JSONRPC, "http://peer/"),
            (BINDING_HTTP_JSON, "http://peer/"),
            (BINDING_GRPC, "peer:50051"),
        ]
        assert select_interfaces(card, prefer=["grpc", "http+json"]) == [
            (BINDING_GRPC, "peer:50051"),
            (BINDING_HTTP_JSON, "http://peer/"),
        ]

    def test_card_without_interfaces_falls_back_to_jsonrpc_at_card_url(self) -> None:
        assert select_interfaces(_card()) == [(BINDING_JSONRPC, "http://testserver")]
        assert select_interfaces(_card(), fallback_url="http://x") == [
            (BINDING_JSONRPC, "http://x")
        ]

    def test_binding_names_are_normalised(self) -> None:
        assert normalise_binding("jsonrpc") == BINDING_JSONRPC
        assert normalise_binding("http+json") == BINDING_HTTP_JSON
        assert normalise_binding("rest") == BINDING_HTTP_JSON
        assert normalise_binding("grpc") == BINDING_GRPC
        with pytest.raises(ValueError, match="unknown A2A protocol binding"):
            normalise_binding("carrier-pigeon")

    def test_make_transport_picks_the_class(self) -> None:
        assert isinstance(make_transport("JSONRPC", "http://x"), JsonRpcClientTransport)
        assert isinstance(make_transport("HTTP+JSON", "http://x"), HttpJsonClientTransport)


class TestClientOverBindings:
    @pytest.mark.parametrize("binding", [BINDING_JSONRPC, BINDING_HTTP_JSON])
    async def test_full_lifecycle(self, binding: str) -> None:
        server = _server()
        async with _http(server) as http:
            client = A2AClient("http://testserver", binding=binding, http_client=http)
            assert client.binding == binding
            task = await client.send_message(
                Message(role="user", parts=[TextPart(text="hi")], messageId="m")
            )
            assert task.status.state == TaskState.completed
            assert task.artifacts[-1].parts[0].text == "done"  # type: ignore[union-attr]

            fetched = await client.get_task(task.id, history_length=1)
            assert fetched.id == task.id
            assert len(fetched.history) == 1
            listed, next_page = await client.list_tasks(status=TaskState.completed, page_size=5)
            assert [t.id for t in listed] == [task.id]
            assert next_page == ""

            with pytest.raises(A2AClientError) as err:
                await client.cancel_task(task.id)
            assert err.value.code == TASK_NOT_CANCELABLE
            assert "A2A error -32002" in str(err.value)

            with pytest.raises(A2AClientError) as missing:
                await client.get_task("nope")
            assert missing.value.code == TASK_NOT_FOUND

            kinds = []
            async for event in client.send_message_streaming(
                Message(role="user", parts=[TextPart(text="again")], messageId="m2")
            ):
                kinds.append(event.get("kind"))
            assert kinds[0] == "task"
            assert kinds[-1] == "status-update"
            assert "artifact-update" in kinds

            frames = [f async for f in client.subscribe(task.id)]
            assert "error" in frames[0]
            assert frames[0]["error"]["code"] == UNSUPPORTED_OPERATION

    async def test_connect_prefers_the_cards_order_and_reports_binding(self) -> None:
        server = _server()
        async with _http(server) as http:
            client = await A2AClient.connect(
                "http://testserver", prefer=["http+json", "jsonrpc"], http_client=http
            )
            assert client.binding == BINDING_HTTP_JSON
            assert client.card is not None
            assert client.card.name == "Locus Agent"
            assert [t.binding for t in client.transports] == [BINDING_HTTP_JSON, BINDING_JSONRPC]
            task = await client.send_message(
                Message(role="user", parts=[TextPart(text="hi")], messageId="m")
            )
            assert task.status.state == TaskState.completed

    async def test_falls_back_when_the_preferred_interface_is_unreachable(self) -> None:
        server = _server(transports=["jsonrpc"])  # no REST routes: HTTP+JSON answers 404/405
        card = _card(
            (BINDING_HTTP_JSON, "http://testserver/"), (BINDING_JSONRPC, "http://testserver/")
        )
        async with _http(server) as http:
            client = A2AClient.from_card(card, prefer=["http+json", "jsonrpc"], http_client=http)
            assert client.binding == BINDING_HTTP_JSON
            task = await client.send_message(
                Message(role="user", parts=[TextPart(text="hi")], messageId="m")
            )
            assert task.status.state == TaskState.completed
            assert client.binding == BINDING_JSONRPC

    async def test_last_transport_raises_unavailable(self) -> None:
        server = _server(transports=["jsonrpc"])
        card = _card((BINDING_HTTP_JSON, "http://testserver/"))
        async with _http(server) as http:
            client = A2AClient.from_card(card, prefer=["http+json"], http_client=http)
            with pytest.raises(A2ATransportUnavailableError):
                await client.get_task("x")

    async def test_grpc_interface_is_skipped_without_grpcio(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import locus.a2a.protocol as protocol_module

        real = protocol_module.make_transport

        def no_grpc(binding: str, url: str, **kwargs: Any) -> Any:
            if binding == BINDING_GRPC:
                raise ImportError("the A2A gRPC binding needs grpcio")
            return real(binding, url, **kwargs)

        monkeypatch.setattr(protocol_module, "make_transport", no_grpc)
        card = _card((BINDING_GRPC, "peer:1"), (BINDING_JSONRPC, "http://testserver/"))
        client = A2AClient.from_card(card, prefer=["grpc", "jsonrpc"])
        assert client.binding == BINDING_JSONRPC
        only_grpc = A2AClient.from_card(_card((BINDING_GRPC, "peer:1")), prefer=["grpc"])
        assert only_grpc.binding == BINDING_JSONRPC
        assert only_grpc.transports[0].url == "http://testserver"

    async def test_legacy_protocol_version_still_uses_legacy_methods(self) -> None:
        server = _server()
        async with _http(server) as http:
            client = A2AClient("http://testserver", protocol_version=None, http_client=http)
            task = await client.send_message(
                Message(role="user", parts=[TextPart(text="hi")], messageId="m")
            )
            assert task.status.state == TaskState.completed
            assert await client.invoke("hello") == "done"
            with pytest.raises(RuntimeError, match=r"requires A2A v1\.0"):
                await client.list_tasks()
            with pytest.raises(RuntimeError, match=r"requires A2A v1\.0"):
                await client.list_push_notification_configs(task.id)


# ---------------------------------------------------------------------------
# Transport edge cases, with a mock HTTP layer.
# ---------------------------------------------------------------------------


def _mock(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _refusing(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


class TestTransportEdges:
    def test_more_binding_spellings(self) -> None:
        assert normalise_binding("HTTP-JSON") == BINDING_HTTP_JSON
        assert normalise_binding("Http+Json") == BINDING_HTTP_JSON

    async def test_jsonrpc_headers_and_owned_client_close(self) -> None:
        transport = JsonRpcClientTransport("http://peer/", api_key="k")
        assert transport.headers()["Authorization"] == "Bearer k"
        assert transport.url == "http://peer"
        client = await transport.client()
        assert isinstance(client, httpx.AsyncClient)
        await transport.aclose()
        assert transport._client is None
        await transport.aclose()

    async def test_sse_frames_without_trailing_blank_line(self) -> None:
        class _Resp:
            async def aiter_lines(self) -> Any:
                for line in ("data: one", "data: two", "", ": comment", "data: three"):
                    yield line

        frames = [f async for f in JsonRpcClientTransport._sse_frames(_Resp())]
        assert frames == ["one\ntwo", "three"]

    async def test_jsonrpc_connection_errors_are_unavailable(self) -> None:
        transport = JsonRpcClientTransport("http://peer", http_client=_mock(_refusing))
        with pytest.raises(A2ATransportUnavailableError):
            await transport.call("GetTask", {"id": "x"})
        with pytest.raises(A2ATransportUnavailableError):
            async for _ in transport.stream("SubscribeToTask", {"id": "x"}):
                pass

    async def test_jsonrpc_404_is_unavailable_for_call_and_stream(self) -> None:
        transport = JsonRpcClientTransport(
            "http://peer", http_client=_mock(lambda request: httpx.Response(404))
        )
        with pytest.raises(A2ATransportUnavailableError):
            await transport.call("GetTask", {"id": "x"})
        with pytest.raises(A2ATransportUnavailableError):
            async for _ in transport.stream("SubscribeToTask", {"id": "x"}):
                pass

    async def test_jsonrpc_json_body_before_stream(self) -> None:
        error = {"jsonrpc": "2.0", "id": "1", "error": {"code": -32001, "message": "gone"}}
        transport = JsonRpcClientTransport(
            "http://peer", http_client=_mock(lambda request: httpx.Response(200, json=error))
        )
        frames = [f async for f in transport.stream("SubscribeToTask", {"id": "x"})]
        assert frames == [{"error": {"code": -32001, "message": "gone"}}]

        silent = JsonRpcClientTransport(
            "http://peer", http_client=_mock(lambda request: httpx.Response(200, json={"ok": 1}))
        )
        assert [f async for f in silent.stream("SubscribeToTask", {"id": "x"})] == []

        broken = JsonRpcClientTransport(
            "http://peer",
            http_client=_mock(
                lambda request: httpx.Response(
                    200, content=b"{not json", headers={"content-type": "application/json"}
                )
            ),
        )
        assert [f async for f in broken.stream("SubscribeToTask", {"id": "x"})] == []

    async def test_jsonrpc_stream_error_frame_and_non_json_frames(self) -> None:
        body = (
            "data: not-json\n\n"
            'data: {"jsonrpc": "2.0", "id": "1", "result": {"task": {"id": "t"}}}\n\n'
            'data: {"jsonrpc": "2.0", "id": "1", "error": {"code": -32004, "message": "no"}}\n\n'
            'data: {"jsonrpc": "2.0", "id": "1", "result": {"never": 1}}\n\n'
        )
        transport = JsonRpcClientTransport(
            "http://peer",
            http_client=_mock(
                lambda request: httpx.Response(
                    200, content=body.encode(), headers={"content-type": "text/event-stream"}
                )
            ),
        )
        frames = [f async for f in transport.stream("SendStreamingMessage", {})]
        assert frames == [{"task": {"id": "t"}}, {"error": {"code": -32004, "message": "no"}}]

    async def test_jsonrpc_non_dict_result_is_wrapped(self) -> None:
        transport = JsonRpcClientTransport(
            "http://peer",
            http_client=_mock(
                lambda request: httpx.Response(
                    200, json={"jsonrpc": "2.0", "id": "1", "result": [1]}
                )
            ),
        )
        assert await transport.call("ListTasks", {}) == {"result": [1]}

    def test_rest_routes(self) -> None:
        transport = HttpJsonClientTransport("http://peer")
        assert transport._route("GetExtendedAgentCard", {}) == (
            "GET",
            "/extendedAgentCard",
            None,
            None,
        )
        assert transport._route("GetTask", {"id": "t", "historyLength": 2, "tenant": None}) == (
            "GET",
            "/tasks/t",
            None,
            {"historyLength": "2"},
        )
        assert transport._route("ListTasks", {"includeArtifacts": True})[3] == {
            "includeArtifacts": "true"
        }
        with pytest.raises(A2AClientError) as err:
            transport._route("Nope", {})
        assert err.value.code == METHOD_NOT_FOUND

    async def test_rest_connection_error_and_unknown_stream_method(self) -> None:
        transport = HttpJsonClientTransport("http://peer", http_client=_mock(_refusing))
        with pytest.raises(A2ATransportUnavailableError):
            await transport.call("GetTask", {"id": "x"})
        with pytest.raises(A2ATransportUnavailableError):
            async for _ in transport.stream("SubscribeToTask", {"id": "x"}):
                pass
        with pytest.raises(A2AClientError) as err:
            async for _ in transport.stream("GetTask", {"id": "x"}):
                pass
        assert err.value.code == METHOD_NOT_FOUND

    async def test_rest_error_bodies(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/tasks/missing":
                return httpx.Response(404, json=http_error_body(TASK_NOT_FOUND, "gone"))
            if request.url.path == "/tasks/broken":
                return httpx.Response(500, content=b"not json")
            if request.url.path == "/tasks/teapot":
                return httpx.Response(418, json={"error": {"message": "short and stout"}})
            if request.url.path == "/tasks/empty":
                return httpx.Response(200, content=b"")
            return httpx.Response(200, json=[1, 2])

        transport = HttpJsonClientTransport("http://peer", http_client=_mock(handler))
        with pytest.raises(A2AClientError) as missing:
            await transport.call("GetTask", {"id": "missing"})
        assert missing.value.code == TASK_NOT_FOUND
        with pytest.raises(httpx.HTTPStatusError):
            await transport.call("GetTask", {"id": "broken"})
        with pytest.raises(A2AClientError) as teapot:
            await transport.call("GetTask", {"id": "teapot"})
        assert teapot.value.message == "short and stout"
        assert await transport.call("GetTask", {"id": "empty"}) == {}
        assert await transport.call("GetTask", {"id": "list"}) == {"result": [1, 2]}

    async def test_rest_stream_errors(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/tasks/missing:subscribe":
                return httpx.Response(404, json=http_error_body(TASK_NOT_FOUND, "gone"))
            if request.url.path == "/tasks/nowhere:subscribe":
                return httpx.Response(404, content=b"")
            if request.url.path == "/tasks/broken:subscribe":
                return httpx.Response(500, content=b"not json")
            body = (
                "data: not-json\n\n"
                'data: {"task": {"id": "t"}}\n\n'
                f"event: error\ndata: {json.dumps(http_error_body(UNSUPPORTED_OPERATION, 'done'))}\n\n"
                'data: {"task": {"id": "never"}}\n\n'
            )
            return httpx.Response(
                200, content=body.encode(), headers={"content-type": "text/event-stream"}
            )

        transport = HttpJsonClientTransport("http://peer", http_client=_mock(handler))
        with pytest.raises(A2AClientError) as missing:
            async for _ in transport.stream("SubscribeToTask", {"id": "missing"}):
                pass
        assert missing.value.code == TASK_NOT_FOUND
        with pytest.raises(A2ATransportUnavailableError):
            async for _ in transport.stream("SubscribeToTask", {"id": "nowhere"}):
                pass
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in transport.stream("SubscribeToTask", {"id": "broken"}):
                pass
        frames = [f async for f in transport.stream("SubscribeToTask", {"id": "ok"})]
        assert frames[0] == {"task": {"id": "t"}}
        assert frames[1]["error"]["code"] == UNSUPPORTED_OPERATION
        assert len(frames) == 2


class TestRestErrorBranches:
    async def test_internal_errors_get_a_correlation_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = _server()

        async def boom(method: str, params: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("boom")

        def boom_stream(method: str, params: dict[str, Any]) -> Any:
            async def gen() -> Any:
                raise RuntimeError("boom")
                yield {}  # pragma: no cover

            return gen()

        monkeypatch.setattr(server, "handle_v1", boom)
        monkeypatch.setattr(server, "handle_v1_stream", boom_stream)
        async with _http(server) as http:
            resp = await http.get("/tasks/x")
            assert resp.status_code == 500
            assert "correlation_id=" in resp.json()["error"]["message"]
            assert "boom" not in resp.text
            stream = await http.post("/message:stream", json=_user("hi"))
            frames = _sse(stream.text)
            assert "correlation_id=" in frames[0]["error"]["message"]
            assert "event: error" in stream.text

    async def test_bad_bodies_and_queries_on_every_route(self) -> None:
        async with _http(_server()) as http:
            bad = {"content": b"{", "headers": {"Content-Type": "application/json"}}
            for path in ("/message:stream", "/tasks/t:cancel", "/tasks/t/pushNotificationConfigs"):
                resp = await http.post(path, **bad)  # type: ignore[arg-type]
                assert resp.status_code == 400, path
                assert resp.json()["error"]["details"][0]["reason"] == "JSON_PARSE_ERROR"
            for path in ("/tasks/t", "/tasks", "/tasks/t/pushNotificationConfigs"):
                resp = await http.get(
                    path, params={"pageSize" if path != "/tasks/t" else "historyLength": "x"}
                )
                assert resp.status_code == 400, path
                assert resp.json()["error"]["details"][0]["reason"] == "INVALID_REQUEST"
