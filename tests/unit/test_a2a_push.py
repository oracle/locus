# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A2A push notifications: signing, delivery, receiving, and the server's config methods."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import pytest

from locus.a2a import (
    A2AClient,
    A2AClientError,
    A2AServer,
    A2AV1TaskPushNotificationConfig,
    Message,
    PushNotificationReceiver,
    PushNotificationSettings,
    PushNotificationSigner,
    TaskState,
    TextPart,
)
from locus.a2a.push import (
    JWKS_PATH,
    NOTIFICATION_TOKEN_HEADER,
    SIGNATURE_HEADER,
    PushNotificationDispatcher,
    PushNotificationError,
    PushNotificationStore,
    PushVerificationError,
    validate_push_url,
    verify_signature,
)
from locus.a2a.spec import INVALID_PARAMS, PUSH_NOTIFICATION_NOT_SUPPORTED, TASK_NOT_FOUND
from locus.core.events import TerminateEvent, ThinkEvent


pytest.importorskip("cryptography")


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


def _receiver_client(receiver: PushNotificationReceiver) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=receiver.app), base_url="http://receiver"
    )


def _server_client(server: A2AServer) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://testserver"
    )


def _message(text: str = "hi") -> Message:
    return Message(role="user", parts=[TextPart(text=text)], messageId="m")


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


class TestSigner:
    def test_sign_and_verify_round_trip(self) -> None:
        signer = PushNotificationSigner.generate(kid="k1")
        body = b'{"statusUpdate": {"taskId": "t1"}}'
        jws = signer.sign(body, task_id="t1", now=1_000_000)
        claims = verify_signature(jws, body, jwks=signer.jwks(), task_id="t1", now=1_000_010)
        assert claims["taskId"] == "t1"
        assert claims["iat"] == 1_000_000
        jwk = signer.public_jwk()
        assert (jwk["kty"], jwk["crv"], jwk["alg"], jwk["kid"]) == ("OKP", "Ed25519", "EdDSA", "k1")

    def test_repr_hides_the_key(self) -> None:
        signer = PushNotificationSigner.generate(kid="k1")
        assert repr(signer) == "PushNotificationSigner(kid='k1')"

    def test_tampered_body_unknown_key_expiry_and_task(self) -> None:
        signer = PushNotificationSigner.generate(kid="k1", ttl=60)
        body = b'{"a": 1}'
        jws = signer.sign(body, task_id="t1", now=100)
        with pytest.raises(PushVerificationError, match="body hash"):
            verify_signature(jws, b'{"a": 2}', jwks=signer.jwks(), now=110)
        with pytest.raises(PushVerificationError, match="unknown signing key"):
            verify_signature(jws, body, jwks={"keys": []}, now=110)
        with pytest.raises(PushVerificationError, match="expired"):
            verify_signature(jws, body, jwks=signer.jwks(), now=1_000)
        with pytest.raises(PushVerificationError, match="another task"):
            verify_signature(jws, body, jwks=signer.jwks(), task_id="t2", now=110)
        other = PushNotificationSigner.generate(kid="k1")
        with pytest.raises(PushVerificationError, match="does not verify"):
            verify_signature(jws, body, jwks=other.jwks(), now=110)
        with pytest.raises(PushVerificationError, match="malformed"):
            verify_signature("nope", body, jwks=signer.jwks())

    def test_from_pem(self) -> None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519, rsa

        key = ed25519.Ed25519PrivateKey.generate()
        pem = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        signer = PushNotificationSigner.from_pem(pem.decode(), kid="pem")
        assert signer.kid == "pem"
        rsa_pem = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        with pytest.raises(TypeError, match="Ed25519"):
            PushNotificationSigner.from_pem(rsa_pem)


# ---------------------------------------------------------------------------
# URL policy, store, dispatcher
# ---------------------------------------------------------------------------


class TestUrlPolicy:
    def test_private_hosts_are_refused_unless_allowed(self) -> None:
        with pytest.raises(PushNotificationError, match="not a public address"):
            validate_push_url("http://127.0.0.1:9/hook")
        with pytest.raises(PushNotificationError, match="not a public address"):
            validate_push_url("http://10.1.2.3/hook")
        validate_push_url("http://127.0.0.1:9/hook", allow_private=True)
        validate_push_url("https://8.8.8.8/hook")

    def test_schemes_and_hosts(self) -> None:
        with pytest.raises(PushNotificationError, match="scheme"):
            validate_push_url("ftp://example.com/x")
        with pytest.raises(PushNotificationError, match="no host"):
            validate_push_url("http:///x")
        with pytest.raises(PushNotificationError, match="not a public address"):
            validate_push_url("http://host.that.does.not.resolve.invalid/x")


class TestStore:
    def test_set_get_list_delete(self) -> None:
        store = PushNotificationStore()
        stored = store.set("t1", A2AV1TaskPushNotificationConfig(url="https://h/x"))
        assert stored.id
        assert stored.taskId == "t1"
        assert store.get("t1", str(stored.id)) == stored
        named = store.set(
            "t1",
            A2AV1TaskPushNotificationConfig(id="c2", url="https://h/y", token="tok"),  # noqa: S106
        )
        assert [c.id for c in store.list("t1")] == [stored.id, "c2"]
        assert store.delete("t1", "c2") is True
        assert store.delete("t1", "c2") is False
        assert named.token == "tok"  # noqa: S105
        store.clear("t1")
        assert store.list("t1") == []


class TestDispatcher:
    async def test_headers_retry_and_logging_never_leak_secrets(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        seen: list[httpx.Request] = []
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(503)
            return httpx.Response(202)

        signer = PushNotificationSigner.generate(kid="k1")
        settings = PushNotificationSettings(
            allow_private_urls=True,
            signer=signer,
            backoff=0.0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        store = PushNotificationStore()
        dispatcher = PushNotificationDispatcher(store, settings)
        dispatcher.register(
            "t1",
            A2AV1TaskPushNotificationConfig(
                url="http://127.0.0.1:9/hook",
                token="secret-token",  # noqa: S106
                authentication={"scheme": "Bearer", "credentials": "secret-cred"},  # type: ignore[arg-type]
            ),
        )
        with caplog.at_level(logging.INFO, logger="locus.a2a.push"):
            dispatcher.notify(
                "t1",
                {
                    "statusUpdate": {
                        "taskId": "t1",
                        "contextId": "c",
                        "status": {"state": "TASK_STATE_WORKING"},
                    }
                },
            )
            await dispatcher.drain()
        assert len(seen) == 2
        request = seen[-1]
        assert request.headers["content-type"] == "application/a2a+json"
        assert request.headers[NOTIFICATION_TOKEN_HEADER] == "secret-token"
        assert request.headers["authorization"] == "Bearer secret-cred"
        body = request.content
        verify_signature(request.headers[SIGNATURE_HEADER], body, jwks=signer.jwks(), task_id="t1")
        assert json.loads(body)["statusUpdate"]["taskId"] == "t1"
        text = caplog.text
        assert "secret-token" not in text
        assert "secret-cred" not in text
        assert request.headers[SIGNATURE_HEADER] not in text
        assert "attempt 1/3 failed" in text
        assert "delivered" in text
        await dispatcher.aclose()

    async def test_transport_errors_are_logged_not_raised(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        settings = PushNotificationSettings(
            allow_private_urls=True,
            max_attempts=2,
            backoff=0.0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        dispatcher = PushNotificationDispatcher(PushNotificationStore(), settings)
        dispatcher.register("t1", A2AV1TaskPushNotificationConfig(url="http://127.0.0.1:9/hook"))
        dispatcher.notify("t1", {"task": {"id": "t1"}})
        dispatcher.notify("t-no-config", {"task": {"id": "t-no-config"}})
        with caplog.at_level(logging.WARNING, logger="locus.a2a.push"):
            await dispatcher.drain()
        assert caplog.text.count("ConnectError") == 2

    def test_register_applies_the_url_policy(self) -> None:
        dispatcher = PushNotificationDispatcher(PushNotificationStore(), PushNotificationSettings())
        with pytest.raises(PushNotificationError):
            dispatcher.register("t1", A2AV1TaskPushNotificationConfig(url="http://localhost/hook"))

    async def test_owned_client_is_created_and_closed(self) -> None:
        dispatcher = PushNotificationDispatcher(PushNotificationStore(), PushNotificationSettings())
        client = await dispatcher._client_for()
        assert isinstance(client, httpx.AsyncClient)
        await dispatcher.aclose()
        assert dispatcher._client is None


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------


class TestReceiver:
    async def test_token_check_and_queue(self) -> None:
        receiver = PushNotificationReceiver(token="tok")  # noqa: S106
        async with _receiver_client(receiver) as http:
            bad = await http.post(
                "/a2a/push",
                json={"task": {"id": "t1"}},
                headers={NOTIFICATION_TOKEN_HEADER: "nope"},
            )
            assert bad.status_code == 401
            not_json = await http.post(
                "/a2a/push", content=b"x", headers={NOTIFICATION_TOKEN_HEADER: "tok"}
            )
            assert not_json.status_code == 401
            no_task = await http.post(
                "/a2a/push",
                json={"message": {"messageId": "m"}},
                headers={NOTIFICATION_TOKEN_HEADER: "tok"},
            )
            assert no_task.status_code == 401
            array = await http.post(
                "/a2a/push", json=[1], headers={NOTIFICATION_TOKEN_HEADER: "tok"}
            )
            assert array.status_code == 401
            wrong_path = await http.post("/elsewhere", json={})
            assert wrong_path.status_code == 404
            ok = await http.post(
                "/a2a/push",
                json={
                    "statusUpdate": {
                        "taskId": "t1",
                        "contextId": "c",
                        "status": {"state": "TASK_STATE_COMPLETED"},
                    }
                },
                headers={NOTIFICATION_TOKEN_HEADER: "tok"},
            )
            assert ok.status_code == 202
        notification = await receiver.next(timeout=1)
        assert notification.task_id == "t1"
        assert notification.kind == "statusUpdate"
        assert notification.state == "TASK_STATE_COMPLETED"
        assert notification.claims is None
        assert receiver.received == [notification]

    async def test_signature_required_when_keys_are_configured(self) -> None:
        signer = PushNotificationSigner.generate(kid="k1")
        receiver = PushNotificationReceiver(jwks=signer.jwks())
        body = json.dumps(
            {"task": {"id": "t1", "contextId": "c", "status": {"state": "TASK_STATE_WORKING"}}}
        ).encode()
        with pytest.raises(PushVerificationError, match="not signed"):
            await receiver.verify({}, body)
        good = await receiver.verify({SIGNATURE_HEADER: signer.sign(body, task_id="t1")}, body)
        assert good.claims is not None
        assert good.claims["taskId"] == "t1"
        assert SIGNATURE_HEADER.lower() not in good.headers
        assert good.state == "TASK_STATE_WORKING"

    async def test_jwks_fetcher_refreshes_once_on_unknown_key(self) -> None:
        old = PushNotificationSigner.generate(kid="old")
        new = PushNotificationSigner.generate(kid="new")
        served = [old.jwks(), new.jwks()]

        async def fetch() -> dict[str, Any]:
            return served.pop(0) if len(served) > 1 else served[0]

        receiver = PushNotificationReceiver(jwks_fetcher=fetch)
        body = b'{"task": {"id": "t1"}}'
        await receiver.verify({SIGNATURE_HEADER: new.sign(body, task_id="t1")}, body)
        assert receiver.requires_signature

    async def test_events_iterator_and_lifespan(self) -> None:
        receiver = PushNotificationReceiver()
        messages = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]
        sent: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return messages.pop(0)

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        await receiver({"type": "lifespan"}, receive, send)
        assert [m["type"] for m in sent] == [
            "lifespan.startup.complete",
            "lifespan.shutdown.complete",
        ]
        await receiver({"type": "websocket"}, receive, send)
        receiver._queue.put_nowait(await receiver.verify({}, b'{"task": {"id": "t9"}}'))
        async for event in receiver.events():
            assert event.task_id == "t9"
            break
        with pytest.raises(TimeoutError):
            await receiver.next(timeout=0.01)


# ---------------------------------------------------------------------------
# Server: push config methods and delivery
# ---------------------------------------------------------------------------


class TestServerPush:
    async def test_end_to_end_signed_delivery_over_jsonrpc(self) -> None:
        signer = PushNotificationSigner.generate(kid="agent-1")
        receiver = PushNotificationReceiver(token="tok", jwks=signer.jwks())  # noqa: S106
        async with _receiver_client(receiver) as hook_client:
            server = A2AServer(
                agent=_agent(),
                allow_unauthenticated=True,
                url="http://testserver",
                push_notifications=PushNotificationSettings(
                    allow_private_urls=True, signer=signer, http_client=hook_client
                ),
            )
            async with _server_client(server) as http:
                card = (await http.get("/.well-known/agent-card.json")).json()
                assert card["capabilities"]["pushNotifications"] is True
                jwks = (await http.get(JWKS_PATH)).json()
                assert jwks["keys"][0]["kid"] == "agent-1"

                client = A2AClient("http://testserver", http_client=http)
                task = await client.send_message(
                    _message(),
                    push_notification_config={"url": "http://127.0.0.1:1/a2a/push", "token": "tok"},
                )
                assert task.status.state == TaskState.completed
                await server.drain()
        kinds = [n.kind for n in receiver.received]
        assert kinds[0] == "statusUpdate"
        assert "artifactUpdate" in kinds
        assert receiver.received[-1].state == "TASK_STATE_COMPLETED"
        assert all(n.task_id == task.id for n in receiver.received)
        assert all(n.claims and n.claims["taskId"] == task.id for n in receiver.received)

    async def test_non_blocking_send_runs_in_background_and_pushes(self) -> None:
        receiver = PushNotificationReceiver(token="tok")  # noqa: S106
        async with _receiver_client(receiver) as hook_client:
            server = A2AServer(
                agent=_agent(),
                allow_unauthenticated=True,
                push_notifications=PushNotificationSettings(
                    allow_private_urls=True, http_client=hook_client
                ),
            )
            async with _server_client(server) as http:
                client = A2AClient("http://testserver", binding="http+json", http_client=http)
                task = await client.send_message(
                    _message(),
                    push_notification_config={"url": "http://127.0.0.1:1/a2a/push", "token": "tok"},
                    blocking=False,
                )
                assert task.status.state in (TaskState.submitted, TaskState.working)
                final = await asyncio.wait_for(_final(receiver), 5)
                assert final.state == "TASK_STATE_COMPLETED"
                await server.drain()
                fetched = await client.get_task(task.id)
                assert fetched.status.state == TaskState.completed

    @pytest.mark.parametrize("binding", ["jsonrpc", "http+json"])
    async def test_config_crud_over_both_http_bindings(self, binding: str) -> None:
        server = A2AServer(agent=_agent(), allow_unauthenticated=True, push_notifications=True)
        async with _server_client(server) as http:
            client = A2AClient("http://testserver", binding=binding, http_client=http)
            task = await client.send_message(_message())
            with pytest.raises(A2AClientError) as private:
                await client.set_push_notification_config(task.id, {"url": "http://127.0.0.1:1/x"})
            assert private.value.code == INVALID_PARAMS
            stored = await client.set_push_notification_config(
                task.id,
                A2AV1TaskPushNotificationConfig(id="c1", url="https://8.8.8.8/a2a"),
            )
            assert stored.id == "c1"
            assert stored.taskId == task.id
            assert (
                await client.get_push_notification_config(task.id, "c1")
            ).url == "https://8.8.8.8/a2a"
            assert [c.id for c in await client.list_push_notification_configs(task.id)] == ["c1"]
            await client.delete_push_notification_config(task.id, "c1")
            await client.delete_push_notification_config(task.id, "c1")  # idempotent
            assert await client.list_push_notification_configs(task.id) == []
            with pytest.raises(A2AClientError) as missing:
                await client.get_push_notification_config(task.id, "c1")
            assert missing.value.code == TASK_NOT_FOUND
            with pytest.raises(A2AClientError) as no_task:
                await client.list_push_notification_configs("nope")
            assert no_task.value.code == TASK_NOT_FOUND

    async def test_push_disabled_answers_not_supported(self) -> None:
        server = A2AServer(agent=_agent(), allow_unauthenticated=True)
        async with _server_client(server) as http:
            client = A2AClient("http://testserver", http_client=http)
            task = await client.send_message(_message())
            with pytest.raises(A2AClientError) as err:
                await client.set_push_notification_config(task.id, {"url": "https://h/x"})
            assert err.value.code == PUSH_NOTIFICATION_NOT_SUPPORTED
            with pytest.raises(A2AClientError) as in_send:
                await client.send_message(
                    _message(), push_notification_config={"url": "https://h/x"}
                )
            assert in_send.value.code == PUSH_NOTIFICATION_NOT_SUPPORTED

    async def test_list_configs_pages_and_rejects_bad_tokens(self) -> None:
        server = A2AServer(agent=_agent(), allow_unauthenticated=True, push_notifications=True)
        async with _server_client(server) as http:
            client = A2AClient("http://testserver", http_client=http)
            task = await client.send_message(_message())
            for i in range(3):
                await client.set_push_notification_config(
                    task.id, {"id": f"c{i}", "url": f"https://8.8.8.8/{i}"}
                )
            page = await server.handle_v1(
                "ListTaskPushNotificationConfigs", {"taskId": task.id, "pageSize": 2}
            )
            assert [c["id"] for c in page["configs"]] == ["c0", "c1"]
            assert page["nextPageToken"] == "2"
            rest = await server.handle_v1(
                "ListTaskPushNotificationConfigs",
                {"taskId": task.id, "pageSize": 2, "pageToken": "2"},
            )
            assert [c["id"] for c in rest["configs"]] == ["c2"]
            assert rest["nextPageToken"] == ""
            with pytest.raises(Exception, match="invalid pageToken"):
                await server.handle_v1(
                    "ListTaskPushNotificationConfigs", {"taskId": task.id, "pageToken": "x"}
                )
            with pytest.raises(Exception, match="taskId is required"):
                await server.handle_v1("CreateTaskPushNotificationConfig", {"url": "https://h/x"})
            with pytest.raises(Exception, match="invalid params"):
                await server.handle_v1("CreateTaskPushNotificationConfig", {"taskId": task.id})
            with pytest.raises(Exception, match="invalid params"):
                await server.handle_v1("GetTaskPushNotificationConfig", {"taskId": task.id})
            with pytest.raises(Exception, match="invalid params"):
                await server.handle_v1("DeleteTaskPushNotificationConfig", {"taskId": task.id})
            with pytest.raises(Exception, match="invalid params"):
                await server.handle_v1("ListTaskPushNotificationConfigs", {})
            with pytest.raises(Exception, match="unknown method"):
                await server.handle_v1("Nope", {})
            with pytest.raises(Exception, match="unknown method"):
                async for _ in server.handle_v1_stream("Nope", {}):
                    pass


async def _final(receiver: PushNotificationReceiver) -> Any:
    async for notification in receiver.events():
        if notification.state in ("TASK_STATE_COMPLETED", "TASK_STATE_FAILED"):
            return notification
    return None  # pragma: no cover
