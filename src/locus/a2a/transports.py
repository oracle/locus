# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Client-side A2A v1.0 transports: one class per protocol binding.

:class:`~locus.a2a.A2AClient` speaks to a peer through whichever binding the
peer's Agent Card advertises. Each transport here turns a v1.0 method name and
its parameters into that binding's wire call and gives back the v1.0 result
as a plain ``dict``:

- :class:`JsonRpcClientTransport`: ``POST <url>`` with a JSON-RPC 2.0
  envelope; streaming methods read server-sent events whose ``data`` is a
  JSON-RPC response.
- :class:`HttpJsonClientTransport`: the HTTP+JSON (REST) binding,
  ``POST /message:send``, ``GET /tasks/{id}`` and so on; streaming methods
  read server-sent events whose ``data`` is a bare ``StreamResponse``.
- The gRPC transport lives in :mod:`locus.a2a.grpc_binding` because it needs
  the optional ``grpcio`` package.

:func:`select_interfaces` orders a card's ``supportedInterfaces`` by a
preference list; the client tries them in that order and moves on when a
binding turns out to be unreachable (:class:`A2ATransportUnavailableError`).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from locus.a2a.protocol_v1 import (
    A2A_VERSION_HEADER,
    code_from_error_details,
)
from locus.a2a.spec import INTERNAL_ERROR, METHOD_NOT_FOUND
from locus.a2a.spec_v1 import (
    A2A_JSON_MEDIA_TYPE,
    A2A_PROTOCOL_BINDINGS,
    A2A_V1_PROTOCOL_VERSION,
    BINDING_GRPC,
    BINDING_HTTP_JSON,
    BINDING_JSONRPC,
)
from locus.core.loop_bound import loop_bound
from locus.observability.propagation import inject_trace_headers


if TYPE_CHECKING:
    import httpx

    from locus.a2a.spec import AgentCard

_logger = logging.getLogger(__name__)

#: The order :class:`~locus.a2a.A2AClient` tries bindings when not told otherwise.
DEFAULT_BINDING_PREFERENCE: tuple[str, ...] = (BINDING_JSONRPC, BINDING_HTTP_JSON, BINDING_GRPC)


class A2AClientError(RuntimeError):
    """The peer answered with an A2A error.

    ``code`` is the JSON-RPC error code the specification assigns to the
    error, whichever binding carried it; ``message`` is the peer's text.
    The string form keeps the ``A2A error <code>: <message>`` shape earlier
    callers matched on.
    """

    def __init__(self, code: int, message: str, *, data: Any | None = None) -> None:
        super().__init__(f"A2A error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class A2ATransportUnavailableError(A2AClientError):
    """A binding could not be reached at all (connection refused, 404, 405, 501, UNIMPLEMENTED).

    The client treats it as "try the next interface", not as an answer.
    """


def select_interfaces(
    card: AgentCard,
    *,
    prefer: Sequence[str] = DEFAULT_BINDING_PREFERENCE,
    fallback_url: str | None = None,
) -> list[tuple[str, str]]:
    """``[(binding, url)]`` from a card's interfaces in preference order.

    Interfaces whose ``protocolVersion`` is not ``1.0`` and bindings not in
    ``prefer`` are left out. A card that advertises nothing usable falls back
    to JSON-RPC at ``fallback_url`` (or the card's ``url``), the binding every
    peer so far has served at its root.
    """
    wanted = [b.upper() for b in prefer]
    ranked: list[tuple[str, str]] = []
    for binding in wanted:
        for interface in card.supportedInterfaces:
            if interface.protocolBinding.upper() != binding:
                continue
            if interface.protocolVersion.split(".")[0] != A2A_V1_PROTOCOL_VERSION.split(".")[0]:
                continue
            pair = (binding, interface.url)
            if pair not in ranked:
                ranked.append(pair)
    if not ranked:
        ranked.append((BINDING_JSONRPC, fallback_url or card.url))
    return ranked


def normalise_binding(binding: str) -> str:
    """The canonical spelling of a binding name (``jsonrpc`` → ``JSONRPC``)."""
    upper = binding.upper().replace("_", "+").replace("REST", BINDING_HTTP_JSON)
    if upper == "HTTP-JSON":
        upper = BINDING_HTTP_JSON
    if upper not in A2A_PROTOCOL_BINDINGS:
        raise ValueError(
            f"unknown A2A protocol binding {binding!r}; one of {A2A_PROTOCOL_BINDINGS}"
        )
    return upper


class ClientTransport:
    """What every binding's client transport provides."""

    binding: str = ""
    url: str = ""

    async def call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: Any = None,  # noqa: ASYNC109 — forwarded to httpx, not an asyncio timer
    ) -> dict[str, Any]:
        """One unary v1.0 method; the result payload, or :class:`A2AClientError`.

        ``timeout`` overrides the transport's default for this call.
        """
        raise NotImplementedError

    def stream(
        self, method: str, params: dict[str, Any], *, timeout: Any = None
    ) -> AsyncIterator[dict[str, Any]]:
        """One streaming v1.0 method; yields ``StreamResponse`` payloads.

        A mid-stream error is yielded as ``{"error": {"code", "message"}}``
        and ends the stream, as the JSON-RPC client always did.
        """
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release the connection, when the transport owns one."""


class _HttpTransport(ClientTransport):
    """Shared httpx plumbing for the two HTTP bindings."""

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client = http_client
        self._owns_client = http_client is None

    def headers(self, **extra: str) -> dict[str, str]:
        """The headers of one call: version, bearer, ``extra`` and the trace context."""
        headers = {A2A_VERSION_HEADER: A2A_V1_PROTOCOL_VERSION}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers.update(extra)
        return inject_trace_headers(headers)

    def _build_client(self) -> httpx.AsyncClient:
        import httpx

        return httpx.AsyncClient(timeout=self._timeout)

    async def client(self) -> httpx.AsyncClient:
        """The transport's client, rebuilt when the running event loop changed.

        An injected client (``http_client=``) is used as given; an owned one
        is bound to the loop that built it, as every cached async client in
        Locus is (``locus.core.loop_bound``).
        """
        if not self._owns_client and self._client is not None:
            return self._client
        return loop_bound(self, "_client", self._build_client)

    @asynccontextmanager
    async def _client_for(
        self,
        timeout: Any,  # noqa: ASYNC109 — an httpx timeout, not an asyncio timer
    ) -> AsyncIterator[httpx.AsyncClient]:
        """The client for one call.

        A per-call ``timeout`` on a transport that owns its client gets a
        client of its own, built with that timeout, so the override reaches
        every layer of httpx; a shared (injected) client is used as is.
        """
        if timeout is None or not self._owns_client:
            yield await self.client()
            return
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            yield client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _unavailable_status(status_code: int) -> bool:
        return status_code in (404, 405, 501)

    @staticmethod
    def _timeout_kw(timeout: Any) -> dict[str, Any]:
        return {} if timeout is None else {"timeout": timeout}

    @staticmethod
    async def _sse_frames(response: Any) -> AsyncIterator[str]:
        """``data`` payloads of a server-sent-event stream, one per event."""
        chunks: list[str] = []
        async for raw in response.aiter_lines():
            line = raw.rstrip("\r\n")
            if not line:
                if chunks:
                    yield "\n".join(chunks)
                chunks = []
                continue
            if line.startswith(":"):
                continue
            key, _, value = line.partition(":")
            if key == "data":
                chunks.append(value.removeprefix(" "))
        if chunks:
            yield "\n".join(chunks)


class JsonRpcClientTransport(_HttpTransport):
    """The JSON-RPC 2.0 binding: every method is a ``POST`` to one URL."""

    binding = BINDING_JSONRPC

    def _envelope(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params}

    async def call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: Any = None,  # noqa: ASYNC109 — forwarded to httpx, not an asyncio timer
    ) -> dict[str, Any]:
        import httpx

        from locus.observability.emit import EV_A2A_CLIENT_RECEIVED, EV_A2A_CLIENT_SEND, emit

        await emit(EV_A2A_CLIENT_SEND, target_url=self.url, method=method, binding=self.binding)
        started = time.perf_counter()
        try:
            async with self._client_for(timeout) as client:
                resp = await client.post(
                    f"{self.url}/",
                    json=self._envelope(method, params),
                    headers=self.headers(),
                    **self._timeout_kw(timeout),
                )
        except httpx.TransportError as e:
            await emit(
                EV_A2A_CLIENT_RECEIVED,
                target_url=self.url,
                method=method,
                binding=self.binding,
                status_code=None,
                duration_ms=(time.perf_counter() - started) * 1000,
                content_length=0,
                error=type(e).__name__,
            )
            raise A2ATransportUnavailableError(INTERNAL_ERROR, f"{self.url}: {e}") from e
        await emit(
            EV_A2A_CLIENT_RECEIVED,
            target_url=self.url,
            method=method,
            binding=self.binding,
            status_code=resp.status_code,
            duration_ms=(time.perf_counter() - started) * 1000,
            content_length=len(resp.content),
            **({"error": "HTTPStatusError"} if resp.is_error else {}),
        )
        if self._unavailable_status(resp.status_code):
            raise A2ATransportUnavailableError(
                METHOD_NOT_FOUND, f"{self.url} answered HTTP {resp.status_code}"
            )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            err = data["error"]
            raise A2AClientError(
                int(err.get("code", INTERNAL_ERROR)),
                str(err.get("message", "")),
                data=err.get("data"),
            )
        result = data.get("result", {})
        return result if isinstance(result, dict) else {"result": result}

    async def stream(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: Any = None,  # noqa: ASYNC109 — forwarded to httpx, not an asyncio timer
    ) -> AsyncIterator[dict[str, Any]]:
        import httpx

        headers = self.headers(Accept="text/event-stream")
        try:
            async with (
                self._client_for(timeout) as client,
                client.stream(
                    "POST",
                    f"{self.url}/",
                    json=self._envelope(method, params),
                    headers=headers,
                    **self._timeout_kw(timeout),
                ) as resp,
            ):
                if self._unavailable_status(resp.status_code):
                    raise A2ATransportUnavailableError(
                        METHOD_NOT_FOUND, f"{self.url} answered HTTP {resp.status_code}"
                    )
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                if content_type.startswith("application/json") or "+json" in content_type:
                    # A refusal before the stream opened: one JSON-RPC error
                    # body instead of events.
                    await resp.aread()
                    try:
                        env = resp.json()
                    except json.JSONDecodeError:
                        env = {}
                    if isinstance(env, dict) and "error" in env:
                        yield {"error": env["error"]}
                    return
                async for payload in self._sse_frames(resp):
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        env = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if "error" in env:
                        yield {"error": env["error"]}
                        break
                    yield env.get("result", env)
        except httpx.TransportError as e:
            raise A2ATransportUnavailableError(INTERNAL_ERROR, f"{self.url}: {e}") from e


class HttpJsonClientTransport(_HttpTransport):
    """The HTTP+JSON binding: resource paths, verbs and bare JSON bodies."""

    binding = BINDING_HTTP_JSON

    @staticmethod
    def _query(params: dict[str, Any], *drop: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in params.items():
            if key in drop or value is None:
                continue
            out[key] = str(value).lower() if isinstance(value, bool) else str(value)
        return out

    def _route(
        self, method: str, params: dict[str, Any]
    ) -> tuple[str, str, dict[str, Any] | None, dict[str, str] | None]:
        """``(verb, path, json_body, query)`` for a unary method."""
        task_id = params.get("id") or params.get("taskId")
        if method == "SendMessage":
            return "POST", "/message:send", params, None
        if method == "GetTask":
            return "GET", f"/tasks/{task_id}", None, self._query(params, "id")
        if method == "ListTasks":
            return "GET", "/tasks", None, self._query(params)
        if method == "CancelTask":
            return "POST", f"/tasks/{task_id}:cancel", params, None
        if method == "CreateTaskPushNotificationConfig":
            return "POST", f"/tasks/{params.get('taskId')}/pushNotificationConfigs", params, None
        if method == "GetTaskPushNotificationConfig":
            return (
                "GET",
                f"/tasks/{params.get('taskId')}/pushNotificationConfigs/{params.get('id')}",
                None,
                self._query(params, "id", "taskId"),
            )
        if method == "ListTaskPushNotificationConfigs":
            return (
                "GET",
                f"/tasks/{params.get('taskId')}/pushNotificationConfigs",
                None,
                self._query(params, "taskId"),
            )
        if method == "DeleteTaskPushNotificationConfig":
            return (
                "DELETE",
                f"/tasks/{params.get('taskId')}/pushNotificationConfigs/{params.get('id')}",
                None,
                self._query(params, "id", "taskId"),
            )
        if method == "GetExtendedAgentCard":
            return "GET", "/extendedAgentCard", None, None
        raise A2AClientError(METHOD_NOT_FOUND, f"unknown method {method!r}")

    def _raise_for_error_body(self, status_code: int, body: Any) -> None:
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            code = code_from_error_details(error.get("details"), default=INTERNAL_ERROR)
            raise A2AClientError(code, str(error.get("message", f"HTTP {status_code}")), data=error)

    async def call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: Any = None,  # noqa: ASYNC109 — forwarded to httpx, not an asyncio timer
    ) -> dict[str, Any]:
        import httpx

        from locus.observability.emit import EV_A2A_CLIENT_RECEIVED, EV_A2A_CLIENT_SEND, emit

        verb, path, body, query = self._route(method, params)
        await emit(EV_A2A_CLIENT_SEND, target_url=self.url, method=method, binding=self.binding)
        started = time.perf_counter()
        content_type = {"Content-Type": A2A_JSON_MEDIA_TYPE} if body is not None else {}
        try:
            async with self._client_for(timeout) as client:
                resp = await client.request(
                    verb,
                    f"{self.url}{path}",
                    json=body,
                    params=query,
                    headers=self.headers(**content_type),
                    **self._timeout_kw(timeout),
                )
        except httpx.TransportError as e:
            await emit(
                EV_A2A_CLIENT_RECEIVED,
                target_url=self.url,
                method=method,
                binding=self.binding,
                status_code=None,
                duration_ms=(time.perf_counter() - started) * 1000,
                content_length=0,
                error=type(e).__name__,
            )
            raise A2ATransportUnavailableError(INTERNAL_ERROR, f"{self.url}: {e}") from e
        await emit(
            EV_A2A_CLIENT_RECEIVED,
            target_url=self.url,
            method=method,
            binding=self.binding,
            status_code=resp.status_code,
            duration_ms=(time.perf_counter() - started) * 1000,
            content_length=len(resp.content),
            **({"error": "HTTPStatusError"} if resp.is_error else {}),
        )
        payload: Any
        try:
            payload = resp.json() if resp.content else {}
        except json.JSONDecodeError:
            payload = {}
        if resp.is_error:
            if isinstance(payload, dict) and "error" in payload:
                self._raise_for_error_body(resp.status_code, payload)
            if self._unavailable_status(resp.status_code):
                raise A2ATransportUnavailableError(
                    METHOD_NOT_FOUND, f"{self.url}{path} answered HTTP {resp.status_code}"
                )
            resp.raise_for_status()
        return payload if isinstance(payload, dict) else {"result": payload}

    async def stream(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: Any = None,  # noqa: ASYNC109 — forwarded to httpx, not an asyncio timer
    ) -> AsyncIterator[dict[str, Any]]:
        import httpx

        if method == "SendStreamingMessage":
            verb, path, body = "POST", "/message:stream", params
        elif method == "SubscribeToTask":
            verb, path, body = "POST", f"/tasks/{params.get('id')}:subscribe", None
        else:
            raise A2AClientError(METHOD_NOT_FOUND, f"unknown streaming method {method!r}")
        headers = self.headers(Accept="text/event-stream")
        if body is not None:
            headers["Content-Type"] = A2A_JSON_MEDIA_TYPE
        try:
            async with (
                self._client_for(timeout) as client,
                client.stream(
                    verb,
                    f"{self.url}{path}",
                    json=body,
                    headers=headers,
                    **self._timeout_kw(timeout),
                ) as resp,
            ):
                if resp.is_error:
                    await resp.aread()
                    try:
                        payload = resp.json()
                    except json.JSONDecodeError:
                        payload = {}
                    if isinstance(payload, dict) and "error" in payload:
                        self._raise_for_error_body(resp.status_code, payload)
                    if self._unavailable_status(resp.status_code):
                        raise A2ATransportUnavailableError(
                            METHOD_NOT_FOUND, f"{self.url}{path} answered HTTP {resp.status_code}"
                        )
                    resp.raise_for_status()
                async for payload_text in self._sse_frames(resp):
                    try:
                        frame = json.loads(payload_text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(frame, dict) and "error" in frame:
                        error = frame["error"]
                        code = code_from_error_details(error.get("details"), default=INTERNAL_ERROR)
                        yield {"error": {"code": code, "message": error.get("message", "")}}
                        break
                    yield frame
        except httpx.TransportError as e:
            raise A2ATransportUnavailableError(INTERNAL_ERROR, f"{self.url}: {e}") from e


def make_transport(
    binding: str,
    url: str,
    *,
    api_key: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    http_client: httpx.AsyncClient | None = None,
    grpc_channel: Any | None = None,
) -> ClientTransport:
    """A client transport for ``binding`` at ``url``.

    gRPC needs the ``a2a-grpc`` extra; asking for it without ``grpcio``
    installed raises ``ImportError`` that names the extra.
    """
    canonical = normalise_binding(binding)
    if canonical == BINDING_JSONRPC:
        return JsonRpcClientTransport(
            url, api_key=api_key, timeout=timeout, http_client=http_client
        )
    if canonical == BINDING_HTTP_JSON:
        return HttpJsonClientTransport(
            url, api_key=api_key, timeout=timeout, http_client=http_client
        )
    from locus.a2a.grpc_binding import GrpcClientTransport

    return GrpcClientTransport(
        url,
        api_key=api_key,
        timeout=timeout if isinstance(timeout, int | float) else None,
        channel=grpc_channel,
    )


__all__ = [
    "DEFAULT_BINDING_PREFERENCE",
    "A2AClientError",
    "A2ATransportUnavailableError",
    "ClientTransport",
    "HttpJsonClientTransport",
    "JsonRpcClientTransport",
    "make_transport",
    "normalise_binding",
    "select_interfaces",
]
