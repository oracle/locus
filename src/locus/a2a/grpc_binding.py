# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The A2A v1.0 gRPC binding: ``lf.a2a.v1.A2AService`` on both ends of the wire.

The specification's protobuf definition ships with Locus as
``proto/a2a_v1.proto`` and its compiled descriptor set ``proto/a2a_v1.desc``.
The descriptors are loaded at run time into a private pool (so a process that
also imports another implementation's generated code sees no symbol clash) and
message classes are built from them; there is no generated Python to keep in
step with a protobuf release. Payloads cross between the binding and the
server's handlers as the same camelCase JSON the HTTP+JSON binding uses, via
``google.protobuf.json_format``.

Server side, :class:`A2AGrpcService` exposes an :class:`~locus.a2a.A2AServer`
through a generic handler (``A2AServer.grpc_server()`` builds the
``grpc.aio`` server for it). Client side, :class:`GrpcClientTransport` is the
transport :class:`~locus.a2a.A2AClient` picks for an interface whose
``protocolBinding`` is ``GRPC``. Errors carry a ``google.rpc.Status`` with the
specification's ``ErrorInfo`` detail in the trailing metadata, as the other
bindings carry it in the body.

Needs the ``a2a-grpc`` extra (``grpcio``, ``grpcio-status``, ``protobuf``);
nothing here is imported until a gRPC interface is asked for.
"""

from __future__ import annotations

import hmac
import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping
from functools import cache
from importlib import resources
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from locus.a2a.protocol_v1 import (
    A2A_ERROR_DOMAIN,
    A2A_VERSION_HEADER,
    A2AV1ProtocolError,
    code_from_error_details,
    error_mapping,
)
from locus.a2a.spec import INTERNAL_ERROR, METHOD_NOT_FOUND, VERSION_NOT_SUPPORTED
from locus.a2a.spec_v1 import A2A_GRPC_SERVICE, A2A_V1_PROTOCOL_VERSION, BINDING_GRPC
from locus.a2a.transports import A2AClientError, A2ATransportUnavailableError, ClientTransport
from locus.observability.propagation import inject_trace_headers, trace_scope


if TYPE_CHECKING:
    from locus.a2a.protocol import A2AServer

_logger = logging.getLogger(__name__)

_PACKAGE = "lf.a2a.v1"
_DESCRIPTOR_FILE = "a2a_v1.desc"

#: ``(request message, response message, server streaming)`` per RPC.
RPC_SIGNATURES: dict[str, tuple[str, str, bool]] = {
    "SendMessage": ("SendMessageRequest", "SendMessageResponse", False),
    "SendStreamingMessage": ("SendMessageRequest", "StreamResponse", True),
    "GetTask": ("GetTaskRequest", "Task", False),
    "ListTasks": ("ListTasksRequest", "ListTasksResponse", False),
    "CancelTask": ("CancelTaskRequest", "Task", False),
    "SubscribeToTask": ("SubscribeToTaskRequest", "StreamResponse", True),
    "CreateTaskPushNotificationConfig": (
        "TaskPushNotificationConfig",
        "TaskPushNotificationConfig",
        False,
    ),
    "GetTaskPushNotificationConfig": (
        "GetTaskPushNotificationConfigRequest",
        "TaskPushNotificationConfig",
        False,
    ),
    "ListTaskPushNotificationConfigs": (
        "ListTaskPushNotificationConfigsRequest",
        "ListTaskPushNotificationConfigsResponse",
        False,
    ),
    "DeleteTaskPushNotificationConfig": (
        "DeleteTaskPushNotificationConfigRequest",
        "google.protobuf.Empty",
        False,
    ),
    "GetExtendedAgentCard": ("GetExtendedAgentCardRequest", "AgentCard", False),
}


def _require_grpc() -> Any:
    try:
        import grpc
    except ImportError as e:
        msg = "the A2A gRPC binding needs grpcio: pip install 'locus-sdk[a2a-grpc]'"
        raise ImportError(msg) from e
    return grpc


class V1Messages:
    """The v1.0 protobuf message classes, built from the shipped descriptor set."""

    def __init__(self) -> None:
        from google.protobuf import (
            descriptor_pb2,
            descriptor_pool,
            empty_pb2,
            message_factory,
            struct_pb2,
            timestamp_pb2,
        )

        raw = resources.files("locus.a2a.proto").joinpath(_DESCRIPTOR_FILE).read_bytes()
        file_set = descriptor_pb2.FileDescriptorSet.FromString(raw)
        pool = descriptor_pool.DescriptorPool()
        for well_known in (struct_pb2, timestamp_pb2, empty_pb2):
            pool.AddSerializedFile(well_known.DESCRIPTOR.serialized_pb)
        for file_proto in file_set.file:
            pool.AddSerializedFile(file_proto.SerializeToString())
        self._pool = pool
        self._factory = message_factory
        self._classes: dict[str, Any] = {}
        self.service = pool.FindServiceByName(A2A_GRPC_SERVICE)

    def __getitem__(self, name: str) -> Any:
        """The message class for ``name`` (short, or fully qualified)."""
        full = name if "." in name else f"{_PACKAGE}.{name}"
        cls = self._classes.get(full)
        if cls is None:
            cls = self._factory.GetMessageClass(self._pool.FindMessageTypeByName(full))
            self._classes[full] = cls
        return cls

    def to_dict(self, message: Any) -> dict[str, Any]:
        """A message as the camelCase JSON the other bindings use."""
        from google.protobuf import json_format

        return json_format.MessageToDict(message)

    def from_dict(self, data: Mapping[str, Any], name: str) -> Any:
        """A ``name`` message from camelCase JSON; unknown keys are dropped."""
        from google.protobuf import json_format

        return json_format.ParseDict(dict(data), self[name](), ignore_unknown_fields=True)


@cache
def v1_messages() -> V1Messages:
    """The process-wide :class:`V1Messages` (descriptors are loaded once)."""
    return V1Messages()


def _rich_status(code: int, message: str) -> Any:
    """A ``google.rpc.Status`` for an A2A error, with its ``ErrorInfo`` detail."""
    from google.rpc import code_pb2, error_details_pb2, status_pb2

    _, grpc_name, reason = error_mapping(code)
    status = status_pb2.Status(code=getattr(code_pb2, grpc_name), message=message)
    info = error_details_pb2.ErrorInfo(reason=reason, domain=A2A_ERROR_DOMAIN)
    info.metadata["message"] = message
    status.details.add().Pack(info)
    return status


def grpc_status_to_error(status: Any | None, fallback: str) -> A2AClientError:
    """The :class:`A2AClientError` a ``google.rpc.Status`` names."""
    if status is None:
        return A2AClientError(INTERNAL_ERROR, fallback)
    from google.rpc import error_details_pb2

    details: list[dict[str, Any]] = []
    for detail in status.details:
        if detail.Is(error_details_pb2.ErrorInfo.DESCRIPTOR):
            info = error_details_pb2.ErrorInfo()
            detail.Unpack(info)
            details.append({"reason": info.reason, "domain": info.domain})
    return A2AClientError(code_from_error_details(details), status.message or fallback)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class A2AGrpcService:
    """``lf.a2a.v1.A2AService`` served from an :class:`~locus.a2a.A2AServer`.

    Add it to a ``grpc.aio`` server with :meth:`add_to_server`. Bearer
    authentication follows the HTTP bindings: when the server has an API key
    the call must carry ``authorization: Bearer <key>`` in its metadata.
    """

    def __init__(self, server: A2AServer) -> None:
        self._server = server
        self._messages = v1_messages()

    # ----- plumbing --------------------------------------------------------

    @staticmethod
    def _metadata(context: Any) -> dict[str, Any]:
        """The call's invocation metadata, keys lowered (gRPC sends them so)."""
        return {key.lower(): value for key, value in (context.invocation_metadata() or ())}

    async def _guard(self, context: Any) -> None:
        grpc = _require_grpc()
        metadata = self._metadata(context)
        expected = self._server.api_key
        if expected is not None:
            presented = str(metadata.get("authorization", ""))
            token = (
                presented.split(" ", 1)[1].strip()
                if presented.lower().startswith("bearer ")
                else ""
            )
            if not token or not hmac.compare_digest(token, expected):
                await context.abort(
                    grpc.StatusCode.UNAUTHENTICATED, "Missing or invalid bearer token"
                )
        version = metadata.get(A2A_VERSION_HEADER.lower())
        if version not in (None, A2A_V1_PROTOCOL_VERSION):
            await self._abort(
                context, VERSION_NOT_SUPPORTED, f"A2A version {version!r} is not supported"
            )

    async def _abort(self, context: Any, code: int, message: str) -> None:
        from grpc_status import rpc_status

        await context.abort_with_status(rpc_status.to_status(_rich_status(code, message)))

    def _unary(self, method: str) -> Callable[[Any, Any], Any]:
        _, response_name, _ = RPC_SIGNATURES[method]

        async def handler(request: Any, context: Any) -> Any:
            await self._guard(context)
            params = self._messages.to_dict(request)
            try:
                # A ``traceparent`` metadata entry puts the run under the caller's trace.
                with trace_scope(self._metadata(context)):
                    result = await self._server.handle_v1(method, params)
            except A2AV1ProtocolError as e:
                await self._abort(context, e.code, e.message)
                raise  # pragma: no cover - abort raises
            except Exception:
                _logger.exception("A2A gRPC %s failed", method)
                await self._abort(context, INTERNAL_ERROR, "internal error")
                raise  # pragma: no cover - abort raises
            return self._messages.from_dict(result, response_name)

        return handler

    def _streaming(self, method: str) -> Callable[[Any, Any], AsyncIterator[Any]]:
        _, response_name, _ = RPC_SIGNATURES[method]

        async def handler(request: Any, context: Any) -> AsyncIterator[Any]:
            await self._guard(context)
            params = self._messages.to_dict(request)
            try:
                with trace_scope(self._metadata(context)):
                    async for frame in self._server.handle_v1_stream(method, params):
                        yield self._messages.from_dict(frame, response_name)
            except A2AV1ProtocolError as e:
                await self._abort(context, e.code, e.message)
            except Exception:
                _logger.exception("A2A gRPC %s stream failed", method)
                await self._abort(context, INTERNAL_ERROR, "internal error")

        return handler

    def generic_handler(self) -> Any:
        """A ``grpc.GenericRpcHandler`` for the whole service."""
        grpc = _require_grpc()
        handlers: dict[str, Any] = {}
        for method, (request_name, response_name, streaming) in RPC_SIGNATURES.items():
            request_cls = self._messages[request_name]
            response_cls = self._messages[response_name]
            if streaming:
                handlers[method] = grpc.unary_stream_rpc_method_handler(
                    self._streaming(method),
                    request_deserializer=request_cls.FromString,
                    response_serializer=response_cls.SerializeToString,
                )
            else:
                handlers[method] = grpc.unary_unary_rpc_method_handler(
                    self._unary(method),
                    request_deserializer=request_cls.FromString,
                    response_serializer=response_cls.SerializeToString,
                )
        return grpc.method_handlers_generic_handler(A2A_GRPC_SERVICE, handlers)

    def add_to_server(self, grpc_server: Any) -> None:
        """Register the service on a ``grpc.aio.Server``."""
        grpc_server.add_generic_rpc_handlers((self.generic_handler(),))


def build_grpc_server(
    server: A2AServer, host: str, port: int, *, credentials: Any | None = None
) -> tuple[Any, int]:
    """A ``grpc.aio`` server with the A2A service bound; returns ``(server, bound_port)``.

    ``port`` ``0`` picks a free port. ``credentials`` (``grpc.ServerCredentials``)
    makes the port TLS; without it the port is plaintext, which the server's
    bind rule limits to loopback unless an API key or
    ``allow_unauthenticated`` is set, as for HTTP.
    """
    grpc = _require_grpc()
    aio_server = grpc.aio.server()
    A2AGrpcService(server).add_to_server(aio_server)
    address = f"{host}:{port}"
    bound = (
        aio_server.add_secure_port(address, credentials)
        if credentials is not None
        else aio_server.add_insecure_port(address)
    )
    return aio_server, int(bound)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def grpc_target(url: str) -> tuple[str, bool]:
    """``(host:port, secure)`` from an interface URL.

    Accepts ``grpc://host:port`` and ``http://host:port`` (plaintext),
    ``grpcs://`` and ``https://`` (TLS), or a bare ``host:port`` (plaintext).
    """
    if "://" not in url:
        return url, False
    parts = urlsplit(url)
    secure = parts.scheme in ("grpcs", "https")
    port = parts.port or (443 if secure else 80)
    return f"{parts.hostname}:{port}", secure


class GrpcClientTransport(ClientTransport):
    """The gRPC binding for :class:`~locus.a2a.A2AClient`."""

    binding = BINDING_GRPC

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        timeout: float | None = None,
        channel: Any | None = None,
    ) -> None:
        self.url = url
        self._api_key = api_key
        self._timeout = timeout
        self._channel = channel
        self._owns_channel = channel is None
        self._messages = v1_messages()

    def _metadata(self) -> list[tuple[str, str]]:
        metadata = [(A2A_VERSION_HEADER.lower(), A2A_V1_PROTOCOL_VERSION)]
        if self._api_key:
            metadata.append(("authorization", f"Bearer {self._api_key}"))
        # The trace context travels as metadata; gRPC keys are lower-case.
        metadata.extend((key.lower(), value) for key, value in inject_trace_headers().items())
        return metadata

    def channel(self) -> Any:
        if self._channel is None:
            grpc = _require_grpc()
            target, secure = grpc_target(self.url)
            self._channel = (
                grpc.aio.secure_channel(target, grpc.ssl_channel_credentials())
                if secure
                else grpc.aio.insecure_channel(target)
            )
        return self._channel

    def _method(self, name: str) -> Any:
        if name not in RPC_SIGNATURES:
            raise A2AClientError(METHOD_NOT_FOUND, f"unknown method {name!r}")
        request_name, response_name, streaming = RPC_SIGNATURES[name]
        path = f"/{A2A_GRPC_SERVICE}/{name}"
        kwargs = {
            "request_serializer": self._messages[request_name].SerializeToString,
            "response_deserializer": self._messages[response_name].FromString,
        }
        channel = self.channel()
        if streaming:
            return channel.unary_stream(path, **kwargs)
        return channel.unary_unary(path, **kwargs)

    def _translate(self, error: Any) -> A2AClientError:
        grpc = _require_grpc()
        from grpc_status import rpc_status

        if error.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNIMPLEMENTED):
            return A2ATransportUnavailableError(METHOD_NOT_FOUND, f"{self.url}: {error.details()}")
        try:
            status = rpc_status.from_call(error)
        except ValueError:
            status = None
        return grpc_status_to_error(status, error.details() or str(error.code()))

    async def call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: Any = None,  # noqa: ASYNC109 — a gRPC deadline, not an asyncio timer
    ) -> dict[str, Any]:
        grpc = _require_grpc()
        from locus.observability.emit import EV_A2A_CLIENT_RECEIVED, EV_A2A_CLIENT_SEND, emit

        request_name, _, _ = RPC_SIGNATURES.get(method, ("", "", False))
        if not request_name:
            raise A2AClientError(METHOD_NOT_FOUND, f"unknown method {method!r}")
        await emit(EV_A2A_CLIENT_SEND, target_url=self.url, method=method, binding=self.binding)
        started = time.perf_counter()
        request = self._messages.from_dict(params, request_name)
        try:
            response = await self._method(method)(
                request,
                metadata=self._metadata(),
                timeout=self._timeout if timeout is None else float(timeout),
            )
        except grpc.aio.AioRpcError as e:
            # ``status_code`` is the gRPC status here, as on the success path.
            await emit(
                EV_A2A_CLIENT_RECEIVED,
                target_url=self.url,
                method=method,
                binding=self.binding,
                status_code=e.code().value[0],
                duration_ms=(time.perf_counter() - started) * 1000,
                error=e.code().name,
            )
            raise self._translate(e) from e
        await emit(
            EV_A2A_CLIENT_RECEIVED,
            target_url=self.url,
            method=method,
            binding=self.binding,
            status_code=0,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        return self._messages.to_dict(response)

    async def stream(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: Any = None,  # noqa: ASYNC109 — a gRPC deadline, not an asyncio timer
    ) -> AsyncIterator[dict[str, Any]]:
        grpc = _require_grpc()
        request_name, _, streaming = RPC_SIGNATURES.get(method, ("", "", False))
        if not request_name or not streaming:
            raise A2AClientError(METHOD_NOT_FOUND, f"unknown streaming method {method!r}")
        request = self._messages.from_dict(params, request_name)
        call = self._method(method)(
            request,
            metadata=self._metadata(),
            timeout=self._timeout if timeout is None else float(timeout),
        )
        try:
            async for response in call:
                yield self._messages.to_dict(response)
        except grpc.aio.AioRpcError as e:
            translated = self._translate(e)
            if isinstance(translated, A2ATransportUnavailableError):
                raise translated from e
            yield {"error": {"code": translated.code, "message": translated.message}}

    async def aclose(self) -> None:
        if self._channel is not None and self._owns_channel:
            await self._channel.close()
            self._channel = None


__all__ = [
    "RPC_SIGNATURES",
    "A2AGrpcService",
    "GrpcClientTransport",
    "V1Messages",
    "build_grpc_server",
    "grpc_status_to_error",
    "grpc_target",
    "v1_messages",
]
