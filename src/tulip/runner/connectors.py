# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""API connectors and MCP mounts, called from inside the box.

A box runner calls three kinds of remote tools itself: an operation of an API
connector (``runs: api:<id>``), a tool on a mounted MCP server
(``runs: mcp:<id>``), and its model. The first two are tool calls, so each one
was admitted by the gateway first, and each carries that admission's one-shot
decision token in the ``x-tulip-decision`` header. The box guard (an OpenShell
supervisor middleware) checks the token against the request it sees on the
wire and strips it before the request leaves; a request without a valid token
never reaches the host.

That check only works if the gateway knows, when it admits a call, exactly
which request the call will make. :func:`api_request` is that mapping, from a
call's arguments to one request: deterministic, so the gateway (which imports
it) and the runner build the same bytes.

Credentials are never values here. A connector or mount names the env var
holding its OpenShell placeholder; the sandbox's proxy swaps the placeholder
for the real credential on that host only.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

import httpx

from tulip.runner.manifest import ApiConnector, ApiOperation, McpMount


__all__ = [
    "DECISION_HEADER",
    "ApiRequest",
    "ConnectorError",
    "McpClient",
    "api_request",
    "call_api",
    "canonical_json",
]

#: The header a decision token travels in, to the box guard.
DECISION_HEADER = "x-tulip-decision"

#: MCP protocol version the runner speaks.
MCP_PROTOCOL_VERSION = "2025-06-18"

#: Seconds one connector or MCP request may take.
DEFAULT_TIMEOUT = 60.0


class ConnectorError(Exception):
    """A connector or MCP call could not be made, or the far side refused it."""


def canonical_json(value: Any) -> bytes:
    """``value`` as compact JSON with sorted keys: the bytes a body is sent as."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _scalar(value: Any) -> str:
    """How an argument is written into a path or a query string."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class ApiRequest:
    """One HTTP request an API connector's operation makes. Credential-free."""

    method: str
    url: str
    host: str
    path: str
    query: tuple[tuple[str, str], ...] = ()
    body: bytes | None = None

    @property
    def query_string(self) -> str:
        """The query as sent (``a=1&b=2``), in the operation's order."""
        return urlencode(self.query)


def api_request(
    connector: ApiConnector, operation: ApiOperation, arguments: Mapping[str, Any]
) -> ApiRequest:
    """The request one call of ``operation`` makes, from its arguments.

    Raises:
        ConnectorError: a path argument is missing, or an argument has
            nowhere to go (``body="none"`` and it is neither in the path nor
            the query).
    """
    args = dict(arguments)
    path = operation.path
    for name in operation.path_params:
        if name not in args:
            raise ConnectorError(f"{operation.method} {operation.path}: argument {name!r} missing")
        path = path.replace("{" + name + "}", quote(_scalar(args.pop(name)), safe=""))
    query = tuple(
        (name, _scalar(args.pop(name)))
        for name in operation.query
        if name in args and args[name] is not None
    )
    for name in operation.query:
        args.pop(name, None)
    body: bytes | None = None
    if operation.body == "json":
        body = canonical_json(args)
    elif args:
        raise ConnectorError(
            f"{operation.method} {operation.path} takes no body; "
            f"unexpected argument(s): {', '.join(sorted(args))}"
        )
    base = urlsplit(connector.base_url)
    full_path = base.path.rstrip("/") + path
    url = f"{base.scheme}://{base.netloc}{full_path}"
    return ApiRequest(
        method=operation.method,
        url=url,
        host=base.hostname or "",
        path=full_path,
        query=query,
        body=body,
    )


def _credential_headers(
    env_var: str | None, header: str, prefix: str, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    if not env_var:
        return {}
    placeholder = (os.environ if environ is None else environ).get(env_var, "")
    if not placeholder:
        raise ConnectorError(f"{env_var} is not set: the box has no credential for this host")
    return {header: f"{prefix}{placeholder}"}


async def call_api(
    connector: ApiConnector,
    operation: ApiOperation,
    arguments: Mapping[str, Any],
    *,
    decision_token: str | None,
    transport: httpx.AsyncBaseTransport | None = None,
    request_timeout: float = DEFAULT_TIMEOUT,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Make one admitted connector call and return the response body as text.

    The credential's placeholder is read from ``environ`` (the process's own
    environment when omitted).

    Raises:
        ConnectorError: no decision token, a bad argument, an unreachable host,
            or a non-2xx answer (its status and the start of its body).
    """
    if not decision_token:
        raise ConnectorError("this call has no decision from the gateway, so it is not made")
    request = api_request(connector, operation, arguments)
    headers = {
        DECISION_HEADER: decision_token,
        "accept": "application/json",
        **_credential_headers(
            connector.credential_env,
            connector.credential_header,
            connector.credential_prefix,
            environ,
        ),
    }
    if request.body is not None:
        headers["content-type"] = "application/json"
    async with httpx.AsyncClient(timeout=request_timeout, transport=transport) as http:
        try:
            response = await http.request(
                request.method,
                request.url,
                params=list(request.query) or None,
                content=request.body,
                headers=headers,
            )
        except httpx.TransportError as exc:
            raise ConnectorError(f"{request.method} {request.host}: {type(exc).__name__}") from None
    if response.status_code >= 300:  # noqa: PLR2004 — HTTP status classes
        raise ConnectorError(
            f"{request.method} {request.host}{request.path} answered "
            f"{response.status_code}: {response.text[:300]}"
        )
    return response.text


def _messages(response: httpx.Response) -> list[dict[str, Any]]:
    """The JSON-RPC messages in a streamable-HTTP answer (plain JSON or SSE)."""
    kind = response.headers.get("content-type", "")
    if "text/event-stream" in kind:
        found: list[dict[str, Any]] = []
        for line in response.text.splitlines():
            if line.startswith("data:"):
                try:
                    message = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if isinstance(message, dict):
                    found.append(message)
        return found
    if not response.content:
        return []
    body = response.json()
    if isinstance(body, list):
        return [m for m in body if isinstance(m, dict)]
    return [body] if isinstance(body, dict) else []


@dataclass
class McpClient:
    """A minimal MCP client over streamable HTTP, for one mount.

    It speaks only what a box runner needs: ``initialize`` once (session
    plumbing, which the box guard lets through without a token) and
    ``tools/call`` with the call's decision token.
    """

    mount: McpMount
    transport: httpx.AsyncBaseTransport | None = None
    timeout: float = DEFAULT_TIMEOUT
    #: Where the credential's placeholder is read from; the process's own when ``None``.
    environ: Mapping[str, str] | None = None
    _session: str | None = field(default=None, init=False, repr=False)
    _ready: bool = field(default=False, init=False)
    _ids: Any = field(default_factory=lambda: itertools.count(1), init=False, repr=False)

    def _headers(self) -> dict[str, str]:
        headers = {
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
            **_credential_headers(
                self.mount.credential_env, "authorization", "Bearer ", self.environ
            ),
        }
        if self._session:
            headers["mcp-session-id"] = self._session
        return headers

    async def _post(
        self, http: httpx.AsyncClient, message: dict[str, Any], extra: Mapping[str, str]
    ) -> httpx.Response:
        try:
            response = await http.post(
                self.mount.url,
                content=canonical_json(message),
                headers={**self._headers(), **extra},
            )
        except httpx.TransportError as exc:
            raise ConnectorError(f"MCP {self.mount.id}: {type(exc).__name__}") from None
        if response.status_code >= 300:  # noqa: PLR2004
            raise ConnectorError(
                f"MCP {self.mount.id} answered {response.status_code}: {response.text[:300]}"
            )
        return response

    async def _initialize(self, http: httpx.AsyncClient) -> None:
        message = {
            "jsonrpc": "2.0",
            "id": next(self._ids),
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "tulip-runner", "version": "1"},
            },
        }
        response = await self._post(http, message, {})
        self._session = response.headers.get("mcp-session-id") or None
        await self._post(http, {"jsonrpc": "2.0", "method": "notifications/initialized"}, {})
        self._ready = True

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any], *, decision_token: str | None
    ) -> str:
        """Call one admitted tool on the mount; return its text content.

        Raises:
            ConnectorError: no decision token, the mount's transport is not
                streamable HTTP, the server failed, or the tool reported an error.
        """
        if not decision_token:
            raise ConnectorError("this call has no decision from the gateway, so it is not made")
        if self.mount.transport != "streamable_http":
            raise ConnectorError(
                f"MCP {self.mount.id}: a box runner speaks streamable HTTP only, "
                f"not {self.mount.transport}"
            )
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as http:
            if not self._ready:
                await self._initialize(http)
            call_id = next(self._ids)
            message = {
                "jsonrpc": "2.0",
                "id": call_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": dict(arguments)},
            }
            response = await self._post(http, message, {DECISION_HEADER: decision_token})
        answer = next((m for m in _messages(response) if m.get("id") == call_id), None)
        if answer is None:
            raise ConnectorError(f"MCP {self.mount.id}: no answer to tools/call {name}")
        if "error" in answer:
            error = answer["error"]
            detail = error.get("message") if isinstance(error, dict) else error
            raise ConnectorError(f"MCP {self.mount.id} {name}: {detail}")
        result = answer.get("result") or {}
        text = "\n".join(
            str(block.get("text", ""))
            for block in result.get("content") or []
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if result.get("isError"):
            raise ConnectorError(f"MCP {self.mount.id} {name}: {text or 'the tool failed'}")
        return text
