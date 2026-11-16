# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OAuth 2.1 sign-in to a remote MCP server, against a real server on loopback.

The MCP app from ``_mcp_loopback_server.py`` sits behind an ASGI gate that
answers ``401`` with a ``WWW-Authenticate`` challenge until a token the fake
authorization server issued arrives. The same gate serves the
protected-resource metadata, the authorization-server metadata, dynamic client
registration and the token endpoint, so the client's discovery, PKCE exchange
and refresh all travel over HTTP the way they would against a real server.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import stat
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from tests._mcp_deps import require_server_deps


uvicorn = require_server_deps()

from locus.integrations.fastmcp import MCPClient  # noqa: E402
from locus.integrations.mcp_auth import (  # noqa: E402
    FileTokenStore,
    InMemoryTokenStore,
    KeyringTokenStore,
    LoopbackCallback,
    MCPOAuth,
    default_token_dir,
)
from tests._mcp_loopback_server import build_server  # noqa: E402
from tests.unit.test_mcp_fidelity import _free_port  # noqa: E402


# ---------------------------------------------------------------------------
# A fake OAuth 2.1 authorization server + resource gate, as one ASGI layer
# ---------------------------------------------------------------------------


class FakeAuthority:
    """Issues codes and tokens; checks PKCE; remembers every token request."""

    def __init__(self, origin: str, mcp_path: str = "/mcp") -> None:
        self.origin = origin
        self.resource = f"{origin}{mcp_path}"
        self.issuer = f"{origin}/auth"
        self.prm_url = f"{origin}/.well-known/oauth-protected-resource{mcp_path}"
        self.clients: dict[str, dict[str, Any]] = {}
        self.codes: dict[str, dict[str, Any]] = {}
        self.access_tokens: dict[str, str] = {}  # token -> client_id
        self.refresh_tokens: dict[str, str] = {}  # refresh -> client_id
        self.token_requests: list[dict[str, str]] = []
        self.authorizations: list[dict[str, str]] = []
        self.expires_in = 3600

    # -- documents -------------------------------------------------------

    def prm(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "authorization_servers": [self.issuer],
            "scopes_supported": ["mcp:tools"],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "issuer": self.issuer,
            "authorization_endpoint": f"{self.issuer}/authorize",
            "token_endpoint": f"{self.issuer}/token",
            "registration_endpoint": f"{self.issuer}/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
            "scopes_supported": ["mcp:tools"],
        }

    # -- the person's side ----------------------------------------------

    def authorize(self, url: str) -> tuple[str, str | None]:
        """What the browser would do at the authorization URL: approve."""
        query = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        self.authorizations.append(query)
        assert query["response_type"] == "code"
        assert query["code_challenge_method"] == "S256"
        assert query["client_id"] in self.clients
        code = secrets.token_urlsafe(16)
        self.codes[code] = query
        return code, query.get("state")

    # -- endpoints -------------------------------------------------------

    def register(self, body: dict[str, Any]) -> dict[str, Any]:
        client_id = f"client-{len(self.clients) + 1}"
        info = {
            "client_id": client_id,
            "redirect_uris": body["redirect_uris"],
            "token_endpoint_auth_method": "none",
            "grant_types": body.get("grant_types", ["authorization_code"]),
            "response_types": ["code"],
            "client_name": body.get("client_name"),
        }
        self.clients[client_id] = info
        return info

    def token(self, form: dict[str, str]) -> tuple[int, dict[str, Any]]:
        self.token_requests.append(form)
        grant = form.get("grant_type")
        if grant == "authorization_code":
            pending = self.codes.pop(form.get("code", ""), None)
            if pending is None:
                return 400, {"error": "invalid_grant"}
            digest = hashlib.sha256(form.get("code_verifier", "").encode()).digest()
            challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
            if challenge != pending["code_challenge"]:
                return 400, {"error": "invalid_grant", "error_description": "bad verifier"}
            if form.get("redirect_uri") != pending["redirect_uri"]:
                return 400, {"error": "invalid_grant", "error_description": "redirect"}
            client_id = pending["client_id"]
        elif grant == "refresh_token":
            client_id_or_none = self.refresh_tokens.pop(form.get("refresh_token", ""), None)
            if client_id_or_none is None:
                return 400, {"error": "invalid_grant"}
            client_id = client_id_or_none
        else:
            return 400, {"error": "unsupported_grant_type"}
        access = f"at-{secrets.token_urlsafe(12)}"
        refresh = f"rt-{secrets.token_urlsafe(12)}"
        self.access_tokens[access] = client_id
        self.refresh_tokens[refresh] = client_id
        return 200, {
            "access_token": access,
            "token_type": "Bearer",
            "expires_in": self.expires_in,
            "refresh_token": refresh,
            "scope": "mcp:tools",
        }

    def accepts(self, authorization: str | None) -> bool:
        if not authorization or not authorization.startswith("Bearer "):
            return False
        return authorization.removeprefix("Bearer ") in self.access_tokens


async def _read_body(receive: Any) -> bytes:
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            return body


async def _send_json(
    send: Any, status: int, payload: Any, headers: list[tuple[bytes, bytes]] | None = None
) -> None:
    data = json.dumps(payload).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(data)).encode()),
                *(headers or []),
            ],
        }
    )
    await send({"type": "http.response.body", "body": data})


class Gate:
    """ASGI app: the fake authority's endpoints, and the MCP app behind a bearer check."""

    def __init__(self, app: Any, authority: FakeAuthority) -> None:
        self.app = app
        self.authority = authority
        self.rejected = 0

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        authority = self.authority
        if path == "/.well-known/oauth-protected-resource/mcp":
            await _send_json(send, 200, authority.prm())
            return
        if path == "/.well-known/oauth-authorization-server/auth":
            await _send_json(send, 200, authority.metadata())
            return
        if path.startswith("/.well-known/"):
            await _send_json(send, 404, {"error": "not_found"})
            return
        if path == "/auth/register":
            body = json.loads(await _read_body(receive) or b"{}")
            await _send_json(send, 201, authority.register(body))
            return
        if path == "/auth/token":
            form = {k: v[0] for k, v in parse_qs((await _read_body(receive)).decode()).items()}
            status, payload = authority.token(form)
            await _send_json(send, status, payload)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        if not authority.accepts(headers.get("authorization")):
            self.rejected += 1
            challenge = f'Bearer resource_metadata="{authority.prm_url}"'
            await _send_json(
                send,
                401,
                {"error": "unauthorized"},
                headers=[(b"www-authenticate", challenge.encode())],
            )
            return
        await self.app(scope, receive, send)


class ProtectedServer:
    """The loopback MCP server behind :class:`Gate`, on a daemon thread."""

    def __init__(self) -> None:
        self.port = _free_port()
        self.origin = f"http://127.0.0.1:{self.port}"
        self.url = f"{self.origin}/mcp"
        self.authority = FakeAuthority(self.origin)
        self.gate: Gate | None = None
        self._server: Any = None
        self._thread: threading.Thread | None = None

    def start(self) -> ProtectedServer:
        app = build_server(self.port).streamable_http_app()
        self.gate = Gate(app, self.authority)
        config = uvicorn.Config(
            self.gate,
            host="127.0.0.1",
            port=self.port,
            log_level="critical",
            lifespan="on",
            timeout_graceful_shutdown=0,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:  # pragma: no cover - environment failure
                raise RuntimeError("protected MCP server did not start")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.should_exit = True
        self._server.force_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None


@pytest.fixture(scope="module")
def server() -> Iterator[ProtectedServer]:
    srv = ProtectedServer().start()
    yield srv
    srv.stop()


class Person:
    """Stands in for the browser: approves at the authorization URL."""

    def __init__(self, authority: FakeAuthority) -> None:
        self.authority = authority
        self.visits = 0
        self._pending: tuple[str, str | None] | None = None

    async def redirect(self, url: str) -> None:
        self.visits += 1
        self._pending = self.authority.authorize(url)

    async def callback(self) -> tuple[str, str | None]:
        assert self._pending is not None
        pending, self._pending = self._pending, None
        return pending


def _oauth(person: Person, **kwargs: Any) -> MCPOAuth:
    return MCPOAuth(
        client_name="locus-test",
        redirect_uri="http://127.0.0.1:1/callback",
        redirect_handler=person.redirect,
        callback_handler=person.callback,
        **kwargs,
    )


def _client(server: ProtectedServer, oauth: MCPOAuth) -> MCPClient:
    return MCPClient(base_url=server.url, verify_url=False, oauth=oauth, connect_timeout=15)


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


async def test_signs_in_with_discovery_pkce_and_registration(
    server: ProtectedServer, caplog: pytest.LogCaptureFixture
) -> None:
    authority = server.authority
    before = len(authority.token_requests)
    person = Person(authority)
    store = InMemoryTokenStore()
    client = _client(server, _oauth(person, store=store))
    with caplog.at_level(logging.DEBUG):
        await client.connect()
        try:
            digest = await client.call_tool("auth_digest", {})
        finally:
            await client.close()

    # One trip to the browser, one code exchange, and the call went through
    # with the token the fake authority issued.
    assert person.visits == 1
    exchange = authority.token_requests[before]
    assert exchange["grant_type"] == "authorization_code"
    assert exchange["code_verifier"]  # PKCE
    assert exchange["resource"] == server.url  # RFC 8707 resource indicator
    tokens = await store.get_tokens()
    assert tokens is not None
    expected = hashlib.sha256(f"Bearer {tokens.access_token}".encode()).hexdigest()[:16]
    assert digest == expected  # the issued token is what reached the server
    # The client registered itself dynamically with the redirect we gave.
    registration = await store.get_client_info()
    assert registration is not None
    assert registration.client_id in authority.clients
    assert str(registration.redirect_uris[0]) == "http://127.0.0.1:1/callback"
    # And no token value reached a log line, at any level — neither from
    # Locus nor from the SDK's wire logging (the tool above digests the
    # header rather than echoing it, so the payload logs hold no token).
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert tokens.access_token not in logged
    assert tokens.refresh_token is not None
    assert tokens.refresh_token not in logged


async def test_an_expired_token_is_refreshed_without_a_person(server: ProtectedServer) -> None:
    authority = server.authority
    person = Person(authority)
    store = InMemoryTokenStore()
    client = _client(server, _oauth(person, store=store))
    await client.connect()
    try:
        first = await client.call_tool("auth_digest", {})
        first_tokens = await store.get_tokens()
        assert first_tokens is not None
        # Make the token look expired to the provider; the next request must
        # refresh it before it is sent, with nobody sent to the browser.
        provider = client._oauth_provider
        provider.context.token_expiry_time = time.time() - 1
        before = len(authority.token_requests)
        second = await client.call_tool("auth_digest", {})
    finally:
        await client.close()

    assert person.visits == 1
    refresh = authority.token_requests[before]
    assert refresh["grant_type"] == "refresh_token"
    assert refresh["refresh_token"] == first_tokens.refresh_token
    rotated = await store.get_tokens()
    assert rotated is not None
    assert rotated.access_token != first_tokens.access_token
    assert second != first  # a different bearer reached the server


async def test_file_store_is_private_and_reused_across_clients(
    server: ProtectedServer, tmp_path: Path
) -> None:
    authority = server.authority
    person = Person(authority)
    path = tmp_path / "tokens" / "server.json"
    store = FileTokenStore(path)

    client = _client(server, _oauth(person, store=store))
    await client.connect()
    try:
        assert "auth=Bearer at-" in await client.call_tool("whoami", {})
    finally:
        await client.close()

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    saved = json.loads(path.read_text())
    assert set(saved) == {"tokens", "client_info"}

    # A fresh client (a new process, in life) reads the file and signs in
    # without a browser trip.
    again = Person(authority)
    second = _client(server, _oauth(again, store=FileTokenStore(path)))
    await second.connect()
    try:
        assert "auth=Bearer at-" in await second.call_tool("whoami", {})
    finally:
        await second.close()
    assert again.visits == 0
    assert person.visits == 1


def test_file_store_default_location_is_per_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert default_token_dir() == tmp_path / "locus" / "mcp-tokens"
    one = FileTokenStore(server_url="https://a.example/mcp")
    two = FileTokenStore(server_url="https://b.example/mcp")
    assert one.path.parent == default_token_dir()
    assert one.path != two.path
    assert one.path.suffix == ".json"
    with pytest.raises(ValueError, match="server_url"):
        FileTokenStore()


async def test_keyring_store_round_trips_through_a_backend() -> None:
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    class Backend:
        def __init__(self) -> None:
            self.entries: dict[tuple[str, str], str] = {}

        def get_password(self, service: str, name: str) -> str | None:
            return self.entries.get((service, name))

        def set_password(self, service: str, name: str, value: str) -> None:
            self.entries[(service, name)] = value

        def delete_password(self, service: str, name: str) -> None:
            del self.entries[(service, name)]

    backend = Backend()
    store = KeyringTokenStore("https://mcp.example/mcp", backend=backend)
    assert await store.get_tokens() is None
    await store.set_tokens(OAuthToken(access_token="at-1", refresh_token="rt-1"))  # noqa: S106
    await store.set_client_info(OAuthClientInformationFull(client_id="c1"))
    assert {name for _, name in backend.entries} == {
        "https://mcp.example/mcp#tokens",
        "https://mcp.example/mcp#client",
    }
    tokens = await store.get_tokens()
    assert tokens is not None
    assert (tokens.access_token, tokens.refresh_token) == ("at-1", "rt-1")
    info = await store.get_client_info()
    assert info is not None
    assert info.client_id == "c1"
    store.clear()
    assert backend.entries == {}
    store.clear()  # nothing left to delete is not an error


def test_keyring_store_names_the_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    def missing(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ImportError, match="pip install keyring"):
        KeyringTokenStore("https://mcp.example/mcp")


def test_oauth_configuration_is_checked_up_front() -> None:
    async def callback() -> tuple[str, str | None]:  # pragma: no cover - never reached
        return "code", None

    with pytest.raises(ValueError, match="redirect_uri"):
        MCPOAuth(callback_handler=callback).provider("https://mcp.example/mcp")
    with pytest.raises(TypeError, match="get_tokens"):
        MCPOAuth(store=object()).provider("https://mcp.example/mcp")
    # The default store is in memory and the default redirect is a loopback
    # listener on a free port.
    provider = MCPOAuth().provider("https://mcp.example/mcp")
    assert isinstance(provider.context.storage, InMemoryTokenStore)
    redirect = str(provider.context.client_metadata.redirect_uris[0])
    assert redirect.startswith("http://127.0.0.1:")
    assert redirect.endswith("/callback")
    assert provider.context.client_metadata.token_endpoint_auth_method == "none"  # noqa: S105


async def test_loopback_listener_receives_the_code_and_answers_the_browser() -> None:
    import httpx

    listener = LoopbackCallback("127.0.0.1", _free_port(), "/callback", timeout=5)
    await listener.start()
    waiting = asyncio.ensure_future(listener.wait())
    async with httpx.AsyncClient() as http:
        missed = await http.get(f"http://127.0.0.1:{listener.port}/elsewhere")
        assert missed.status_code == 404
        page = await http.get(f"{listener.redirect_uri}?code=abc123&state=xyz")
    assert page.status_code == 200
    assert "Signed in" in page.text
    assert await waiting == ("abc123", "xyz")


async def test_loopback_listener_reports_a_denied_sign_in() -> None:
    import httpx

    listener = LoopbackCallback("127.0.0.1", _free_port(), "/callback", timeout=5)
    await listener.start()
    waiting = asyncio.ensure_future(listener.wait())
    async with httpx.AsyncClient() as http:
        await http.get(
            f"{listener.redirect_uri}?error=access_denied&error_description=user+said+no"
        )
    with pytest.raises(PermissionError, match="user said no"):
        await waiting
