# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OAuth 2.1 for remote MCP servers.

A Streamable HTTP MCP server that answers ``401`` is asking the client to
sign in. :class:`MCPOAuth` describes how :class:`~locus.integrations.MCPClient`
does that: the authorization-code flow with PKCE, discovery of the server's
protected-resource metadata and of its authorization server's metadata, dynamic
client registration when the server offers it, and token refresh. The flow
itself is the ``mcp`` package's; this module configures it, supplies the
browser hand-off, and keeps the tokens somewhere sensible.

Where tokens live is pluggable (:class:`MCPTokenStore`):

* :class:`InMemoryTokenStore` — the default; the process signs in once and
  forgets on exit.
* :class:`FileTokenStore` — a JSON file under the user's config directory
  (``$XDG_CONFIG_HOME/locus/mcp-tokens`` or ``~/.config/locus/mcp-tokens``),
  created with mode ``0600``.
* :class:`KeyringTokenStore` — the OS keyring, through the optional
  ``keyring`` package.

No token value is ever logged by this module.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import os
import socket
import webbrowser
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

#: ``(authorization_url) -> None`` — show the person where to sign in.
RedirectHandler = Callable[[str], Awaitable[None] | None]

#: ``() -> (code, state)`` — wait for the authorization response.
CallbackHandler = Callable[[], Awaitable[tuple[str, str | None]]]


@runtime_checkable
class MCPTokenStore(Protocol):
    """Where an :class:`MCPOAuth` keeps tokens and the registered client.

    The same shape as the ``mcp`` package's ``TokenStorage``, so any store
    written for it works here, and vice versa. Token objects are the ``mcp``
    package's ``OAuthToken`` / ``OAuthClientInformationFull`` models.
    """

    async def get_tokens(self) -> Any | None:
        """The stored tokens, or ``None``."""

    async def set_tokens(self, tokens: Any) -> None:
        """Store ``tokens``."""

    async def get_client_info(self) -> Any | None:
        """The stored client registration, or ``None``."""

    async def set_client_info(self, client_info: Any) -> None:
        """Store ``client_info``."""


def _token_model() -> Any:
    from mcp.shared.auth import OAuthToken

    return OAuthToken


def _client_info_model() -> Any:
    from mcp.shared.auth import OAuthClientInformationFull

    return OAuthClientInformationFull


def _dump(model: Any) -> dict[str, Any]:
    dumped: dict[str, Any] = model.model_dump(mode="json", exclude_none=True)
    return dumped


class InMemoryTokenStore:
    """Tokens for the life of the process — the default store."""

    def __init__(self) -> None:
        self._tokens: Any | None = None
        self._client_info: Any | None = None

    async def get_tokens(self) -> Any | None:
        return self._tokens

    async def set_tokens(self, tokens: Any) -> None:
        self._tokens = tokens

    async def get_client_info(self) -> Any | None:
        return self._client_info

    async def set_client_info(self, client_info: Any) -> None:
        self._client_info = client_info

    def clear(self) -> None:
        """Forget the tokens and the registration."""
        self._tokens = None
        self._client_info = None


def default_token_dir() -> Path:
    """Where :class:`FileTokenStore` keeps files when given no path.

    ``$XDG_CONFIG_HOME/locus/mcp-tokens`` when the variable is set, otherwise
    ``~/.config/locus/mcp-tokens``.
    """
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "locus" / "mcp-tokens"


class FileTokenStore:
    """Tokens in a JSON file only the user can read (mode ``0600``).

    Args:
        path: The file. When omitted, one is derived from ``server_url`` under
            :func:`default_token_dir`, so each server has its own file.
        server_url: The MCP server the tokens belong to; required when ``path``
            is omitted.
    """

    def __init__(self, path: str | Path | None = None, *, server_url: str | None = None) -> None:
        if path is None:
            if not server_url:
                msg = "FileTokenStore needs a path or the server_url to derive one from"
                raise ValueError(msg)
            digest = hashlib.sha256(server_url.encode("utf-8")).hexdigest()[:24]
            path = default_token_dir() / f"{digest}.json"
        self.path = Path(path)

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            logger.warning("MCP token file %s is unreadable; signing in again", self.path)
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, Any]) -> None:
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True)
        try:
            directory.chmod(0o700)
        except OSError:  # a shared or read-only parent; the file mode still protects it
            logger.debug("could not restrict %s to the user", directory)
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        tmp.chmod(0o600)
        tmp.replace(self.path)

    async def get_tokens(self) -> Any | None:
        data = self._read().get("tokens")
        return _token_model().model_validate(data) if isinstance(data, dict) else None

    async def set_tokens(self, tokens: Any) -> None:
        data = self._read()
        data["tokens"] = _dump(tokens)
        self._write(data)

    async def get_client_info(self) -> Any | None:
        data = self._read().get("client_info")
        return _client_info_model().model_validate(data) if isinstance(data, dict) else None

    async def set_client_info(self, client_info: Any) -> None:
        data = self._read()
        data["client_info"] = _dump(client_info)
        self._write(data)

    def clear(self) -> None:
        """Delete the file."""
        self.path.unlink(missing_ok=True)


class KeyringTokenStore:
    """Tokens in the operating system's keyring.

    Needs the optional ``keyring`` package (``pip install keyring``) unless a
    ``backend`` with ``get_password`` / ``set_password`` / ``delete_password``
    is passed.

    Args:
        server_url: The MCP server the tokens belong to; part of the entry name.
        service: The keyring service name the entries are filed under.
        backend: A ``keyring``-compatible module or object; defaults to the
            ``keyring`` package.
    """

    def __init__(
        self, server_url: str, *, service: str = "locus-mcp", backend: Any | None = None
    ) -> None:
        if backend is None:
            try:
                backend = importlib.import_module("keyring")
            except ImportError as exc:
                msg = "KeyringTokenStore needs the optional keyring package: pip install keyring"
                raise ImportError(msg) from exc
        self._backend: Any = backend
        self._service = service
        self._account = server_url

    def _get(self, kind: str) -> dict[str, Any] | None:
        raw = self._backend.get_password(self._service, f"{self._account}#{kind}")
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def _set(self, kind: str, model: Any) -> None:
        self._backend.set_password(
            self._service, f"{self._account}#{kind}", json.dumps(_dump(model))
        )

    async def get_tokens(self) -> Any | None:
        data = self._get("tokens")
        return _token_model().model_validate(data) if data else None

    async def set_tokens(self, tokens: Any) -> None:
        self._set("tokens", tokens)

    async def get_client_info(self) -> Any | None:
        data = self._get("client")
        return _client_info_model().model_validate(data) if data else None

    async def set_client_info(self, client_info: Any) -> None:
        self._set("client", client_info)

    def clear(self) -> None:
        """Remove both entries."""
        for kind in ("tokens", "client"):
            try:
                self._backend.delete_password(self._service, f"{self._account}#{kind}")
            except Exception:  # noqa: BLE001 — keyring backends raise their own "no such entry"
                logger.debug("no keyring entry to delete for %s", kind)


# ---------------------------------------------------------------------------
# The browser hand-off
# ---------------------------------------------------------------------------

_CALLBACK_PAGE = (
    "<!doctype html><html><head><meta charset='utf-8'><title>Signed in</title></head>"
    "<body style='font-family: system-ui, sans-serif; margin: 3em'>"
    "<h1>Signed in</h1><p>You can close this tab and return to Locus.</p></body></html>"
)


def free_port(host: str = "127.0.0.1") -> int:
    """A TCP port nothing is listening on right now."""
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class LoopbackCallback:
    """Receive the authorization response on a loopback HTTP listener.

    The authorization server redirects the browser to
    ``http://<host>:<port><path>?code=…&state=…``; this listens for exactly that
    one request, answers it with a small page and hands back the code.
    """

    def __init__(self, host: str, port: int, path: str, timeout: float) -> None:
        self.host = host
        self.port = port
        self.path = path
        self.timeout = timeout
        self._server: asyncio.AbstractServer | None = None
        self._result: asyncio.Future[tuple[str, str | None]] | None = None

    @property
    def redirect_uri(self) -> str:
        return f"http://{self.host}:{self.port}{self.path}"

    async def start(self) -> None:
        """Listen, so the redirect finds someone home before the browser opens."""
        if self._server is not None:
            return
        self._result = asyncio.get_running_loop().create_future()
        self._server = await asyncio.start_server(self._handle, self.host, self.port)

    async def wait(self) -> tuple[str, str | None]:
        """The ``(code, state)`` of the authorization response."""
        if self._result is None:
            await self.start()
        assert self._result is not None
        try:
            return await asyncio.wait_for(asyncio.shield(self._result), self.timeout)
        except TimeoutError as exc:
            msg = f"nobody completed the sign-in within {self.timeout:g}s"
            raise TimeoutError(msg) from exc
        finally:
            await self.close()

    async def close(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), 10)
            while True:  # drain the headers; nothing in them matters here
                line = await asyncio.wait_for(reader.readline(), 10)
                if line in (b"\r\n", b"\n", b""):
                    break
            parts = request_line.decode("latin-1").split()
            target = parts[1] if len(parts) >= 2 else "/"
            url = urlsplit(target)
            if url.path != self.path:
                writer.write(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
                return
            query = parse_qs(url.query)
            body = _CALLBACK_PAGE.encode("utf-8")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                + f"Content-Length: {len(body)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
                + body
            )
            if self._result is not None and not self._result.done():
                error = query.get("error", [None])[0]
                code = query.get("code", [None])[0]
                state = query.get("state", [None])[0]
                if error or not code:
                    description = query.get("error_description", [error or "no code"])[0]
                    self._result.set_exception(PermissionError(f"sign-in failed: {description}"))
                else:
                    self._result.set_result((code, state))
        finally:
            try:
                await writer.drain()
            except (ConnectionError, OSError):  # the browser hung up first
                logger.debug("callback response not delivered")
            writer.close()


class MCPOAuth(BaseModel):
    """How an :class:`~locus.integrations.MCPClient` signs in to a remote server.

    Pass it as ``MCPClient(base_url=..., oauth=MCPOAuth(...))``. The first
    request the server answers with ``401`` starts the flow: the client reads
    the protected-resource metadata the challenge points at, the authorization
    server's metadata, registers itself when no ``client_metadata_url`` is
    given, sends the person to the authorization URL, trades the returned code
    (with the PKCE verifier) for tokens and retries the request. Expired tokens
    are refreshed without a person; a refresh that fails starts the flow again.

    By default the person is sent to their browser and the code comes back to a
    loopback listener. A headless process supplies ``redirect_handler`` (show
    or forward the URL) and ``callback_handler`` (return the code) together
    with the ``redirect_uri`` the authorization server will call.

    Attributes:
        client_name: Shown on the consent screen; sent at registration.
        scopes: Space-separated scopes to ask for; ``None`` asks for what the
            server advertises.
        store: An :class:`MCPTokenStore`; ``None`` is an
            :class:`InMemoryTokenStore`.
        redirect_host: Loopback address for the default listener.
        redirect_port: Its port; ``0`` picks a free one when the client is
            built.
        redirect_path: Its path.
        redirect_uri: The full redirect URI, replacing the loopback default.
            Required with a custom ``callback_handler``.
        client_metadata_url: A hosted client-metadata document to identify
            with instead of registering dynamically (an ``https`` URL).
        open_browser: Whether the default redirect handler opens the browser;
            the URL is logged at INFO either way.
        timeout: Seconds to wait for the person to finish signing in.
        redirect_handler: Replaces the browser hand-off.
        callback_handler: Replaces the loopback listener; returns
            ``(code, state)``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client_name: str = "Locus"
    scopes: str | None = None
    store: Any = Field(default=None, repr=False)
    redirect_host: str = "127.0.0.1"
    redirect_port: int = Field(default=0, ge=0, le=65535)
    redirect_path: str = "/callback"
    redirect_uri: str | None = None
    client_metadata_url: str | None = None
    open_browser: bool = True
    timeout: float = Field(default=300.0, gt=0)
    redirect_handler: RedirectHandler | None = Field(default=None, repr=False)
    callback_handler: CallbackHandler | None = Field(default=None, repr=False)

    def provider(self, server_url: str) -> Any:
        """The ``httpx`` auth object that runs the flow for ``server_url``.

        An ``mcp.client.auth.OAuthClientProvider``; attach it to the HTTP
        client that carries the MCP session. :class:`MCPClient` does this for
        you — call it directly only to drive the flow yourself.
        """
        try:
            from mcp.client.auth import OAuthClientProvider
            from mcp.shared.auth import OAuthClientMetadata
        except ImportError as exc:
            msg = "OAuth for MCP servers needs the mcp package: pip install 'locus-sdk[mcp]'"
            raise ImportError(msg) from exc
        from pydantic import AnyUrl

        if self.callback_handler is not None and not self.redirect_uri:
            msg = "MCPOAuth(callback_handler=...) also needs redirect_uri"
            raise ValueError(msg)

        store = self.store if self.store is not None else InMemoryTokenStore()
        if not isinstance(store, MCPTokenStore):
            msg = "MCPOAuth.store must implement get_tokens/set_tokens/get_client_info/set_client_info"
            raise TypeError(msg)

        listener: LoopbackCallback | None = None
        redirect_uri = self.redirect_uri
        if redirect_uri is None:
            port = self.redirect_port or free_port(self.redirect_host)
            listener = LoopbackCallback(self.redirect_host, port, self.redirect_path, self.timeout)
            redirect_uri = listener.redirect_uri

        metadata = OAuthClientMetadata(
            client_name=self.client_name,
            redirect_uris=[AnyUrl(redirect_uri)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope=self.scopes,
            token_endpoint_auth_method="none",  # noqa: S106 — a method name, not a secret
        )

        redirect = self.redirect_handler
        callback = self.callback_handler
        open_browser = self.open_browser

        async def _redirect(url: str) -> None:
            if listener is not None:
                await listener.start()
            if redirect is not None:
                outcome = redirect(url)
                if outcome is not None:
                    await outcome
                return
            logger.info("Sign in to %s at: %s", server_url, url)
            if open_browser:
                try:
                    webbrowser.open(url)
                except Exception:  # noqa: BLE001 — no browser is not a failure; the URL is logged
                    logger.debug("could not open a browser", exc_info=True)

        async def _callback() -> Any:
            if callback is not None:
                code, state = await callback()
            else:
                assert listener is not None
                code, state = await listener.wait()
            try:
                from mcp.shared.auth import AuthorizationCodeResult
            except ImportError:  # mcp 1.x hands back the bare tuple
                return (code, state)
            return AuthorizationCodeResult(code=code, state=state)

        kwargs: dict[str, Any] = {}
        if self.client_metadata_url:
            kwargs["client_metadata_url"] = self.client_metadata_url
        return OAuthClientProvider(
            server_url=server_url,
            client_metadata=metadata,
            storage=store,
            redirect_handler=_redirect,
            callback_handler=_callback,
            **kwargs,
        )


__all__ = [
    "CallbackHandler",
    "FileTokenStore",
    "InMemoryTokenStore",
    "KeyringTokenStore",
    "LoopbackCallback",
    "MCPOAuth",
    "MCPTokenStore",
    "RedirectHandler",
    "default_token_dir",
    "free_port",
]
