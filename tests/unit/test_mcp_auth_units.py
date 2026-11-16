# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The pieces of ``locus.integrations.mcp_auth`` on their own.

The flow against a live server is in ``test_mcp_oauth.py``; this file pins
the edges of the token stores, the loopback listener and the default
browser hand-off that the flow test does not reach.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from locus.integrations.mcp_auth import (
    FileTokenStore,
    InMemoryTokenStore,
    KeyringTokenStore,
    LoopbackCallback,
    MCPOAuth,
    default_token_dir,
    free_port,
)


pytest.importorskip("mcp")

ACCESS = "at-x"  # noqa: S105 — a test fixture value, not a credential
REFRESH = "rt-x"  # noqa: S105


def _tokens(**overrides: Any) -> Any:
    from mcp.shared.auth import OAuthToken

    return OAuthToken(access_token=ACCESS, refresh_token=REFRESH, **overrides)


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


async def test_in_memory_store_round_trips_and_clears() -> None:
    from mcp.shared.auth import OAuthClientInformationFull

    store = InMemoryTokenStore()
    assert await store.get_tokens() is None
    assert await store.get_client_info() is None
    await store.set_tokens(_tokens())
    await store.set_client_info(OAuthClientInformationFull(client_id="c"))
    assert (await store.get_tokens()).access_token == ACCESS
    assert (await store.get_client_info()).client_id == "c"
    store.clear()
    assert await store.get_tokens() is None
    assert await store.get_client_info() is None


async def test_file_store_treats_a_damaged_file_as_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "tokens.json"
    path.write_text("{not json")
    store = FileTokenStore(path)
    with caplog.at_level(logging.WARNING):
        assert await store.get_tokens() is None
    assert any("unreadable" in r.getMessage() for r in caplog.records)
    # A file holding something other than an object is also "nothing saved".
    path.write_text("[1, 2, 3]")
    assert await store.get_tokens() is None
    assert await store.get_client_info() is None
    # Saving replaces the damaged file whole and keeps both halves.
    await store.set_tokens(_tokens())
    assert json.loads(path.read_text())["tokens"]["access_token"] == ACCESS
    store.clear()
    assert not path.exists()
    store.clear()  # already gone


async def test_file_store_survives_a_directory_it_cannot_restrict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_chmod = Path.chmod

    def picky_chmod(self: Path, mode: int, *args: Any, **kwargs: Any) -> None:
        if mode == 0o700:
            raise PermissionError("not yours")
        real_chmod(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", picky_chmod)
    store = FileTokenStore(tmp_path / "shared" / "tokens.json")
    await store.set_tokens(_tokens())
    assert (await store.get_tokens()).access_token == ACCESS


async def test_file_store_leaves_no_temp_file_when_the_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_dump(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(json, "dump", broken_dump)
    store = FileTokenStore(tmp_path / "tokens.json")
    with pytest.raises(OSError, match="disk full"):
        await store.set_tokens(_tokens())
    assert list(tmp_path.iterdir()) == []


def test_default_token_dir_without_xdg_is_under_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert default_token_dir() == tmp_path / ".config" / "locus" / "mcp-tokens"


async def test_keyring_store_ignores_entries_it_cannot_read() -> None:
    class Backend:
        def __init__(self) -> None:
            self.entries: dict[str, str] = {
                "https://s/mcp#tokens": "{broken",
                "https://s/mcp#client": "[]",
            }

        def get_password(self, service: str, name: str) -> str | None:
            return self.entries.get(name)

        def set_password(self, service: str, name: str, value: str) -> None:
            self.entries[name] = value

        def delete_password(self, service: str, name: str) -> None:
            if name not in self.entries:
                raise LookupError(name)
            del self.entries[name]

    store = KeyringTokenStore("https://s/mcp", backend=Backend())
    assert await store.get_tokens() is None
    assert await store.get_client_info() is None
    store.clear()  # a backend that raises on a missing entry is tolerated
    store.clear()


# ---------------------------------------------------------------------------
# The loopback listener
# ---------------------------------------------------------------------------


class _Writer:
    """A StreamWriter stand-in that records what was written."""

    def __init__(self, *, hang_up: bool = False) -> None:
        self.data = b""
        self.hang_up = hang_up
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        if self.hang_up:
            raise ConnectionResetError("browser went away")

    def close(self) -> None:
        self.closed = True


def _request(target: str) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(f"GET {target} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
    reader.feed_eof()
    return reader


async def test_listener_answers_other_paths_with_404_and_tolerates_a_hang_up() -> None:
    listener = LoopbackCallback("127.0.0.1", free_port(), "/callback", timeout=1)
    writer = _Writer(hang_up=True)
    await listener._handle(_request("/favicon.ico"), writer)
    assert writer.data.startswith(b"HTTP/1.1 404")
    assert writer.closed


async def test_listener_reports_a_response_without_a_code() -> None:
    listener = LoopbackCallback("127.0.0.1", free_port(), "/callback", timeout=1)
    await listener.start()
    await listener.start()  # idempotent
    waiting = asyncio.ensure_future(listener.wait())
    await listener._handle(_request("/callback?state=s"), _Writer())
    with pytest.raises(PermissionError, match="no code"):
        await waiting


async def test_listener_wait_starts_itself_and_times_out() -> None:
    listener = LoopbackCallback("127.0.0.1", free_port(), "/callback", timeout=0.05)
    await listener.close()  # nothing to close yet
    with pytest.raises(TimeoutError, match="nobody completed"):
        await listener.wait()
    assert listener._server is None  # closed on the way out


# ---------------------------------------------------------------------------
# MCPOAuth.provider: the default hand-off and the version shims
# ---------------------------------------------------------------------------


async def test_default_handlers_open_the_browser_and_take_the_loopback_code(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import httpx

    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    provider = MCPOAuth(redirect_port=free_port(), timeout=5).provider("https://mcp.example/mcp")
    redirect_uri = str(provider.context.client_metadata.redirect_uris[0])

    with caplog.at_level(logging.INFO):
        await provider.context.redirect_handler("https://as.example/authorize?state=s1")
    assert opened == ["https://as.example/authorize?state=s1"]
    assert any("Sign in to https://mcp.example/mcp" in r.getMessage() for r in caplog.records)

    waiting = asyncio.ensure_future(provider.context.callback_handler())
    async with httpx.AsyncClient() as http:
        page = await http.get(f"{redirect_uri}?code=c-1&state=s1")
    assert page.status_code == 200
    result = await waiting
    assert (result.code, result.state) == ("c-1", "s1")


async def test_a_browser_that_cannot_open_is_not_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_browser(url: str) -> None:
        raise RuntimeError("no display")

    monkeypatch.setattr("webbrowser.open", no_browser)
    provider = MCPOAuth(redirect_port=free_port()).provider("https://mcp.example/mcp")
    await provider.context.redirect_handler("https://as.example/authorize")  # no raise


async def test_open_browser_false_only_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_open(url: str) -> None:
        raise AssertionError("browser opened")

    monkeypatch.setattr("webbrowser.open", must_not_open)
    provider = MCPOAuth(redirect_port=free_port(), open_browser=False).provider(
        "https://mcp.example/mcp"
    )
    await provider.context.redirect_handler("https://as.example/authorize")


async def test_a_sync_redirect_handler_and_a_custom_callback_are_honoured() -> None:
    seen: list[str] = []

    def show(url: str) -> None:
        seen.append(url)

    async def callback() -> tuple[str, str | None]:
        return "c-2", None

    oauth = MCPOAuth(
        redirect_handler=show,
        callback_handler=callback,
        redirect_uri="https://app.example/oauth/callback",
        client_metadata_url="https://app.example/oauth/client.json",
        scopes="mcp:tools offline_access",
    )
    provider = oauth.provider("https://mcp.example/mcp")
    assert provider.context.client_metadata_url == "https://app.example/oauth/client.json"
    assert provider.context.client_metadata.scope == "mcp:tools offline_access"
    assert str(provider.context.client_metadata.redirect_uris[0]) == (
        "https://app.example/oauth/callback"
    )
    await provider.context.redirect_handler("https://as.example/authorize")
    assert seen == ["https://as.example/authorize"]
    result = await provider.context.callback_handler()
    assert (result.code, result.state) == ("c-2", None)


async def test_callback_hands_back_a_tuple_on_an_sdk_without_the_result_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def callback() -> tuple[str, str | None]:
        return "c-3", "s3"

    provider = MCPOAuth(callback_handler=callback, redirect_uri="https://app.example/cb").provider(
        "https://mcp.example/mcp"
    )
    # mcp 1.x has no ``AuthorizationCodeResult``; the provider wants the tuple.
    monkeypatch.setitem(sys.modules, "mcp.shared.auth", types.ModuleType("mcp.shared.auth"))
    assert await provider.context.callback_handler() == ("c-3", "s3")


def test_provider_names_the_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mcp.client.auth", None)
    with pytest.raises(ImportError, match="locus-sdk\\[mcp\\]"):
        MCPOAuth().provider("https://mcp.example/mcp")
