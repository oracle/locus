# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The browser tools against real Chromium and a local site.

Skipped unless Playwright is installed and its Chromium launches.
"""

from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import pytest

from tulip.tools.browser import BrowserError, BrowserSession, browser_toolset


if TYPE_CHECKING:
    from collections.abc import Iterator


def _chromium_launches() -> bool:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return False

    async def probe() -> bool:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            await browser.close()
        return True

    try:
        return asyncio.run(probe())
    except Exception:  # noqa: BLE001 — any launch failure means skip
        return False


pytestmark = pytest.mark.skipif(not _chromium_launches(), reason="needs Playwright with Chromium")

_FORM = (
    "<html><head><title>Refunds</title></head><body><h1>Refund order 4821</h1>"
    '<form action="/done" method="get"><input id="reason" name="reason">'
    '<button id="send" type="submit">Send</button></form>'
    '<a id="away" href="http://localhost.invalid:9/">elsewhere</a></body></html>'
)


class _Site(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/done":
            reason = parse_qs(parsed.query).get("reason", [""])[0]
            body = (
                f"<html><head><title>Done</title></head><body>Refund sent: {reason}</body></html>"
            )
        else:
            body = _FORM
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:
        return


@pytest.fixture
def site() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_open_type_and_submit_a_real_form(site: str) -> None:
    async with BrowserSession(allowed_domains=["127.0.0.1"]) as session:
        tools = {t.name: t for t in browser_toolset(session, allow_submit=True)}

        opened = await tools["browser_open"].fn(url=f"{site}/")
        await tools["browser_type"].fn(selector="#reason", text="damaged in transit")
        submitted = await tools["browser_submit"].fn(selector="#send")

    assert opened.startswith("Refunds")
    assert "Refund order 4821" in opened
    assert "Refund sent: damaged in transit" in submitted


@pytest.mark.asyncio
async def test_an_off_list_host_is_refused_before_loading(site: str) -> None:
    async with BrowserSession(allowed_domains=["127.0.0.1"]) as session:
        tools = {t.name: t for t in browser_toolset(session)}
        with pytest.raises(BrowserError):
            await tools["browser_open"].fn(url="http://localhost.invalid:9/")
