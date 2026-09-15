# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The computer tool against real Chromium and a local page.

Skipped unless Playwright is installed and its Chromium launches.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

import pytest

from tulip.core.media import images
from tulip.tools.browser import BrowserSession
from tulip.tools.computer import computer_tool


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


pytestmark = pytest.mark.skipif(not _chromium_launches(), reason="Playwright Chromium unavailable")


_PAGE = b"""<!doctype html><html><body style="margin:0">
<input id="q" style="position:absolute;left:20px;top:20px;width:300px;height:30px">
<button id="go" style="position:absolute;left:20px;top:80px;width:120px;height:40px"
  onclick="document.getElementById('out').textContent = 'searched ' + document.getElementById('q').value">
  Search</button>
<div id="out" style="position:absolute;left:20px;top:140px"></div>
<div style="height:3000px"></div>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(_PAGE)

    def log_message(self, *args: Any) -> None:
        return


@pytest.fixture
def site() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_click_type_and_screenshot_in_chromium(site: str) -> None:
    async with BrowserSession(allowed_domains=["127.0.0.1"]) as session:
        computer = computer_tool(session, display_width=800, display_height=600, start_url=site)

        first = await computer.execute(action="screenshot")
        png = base64.b64decode(images(first)[0].data)
        assert png.startswith(b"\x89PNG")
        assert int.from_bytes(png[16:20], "big") == 800

        await computer.execute(action="left_click", coordinate=[100, 35])
        await computer.execute(action="type", text="tulips")
        await computer.execute(action={"type": "click", "x": 60, "y": 100})
        page = await session.page()
        assert await page.inner_text("#out") == "searched tulips"

        await computer.execute(action="key", text="ctrl+a")
        await computer.execute(action={"type": "keypress", "keys": ["BACKSPACE"]})
        await computer.execute(action="scroll", coordinate=[400, 300], scroll_direction="down")
        await asyncio.sleep(0.2)
        assert await page.evaluate("window.scrollY") > 0
