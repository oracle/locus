# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Browser tools against a fake page: what the model can reach, and what it can't.

The live behaviour against real Chromium is in
``tests/integration/test_browser_live.py``.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from tulip.control import Action, ControlPolicy, InMemoryApprovals, gate_tool
from tulip.tools.browser import (
    READ_LABELS,
    SUBMIT_LABELS,
    WRITE_LABELS,
    BrowserError,
    BrowserSession,
    browser_toolset,
)


class _Page:
    """A page that knows a few addresses and where some selectors lead."""

    def __init__(self, site: dict[str, dict[str, Any]]) -> None:
        self.site = site
        self.url = "about:blank"
        self.filled: dict[str, str] = {}
        self.clicks: list[str] = []

    async def goto(self, url: str) -> None:
        self.url = url

    async def title(self) -> str:
        return str(self.site.get(self.url, {}).get("title", ""))

    async def inner_text(self, selector: str) -> str:
        return str(self.site.get(self.url, {}).get("text", ""))

    async def fill(self, selector: str, text: str) -> None:
        self.filled[selector] = text

    async def click(self, selector: str) -> None:
        self.clicks.append(selector)
        target = self.site.get(self.url, {}).get("links", {}).get(selector)
        if target:
            self.url = target


_SITE = {
    "https://shop.example.com/orders": {
        "title": "Orders",
        "text": "Order 4821 — refund pending",
        "links": {"#refund": "https://shop.example.com/done", "#away": "https://evil.example.net/"},
    },
    "https://shop.example.com/done": {"title": "Done", "text": "Refund issued"},
}


def _session(**options: Any) -> tuple[BrowserSession, _Page]:
    page = _Page(_SITE)

    async def factory() -> _Page:
        return page

    return BrowserSession(
        allowed_domains=["shop.example.com"], page_factory=factory, **options
    ), page


def _tools(session: BrowserSession, **options: Any) -> dict[str, Any]:
    return {t.name: t for t in browser_toolset(session, **options)}


@pytest.mark.asyncio
async def test_open_and_read_an_allowed_page() -> None:
    session, _ = _session()
    tools = _tools(session)

    opened = await tools["browser_open"].fn(url="https://shop.example.com/orders")

    assert opened.startswith("Orders (https://shop.example.com/orders)")
    assert "refund pending" in opened
    assert await tools["browser_read"].fn() == "Order 4821 — refund pending"


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example.net/",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "https://example.com.evil.net/",
    ],
)
@pytest.mark.asyncio
async def test_other_hosts_and_schemes_are_refused(url: str) -> None:
    session, page = _session()

    with pytest.raises(BrowserError):
        await _tools(session)["browser_open"].fn(url=url)
    assert page.url == "about:blank"


def test_subdomains_of_an_allowed_host_are_allowed() -> None:
    session, _ = _session()

    session.check_url("https://eu.shop.example.com/orders")
    BrowserSession().check_url("https://anything.example.org/")


@pytest.mark.asyncio
async def test_a_click_that_leaves_the_allowed_domains_is_undone() -> None:
    session, page = _session()
    tools = _tools(session)
    await tools["browser_open"].fn(url="https://shop.example.com/orders")

    with pytest.raises(BrowserError, match=r"evil\.example\.net"):
        await tools["browser_click"].fn(selector="#away")
    assert page.url == "about:blank"


@pytest.mark.asyncio
async def test_long_text_is_cut_with_a_note() -> None:
    session, page = _session(max_text_chars=10)
    await _tools(session)["browser_open"].fn(url="https://shop.example.com/orders")

    text = await _tools(session)["browser_read"].fn(selector="body")

    assert text.startswith("Order 4821")
    assert "more characters]" in text


def test_writes_and_submit_are_opt_in_and_labelled() -> None:
    session, _ = _session()

    assert set(_tools(session, allow_writes=False)) == {"browser_open", "browser_read"}
    tools = _tools(session, allow_submit=True)
    assert set(tools) == {
        "browser_open",
        "browser_read",
        "browser_click",
        "browser_type",
        "browser_submit",
    }
    assert tools["browser_read"].labels == READ_LABELS
    assert tools["browser_type"].labels == WRITE_LABELS
    assert tools["browser_submit"].labels == SUBMIT_LABELS


@pytest.mark.asyncio
async def test_a_gated_submit_waits_for_a_person() -> None:
    session, page = _session()
    tools = _tools(session, allow_submit=True)
    await tools["browser_open"].fn(url="https://shop.example.com/orders")
    await tools["browser_type"].fn(selector="#reason", text="damaged")
    store = InMemoryApprovals()
    submit = gate_tool(
        tools["browser_submit"],
        policy=ControlPolicy(
            require_verification_score=0.0, require_human_for=frozenset({"browser-submit"})
        ),
        action=lambda name, args: Action(
            name=name, asset=args["selector"], tags=frozenset({"browser-submit"})
        ),
        approval=store,
        on_refusal="interrupt",
    )

    held = json.loads(await submit.execute(selector="#refund"))
    assert held["__interrupt__"] is True
    assert page.clicks == []

    store.decide(held["metadata"]["approval_id"], "approved", by="alice")
    result = await submit.execute(selector="#refund")

    assert page.clicks == ["#refund"]
    assert page.filled == {"#reason": "damaged"}
    assert "Refund issued" in str(result)


@pytest.mark.asyncio
async def test_without_playwright_the_error_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.async_api", None)

    with pytest.raises(BrowserError, match=r"tulip-agents\[browser\]"):
        await BrowserSession().page()


@pytest.mark.asyncio
async def test_closing_a_session_that_never_launched_is_harmless() -> None:
    session, _ = _session()
    async with session:
        await session.page()
    assert session._page is None


@pytest.mark.asyncio
async def test_closing_a_launched_session_closes_the_browser_and_playwright() -> None:
    closed: list[str] = []

    class _Closable:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            closed.append(self.name)

        async def stop(self) -> None:
            closed.append(self.name)

    session = BrowserSession()
    session._browser = _Closable("browser")
    session._playwright = _Closable("playwright")
    session._page = object()

    await session.close()

    assert closed == ["browser", "playwright"]
    assert session._page is None
