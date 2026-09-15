# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Browser tools: open, read, click, type and submit, each an ordinary tool.

An agent that can drive a browser can also submit a form it should not. These
tools are plain :class:`~tulip.tools.decorator.Tool` objects, so hooks,
``gate_tool`` and the audit trail see every action, and each carries labels a
policy can match: ``browser-read`` for opening and reading, ``browser-write`` for
clicking and typing, ``browser-submit`` for submitting::

    session = BrowserSession(allowed_domains=["shop.example.com"])
    tools = browser_toolset(session, allow_submit=True)
    submit = next(t for t in tools if t.name == "browser_submit")
    tools = [t for t in tools if t is not submit] + [
        gate_tool(
            submit,
            policy=ControlPolicy(require_human_for=frozenset({"browser-submit"})),
            action=lambda name, args: Action(
                name=name,
                asset=args["selector"],
                tags=frozenset({"browser-submit"}),
            ),
            approval=store,
            on_refusal="interrupt",
        )
    ]

The session only opens ``http``/``https`` URLs. With ``allowed_domains`` it
refuses any other host, including one a click navigates to: the page is reset to
``about:blank`` and the tool raises. Needs Playwright
(``pip install "tulip-agents[browser]"`` then ``playwright install chromium``).
Run it inside a sandbox for untrusted sites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from tulip.tools.decorator import Tool, tool


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable


READ_LABELS = frozenset({"browser", "browser-read"})
WRITE_LABELS = frozenset({"browser", "browser-write"})
SUBMIT_LABELS = frozenset({"browser", "browser-write", "browser-submit"})

_BLANK = "about:blank"


class BrowserError(Exception):
    """The browser refused or failed an action."""


class BrowserSession:
    """One browser page shared by the browser tools.

    Args:
        allowed_domains: Hosts the page may be on; a subdomain of a listed host
            is allowed. ``None`` allows any ``http``/``https`` host.
        browser: ``"chromium"``, ``"firefox"`` or ``"webkit"``.
        headless: Run without a window.
        timeout_ms: Default timeout for every page action.
        max_text_chars: Longest page text returned to the model.
        page_factory: An async callable returning a page object, instead of
            launching Playwright; for tests and for bringing your own browser.
    """

    def __init__(
        self,
        *,
        allowed_domains: Iterable[str] | None = None,
        browser: str = "chromium",
        headless: bool = True,
        timeout_ms: int = 15_000,
        max_text_chars: int = 4_000,
        page_factory: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        self.allowed_domains = (
            None if allowed_domains is None else frozenset(d.lower() for d in allowed_domains)
        )
        self.browser = browser
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.max_text_chars = max_text_chars
        self._page_factory = page_factory
        self._page: Any = None
        self._playwright: Any = None
        self._browser: Any = None

    async def page(self) -> Any:
        """The live page, launching the browser on first use."""
        if self._page is not None:
            return self._page
        if self._page_factory is not None:
            self._page = await self._page_factory()
            return self._page
        try:
            from playwright.async_api import async_playwright  # noqa: PLC0415
        except ImportError as exc:
            raise BrowserError(
                'the browser tools need Playwright: pip install "tulip-agents[browser]" '
                "and then playwright install chromium"
            ) from exc
        self._playwright = await async_playwright().start()
        launcher = getattr(self._playwright, self.browser)
        self._browser = await launcher.launch(headless=self.headless)
        self._page = await self._browser.new_page()
        self._page.set_default_timeout(self.timeout_ms)
        return self._page

    def check_url(self, url: str) -> None:
        """Raise :class:`BrowserError` unless the page may be at ``url``."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise BrowserError(f"only http and https URLs can be opened, not {url!r}")
        if self.allowed_domains is None:
            return
        host = (parsed.hostname or "").lower()
        if not any(host == d or host.endswith("." + d) for d in self.allowed_domains):
            raise BrowserError(f"{host or url!r} is not one of the allowed domains")

    async def ensure_allowed(self) -> None:
        """After an action: if the page left the allowed domains, reset it and raise."""
        page = await self.page()
        url = str(page.url)
        if url == _BLANK:
            return
        try:
            self.check_url(url)
        except BrowserError as exc:
            await page.goto(_BLANK)
            raise BrowserError(f"the page navigated to {url}, which is not allowed: {exc}") from exc

    def excerpt(self, text: str) -> str:
        """``text``, cut to ``max_text_chars`` with a note of what was dropped."""
        if len(text) <= self.max_text_chars:
            return text
        dropped = len(text) - self.max_text_chars
        return f"{text[: self.max_text_chars]}\n[... {dropped} more characters]"

    async def close(self) -> None:
        """Close the browser, if this session launched one."""
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()
        self._page = self._browser = self._playwright = None

    async def __aenter__(self) -> BrowserSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


def browser_toolset(
    session: BrowserSession,
    *,
    allow_writes: bool = True,
    allow_submit: bool = False,
) -> list[Tool]:
    """The browser tools, bound to ``session``.

    Opening and reading are always included. Clicking and typing come with
    ``allow_writes``; submitting is opt-in with ``allow_submit`` and carries the
    ``browser-submit`` label, so it is the one to gate.
    """

    @tool(name="browser_open", labels=READ_LABELS, idempotent=True)
    async def browser_open(url: str) -> str:
        """Open a web page and return its title, address and text."""
        session.check_url(url)
        page = await session.page()
        await page.goto(url)
        await session.ensure_allowed()
        text = await page.inner_text("body")
        return f"{await page.title()} ({page.url})\n\n{session.excerpt(text)}"

    @tool(name="browser_read", labels=READ_LABELS, idempotent=True)
    async def browser_read(selector: str = "body") -> str:
        """Return the visible text of the element matching a CSS selector."""
        page = await session.page()
        return session.excerpt(await page.inner_text(selector))

    @tool(name="browser_click", labels=WRITE_LABELS)
    async def browser_click(selector: str) -> str:
        """Click the element matching a CSS selector."""
        page = await session.page()
        await page.click(selector)
        await session.ensure_allowed()
        return f"clicked {selector}; the page is at {page.url}"

    @tool(name="browser_type", labels=WRITE_LABELS)
    async def browser_type(selector: str, text: str) -> str:
        """Type text into the input matching a CSS selector, replacing its value."""
        page = await session.page()
        await page.fill(selector, text)
        return f"typed into {selector}"

    @tool(name="browser_submit", labels=SUBMIT_LABELS)
    async def browser_submit(selector: str) -> str:
        """Submit a form by clicking its submit control; returns the resulting page."""
        page = await session.page()
        await page.click(selector)
        await session.ensure_allowed()
        text = await page.inner_text("body")
        return f"submitted with {selector}; the page is at {page.url}\n\n{session.excerpt(text)}"

    tools = [browser_open, browser_read]
    if allow_writes:
        tools += [browser_click, browser_type]
    if allow_submit:
        tools.append(browser_submit)
    return tools


__all__ = [
    "READ_LABELS",
    "SUBMIT_LABELS",
    "WRITE_LABELS",
    "BrowserError",
    "BrowserSession",
    "browser_toolset",
]
