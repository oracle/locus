# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Web-fetch provider protocol + ``httpx`` implementation.

The protocol is one method: ``async fetch(url) -> WebPage``. The default
:class:`HTTPXWebFetcher` implementation uses the ``httpx`` dep that's
already in core, plus a tiny stdlib HTML→text shim so we don't pull in
``html2text`` / ``beautifulsoup`` for the common case.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Protocol, runtime_checkable

from locus.providers.types import WebPage


@runtime_checkable
class BaseWebFetchProvider(Protocol):
    """Protocol every web-fetch provider must implement."""

    async def fetch(
        self,
        url: str,
        *,
        max_chars: int = 50_000,
        keep_html: bool = False,
    ) -> WebPage:
        """Fetch ``url`` and return a normalized :class:`WebPage`.

        Implementations should follow redirects, time out within a
        reasonable budget, and cap the returned ``text`` at ``max_chars``
        to keep it agent-context friendly.
        """
        ...


class _HTMLToText(HTMLParser):
    """Minimal HTML → plain-text converter.

    Skips the contents of ``<script>`` / ``<style>`` blocks, collapses
    runs of whitespace, and emits one line per block-level element. This
    is sufficient for an agent reading a page; it doesn't preserve layout
    or tables. Production users who need richer extraction should ship
    a custom :class:`BaseWebFetchProvider` that wraps ``trafilatura`` or
    ``html2text``.
    """

    _BLOCK_TAGS = frozenset(
        {
            "p",
            "div",
            "br",
            "li",
            "tr",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "section",
            "article",
            "header",
            "footer",
            "main",
            "nav",
            "blockquote",
        }
    )
    _SKIP_TAGS = frozenset({"script", "style", "noscript", "iframe"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._buf: list[str] = []
        self._title: str | None = None
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in self._BLOCK_TAGS:
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in self._BLOCK_TAGS:
            self._buf.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title = (self._title or "") + data
            return
        self._buf.append(data)

    def text(self) -> str:
        joined = "".join(self._buf)
        joined = html.unescape(joined)
        joined = re.sub(r"[ \t\r\f\v]+", " ", joined)
        joined = re.sub(r"\n[ \t]+", "\n", joined)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        return joined.strip()

    def title(self) -> str:
        return (self._title or "").strip()


class HTTPXWebFetcher:
    """Default web-fetch provider using ``httpx`` + a stdlib HTML→text shim.

    Args:
        timeout_seconds: Per-request timeout. Default 10s.
        user_agent: ``User-Agent`` header. Default ``locus-web-fetch/1.0``.
        follow_redirects: Whether to follow redirects. Default True.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        user_agent: str = "locus-web-fetch/1.0",
        follow_redirects: bool = True,
    ) -> None:
        self._timeout = timeout_seconds
        self._ua = user_agent
        self._follow = follow_redirects

    async def fetch(
        self,
        url: str,
        *,
        max_chars: int = 50_000,
        keep_html: bool = False,
    ) -> WebPage:
        import httpx

        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=self._follow,
            headers={"User-Agent": self._ua},
        ) as client:
            resp = await client.get(url)

        body = resp.text or ""
        ctype = resp.headers.get("content-type", "")
        if "html" in ctype.lower():
            parser = _HTMLToText()
            try:
                parser.feed(body)
                parser.close()
                text = parser.text()
                title = parser.title()
            except (ValueError, TypeError, AttributeError):
                # Defensive: stdlib HTMLParser can choke on truly malformed
                # markup. Fall back to the raw body.
                text = body
                title = ""
        else:
            text = body
            title = ""

        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars]
            truncated = True

        return WebPage(
            url=str(resp.url),
            status=resp.status_code,
            title=title,
            text=text,
            html=body if keep_html else None,
            truncated=truncated,
        )


__all__ = ["BaseWebFetchProvider", "HTTPXWebFetcher"]
