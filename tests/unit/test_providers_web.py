# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage tests for web providers (``locus.providers.web_fetch`` and
``locus.providers.web_search``).

The fetch provider is HTTP-bound, so we use ``respx`` to mock httpx.
The search provider drives an OpenAI-style model — we mock that with
``AsyncMock`` returning a canned ``ModelResponse``-shape.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from locus.providers.types import WebPage
from locus.providers.web_fetch import (
    BaseWebFetchProvider,
    HTTPXWebFetcher,
    _HTMLToText,
)
from locus.providers.web_search import (
    BaseWebSearchProvider,
    OpenAISearchPreviewProvider,
)


# ---------------------------------------------------------------------------
# _HTMLToText helper
# ---------------------------------------------------------------------------


class TestHTMLToText:
    def test_extracts_title_and_text(self) -> None:
        parser = _HTMLToText()
        parser.feed(
            "<html><head><title>Hello</title></head><body><p>First</p><p>Second</p></body></html>"
        )
        parser.close()
        assert parser.title() == "Hello"
        text = parser.text()
        assert "First" in text
        assert "Second" in text

    def test_skips_script_and_style(self) -> None:
        parser = _HTMLToText()
        parser.feed(
            "<html><body>"
            "<script>alert('x')</script>"
            "<style>body { color: red; }</style>"
            "<p>visible</p>"
            "</body></html>"
        )
        parser.close()
        text = parser.text()
        assert "alert" not in text
        assert "color" not in text
        assert "visible" in text

    def test_collapses_whitespace(self) -> None:
        parser = _HTMLToText()
        parser.feed("<p>a    b\t\tc</p>")
        parser.close()
        # whitespace runs collapse to single space
        assert "a b c" in parser.text()

    def test_block_tags_emit_newlines(self) -> None:
        parser = _HTMLToText()
        parser.feed("<div>one</div><div>two</div>")
        parser.close()
        text = parser.text()
        # Two block-level divs should be on separate lines.
        assert "one" in text
        assert "two" in text
        assert "\n" in text

    def test_no_title_returns_empty(self) -> None:
        parser = _HTMLToText()
        parser.feed("<p>no title here</p>")
        parser.close()
        assert parser.title() == ""

    def test_unescapes_html_entities(self) -> None:
        parser = _HTMLToText()
        parser.feed("<p>&amp;quot;hi&amp;quot;</p>")
        parser.close()
        # &amp; → &, then the inner &quot; → "
        assert '"hi"' in parser.text()


# ---------------------------------------------------------------------------
# HTTPXWebFetcher
# ---------------------------------------------------------------------------


class TestHTTPXWebFetcher:
    @pytest.mark.asyncio
    async def test_fetch_html_extracts_text(self) -> None:
        fetcher = HTTPXWebFetcher()
        with respx.mock(assert_all_called=True) as router:
            router.get("https://example.com/").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/html; charset=utf-8"},
                    text="<html><head><title>T</title></head><body><p>Body text</p></body></html>",
                )
            )
            page = await fetcher.fetch("https://example.com/")
        assert isinstance(page, WebPage)
        assert page.status == 200
        assert page.title == "T"
        assert "Body text" in page.text
        assert page.html is None  # keep_html defaults to False
        assert page.truncated is False

    @pytest.mark.asyncio
    async def test_fetch_keep_html_returns_raw(self) -> None:
        fetcher = HTTPXWebFetcher()
        raw = "<html><body><p>X</p></body></html>"
        with respx.mock() as router:
            router.get("https://example.com/").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                    text=raw,
                )
            )
            page = await fetcher.fetch("https://example.com/", keep_html=True)
        assert page.html == raw

    @pytest.mark.asyncio
    async def test_fetch_non_html_returns_body(self) -> None:
        fetcher = HTTPXWebFetcher()
        with respx.mock() as router:
            router.get("https://example.com/api").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "application/json"},
                    text='{"ok": true}',
                )
            )
            page = await fetcher.fetch("https://example.com/api")
        # Non-HTML: text is the raw body and title is empty
        assert page.text == '{"ok": true}'
        assert page.title == ""

    @pytest.mark.asyncio
    async def test_fetch_truncates_when_over_max_chars(self) -> None:
        fetcher = HTTPXWebFetcher()
        big = "x" * 1000
        with respx.mock() as router:
            router.get("https://example.com/").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/plain"},
                    text=big,
                )
            )
            page = await fetcher.fetch("https://example.com/", max_chars=50)
        assert page.truncated is True
        assert len(page.text) == 50

    @pytest.mark.asyncio
    async def test_fetch_falls_back_when_html_parsing_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the stdlib parser raises, the fetcher returns the raw body."""

        class _BoomParser:
            def feed(self, _: str) -> None:
                raise ValueError("malformed")

            def close(self) -> None:
                pass

            def text(self) -> str:  # pragma: no cover — unreachable
                return ""

            def title(self) -> str:  # pragma: no cover — unreachable
                return ""

        from locus.providers import web_fetch as web_fetch_mod

        monkeypatch.setattr(web_fetch_mod, "_HTMLToText", _BoomParser)

        fetcher = HTTPXWebFetcher()
        with respx.mock() as router:
            router.get("https://example.com/").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                    text="<broken>",
                )
            )
            page = await fetcher.fetch("https://example.com/")
        assert page.text == "<broken>"
        assert page.title == ""

    def test_provider_is_runtime_checkable(self) -> None:
        """``HTTPXWebFetcher`` must satisfy the protocol via duck typing."""
        assert isinstance(HTTPXWebFetcher(), BaseWebFetchProvider)


# ---------------------------------------------------------------------------
# OpenAISearchPreviewProvider
# ---------------------------------------------------------------------------


def _build_response(content: str) -> Any:
    """Stand-in for ``ModelResponse`` returned by ``OpenAIModel.complete``."""
    msg = MagicMock()
    msg.content = content
    resp = MagicMock()
    resp.message = msg
    return resp


class TestOpenAISearchPreviewProvider:
    @pytest.mark.asyncio
    async def test_returns_parsed_results(self) -> None:
        model = MagicMock()
        model.complete = AsyncMock(
            return_value=_build_response(
                '{"results":['
                '{"title":"A","url":"https://a.example","snippet":"alpha"},'
                '{"title":"B","url":"https://b.example","snippet":"beta"}'
                "]}"
            )
        )
        provider = OpenAISearchPreviewProvider(model, max_chars_per_snippet=100)
        out = await provider.search("hello", max_results=2)
        assert len(out) == 2
        assert out[0].url == "https://a.example"
        assert out[1].title == "B"

    @pytest.mark.asyncio
    async def test_returns_empty_on_unparseable_content(self) -> None:
        model = MagicMock()
        model.complete = AsyncMock(return_value=_build_response("not-json-at-all"))
        provider = OpenAISearchPreviewProvider(model)
        out = await provider.search("query")
        assert out == []

    @pytest.mark.asyncio
    async def test_caps_each_snippet_to_max_chars(self) -> None:
        model = MagicMock()
        long_snippet = "z" * 1000
        model.complete = AsyncMock(
            return_value=_build_response(
                '{"results":[{"title":"T","url":"https://x.example",'
                f'"snippet":"{long_snippet}"}}]}}'
            )
        )
        provider = OpenAISearchPreviewProvider(model, max_chars_per_snippet=20)
        out = await provider.search("q")
        assert len(out[0].snippet) == 20

    @pytest.mark.asyncio
    async def test_truncates_to_max_results(self) -> None:
        model = MagicMock()
        model.complete = AsyncMock(
            return_value=_build_response(
                '{"results":['
                '{"title":"1","url":"https://1.example","snippet":""},'
                '{"title":"2","url":"https://2.example","snippet":""},'
                '{"title":"3","url":"https://3.example","snippet":""}'
                "]}"
            )
        )
        provider = OpenAISearchPreviewProvider(model)
        out = await provider.search("q", max_results=2)
        assert len(out) == 2
        assert out[0].title == "1"
        assert out[1].title == "2"

    def test_provider_is_runtime_checkable(self) -> None:
        model = MagicMock()
        provider = OpenAISearchPreviewProvider(model)
        assert isinstance(provider, BaseWebSearchProvider)
