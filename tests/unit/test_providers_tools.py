# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage tests for ``locus.providers.tools`` — the auto-tool factories.

These factories build :class:`locus.tools.decorator.Tool` wrappers around
the multi-modal provider protocols. We use minimal stub providers so the
tests don't pull any SDKs.
"""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import MagicMock

import pytest

from locus.providers.image import ImageResult
from locus.providers.tools import (
    auto_register,
    make_image_generation_tool,
    make_speech_tools,
    make_web_fetch_tool,
    make_web_search_tool,
)
from locus.providers.types import SearchResult, WebPage


# ---------------------------------------------------------------------------
# Stub providers
# ---------------------------------------------------------------------------


class _StubSearch:
    def __init__(self, hits: list[SearchResult]) -> None:
        self._hits = hits

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        # Mirror the cap behavior of the real factory.
        return self._hits[:max_results]


class _StubFetch:
    def __init__(self, page: WebPage) -> None:
        self._page = page

    async def fetch(self, url: str, *, max_chars: int = 50_000, keep_html: bool = False) -> WebPage:
        return self._page


class _StubImage:
    def __init__(self, results: list[ImageResult]) -> None:
        self._results = results

    async def generate(self, prompt: str, *, size: str, n: int) -> list[ImageResult]:
        return self._results[:n]


class _StubSpeech:
    def __init__(self, caps: frozenset[str]) -> None:
        self.capabilities = caps

    async def speak(self, text: str, *, voice: str | None = None) -> Any:
        from locus.providers.speech import SynthesizedAudio

        return SynthesizedAudio(text=text, audio_bytes=b"\x00" * 64, content_type="audio/mpeg")

    async def transcribe(self, audio_bytes: bytes, *, content_type: str = "audio/mpeg") -> Any:
        from locus.providers.speech import SpeechTranscript

        return SpeechTranscript(text=f"len={len(audio_bytes)}")


# ---------------------------------------------------------------------------
# make_web_search_tool
# ---------------------------------------------------------------------------


class TestMakeWebSearchTool:
    @pytest.mark.asyncio
    async def test_returns_numbered_results(self) -> None:
        provider = _StubSearch(
            [
                SearchResult(title="A", url="https://a.example", snippet="alpha"),
                SearchResult(title="B", url="https://b.example", snippet="beta"),
            ]
        )
        t = make_web_search_tool(provider)
        out = await t.execute(query="x")
        assert "1. A" in out
        assert "https://a.example" in out
        assert "2. B" in out

    @pytest.mark.asyncio
    async def test_no_hits_returns_descriptive_message(self) -> None:
        t = make_web_search_tool(_StubSearch([]))
        out = await t.execute(query="empty")
        assert "No results" in out
        assert "empty" in out

    @pytest.mark.asyncio
    async def test_max_results_clamped_to_20(self) -> None:
        # 30 hits, clamp to 20.
        hits = [
            SearchResult(title=f"T{i}", url=f"https://x{i}.example", snippet="") for i in range(30)
        ]
        t = make_web_search_tool(_StubSearch(hits))
        out = await t.execute(query="q", max_results=99)
        # Twenty distinct entries → twenty leading "N." numbers.
        for i in range(1, 21):
            assert f"{i}. T" in out
        assert "21. T" not in out

    @pytest.mark.asyncio
    async def test_max_results_clamped_to_1(self) -> None:
        hits = [SearchResult(title="A", url="https://a.example", snippet="alpha")]
        t = make_web_search_tool(_StubSearch(hits))
        out = await t.execute(query="q", max_results=0)
        assert "1. A" in out


# ---------------------------------------------------------------------------
# make_web_fetch_tool
# ---------------------------------------------------------------------------


class TestMakeWebFetchTool:
    @pytest.mark.asyncio
    async def test_renders_title_url_text(self) -> None:
        page = WebPage(url="https://x.example/", status=200, title="Title", text="Body")
        t = make_web_fetch_tool(_StubFetch(page))
        out = await t.execute(url="https://x.example/")
        assert "# Title" in out
        assert "https://x.example/" in out
        assert "Body" in out

    @pytest.mark.asyncio
    async def test_no_title_uses_url_only_prefix(self) -> None:
        page = WebPage(url="https://x.example/", status=200, title="", text="raw")
        t = make_web_fetch_tool(_StubFetch(page))
        out = await t.execute(url="https://x.example/")
        assert "# " not in out
        assert "raw" in out

    @pytest.mark.asyncio
    async def test_truncated_marker_appended(self) -> None:
        page = WebPage(
            url="https://x.example/",
            status=200,
            title="T",
            text="x",
            truncated=True,
        )
        t = make_web_fetch_tool(_StubFetch(page))
        out = await t.execute(url="https://x.example/")
        assert "[truncated]" in out

    @pytest.mark.asyncio
    async def test_http_error_yields_status_and_excerpt(self) -> None:
        page = WebPage(
            url="https://x.example/",
            status=503,
            title="",
            text="overload" * 200,
        )
        t = make_web_fetch_tool(_StubFetch(page))
        out = await t.execute(url="https://x.example/")
        assert out.startswith("HTTP 503")
        # Body excerpt capped at 1000 chars
        assert len(out) <= 2000


# ---------------------------------------------------------------------------
# make_image_generation_tool
# ---------------------------------------------------------------------------


class TestMakeImageGenerationTool:
    @pytest.mark.asyncio
    async def test_returns_urls(self) -> None:
        provider = _StubImage(
            [
                ImageResult(prompt="p", url="https://i.example/a.png"),
                ImageResult(prompt="p", url="https://i.example/b.png"),
            ]
        )
        t = make_image_generation_tool(provider)
        out = await t.execute(prompt="two robots", n=2)
        assert "https://i.example/a.png" in out
        assert "https://i.example/b.png" in out

    @pytest.mark.asyncio
    async def test_clamps_n_to_4(self) -> None:
        provider = _StubImage([ImageResult(prompt="p", url=f"https://x{i}") for i in range(10)])
        t = make_image_generation_tool(provider)
        out = await t.execute(prompt="p", n=99)
        # 4-line cap
        assert out.count("\n") + 1 <= 5

    @pytest.mark.asyncio
    async def test_inline_base64_rendered_as_size_summary(self) -> None:
        b64 = base64.b64encode(b"\x00" * 100).decode("ascii")
        provider = _StubImage([ImageResult(prompt="p", b64_png=b64)])
        t = make_image_generation_tool(provider)
        out = await t.execute(prompt="p")
        assert "(inline base64 PNG, 100 bytes)" in out

    @pytest.mark.asyncio
    async def test_revised_prompt_appended(self) -> None:
        provider = _StubImage(
            [ImageResult(prompt="p", url="https://x.example/a.png", revised_prompt="rewritten")]
        )
        t = make_image_generation_tool(provider)
        out = await t.execute(prompt="p")
        assert "revised_prompt: rewritten" in out

    @pytest.mark.asyncio
    async def test_no_results_falls_back_to_message(self) -> None:
        provider = _StubImage([])
        t = make_image_generation_tool(provider)
        out = await t.execute(prompt="p")
        assert out == "(no images returned)"


# ---------------------------------------------------------------------------
# make_speech_tools
# ---------------------------------------------------------------------------


class TestMakeSpeechTools:
    @pytest.mark.asyncio
    async def test_tts_only_returns_one_tool(self) -> None:
        tools = make_speech_tools(_StubSpeech(frozenset({"tts"})))
        assert len(tools) == 1
        assert tools[0].name == "speak"
        out = await tools[0].execute(text="hi")
        assert "audio:" in out
        assert "audio/mpeg" in out

    @pytest.mark.asyncio
    async def test_stt_only_returns_one_tool(self) -> None:
        tools = make_speech_tools(_StubSpeech(frozenset({"stt"})))
        assert len(tools) == 1
        assert tools[0].name == "transcribe"
        b64 = base64.b64encode(b"abc").decode()
        out = await tools[0].execute(audio_b64=b64)
        assert out == "len=3"

    def test_no_caps_returns_empty_list(self) -> None:
        tools = make_speech_tools(_StubSpeech(frozenset()))
        assert tools == []

    def test_provider_without_capabilities_attr(self) -> None:
        # The factory uses ``getattr(provider, "capabilities", frozenset())``
        # so a provider that doesn't expose the attribute yields no tools.
        class _NoCaps:
            pass

        tools = make_speech_tools(_NoCaps())  # type: ignore[arg-type]
        assert tools == []


# ---------------------------------------------------------------------------
# auto_register
# ---------------------------------------------------------------------------


class TestAutoRegister:
    def test_registers_each_provider_kind(self) -> None:
        registry = MagicMock()
        registry.register = MagicMock()
        names = auto_register(
            tool_registry=registry,
            web_search=_StubSearch([]),
            web_fetch=_StubFetch(WebPage(url="x", status=200)),
            image_generator=_StubImage([]),
            speech_provider=_StubSpeech(frozenset({"tts", "stt"})),
        )
        # Five tools registered: web_search, web_fetch, generate_image, speak, transcribe.
        assert names == ["web_search", "web_fetch", "generate_image", "speak", "transcribe"]
        assert registry.register.call_count == 5

    def test_no_providers_registers_nothing(self) -> None:
        registry = MagicMock()
        names = auto_register(tool_registry=registry)
        assert names == []
        registry.register.assert_not_called()
