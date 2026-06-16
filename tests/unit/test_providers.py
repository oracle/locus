# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the multi-modal provider registry.

Covers:

- Pydantic types in ``locus.providers.types`` are frozen and validate.
- The four ``BaseXxxProvider`` Protocols accept duck-typed implementations
  via ``runtime_checkable``.
- The auto-tool factories in ``locus.providers.tools`` produce real
  ``@tool``-decorated callables that delegate to the underlying provider
  and return the documented tool-string format.
- ``auto_register`` on a real ``ToolRegistry`` registers exactly the
  tools whose providers are non-None and skips the rest.
- ``AgentConfig`` accepts the new fields and ``Agent._initialize`` wires
  ``auto_register`` so a configured provider becomes a model-callable tool.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from locus.providers.image import BaseImageGenerationProvider, ImageResult
from locus.providers.speech import (
    BaseSpeechProvider,
    SpeechTranscript,
    SynthesizedAudio,
)
from locus.providers.tools import (
    auto_register,
    make_image_generation_tool,
    make_speech_tools,
    make_web_fetch_tool,
    make_web_search_tool,
)
from locus.providers.types import SearchResult, WebPage
from locus.providers.web_fetch import BaseWebFetchProvider
from locus.providers.web_search import BaseWebSearchProvider
from locus.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fakes — duck-typed provider stand-ins. Each tracks the calls it receives
# so the tests can assert the tool delegated correctly.
# ---------------------------------------------------------------------------


class _FakeWebSearch:
    """Returns a fixed list of hits and records the queries it saw."""

    def __init__(self, hits: list[SearchResult]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> list[SearchResult]:
        self.calls.append((query, max_results))
        return list(self.hits[:max_results])


class _FakeWebFetcher:
    """Returns a fixed page and records the URLs it saw."""

    def __init__(self, page: WebPage) -> None:
        self.page = page
        self.calls: list[tuple[str, int]] = []

    async def fetch(
        self,
        url: str,
        *,
        max_chars: int = 50_000,
        keep_html: bool = False,
    ) -> WebPage:
        self.calls.append((url, max_chars))
        return self.page


class _FakeImageGen:
    """Returns a fixed list of results and records the prompts it saw."""

    def __init__(self, results: list[ImageResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, str, int]] = []

    async def generate(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        n: int = 1,
        **kwargs: Any,
    ) -> list[ImageResult]:
        self.calls.append((prompt, size, n))
        return list(self.results[:n])


class _FakeSpeech:
    """Configurable TTS / STT fake."""

    def __init__(
        self,
        *,
        tts: bool = True,
        stt: bool = True,
        synthesized: SynthesizedAudio | None = None,
        transcript: SpeechTranscript | None = None,
    ) -> None:
        caps: set[str] = set()
        if tts:
            caps.add("tts")
        if stt:
            caps.add("stt")
        self.capabilities = frozenset(caps)
        self._synth = synthesized
        self._tx = transcript
        self.spoke: list[tuple[str, str | None]] = []
        self.transcribed: list[tuple[bytes, str]] = []

    async def speak(
        self,
        text: str,
        *,
        voice: str | None = None,
        **_: Any,
    ) -> SynthesizedAudio:
        if "tts" not in self.capabilities or self._synth is None:
            raise NotImplementedError
        self.spoke.append((text, voice))
        return self._synth

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        content_type: str = "audio/mpeg",
        **_: Any,
    ) -> SpeechTranscript:
        if "stt" not in self.capabilities or self._tx is None:
            raise NotImplementedError
        self.transcribed.append((audio_bytes, content_type))
        return self._tx


# ---------------------------------------------------------------------------
# Pydantic types
# ---------------------------------------------------------------------------


class TestProviderTypes:
    def test_search_result_is_frozen(self) -> None:
        r = SearchResult(title="t", url="https://x", snippet="s")
        with pytest.raises((TypeError, ValueError)):
            r.title = "z"  # type: ignore[misc]

    def test_web_page_defaults(self) -> None:
        p = WebPage(url="https://x", status=200)
        assert p.title == ""
        assert p.text == ""
        assert p.html is None
        assert p.truncated is False

    def test_image_result_optional_fields(self) -> None:
        r = ImageResult(prompt="cat", url="https://img/cat.png")
        assert r.b64_png is None
        assert r.revised_prompt is None

    def test_synthesized_audio_round_trips_bytes(self) -> None:
        a = SynthesizedAudio(text="hi", audio_bytes=b"\x00\x01\x02")
        assert a.audio_bytes == b"\x00\x01\x02"


# ---------------------------------------------------------------------------
# Protocol runtime-checkability — fakes pass isinstance() against the
# protocol. Guards the ``@runtime_checkable`` decoration on each protocol.
# ---------------------------------------------------------------------------


class TestProtocolsAreRuntimeCheckable:
    def test_web_search_protocol(self) -> None:
        assert isinstance(_FakeWebSearch([]), BaseWebSearchProvider)

    def test_web_fetch_protocol(self) -> None:
        page = WebPage(url="https://x", status=200)
        assert isinstance(_FakeWebFetcher(page), BaseWebFetchProvider)

    def test_image_generation_protocol(self) -> None:
        assert isinstance(_FakeImageGen([]), BaseImageGenerationProvider)

    def test_speech_protocol(self) -> None:
        assert isinstance(_FakeSpeech(), BaseSpeechProvider)


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    @pytest.mark.asyncio
    async def test_returns_numbered_list(self) -> None:
        provider = _FakeWebSearch(
            [
                SearchResult(title="A", url="https://a", snippet="aa"),
                SearchResult(title="B", url="https://b", snippet="bb"),
            ]
        )
        t = make_web_search_tool(provider)
        assert t.name == "web_search"
        out = await t.fn(query="anything", max_results=2)
        assert "1. A" in out
        assert "https://a" in out
        assert "2. B" in out
        assert provider.calls == [("anything", 2)]

    @pytest.mark.asyncio
    async def test_clamps_max_results(self) -> None:
        provider = _FakeWebSearch([SearchResult(title="A", url="https://a")])
        t = make_web_search_tool(provider)
        await t.fn(query="q", max_results=999)
        assert provider.calls[-1][1] == 20

    @pytest.mark.asyncio
    async def test_handles_no_hits(self) -> None:
        provider = _FakeWebSearch([])
        t = make_web_search_tool(provider)
        out = await t.fn(query="empty")
        assert "No results" in out


class TestWebFetchTool:
    @pytest.mark.asyncio
    async def test_formats_page_with_title(self) -> None:
        page = WebPage(url="https://x", status=200, title="Hello", text="world")
        t = make_web_fetch_tool(_FakeWebFetcher(page))
        assert t.name == "web_fetch"
        out = await t.fn(url="https://x")
        assert out.startswith("# Hello")
        assert "https://x" in out
        assert "world" in out

    @pytest.mark.asyncio
    async def test_marks_truncated(self) -> None:
        page = WebPage(url="https://x", status=200, text="long", truncated=True)
        t = make_web_fetch_tool(_FakeWebFetcher(page))
        out = await t.fn(url="https://x")
        assert out.endswith("[truncated]")

    @pytest.mark.asyncio
    async def test_renders_http_error(self) -> None:
        page = WebPage(url="https://x", status=503, text="oops")
        t = make_web_fetch_tool(_FakeWebFetcher(page))
        out = await t.fn(url="https://x")
        assert "HTTP 503" in out


class TestImageTool:
    @pytest.mark.asyncio
    async def test_returns_urls(self) -> None:
        provider = _FakeImageGen(
            [
                ImageResult(prompt="cat", url="https://img/1.png"),
                ImageResult(prompt="cat", url="https://img/2.png"),
            ]
        )
        t = make_image_generation_tool(provider)
        assert t.name == "generate_image"
        out = await t.fn(prompt="cat", n=2)
        assert "https://img/1.png" in out
        assert "https://img/2.png" in out

    @pytest.mark.asyncio
    async def test_returns_b64_summary_when_no_url(self) -> None:
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        provider = _FakeImageGen(
            [
                ImageResult(
                    prompt="cat",
                    b64_png=base64.b64encode(png_bytes).decode("ascii"),
                ),
            ]
        )
        t = make_image_generation_tool(provider)
        out = await t.fn(prompt="cat")
        assert "inline base64 PNG" in out
        assert str(len(png_bytes)) in out

    @pytest.mark.asyncio
    async def test_includes_revised_prompt(self) -> None:
        provider = _FakeImageGen(
            [
                ImageResult(
                    prompt="cat",
                    url="https://img/1.png",
                    revised_prompt="a cute orange tabby",
                )
            ]
        )
        t = make_image_generation_tool(provider)
        out = await t.fn(prompt="cat")
        assert "revised_prompt: a cute orange tabby" in out


class TestSpeechTools:
    @pytest.mark.asyncio
    async def test_tts_only_provider_yields_speak(self) -> None:
        provider = _FakeSpeech(
            tts=True,
            stt=False,
            synthesized=SynthesizedAudio(text="hi", audio_bytes=b"abc"),
        )
        tools = make_speech_tools(provider)
        assert {t.name for t in tools} == {"speak"}
        out = await tools[0].fn(text="hi", voice="alloy")
        assert "audio:" in out
        assert "audio/mpeg" in out
        assert "voice=alloy" in out

    @pytest.mark.asyncio
    async def test_stt_only_provider_yields_transcribe(self) -> None:
        provider = _FakeSpeech(
            tts=False,
            stt=True,
            transcript=SpeechTranscript(text="hello world"),
        )
        tools = make_speech_tools(provider)
        assert {t.name for t in tools} == {"transcribe"}
        b64 = base64.b64encode(b"\x00\x01").decode("ascii")
        out = await tools[0].fn(audio_b64=b64)
        assert out == "hello world"

    def test_no_capabilities_yields_no_tools(self) -> None:
        provider = _FakeSpeech(tts=False, stt=False)
        assert make_speech_tools(provider) == []


# ---------------------------------------------------------------------------
# auto_register
# ---------------------------------------------------------------------------


class TestAutoRegister:
    def test_registers_only_set_providers(self) -> None:
        registry = ToolRegistry()
        names = auto_register(
            tool_registry=registry,
            web_search=_FakeWebSearch([]),
            web_fetch=None,
            image_generator=None,
            speech_provider=None,
        )
        assert names == ["web_search"]
        assert "web_search" in registry.tools
        assert "web_fetch" not in registry.tools

    def test_registers_all_four_modalities(self) -> None:
        registry = ToolRegistry()
        names = auto_register(
            tool_registry=registry,
            web_search=_FakeWebSearch([]),
            web_fetch=_FakeWebFetcher(WebPage(url="https://x", status=200)),
            image_generator=_FakeImageGen([]),
            speech_provider=_FakeSpeech(
                tts=True,
                stt=True,
                synthesized=SynthesizedAudio(text="", audio_bytes=b""),
                transcript=SpeechTranscript(text=""),
            ),
        )
        assert set(names) == {"web_search", "web_fetch", "generate_image", "speak", "transcribe"}
        assert set(registry.tools.keys()) >= {
            "web_search",
            "web_fetch",
            "generate_image",
            "speak",
            "transcribe",
        }


# ---------------------------------------------------------------------------
# OpenAIModel.search-preview detection — the search provider relies on
# ``_rejects_sampling_params`` to drop temperature/top_p, so guard it here.
# ---------------------------------------------------------------------------


class TestSearchPreviewSamplingGate:
    def test_detects_native_search_preview(self) -> None:
        from locus.models.native.openai import OpenAIModel

        assert OpenAIModel._rejects_sampling_params("gpt-4o-search-preview")
        assert OpenAIModel._rejects_sampling_params("gpt-4o-mini-search-preview")

    def test_detects_oci_namespaced_search_preview(self) -> None:
        from locus.models.native.openai import OpenAIModel

        assert OpenAIModel._rejects_sampling_params("openai.gpt-4o-search-preview")

    def test_regular_models_unaffected(self) -> None:
        from locus.models.native.openai import OpenAIModel

        assert not OpenAIModel._rejects_sampling_params("gpt-4o-mini")
        assert not OpenAIModel._rejects_sampling_params("gpt-5.1")
        assert not OpenAIModel._rejects_sampling_params("openai.gpt-4o")


# ---------------------------------------------------------------------------
# AgentConfig + Agent._initialize wiring
# ---------------------------------------------------------------------------


class TestAgentConfigWiring:
    def test_config_accepts_provider_fields(self) -> None:
        from locus.agent.config import AgentConfig

        cfg = AgentConfig(
            model="openai:gpt-4o-mini",
            web_search=_FakeWebSearch([]),
            web_fetch=_FakeWebFetcher(WebPage(url="https://x", status=200)),
            image_generator=_FakeImageGen([]),
            speech_provider=_FakeSpeech(
                tts=True,
                stt=True,
                synthesized=SynthesizedAudio(text="", audio_bytes=b""),
                transcript=SpeechTranscript(text=""),
            ),
        )
        assert cfg.web_search is not None
        assert cfg.web_fetch is not None
        assert cfg.image_generator is not None
        assert cfg.speech_provider is not None

    def test_agent_registers_provider_tools_on_initialize(self) -> None:
        from locus.agent import Agent

        class _StubModel:
            async def complete(self, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError

            async def stream(self, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError

        agent = Agent(
            model=_StubModel(),
            web_search=_FakeWebSearch([]),
            web_fetch=_FakeWebFetcher(WebPage(url="https://x", status=200)),
            image_generator=_FakeImageGen([]),
        )
        agent._initialize()
        registered = set(agent._tool_registry.tools.keys())
        assert {"web_search", "web_fetch", "generate_image"} <= registered
