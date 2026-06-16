# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Live integration tests for the multi-modal provider registry.

These hit real services and are auto-skipped when credentials / network
aren't available:

- ``HTTPXWebFetcher`` against ``https://example.com`` (network-only).
- ``OpenAISearchPreviewProvider`` against ``gpt-4o-search-preview``
  (needs ``OPENAI_API_KEY``).
- ``OpenAIImageProvider`` against ``gpt-image-1`` (needs ``OPENAI_API_KEY``,
  costs cents per call — gated behind ``LOCUS_LIVE_IMAGE=1``).
- ``OpenAISpeechProvider`` round-trip TTS→STT (needs ``OPENAI_API_KEY``,
  costs cents per call — gated behind ``LOCUS_LIVE_SPEECH=1``).
- ``Agent`` end-to-end: configure with ``web_fetch=`` + ``web_search=``
  and verify the model actually calls the auto-registered tools
  (needs ``OPENAI_API_KEY``).
"""

from __future__ import annotations

import os
import socket

import pytest

from tests.integration.conftest import skip_without_openai


def _network_available() -> bool:
    try:
        with socket.create_connection(("example.com", 443), timeout=2.0):
            return True
    except (OSError, TimeoutError):
        return False


skip_without_network = pytest.mark.skipif(
    not _network_available(),
    reason="No outbound network to example.com",
)


# ---------------------------------------------------------------------------
# Web fetch — network only, no API key.
# ---------------------------------------------------------------------------


@skip_without_network
@pytest.mark.asyncio
async def test_httpx_web_fetcher_against_example_com() -> None:
    from locus.providers.web_fetch import HTTPXWebFetcher

    fetcher = HTTPXWebFetcher(timeout_seconds=10.0)
    page = await fetcher.fetch("https://example.com", max_chars=2000)

    assert page.status == 200
    assert "example.com" in page.url.lower() or page.url.startswith("https://")
    assert "Example Domain" in page.text
    assert page.title.lower().startswith("example domain")


@skip_without_network
@pytest.mark.asyncio
async def test_web_fetch_tool_renders_real_page() -> None:
    from locus.providers.tools import make_web_fetch_tool
    from locus.providers.web_fetch import HTTPXWebFetcher

    tool = make_web_fetch_tool(HTTPXWebFetcher(timeout_seconds=10.0))
    out = await tool.fn(url="https://example.com", max_chars=1000)
    assert "Example Domain" in out
    assert "https://example.com" in out


# ---------------------------------------------------------------------------
# Web search — needs OPENAI_API_KEY.
# ---------------------------------------------------------------------------


@skip_without_openai
@pytest.mark.asyncio
async def test_openai_search_preview_returns_results() -> None:
    from locus.models.native.openai import OpenAIModel
    from locus.providers.web_search import OpenAISearchPreviewProvider

    model = OpenAIModel(model="gpt-4o-search-preview", max_tokens=1024)
    provider = OpenAISearchPreviewProvider(model)
    hits = await provider.search("Oracle Generative AI service", max_results=3)
    assert len(hits) >= 1
    assert all(h.url.startswith(("http://", "https://")) for h in hits)


# ---------------------------------------------------------------------------
# Image generation — gated, costs money.
# ---------------------------------------------------------------------------


@skip_without_openai
@pytest.mark.skipif(
    os.getenv("LOCUS_LIVE_IMAGE") != "1",
    reason="Set LOCUS_LIVE_IMAGE=1 to run live DALL-E calls (costs ~$0.04)",
)
@pytest.mark.asyncio
async def test_openai_image_provider_round_trip() -> None:
    from locus.providers.image import OpenAIImageProvider

    provider = OpenAIImageProvider(model="gpt-image-1")
    results = await provider.generate(
        "a tiny cartoon orange octopus holding a wrench",
        size="1024x1024",
        n=1,
    )
    assert len(results) == 1
    r = results[0]
    assert r.url is not None or r.b64_png is not None
    if r.url is not None:
        assert r.url.startswith("https://")


# ---------------------------------------------------------------------------
# Speech — gated, costs money.
# ---------------------------------------------------------------------------


@skip_without_openai
@pytest.mark.skipif(
    os.getenv("LOCUS_LIVE_SPEECH") != "1",
    reason="Set LOCUS_LIVE_SPEECH=1 to run live TTS+Whisper calls",
)
@pytest.mark.asyncio
async def test_openai_speech_round_trip_tts_then_stt() -> None:
    from locus.providers.speech import OpenAISpeechProvider

    provider = OpenAISpeechProvider()
    audio = await provider.speak(
        "The quick brown fox jumps over the lazy dog.",
        voice="alloy",
    )
    assert len(audio.audio_bytes) > 1000  # MP3 with a real sentence
    assert audio.content_type == "audio/mpeg"

    transcript = await provider.transcribe(
        audio.audio_bytes,
        content_type="audio/mpeg",
    )
    text = transcript.text.lower()
    # Whisper isn't word-perfect; check for the distinctive nouns.
    assert "fox" in text
    assert "dog" in text


# ---------------------------------------------------------------------------
# End-to-end: an Agent picks up the auto-registered web_fetch tool and
# uses it to answer a question about a real page.
# ---------------------------------------------------------------------------


@skip_without_openai
@skip_without_network
@pytest.mark.asyncio
async def test_agent_uses_auto_registered_web_fetch() -> None:
    from locus.agent import Agent
    from locus.core.events import TerminateEvent, ToolCompleteEvent
    from locus.models.native.openai import OpenAIModel
    from locus.providers.web_fetch import HTTPXWebFetcher

    agent = Agent(
        model=OpenAIModel(model="gpt-4o-mini", max_tokens=512),
        web_fetch=HTTPXWebFetcher(timeout_seconds=10.0),
        max_iterations=4,
        system_prompt=(
            "You have a web_fetch(url) tool. When asked about a page, "
            "call web_fetch and answer from its content."
        ),
    )
    final_message = ""
    tool_calls: list[str] = []
    async for event in agent.run(
        "Use web_fetch on https://example.com and tell me the page's title."
    ):
        if isinstance(event, ToolCompleteEvent):
            tool_calls.append(event.tool_name)
        if isinstance(event, TerminateEvent):
            final_message = event.final_message or ""

    assert "web_fetch" in tool_calls, f"agent never called web_fetch (saw: {tool_calls})"
    assert "example domain" in final_message.lower()
