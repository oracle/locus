# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 57: Multi-modal phishing triage — lure pages, intel, image, speech.

A reported phishing case is rarely just text. The analyst has a lure URL
to fetch, a threat-intel feed to query, a screenshot of the lure page to
reproduce for the awareness-training corpus, and a vishing voicemail to
transcribe. Set a provider on the Agent kwargs (web_search, web_fetch,
image_generator, speech_provider) and Tulip auto-registers a matching
@tool. The model calls it the same way it calls a hand-written tool —
you don't write the wrapper.

- Four Protocols under tulip.providers: search, fetch, image, speech.
- Live demo with HTTPXWebFetcher (no API key needed) — example.com
  stands in for a reported suspicious lure URL.
- Bring-your-own: any duck-typed object that implements the protocol
  method, e.g. an internal threat-intel search.
- Optional OpenAI-backed providers (image, speech, search-preview).

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_57_multimodal_providers.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_57_multimodal_providers.py

Optional: set OPENAI_API_KEY to exercise the OpenAI-backed providers.
"""

from __future__ import annotations

import asyncio
import os

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.providers.web_fetch import HTTPXWebFetcher


# Part 1: the four provider Protocols. Implement one and Tulip accepts it.


def example_protocols():
    """Print the four Protocols you can implement to plug a backend."""
    print("=== Part 1: The four provider Protocols ===\n")

    print("tulip.providers exposes four runtime_checkable Protocols:")
    print()
    print("  BaseWebSearchProvider:    async search(query, max_results)")
    print("                            -> list[SearchResult]")
    print("  BaseWebFetchProvider:     async fetch(url, max_chars, keep_html)")
    print("                            -> WebPage")
    print("  BaseImageGenerationProvider: async generate(prompt, size, n)")
    print("                            -> list[ImageResult]")
    print("  BaseSpeechProvider:       capabilities: frozenset[str]")
    print("                            async speak(text, voice)")
    print("                            async transcribe(audio_bytes, content_type)")
    print()
    print("Any duck-typed object implementing the methods passes")
    print("`isinstance(obj, BaseXxxProvider)` — no subclassing required.")


# Part 2: setting a provider on AgentConfig auto-registers the tool.


def example_auto_register():
    """Configure providers; tulip registers the tools."""
    print("\n=== Part 2: Auto-registered tools ===\n")

    # HTTPXWebFetcher is the only built-in that needs no API key.
    fetcher = HTTPXWebFetcher(timeout_seconds=10.0)

    agent = Agent(
        config=AgentConfig(
            model=get_model(),
            system_prompt="Use web_fetch to retrieve reported URLs for analysis when asked.",
            max_iterations=4,
            web_fetch=fetcher,
        )
    )

    print(f"Registered tools: {sorted(agent._tool_registry.tools.keys())}")
    print()
    print("Notice `web_fetch` appears even though we didn't pass it via `tools=`.")
    print("Setting `web_fetch=` is enough — tulip auto-registered the wrapper.")
    print()
    print("The same kwargs work for the other modalities:")
    print("  Agent(web_search=..., web_fetch=..., image_generator=..., speech_provider=...)")
    print()
    print("Each provider becomes one tool (or two — `speech_provider` yields")
    print("`speak` and/or `transcribe` depending on `provider.capabilities`).")


# Part 3: live demo — fetch the page behind a reported link through the
# auto-registered tool. example.com stands in for the suspicious URL.


async def example_live_fetch():
    """Use the registered tool directly to verify the wiring."""
    print("\n=== Part 3: Live fetch via the registered tool ===\n")

    fetcher = HTTPXWebFetcher(timeout_seconds=10.0)
    agent = Agent(
        config=AgentConfig(
            model=get_model(),
            system_prompt="(unused — we'll call the tool directly)",
            web_fetch=fetcher,
        )
    )

    tool = agent._tool_registry.get("web_fetch")
    assert tool is not None, "web_fetch tool was not registered"

    # Calling tool.fn directly bypasses the model so we can verify wiring
    # without spending a round-trip on a trivial fetch. In a real triage
    # run the model would call this with the URL from the phishing report.
    rendered = await tool.fn(url="https://example.com", max_chars=400)
    print("First 200 chars of the rendered tool output:")
    print(rendered[:200])
    print("...")


# Part 4: any duck-typed object implementing the protocol method works —
# here, a stand-in for an internal threat-intel search service.


def example_byo_backend():
    """A toy threat-intel search provider — any duck-typed class works."""
    print("\n=== Part 4: Bring your own intel backend ===\n")

    from tulip.providers.types import SearchResult

    class StaticIntelSearch:
        """Hard-coded intel hits — swap for your real intel-platform client."""

        async def search(self, query, *, max_results=5):
            return [
                SearchResult(
                    title=f"Intel hit {i + 1} for {query!r}",
                    url=f"https://intel.example/advisories/{i}",
                    snippet="advisory snippet (mock)",
                )
                for i in range(min(max_results, 3))
            ]

    agent = Agent(
        config=AgentConfig(
            model=get_model(),
            system_prompt="Use web_search to query the intel feed when asked about an indicator.",
            web_search=StaticIntelSearch(),
        )
    )
    print(f"Registered tools: {sorted(agent._tool_registry.tools.keys())}")
    print()
    print("The model now has a `web_search(query, max_results)` tool that")
    print("calls our StaticIntelSearch.search() under the hood. Swap it for")
    print("your intel-platform client and the agent queries live feeds.")


# Part 5: OpenAI-backed providers (only if OPENAI_API_KEY is set).


def example_openai_providers():
    """Show the wiring for the built-in OpenAI implementations."""
    print("\n=== Part 5: OpenAI-backed providers (optional) ===\n")

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — printing the wiring without instantiating.")
        print()
        print("Snippet you'd use with a key:")
        print("""
  from tulip.providers.image import OpenAIImageProvider
  from tulip.providers.speech import OpenAISpeechProvider
  from tulip.providers.web_search import OpenAISearchPreviewProvider
  from tulip.models.native.openai import OpenAIModel

  agent = Agent(config=AgentConfig(
      model=get_model(),
      web_search=OpenAISearchPreviewProvider(
          OpenAIModel("gpt-4o-search-preview")
      ),
      image_generator=OpenAIImageProvider(model="dall-e-3"),
      speech_provider=OpenAISpeechProvider(),  # tts-1 + whisper-1
  ))
""")
        return

    from tulip.providers.image import OpenAIImageProvider
    from tulip.providers.speech import OpenAISpeechProvider

    image = OpenAIImageProvider(model="dall-e-3")
    speech = OpenAISpeechProvider()
    print(f"Image provider: {type(image).__name__}, model=dall-e-3")
    print(f"Speech provider: {type(speech).__name__}, capabilities={speech.capabilities}")
    print()
    print("Set them on AgentConfig and the agent gets `generate_image`, ")
    print("`speak`, and `transcribe` tools without extra wiring — e.g.")
    print("reproducing a reported lure screenshot for the awareness-training")
    print("corpus, spoken advisories, and vishing-voicemail transcription")
    print("in one phishing-triage agent.")


if __name__ == "__main__":
    example_protocols()
    example_auto_register()
    asyncio.run(example_live_fetch())
    example_byo_backend()
    example_openai_providers()
