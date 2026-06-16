# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 52: Multi-modal providers — web search, web fetch, image, speech.

Set a provider on the Agent kwargs (web_search, web_fetch, image_generator,
speech_provider) and Locus auto-registers a matching @tool. The model
calls it the same way it calls a hand-written tool — you don't write the
wrapper.

- Four Protocols under locus.providers: search, fetch, image, speech.
- Live demo with HTTPXWebFetcher (no API key needed) against example.com.
- Bring-your-own: any duck-typed object that implements the protocol method.
- Optional OpenAI-backed providers (image, speech, search-preview).

Run it
    # Default: OCI Generative AI auto-detected from ~/.oci/config
    python examples/notebook_57_multimodal_providers.py

    # Offline / no credentials:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_57_multimodal_providers.py

Optional: set OPENAI_API_KEY to exercise the OpenAI-backed providers.
"""

from __future__ import annotations

import asyncio
import os

from config import get_model

from locus.agent import Agent, AgentConfig
from locus.providers.web_fetch import HTTPXWebFetcher


# Part 1: the four provider Protocols. Implement one and Locus accepts it.


def example_protocols():
    """Print the four Protocols you can implement to plug a backend."""
    print("=== Part 1: The four provider Protocols ===\n")

    print("locus.providers exposes four runtime_checkable Protocols:")
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
    """Configure providers; locus registers the tools."""
    print("\n=== Part 2: Auto-registered tools ===\n")

    # HTTPXWebFetcher is the only built-in that needs no API key.
    fetcher = HTTPXWebFetcher(timeout_seconds=10.0)

    agent = Agent(
        config=AgentConfig(
            model=get_model(),
            system_prompt="Use web_fetch to look up pages when asked.",
            max_iterations=4,
            web_fetch=fetcher,
        )
    )

    print(f"Registered tools: {sorted(agent._tool_registry.tools.keys())}")
    print()
    print("Notice `web_fetch` appears even though we didn't pass it via `tools=`.")
    print("Setting `web_fetch=` is enough — locus auto-registered the wrapper.")
    print()
    print("The same kwargs work for the other modalities:")
    print("  Agent(web_search=..., web_fetch=..., image_generator=..., speech_provider=...)")
    print()
    print("Each provider becomes one tool (or two — `speech_provider` yields")
    print("`speak` and/or `transcribe` depending on `provider.capabilities`).")


# Part 3: live demo — fetch example.com through the auto-registered tool.


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
    # without spending a round-trip on a trivial fetch.
    rendered = await tool.fn(url="https://example.com", max_chars=400)
    print("First 200 chars of the rendered tool output:")
    print(rendered[:200])
    print("...")


# Part 4: any duck-typed object implementing the protocol method works.


def example_byo_backend():
    """A toy custom search provider — any duck-typed class works."""
    print("\n=== Part 4: Bring your own backend ===\n")

    from locus.providers.types import SearchResult

    class StaticSearch:
        """Hard-coded results — swap for a real Bing / Tavily client."""

        async def search(self, query, *, max_results=5):
            return [
                SearchResult(
                    title=f"Result {i + 1} for {query!r}",
                    url=f"https://example.org/{i}",
                    snippet="snippet text",
                )
                for i in range(min(max_results, 3))
            ]

    agent = Agent(
        config=AgentConfig(
            model=get_model(),
            system_prompt="Use web_search when asked to look something up.",
            web_search=StaticSearch(),
        )
    )
    print(f"Registered tools: {sorted(agent._tool_registry.tools.keys())}")
    print()
    print("The model now has a `web_search(query, max_results)` tool that")
    print("calls our StaticSearch.search() under the hood. Swap StaticSearch")
    print("for a real client and you have a Bing / Tavily / DuckDuckGo agent.")


# Part 5: OpenAI-backed providers (only if OPENAI_API_KEY is set).


def example_openai_providers():
    """Show the wiring for the built-in OpenAI implementations."""
    print("\n=== Part 5: OpenAI-backed providers (optional) ===\n")

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — printing the wiring without instantiating.")
        print()
        print("Snippet you'd use with a key:")
        print("""
  from locus.providers.image import OpenAIImageProvider
  from locus.providers.speech import OpenAISpeechProvider
  from locus.providers.web_search import OpenAISearchPreviewProvider
  from locus.models.native.openai import OpenAIModel

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

    from locus.providers.image import OpenAIImageProvider
    from locus.providers.speech import OpenAISpeechProvider

    image = OpenAIImageProvider(model="dall-e-3")
    speech = OpenAISpeechProvider()
    print(f"Image provider: {type(image).__name__}, model=dall-e-3")
    print(f"Speech provider: {type(speech).__name__}, capabilities={speech.capabilities}")
    print()
    print("Set them on AgentConfig and the agent gets `generate_image`, ")
    print("`speak`, and `transcribe` tools without any extra wiring.")


if __name__ == "__main__":
    example_protocols()
    example_auto_register()
    asyncio.run(example_live_fetch())
    example_byo_backend()
    example_openai_providers()
