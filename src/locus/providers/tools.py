# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Auto-tool factories for multi-modal providers.

When an ``AgentConfig`` carries ``web_search``, ``web_fetch``,
``image_generator``, or ``speech_provider``, the agent registers a
matching ``@tool`` so the model can invoke them without the user
hand-rolling a wrapper. These functions build those tools.

Each factory returns a :class:`locus.tools.decorator.Tool` rather than a
bare function so the agent's tool registry treats them uniformly with
hand-written tools.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from locus.tools.decorator import Tool, tool


if TYPE_CHECKING:
    from locus.providers.image import BaseImageGenerationProvider
    from locus.providers.speech import BaseSpeechProvider
    from locus.providers.web_fetch import BaseWebFetchProvider
    from locus.providers.web_search import BaseWebSearchProvider


def make_web_search_tool(provider: BaseWebSearchProvider) -> Tool:
    """Wrap a ``BaseWebSearchProvider`` as a ``web_search`` tool.

    Tool signature exposed to the model::

        web_search(query: str, max_results: int = 5) -> str
    """

    @tool(name="web_search")
    async def web_search(query: str, max_results: int = 5) -> str:
        """Search the web. Returns a numbered list of results
        (title + URL + snippet) for the given query.

        Args:
            query: The search query.
            max_results: Maximum results to return (1..20).
        """
        max_results = max(1, min(int(max_results), 20))
        hits = await provider.search(query, max_results=max_results)
        if not hits:
            return f"No results for {query!r}."
        lines: list[str] = []
        for i, h in enumerate(hits, 1):
            lines.append(f"{i}. {h.title}\n   {h.url}\n   {h.snippet}")
        return "\n".join(lines)

    return web_search


def make_web_fetch_tool(provider: BaseWebFetchProvider) -> Tool:
    """Wrap a ``BaseWebFetchProvider`` as a ``web_fetch`` tool.

    Tool signature::

        web_fetch(url: str, max_chars: int = 50000) -> str
    """

    @tool(name="web_fetch")
    async def web_fetch(url: str, max_chars: int = 50_000) -> str:
        """Fetch a URL and return its cleaned text content.

        Args:
            url: The full URL to fetch (must include scheme).
            max_chars: Cap the returned text at this many characters.
        """
        page = await provider.fetch(url, max_chars=int(max_chars))
        if page.status >= 400:
            return f"HTTP {page.status} fetching {page.url}\n{page.text[:1000]}"
        prefix = f"# {page.title}\n{page.url}\n\n" if page.title else f"{page.url}\n\n"
        suffix = "\n\n[truncated]" if page.truncated else ""
        return prefix + page.text + suffix

    return web_fetch


def make_image_generation_tool(provider: BaseImageGenerationProvider) -> Tool:
    """Wrap a ``BaseImageGenerationProvider`` as a ``generate_image`` tool.

    Tool signature::

        generate_image(prompt: str, size: str = "1024x1024", n: int = 1) -> str
    """

    @tool(name="generate_image")
    async def generate_image(
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
    ) -> str:
        """Generate an image (or images) from a text prompt.

        Returns one URL per generated image, separated by newlines. When
        the provider returns base64 bytes (no hosted URL), the tool
        returns "(inline base64, NN bytes)" so the agent stops prompting
        for a URL.

        Args:
            prompt: The image description.
            size: Provider-specific size token (e.g. "1024x1024").
            n: How many images to generate.
        """
        results = await provider.generate(prompt, size=size, n=max(1, min(int(n), 4)))
        lines: list[str] = []
        for r in results:
            if r.url:
                lines.append(r.url)
            elif r.b64_png:
                # Don't echo the base64 payload — it's huge and useless to
                # the model. Tell the agent it exists.
                size_bytes = len(base64.b64decode(r.b64_png, validate=False))
                lines.append(f"(inline base64 PNG, {size_bytes} bytes)")
        if results and results[0].revised_prompt:
            lines.append(f"\nrevised_prompt: {results[0].revised_prompt}")
        return "\n".join(lines) or "(no images returned)"

    return generate_image


def make_speech_tools(provider: BaseSpeechProvider) -> list[Tool]:
    """Wrap a ``BaseSpeechProvider`` as ``speak`` and/or ``transcribe`` tools.

    Returns 0-2 tools depending on ``provider.capabilities``:

    - ``"tts"`` -> ``speak(text, voice=None) -> "(audio: NN bytes, audio/mpeg)"``
    - ``"stt"`` -> ``transcribe(audio_b64, content_type="audio/mpeg") -> str``

    Both tools surface non-text artifacts as terse strings so the model
    isn't fed audio bytes; the audio is held on the provider side and
    real callers should retrieve it via the provider directly when they
    need to emit it on a channel.
    """

    tools: list[Tool] = []
    caps: frozenset[str] = getattr(provider, "capabilities", frozenset())

    if "tts" in caps:

        @tool(name="speak")
        async def speak(text: str, voice: str | None = None) -> str:
            """Synthesize ``text`` to audio. Returns a one-line summary."""
            audio = await provider.speak(text, voice=voice)
            return (
                f"(audio: {len(audio.audio_bytes)} bytes, "
                f"{audio.content_type}, voice={voice or 'default'})"
            )

        tools.append(speak)

    if "stt" in caps:

        @tool(name="transcribe")
        async def transcribe(
            audio_b64: str,
            content_type: str = "audio/mpeg",
        ) -> str:
            """Transcribe base64-encoded audio bytes to text."""
            data = base64.b64decode(audio_b64)
            t = await provider.transcribe(data, content_type=content_type)
            return t.text

        tools.append(transcribe)

    return tools


def auto_register(
    *,
    tool_registry: Any,
    web_search: BaseWebSearchProvider | None = None,
    web_fetch: BaseWebFetchProvider | None = None,
    image_generator: BaseImageGenerationProvider | None = None,
    speech_provider: BaseSpeechProvider | None = None,
) -> list[str]:
    """Register provider-backed tools on a ``ToolRegistry``.

    Returns the list of tool names that were registered, so the agent's
    initialization can log / surface them. Skipping a kwarg leaves the
    corresponding tool unregistered.
    """
    names: list[str] = []
    if web_search is not None:
        t = make_web_search_tool(web_search)
        tool_registry.register(t)
        names.append(t.name)
    if web_fetch is not None:
        t = make_web_fetch_tool(web_fetch)
        tool_registry.register(t)
        names.append(t.name)
    if image_generator is not None:
        t = make_image_generation_tool(image_generator)
        tool_registry.register(t)
        names.append(t.name)
    if speech_provider is not None:
        for t in make_speech_tools(speech_provider):
            tool_registry.register(t)
            names.append(t.name)
    return names


__all__ = [
    "auto_register",
    "make_image_generation_tool",
    "make_speech_tools",
    "make_web_fetch_tool",
    "make_web_search_tool",
]
