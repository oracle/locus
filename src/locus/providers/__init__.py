# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Multi-modal provider registry.

Locus's first-class providers are LLM model providers (under
:mod:`locus.models`), embedding providers (under :mod:`locus.rag.embeddings`),
and vector stores (under :mod:`locus.rag.stores`). This package adds the
seam for **non-LLM** providers — image generation, text-to-speech, speech
recognition, web search, web fetch — so an agent can be wired with any of
those capabilities the same way it's wired with a model or a checkpointer.

Each provider type is a small Pydantic-friendly Protocol with one or two
async methods. We ship at least one concrete implementation per protocol;
users plug their own by implementing the protocol.

Headlines
---------

- :class:`BaseWebSearchProvider` — ``async search(query, max_results)``
- :class:`BaseWebFetchProvider`  — ``async fetch(url)``
- :class:`BaseImageGenerationProvider` — ``async generate(prompt, ...)``
- :class:`BaseSpeechProvider` — ``async speak(text, ...)`` + ``async transcribe(audio, ...)``

Auto-tool wiring
----------------

Setting any of these on :class:`AgentConfig` (``web_search=``,
``web_fetch=``, ``image_generator=``, ``speech_provider=``) auto-registers
a corresponding ``@tool`` so the model can call it without the user
hand-rolling a wrapper. See :func:`locus.providers.tools.auto_register`.
"""

from __future__ import annotations

from locus.providers.image import BaseImageGenerationProvider, ImageResult
from locus.providers.speech import (
    BaseSpeechProvider,
    SpeechTranscript,
    SynthesizedAudio,
)
from locus.providers.types import SearchResult, WebPage
from locus.providers.web_fetch import BaseWebFetchProvider, HTTPXWebFetcher
from locus.providers.web_search import (
    BaseWebSearchProvider,
    OpenAISearchPreviewProvider,
)


__all__ = [
    "BaseImageGenerationProvider",
    "ImageResult",
    "BaseSpeechProvider",
    "SpeechTranscript",
    "SynthesizedAudio",
    "BaseWebFetchProvider",
    "HTTPXWebFetcher",
    "BaseWebSearchProvider",
    "OpenAISearchPreviewProvider",
    "SearchResult",
    "WebPage",
]
