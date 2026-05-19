# Multi-modal providers

Non-LLM provider Protocols — web search, web fetch, image generation,
text-to-speech, speech recognition. Setting any of them on
`AgentConfig` (`web_search=`, `web_fetch=`, `image_generator=`,
`speech_provider=`) auto-registers a matching `@tool` so the model
can call the capability the same way it calls any other tool.

For LLM providers, see [Models](models.md). For embedding providers
and vector stores, see [RAG](rag.md).

## Web search

::: locus.providers.web_search.BaseWebSearchProvider
::: locus.providers.web_search.OpenAISearchPreviewProvider

## Web fetch

::: locus.providers.web_fetch.BaseWebFetchProvider
::: locus.providers.web_fetch.HTTPXWebFetcher

## Image generation

::: locus.providers.image.BaseImageGenerationProvider
::: locus.providers.image.ImageResult

## Speech (TTS + ASR)

::: locus.providers.speech.BaseSpeechProvider
::: locus.providers.speech.SynthesizedAudio
::: locus.providers.speech.SpeechTranscript

## Shared types

::: locus.providers.types.SearchResult
::: locus.providers.types.WebPage
