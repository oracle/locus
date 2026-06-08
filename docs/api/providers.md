# Multi-modal providers

Non-LLM provider Protocols — web search, web fetch, image generation,
text-to-speech, speech recognition. Setting any of them on
`AgentConfig` (`web_search=`, `web_fetch=`, `image_generator=`,
`speech_provider=`) auto-registers a matching `@tool` so the model
can call the capability the same way it calls any other tool.

For LLM providers, see [Models](models.md). For embedding providers
and vector stores, see [RAG](rag.md).

## Web search

::: tulip.providers.web_search.BaseWebSearchProvider
::: tulip.providers.web_search.OpenAISearchPreviewProvider

## Web fetch

::: tulip.providers.web_fetch.BaseWebFetchProvider
::: tulip.providers.web_fetch.HTTPXWebFetcher

## Image generation

::: tulip.providers.image.BaseImageGenerationProvider
::: tulip.providers.image.ImageResult

## Speech (TTS + ASR)

::: tulip.providers.speech.BaseSpeechProvider
::: tulip.providers.speech.SynthesizedAudio
::: tulip.providers.speech.SpeechTranscript

## Shared types

::: tulip.providers.types.SearchResult
::: tulip.providers.types.WebPage
