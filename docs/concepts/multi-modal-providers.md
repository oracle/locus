# Multi-modal providers

The model is one provider an agent depends on. Production agents pull
from more: a web index, a page fetcher, an image renderer, a speech
synthesiser. Tulip exposes
those as a small set of **Protocol** types under `tulip.providers` and
an opt-in auto-registration step that turns each one into a
model-callable tool.

```python
from tulip.agent import Agent
from tulip.providers.web_fetch import HTTPXWebFetcher
from tulip.providers.web_search import OpenAISearchPreviewProvider
from tulip.providers.image import OpenAIImageProvider
from tulip.providers.speech import OpenAISpeechProvider
from tulip.models.native.openai import OpenAIModel

agent = Agent(
    model="openai:gpt-4o-mini",
    web_search=OpenAISearchPreviewProvider(OpenAIModel("gpt-4o-search-preview")),
    web_fetch=HTTPXWebFetcher(),
    image_generator=OpenAIImageProvider(model="dall-e-3"),
    speech_provider=OpenAISpeechProvider(),
)
```

Setting any of those four kwargs on `Agent` (or `AgentConfig`) registers
a matching `@tool`:

| Provider kwarg | Auto-registered tool(s) | Signature |
|---|---|---|
| `web_search=` | `web_search` | `query: str, max_results: int = 5` |
| `web_fetch=` | `web_fetch` | `url: str, max_chars: int = 50000` |
| `image_generator=` | `generate_image` | `prompt: str, size: str = "1024x1024", n: int = 1` |
| `speech_provider=` | `speak` and/or `transcribe` | depends on `provider.capabilities` |

The model can call these alongside hand-written `@tool` functions — they
share the same registry, the same idempotency machinery, the same hooks.

## The protocols

Each provider is a one- or two-method `typing.Protocol` decorated with
`@runtime_checkable`, so any duck-typed object that implements the
methods is accepted. You don't need to subclass.

- `BaseWebSearchProvider`: `async search(query, max_results)` →
  `list[SearchResult]`.
- `BaseWebFetchProvider`: `async fetch(url, max_chars, keep_html)` →
  `WebPage`.
- `BaseImageGenerationProvider`: `async generate(prompt, size, n)` →
  `list[ImageResult]`.
- `BaseSpeechProvider`: `capabilities: frozenset[str]` plus
  `async speak(text, voice)` and/or `async transcribe(audio_bytes,
  content_type)`.

The shared Pydantic types live in `tulip.providers.types` (`SearchResult`,
`WebPage`) and beside each protocol (`ImageResult`, `SynthesizedAudio`,
`SpeechTranscript`).

## Built-in implementations

- `HTTPXWebFetcher` — uses the `httpx` dep that's already in core, plus a
  stdlib `HTMLParser` shim that strips `<script>` / `<style>` and
  collapses whitespace. No `beautifulsoup` dep.
- `OpenAISearchPreviewProvider` — wraps OpenAI's `gpt-4o-search-preview`
  chat-completions model. The model performs the retrieval itself and
  returns annotated results; the provider pins them through a strict
  JSON schema and returns a list of `SearchResult`.
- `OpenAIImageProvider` — `images.generate` (`dall-e-3` /
  `gpt-image-1`). Surfaces hosted URLs when the API returns them and
  base64 PNG bytes otherwise.
- `OpenAISpeechProvider` — `audio.speech.create` (TTS,
  default `tts-1`) plus `audio.transcriptions.create` (Whisper, default
  `whisper-1`). Round-trips text → audio → text.

All four lazy-import `openai` / `httpx` so the SDK core stays free of
optional dependencies until you actually wire one of these in.

## Bring your own

The protocols are the contract — implement them and you're in. A
production user might wrap Bing for search, `trafilatura` for fetch,
a cloud vision API for image generation, or a speech API for STT/TTS. The agent
glue stays identical: set the kwarg on `AgentConfig`, the SDK registers
the tool.

```python
class BingSearch:
    async def search(self, query, *, max_results=5):
        ...  # call Bing, return list[SearchResult]

agent = Agent(
    model=...,
    web_search=BingSearch(),  # picked up via runtime_checkable Protocol
)
```

## What this is not

- **Not a replacement for `@tool`.** Hand-written tools still call your
  internal APIs and DBs. The provider registry is for the small set of
  modalities almost every agent needs.
- **Not multi-modal model wiring.** This is *capability* wiring — the
  model itself is still text-in / text-out. If you want a vision model
  reading screenshots, configure that on the model side.
- **Not a multi-modal output channel.** `speak` returns a tool-string
  summary so the model isn't fed raw audio bytes; the actual audio
  lives on the provider and your application code retrieves it from
  there when it's time to emit on a voice channel.

## Source and tests

- `src/tulip/providers/` — the four protocols, four implementations,
  and the `auto_register()` glue.
- `tests/unit/test_providers.py` — runtime-checkable protocols, tool
  factories, `AgentConfig` wiring.
- `tests/integration/test_providers_live.py` — live `httpx` fetch,
  live OpenAI search / image / speech (gated behind env vars).
