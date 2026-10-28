# Multi-modal providers

The model is only one provider an agent depends on. A support agent
may also need web search for order lookups, a fetcher for policy
pages, image generation for report figures, and transcription for
call recordings. Locus exposes those as a small set of **Protocol**
types under `locus.providers` and an opt-in auto-registration step
that turns each one into a model-callable tool.

```python
from locus.agent import Agent
from locus.providers.web_fetch import HTTPXWebFetcher
from locus.providers.web_search import OpenAISearchPreviewProvider
from locus.providers.image import OpenAIImageProvider
from locus.providers.speech import OpenAISpeechProvider
from locus.models import OpenAIModel

agent = Agent(
    model="openai:gpt-4o-mini",
    web_search=OpenAISearchPreviewProvider(OpenAIModel("gpt-5-search-api")),  # order / product lookups
    web_fetch=HTTPXWebFetcher(),         # pull policy pages, docs, status pages
    image_generator=OpenAIImageProvider(),  # gpt-image-2 by default — render report figures
    speech_provider=OpenAISpeechProvider(),  # transcribe call recordings
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

The model can call these alongside your own `@tool` set —
`lookup_order`, `issue_refund`, `deploy_service` —
sharing the same registry, the same idempotency machinery, the same hooks.

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

The shared Pydantic types live in `locus.providers.types` (`SearchResult`,
`WebPage`) and beside each protocol (`ImageResult`, `SynthesizedAudio`,
`SpeechTranscript`).

## Built-in implementations

- `HTTPXWebFetcher` — uses the `httpx` dep that's already in core, plus a
  stdlib `HTMLParser` shim that strips `<script>` / `<style>` and
  collapses whitespace. No `beautifulsoup` dep. Good for pulling
  policy pages, vendor docs, and status pages.
- `OpenAISearchPreviewProvider` — wraps an OpenAI search model
  (`gpt-5-search-api`) on chat-completions. The model performs the
  retrieval itself; the provider pins its results to a strict
  JSON schema and returns a list of `SearchResult` — handy for a quick
  open-web pivot before you commit to a hard API call.
- `OpenAIImageProvider` — `images.generate`, defaulting to
  `gpt-image-2` (`gpt-image-1` is retired). `dall-e-3` / `dall-e-2` are deprecated by OpenAI; pass
  one explicitly only if you still need it. Surfaces hosted URLs when
  the API returns them and base64 PNG bytes otherwise. Use it to render
  figures and diagrams for a report.
- `OpenAISpeechProvider` — `audio.speech.create` (TTS,
  default `tts-1`) plus `audio.transcriptions.create` (Whisper, default
  `whisper-1`). Transcribe call recordings into text the agent can
  reason over.

All four lazy-import `openai` / `httpx` so the SDK core stays free of
optional dependencies until you actually wire one of these in.

## Bring your own

The protocols are the contract — implement them and you're in. A
support team might wrap its own catalog search API, `trafilatura` for
fetching policy pages, a charting API for report figures, or a
transcription API for call recordings; a security team might wrap a
threat-intel API for search the same way. The agent glue stays
identical: set the kwarg on `AgentConfig`, the SDK registers the tool.

```python
class CatalogSearch:
    async def search(self, query, *, max_results=5):
        ...  # call your commerce API, return list[SearchResult]
              # of matching products / orders

agent = Agent(
    model=...,
    web_search=CatalogSearch(),  # picked up via runtime_checkable Protocol
)
```

## What this is not

- **Not a replacement for `@tool`.** Hand-written tools still call
  your billing, CRM, and infra APIs (`lookup_order`, `issue_refund`,
  `deploy_service`). The provider registry is for the
  small set of modalities almost every agent needs.
- **Not multi-modal model wiring.** This is *capability* wiring — the
  model itself is still text-in / text-out. If you want a vision model
  reading screenshots, configure that on the model side.
- **Not a multi-modal output channel.** `speak` returns a tool-string
  summary so the model isn't fed raw audio bytes; the actual audio
  lives on the provider and your application code retrieves it from
  there when it's time to emit on a voice channel.

## Source and tests

- `src/locus/providers/` — the four protocols, four implementations,
  and the `auto_register()` glue.
- `tests/unit/test_providers.py` — runtime-checkable protocols, tool
  factories, `AgentConfig` wiring.
- `tests/integration/test_providers_live.py` — live `httpx` fetch,
  live OpenAI search / image / speech (gated behind env vars).
