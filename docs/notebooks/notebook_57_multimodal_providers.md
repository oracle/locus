# Multi-Modal Providers

Set a provider on the Agent kwargs (`web_search`, `web_fetch`,
`image_generator`, `speech_provider`) and Tulip auto-registers a
matching `@tool`. The model calls it the same way it calls a
hand-written tool — you don't write the wrapper.

- Four Protocols under `tulip.providers`: search, fetch, image, speech.
- Live demo with `HTTPXWebFetcher` (no API key needed) against
  example.com.
- Bring-your-own: any duck-typed object that implements the protocol
  method.
- Optional OpenAI-backed providers (image, speech, search-preview).

Run it (defaults to the bundled mock model; set `TULIP_MODEL_PROVIDER` to `openai` / `anthropic` for a live model):

    python examples/notebook_57_multimodal_providers.py

Offline:

    TULIP_MODEL_PROVIDER=mock python examples/notebook_57_multimodal_providers.py

Optional: set `OPENAI_API_KEY` to exercise the OpenAI-backed providers.

## Source

```python
--8<-- "examples/notebook_57_multimodal_providers.py"
```
