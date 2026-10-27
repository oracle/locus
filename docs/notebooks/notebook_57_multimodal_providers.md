# Multi-Modal Providers

Set a provider on the Agent kwargs (`web_search`, `web_fetch`,
`image_generator`, `speech_provider`) and Locus auto-registers a
matching `@tool`. The model calls it the same way it calls a
hand-written tool — you don't write the wrapper.

A chargeback case is rarely just text: the payments analyst has a
merchant order URL to fetch, an internal transactions ledger to query,
a receipt image to regenerate for the dispute packet, and a customer
voicemail to transcribe. Each modality is one provider kwarg away.

- Four Protocols under `locus.providers`: search, fetch, image, speech.
- Live demo with `HTTPXWebFetcher` (no API key needed) against
  example.com, standing in for a merchant's order-confirmation URL.
- Bring-your-own: any duck-typed object that implements the protocol
  method — here, an internal transaction-ledger search.
- Optional OpenAI-backed providers (image, speech, web search).

Run it (defaults to the bundled mock model; set `LOCUS_MODEL_PROVIDER` to `openai` / `anthropic` for a live model):

    python examples/notebook_57_multimodal_providers.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_57_multimodal_providers.py

Optional: set `OPENAI_API_KEY` to exercise the OpenAI-backed providers.

## Source

````python
--8<-- "examples/notebook_57_multimodal_providers.py"
````
