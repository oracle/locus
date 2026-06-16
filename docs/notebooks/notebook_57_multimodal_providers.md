# Multi-Modal Providers

Set a provider on the Agent kwargs (`web_search`, `web_fetch`,
`image_generator`, `speech_provider`) and Locus auto-registers a
matching `@tool`. The model calls it the same way it calls a
hand-written tool — you don't write the wrapper.

- Four Protocols under `locus.providers`: search, fetch, image, speech.
- Live demo with `HTTPXWebFetcher` (no API key needed) against
  example.com.
- Bring-your-own: any duck-typed object that implements the protocol
  method.
- Optional OpenAI-backed providers (image, speech, search-preview).

Run it (OCI Generative AI is the default — the notebook harness picks it when `~/.oci/config` exists; set `OCI_PROFILE` to choose a non-default profile):

    python examples/notebook_57_multimodal_providers.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_57_multimodal_providers.py

Optional: set `OPENAI_API_KEY` to exercise the OpenAI-backed providers.

## Source

```python
--8<-- "examples/notebook_57_multimodal_providers.py"
```
