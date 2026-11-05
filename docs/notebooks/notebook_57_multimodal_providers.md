# Multi-modal providers

A chargeback case is rarely just text. The payments analyst has a
merchant order URL to fetch, an internal transactions ledger to query, a
receipt image to regenerate for the dispute packet and a customer
voicemail to transcribe. This example walks through the provider seam
under `locus.providers` that gives a triage agent each of those
capabilities as an ordinary tool, rather than through a wrapper the team
writes and maintains for every modality.

Set a provider on `AgentConfig` (`web_search`, `web_fetch`,
`image_generator`, `speech_provider`; the same names work as `Agent`
keyword arguments) and the SDK registers a matching `@tool` while the agent
initializes: `web_search`, `web_fetch`, `generate_image`, and `speak`
and/or `transcribe` for a speech provider, according to its
`capabilities`. The model calls them the same way it calls a hand-written
tool; there is no wrapper to write.

What this notebook covers:

- The four runtime-checkable Protocols under `locus.providers`
  (`BaseWebSearchProvider`, `BaseWebFetchProvider`,
  `BaseImageGenerationProvider`, `BaseSpeechProvider`), checked with
  `isinstance` against real objects.
- Auto-registration: an agent built with `web_fetch=` lists `web_fetch`
  in its tool registry although nothing was passed through `tools=`.
- A live fetch of `https://example.com`, standing in for a merchant's
  order-confirmation URL, through the registered tool and the bundled
  `HTTPXWebFetcher`, which needs no API key.
- Bring your own backend: `StaticLedgerSearch`, a class that subclasses
  nothing, becomes the agent's `web_search` tool because it implements
  `search`.
- The optional OpenAI-backed providers for web search, image generation
  and speech, printed as a wiring snippet unless `OPENAI_API_KEY` is set.

## Key lines

A provider is any object with the protocol's method: `StaticLedgerSearch`
subclasses nothing, yet the `isinstance` check in Part 1 passes it as a
`BaseWebSearchProvider` and Part 4 hands it to the agent as `web_search=`:

{{ locus_source examples/notebook_57_multimodal_providers.py::StaticLedgerSearch }}

Setting `web_fetch=` on `AgentConfig` is the whole registration: the
agent's tool registry lists `web_fetch` although nothing was passed
through `tools=`, and the same kwargs cover the other three modalities:

{{ locus_source examples/notebook_57_multimodal_providers.py::example_auto_register }}

Part 3 reaches the registered tool through `agent.tools.get("web_fetch")`
and awaits its function directly, so the wiring is checked with a real
fetch and no model round-trip; in a triage run the model would make the
same call with the order URL from the dispute case:

{{ locus_source examples/notebook_57_multimodal_providers.py::example_live_fetch }}

## Run it

```bash
python examples/notebook_57_multimodal_providers.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so
the example needs no model credentials. Set `LOCUS_MODEL_PROVIDER` (`oci`,
`openai`, `anthropic` or `mock`) to force one.

None of the five parts sends a prompt to the model: the agents are built
only to show which tools they register, and Part 3 calls the registered
tool's function directly. That call fetches `https://example.com`, so the
run needs outbound network access even when the mock model is forced;
without a route to example.com the fetch raises and the script stops
before Parts 4 and 5. To use the mock model regardless of the local OCI
configuration:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_57_multimodal_providers.py
```

Part 5 instantiates the OpenAI-backed image and speech providers only
when `OPENAI_API_KEY` is set, and then prints each provider's class, the
image model (`dall-e-3`, passed explicitly; the provider's default is
`gpt-image-2`) and the speech provider's `capabilities`. Without the key
the part prints the wiring as a snippet instead and returns, so the
default run needs no OpenAI account.

## Output

Running it offline — no credentials, bundled mock model — prints the four `isinstance` checks against the provider Protocols, the auto-registered tool names (`web_fetch`, then `web_search`) and the first lines of the example.com page, which Part 3 fetches live, so even this keyless run needs outbound network access:

```text
=== Part 1: The four provider Protocols ===

  isinstance(HTTPXWebFetcher(), BaseWebFetchProvider) = True
  isinstance(StaticLedgerSearch(), BaseWebSearchProvider) = True
  isinstance(HTTPXWebFetcher(), BaseSpeechProvider) = False
  isinstance(StaticLedgerSearch(), BaseImageGenerationProvider) = False

The Protocols are structural: StaticLedgerSearch subclasses nothing,
yet it passes as a web-search provider because it has `search`.

=== Part 2: Auto-registered tools ===

Registered tools: ['web_fetch']

Notice `web_fetch` appears even though we didn't pass it via `tools=`.
Setting `web_fetch=` is enough — locus auto-registered the wrapper.

The same kwargs work for the other modalities:
  Agent(web_search=..., web_fetch=..., image_generator=..., speech_provider=...)

Each provider becomes one tool (or two — `speech_provider` yields
`speak` and/or `transcribe` depending on `provider.capabilities`).

=== Part 3: Live fetch via the registered tool ===

First 200 chars of the rendered tool output:
# Example Domain
https://example.com

This domain is for use in documentation examples without needing permission. This is not a service; avoid relying on it for testing and monitoring purposes.
...

=== Part 4: Bring your own ledger backend ===

Registered tools: ['web_search']

The model now has a `web_search(query, max_results)` tool that
calls our StaticLedgerSearch.search() under the hood. Swap it for
your payments-ledger client and the agent queries live transactions.

=== Part 5: OpenAI-backed providers (optional) ===

OPENAI_API_KEY not set — printing the wiring without instantiating.

Snippet you'd use with a key:

  from locus.providers.image import OpenAIImageProvider
  from locus.providers.speech import OpenAISpeechProvider
  from locus.providers.web_search import OpenAISearchPreviewProvider
  from locus.models.native.openai import OpenAIModel

  agent = Agent(config=AgentConfig(
      model=get_model(),
      web_search=OpenAISearchPreviewProvider(
          OpenAIModel("gpt-5-search-api")
      ),
      image_generator=OpenAIImageProvider(model="dall-e-3"),
      speech_provider=OpenAISpeechProvider(),  # tts-1 + whisper-1
  ))
```
<!-- notebook-output:end -->

## See also

- [Multi-modal providers](../concepts/multi-modal-providers.md) — the four
  protocols, the built-in implementations and the bring-your-own contract.
- [Build a custom tool](../how-to/custom-tools.md) — the hand-written
  `@tool` path that the auto-registered tools sit beside in the registry.
- [Notebook 58 — Audio response](notebook_58_audio_response.md) — an
  `OpenAISpeechProvider` speaking an agent's reply, next in the series.

## Source

??? note "Full source — `examples/notebook_57_multimodal_providers.py`"

    ````python
    --8<-- "examples/notebook_57_multimodal_providers.py"
    ````
