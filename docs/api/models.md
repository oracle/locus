# Models

Native providers: **OpenAI** (`openai:`), **Anthropic** (`anthropic:`),
routed by name: a table of OpenAI-compatible prefixes, such as `ollama:`,
`vllm:`, `lmstudio:`, `groq:`, `together:`, `openrouter:`, `deepseek:`,
`mistral:` and `gemini:`. Each prefix has its own default endpoint and
API-key variable. Any other OpenAI-compatible endpoint works through
`openai-compatible:<model>` with an explicit `base_url=`. See
[OpenAI-compatible providers](../concepts/providers/openai-compatible.md).
A model is a string — the prefix before the colon selects the provider.

For the concepts, start with [Model providers](../concepts/models.md).

## Registry

String factory — routes `"openai:gpt-4o"`,
`"anthropic:claude-sonnet-5-5"`, etc. to the right client.

::: locus.models.registry.get_model
::: locus.models.registry.list_providers
::: locus.models.registry.register_provider
::: locus.models.providers.CompatibleProvider
::: locus.models.providers.register_compatible_providers

## Base contract

Every model provider implements `ModelProtocol`. `RequestBuilder` and
`ResponseParser` are the per-provider seams for translating between
Locus's `ModelConfig` / `Message` types and the provider's wire
format.

::: locus.models.base.ModelProtocol
::: locus.models.base.ModelConfig
::: locus.models.base.ModelResponse
::: locus.models.base.RequestBuilder
::: locus.models.base.ResponseParser

## OCI Generative AI

Needs the `oci` extra (`pip install "locus-sdk[oci]"`). See
[OCI Generative AI](../concepts/providers/oci.md) and
[OCI GenAI models](../how-to/oci-models.md). `get_model("oci:<model-id>")`
picks the transport automatically.

### OpenAI-compatible transport (V1)

`OCIChatCompletionsModel` targets the `/openai/v1` transport — real SSE
streaming, day-0 support for new OpenAI / Meta / xAI / Mistral / Gemini
releases. Use this for everything except Cohere models (any `cohere.*`
id) and Dedicated AI Cluster endpoints.

::: locus.models.providers.oci.openai_compat.OCIChatCompletionsModel
::: locus.models.providers.oci.OCIChatCompletionsConfig

### Cohere R-series and Dedicated AI Cluster transport (SDK)

`OCIModel` targets the OCI SDK's `/20231130/actions/v1` endpoint —
required for `cohere.command-r-*` (Cohere's Chat API shape isn't served
on the `/openai/v1` transport) and for `ocid1.generativeaiendpoint.…`
Dedicated AI Cluster endpoints.

::: locus.models.providers.oci.OCIModel
::: locus.models.providers.oci.OCIConfig
::: locus.models.providers.oci.OCIAuthType

### Responses API transport (opt-in)

`OCIResponsesModel` targets the server-stateful `/v1/responses` endpoint.
See [OCI Responses API](../concepts/oci-responses.md).

::: locus.models.providers.oci.responses.OCIResponsesModel
::: locus.models.providers.oci.responses.OCIResponsesConfig

## OpenAI

::: locus.models.native.openai.OpenAIModel
::: locus.models.native.openai.OpenAIConfig

## Anthropic

::: locus.models.native.anthropic.AnthropicModel
::: locus.models.native.anthropic.AnthropicConfig
