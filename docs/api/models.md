# Models

Two categories of provider: **native** (direct API connections —
OpenAI, Anthropic, Ollama) and **hosted** (OCI Generative AI, which
fronts OpenAI / Meta / xAI / Mistral / Gemini / Cohere via two
transports).

For Oracle deployments the default path is `OCIChatCompletionsModel` — the
OpenAI-compatible `/openai/v1` transport with real SSE streaming, day-0
model support, and the OCI auth chain (instance principal, resource
principal, session token, API key).

## Registry

String factory — auto-routes `"oci:openai.gpt-5"`, `"openai:gpt-4o"`,
`"anthropic:claude-3-5-sonnet"`, etc. to the right transport.

::: locus.models.registry.get_model
::: locus.models.registry.list_providers
::: locus.models.registry.register_provider

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

### OpenAI-compatible transport (V1)

`OCIChatCompletionsModel` targets the `/openai/v1` transport — real SSE streaming,
day-0 support for new OpenAI / Meta / xAI / Mistral / Gemini releases.
Use this for everything except Cohere R-series.

::: locus.models.providers.oci.openai_compat.OCIChatCompletionsModel
::: locus.models.providers.oci.OCIChatCompletionsConfig

### Cohere R-series transport

`OCIModel` targets the OCI SDK's `/20231130/actions/v1` endpoint —
required for `cohere.command-r-*` (Cohere's Chat API shape isn't
served on the `/openai/v1` transport).

::: locus.models.providers.oci.OCIModel
::: locus.models.providers.oci.OCIConfig
::: locus.models.providers.oci.OCIAuthType

## OpenAI

::: locus.models.native.openai.OpenAIModel
::: locus.models.native.openai.OpenAIConfig

## Anthropic

::: locus.models.native.anthropic.AnthropicModel

## Ollama

::: locus.models.native.ollama.OllamaModel
