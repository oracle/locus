# Models

Direct API providers: **OpenAI**, **Anthropic**. A
model is a string — the prefix before the colon selects the provider.

## Registry

String factory — routes `"openai:gpt-4o"`,
`"anthropic:claude-sonnet-4-6"`, `"anthropic:claude-sonnet-4-6"`, etc. to the right
client.

::: tulip.models.registry.get_model
::: tulip.models.registry.list_providers
::: tulip.models.registry.register_provider

## Base contract

Every model provider implements `ModelProtocol`. `RequestBuilder` and
`ResponseParser` are the per-provider seams for translating between
Tulip's `ModelConfig` / `Message` types and the provider's wire
format.

::: tulip.models.base.ModelProtocol
::: tulip.models.base.ModelConfig
::: tulip.models.base.ModelResponse
::: tulip.models.base.RequestBuilder
::: tulip.models.base.ResponseParser

## OpenAI

::: tulip.models.native.openai.OpenAIModel
::: tulip.models.native.openai.OpenAIConfig

## Anthropic

::: tulip.models.native.anthropic.AnthropicModel
