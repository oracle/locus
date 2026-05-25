# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Native model providers for Locus.

Native providers connect directly to model vendor APIs:
- OpenAI → GPT models (Oracle partnership)
- Anthropic → Claude models
- Ollama → Local LLMs (Llama, Mistral, Gemma, etc.)
- LiteLLM → Unified gateway covering 100+ providers, including the full
  Oracle Generative AI Infrastructure catalog (xAI Grok, Cohere Command,
  Cohere Embed, Meta Llama, Google Gemini, OpenAI gpt-5).
"""

from locus.models.native.openai import OpenAIConfig, OpenAIModel


__all__ = [
    "OpenAIModel",
    "OpenAIConfig",
    # Anthropic, Ollama, and LiteLLM are lazy imports to avoid hard
    # dependencies on packages users may not have installed:
    #   from locus.models.native.anthropic import AnthropicModel
    #   from locus.models.native.ollama import OllamaModel
    #   from locus.models.native.litellm import LiteLLMModel
]
