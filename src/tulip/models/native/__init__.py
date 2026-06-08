# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Native model providers for Tulip.

Native providers connect directly to model vendor APIs:
- OpenAI → GPT models (and OpenAI-compatible gateways via ``base_url``)
- Anthropic → Claude models
"""

from tulip.models.native.openai import OpenAIConfig, OpenAIModel


__all__ = [
    "OpenAIModel",
    "OpenAIConfig",
    # Anthropic is a lazy import to avoid a hard dependency:
    #   from tulip.models.native.anthropic import AnthropicModel
]
