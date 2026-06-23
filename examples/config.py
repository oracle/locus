# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Shared configuration for Tulip notebooks.

The notebooks default to a built-in mock model so they run end-to-end
on a clean machine with zero setup, and upgrade to a live provider —
OpenAI, Anthropic — by setting a single environment variable.
That means:

  - On a clean machine (no env vars), notebooks run end-to-end against
    the mock model so you can read the output and understand the shape
    before authenticating.
  - Set ``TULIP_MODEL_PROVIDER`` + the matching credentials to run
    against a live provider.

Environment Variables:
    TULIP_MODEL_PROVIDER   - "mock" (default), "openai", "anthropic", or
"openai" or "anthropic".
    TULIP_MODEL_ID         - Model identifier (provider-specific)

    # OpenAI
    OPENAI_API_KEY         - OpenAI API key

    # Anthropic
    ANTHROPIC_API_KEY      - Anthropic API key

Examples:
    # Run with mock (default - no credentials needed):
    python examples/notebook_06_basic_agent.py

    # Run with OpenAI:
    export TULIP_MODEL_PROVIDER=openai
    export OPENAI_API_KEY=sk-...
    python examples/notebook_06_basic_agent.py

    # Run with Anthropic:
    export TULIP_MODEL_PROVIDER=anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/notebook_06_basic_agent.py

    export TULIP_MODEL_ID=llama3.2
    python examples/notebook_06_basic_agent.py
"""

import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tulip.core.events import ModelChunkEvent
from tulip.core.messages import Message
from tulip.models.base import ModelResponse


class MockModel(BaseModel):
    """
    Mock model for testing notebooks without API calls.

    Returns predetermined responses for common prompts.
    """

    max_tokens: int = 100
    temperature: float = 0.7

    # Simulated responses — security-flavoured so the offline output reads
    # like the real thing. Keyed on words that show up in SOC/IR prompts.
    _responses: dict[str, str] = {
        "default": "This is a mock response for testing purposes.",
        "triage": "Escalate: the indicators line up with an active phishing campaign.",
        "phishing": "Classic phishing markers (lookalike domain, urgent lure) — treat as malicious.",
        "alert": "Alert assessment: likely true positive. Recommend containment.",
        "severity": "Severity: HIGH — the exposure is reachable and exploitable.",
        "ioc": "Indicator enriched: multiple vendor detections, first seen 2 days ago.",
        "abstain": "Insufficient grounded evidence — abstaining rather than guessing.",
    }

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Return a mock response based on the last message."""
        last_msg = messages[-1].content or "" if messages else ""
        response = self._get_response(last_msg.lower(), tools)
        return ModelResponse(
            message=Message.assistant(content=response),
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            stop_reason="end_turn",
        )

    def _get_response(self, prompt: str, tools: list[dict[str, Any]] | None) -> str:
        """Get appropriate response based on prompt content."""
        # Check for tool calls — fire when a tool-bound prompt looks like a
        # SOC task the agent would reach for a tool to answer.
        tool_hints = (
            "triage",
            "alert",
            "domain",
            "ioc",
            "lookup",
            "enrich",
            "phishing",
            "reputation",
            "scan",
        )
        if tools and any(hint in prompt for hint in tool_hints):
            return self._get_tool_response(prompt, tools)

        # Match keywords to responses
        for keyword, response in self._responses.items():
            if keyword in prompt:
                return response
        return self._responses["default"]

    def _get_tool_response(self, prompt: str, tools: list[dict[str, Any]]) -> str:
        """Simulate tool usage response."""
        return "I'll use the available tools to help with that."

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        """Stream mock response in chunks."""
        response = await self.complete(messages, tools, **kwargs)
        content = response.content or ""

        # Yield in small chunks
        chunk_size = 10
        for i in range(0, len(content), chunk_size):
            yield ModelChunkEvent(content=content[i : i + chunk_size])
        yield ModelChunkEvent(done=True)


def check_structured_output_capable() -> None:
    """Exit cleanly with guidance if the current model cannot produce JSON.

    Guards against MockModel (plain text).
    """
    provider = os.environ.get("TULIP_MODEL_PROVIDER", "mock").lower()

    if provider != "mock":
        return

    print(
        "\n⚠  This notebook requires structured-output (JSON schema) support.\n"
        "   MockModel returns plain text and cannot demonstrate these features.\n\n"
        "   Run with a model that supports constrained decoding:\n\n"
        "     export TULIP_MODEL_PROVIDER=openai\n"
        "     export OPENAI_API_KEY=sk-...\n"
        "     export TULIP_MODEL_ID=gpt-4o\n"
        f"     python {Path(sys.argv[0]).name}\n"
    )
    sys.exit(0)


def get_model(**kwargs: Any) -> Any:
    """Return the configured model for the current notebook.

    Reads ``TULIP_MODEL_PROVIDER`` first. When it isn't set, falls back
    to the bundled mock so notebooks run end-to-end with no credentials.

    Args:
        **kwargs: Override any model parameters (max_tokens, temperature, …).
            Pass ``model_id="..."`` to use a specific model id without
            changing ``TULIP_MODEL_ID``.
    """
    provider = os.environ.get("TULIP_MODEL_PROVIDER", "").lower() or "mock"

    if provider == "mock":
        kwargs.pop("model_id", None)  # MockModel ignores model_id
        return MockModel(**kwargs)
    elif provider == "openai":
        return _get_openai_model(**kwargs)
    elif provider == "anthropic":
        return _get_anthropic_model(**kwargs)
    else:
        raise ValueError(
            f"Unknown model provider: {provider}. Use 'mock', 'openai', or 'anthropic'."
        )


def get_model_b(**kwargs: Any) -> Any:
    """Secondary model slot — typically a cheaper/faster variant for
    triage, routing, or color commentary in multi-agent notebooks.

    Reads ``TULIP_MODEL_ID_B`` (set by the workbench's "Model B" slot).
    Falls back to ``TULIP_MODEL_ID`` (= slot A) when unset, so notebooks
    that call ``get_model_b()`` still work in plain CLI runs where only
    one model is configured.
    """
    kwargs.setdefault(
        "model_id",
        os.environ.get("TULIP_MODEL_ID_B") or os.environ.get("TULIP_MODEL_ID", ""),
    )
    return get_model(**kwargs)


def get_model_c(**kwargs: Any) -> Any:
    """Tertiary model slot — same fall-through rules as :func:`get_model_b`,
    typically used for a judge / critic role distinct from both A and B."""
    kwargs.setdefault(
        "model_id",
        os.environ.get("TULIP_MODEL_ID_C") or os.environ.get("TULIP_MODEL_ID", ""),
    )
    return get_model(**kwargs)


def _get_openai_model(**kwargs: Any) -> Any:
    """Get OpenAI model."""
    from tulip.models import OpenAIModel

    model_id = kwargs.pop("model_id", os.environ.get("TULIP_MODEL_ID", "gpt-4o"))
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")

    return OpenAIModel(
        model=model_id,
        api_key=api_key,
        **kwargs,
    )


def _get_anthropic_model(**kwargs: Any) -> Any:
    """Get Anthropic model."""
    from tulip.models.native.anthropic import AnthropicModel

    model_id = kwargs.pop("model_id", os.environ.get("TULIP_MODEL_ID", "claude-sonnet-4-6"))
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable required")

    return AnthropicModel(
        model=model_id,
        api_key=api_key,
        **kwargs,
    )


def print_config():
    """Print current configuration for debugging."""
    provider = os.environ.get("TULIP_MODEL_PROVIDER", "").lower() or "mock"
    model_id = os.environ.get("TULIP_MODEL_ID", "(default)")

    print(f"Model Provider: {provider}")

    if provider == "mock":
        print("Using mock model (no API calls)")
    else:
        print(f"Model ID: {model_id}")
