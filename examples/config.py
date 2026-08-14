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
    TULIP_MODEL_PROVIDER   - "mock" (default), "openai", or "anthropic".
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

    # Pick a specific model for the selected provider:
    export TULIP_MODEL_ID=gpt-4o-mini
    python examples/notebook_06_basic_agent.py
"""

import os
import re
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
        """Return a mock response, calling a tool when one is bound.

        A mock that only ever returns prose makes every tool-centric example
        print ``Tool calls made: 0``, which is the opposite of what those
        notebooks exist to show. So on the first turn of a tool-bound run this
        emits a real ``ToolCall`` with arguments synthesised from the tool's
        own JSON schema; once a result comes back it answers in prose, which
        also terminates the loop.
        """
        if tools and not self._has_tool_result(messages):
            # Original case, not lowered: argument synthesis reads proper nouns
            # out of the prompt so the printed trace says "Paris", not "sample".
            call = self._synth_tool_call(self._last_user_text(messages), tools)
            if call is not None:
                return ModelResponse(
                    message=Message.assistant(content=None, tool_calls=[call]),
                    usage={"prompt_tokens": 10, "completion_tokens": 20},
                    stop_reason="tool_calls",
                )

        response = self._get_response(self._last_user_text(messages).lower(), tools)
        return ModelResponse(
            message=Message.assistant(content=response),
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            stop_reason="end_turn",
        )

    @staticmethod
    def _last_user_text(messages: list[Message]) -> str:
        """The most recent user turn, or the last message if there is none."""
        for msg in reversed(messages):
            if getattr(msg.role, "value", msg.role) == "user":
                return msg.content or ""
        return (messages[-1].content or "") if messages else ""

    @staticmethod
    def _has_tool_result(messages: list[Message]) -> bool:
        """Whether a tool has already answered in this run.

        Without this the mock would call the same tool every turn and only
        stop at the iteration cap.
        """
        return any(getattr(m.role, "value", m.role) == "tool" for m in messages)

    def _synth_tool_call(self, prompt: str, tools: list[dict[str, Any]]) -> Any:
        """Build a ToolCall for the tool that best matches the prompt.

        Preference goes to a tool whose name shares a word with the prompt —
        ``get_weather`` for "what's the weather in Tokyo" — so the chosen tool
        reads as a deliberate decision rather than a coin toss. Falls back to
        the first bound tool, because demonstrating the mechanism matters more
        than picking correctly.
        """
        from tulip.core.messages import ToolCall  # local: keeps import cost off the top

        specs = [self._tool_spec(t) for t in tools]
        specs = [s for s in specs if s.get("name")]
        if not specs:
            return None

        words = {w for w in re.split(r"\W+", prompt.lower()) if len(w) > 2}
        best = next(
            (
                s
                for s in specs
                if words & {w for w in re.split(r"\W+", s["name"].lower()) if len(w) > 2}
            ),
            specs[0],
        )
        return ToolCall(name=best["name"], arguments=self._synth_arguments(best, prompt))

    @staticmethod
    def _tool_spec(tool: dict[str, Any]) -> dict[str, Any]:
        """Normalise the two shapes a tool payload arrives in.

        OpenAI nests under ``function``; Anthropic-style payloads are flat.
        """
        inner = tool.get("function")
        return inner if isinstance(inner, dict) else tool

    @staticmethod
    def _synth_arguments(spec: dict[str, Any], prompt: str) -> dict[str, Any]:
        """Fill the tool's required parameters with schema-valid values.

        Only required parameters are supplied, so defaults keep their meaning.
        String arguments reuse the prompt's last capitalised word when there is
        one — "Weather in Tokyo?" yields ``{"city": "Tokyo"}``, which makes the
        printed trace legible instead of full of placeholders.
        """
        schema = spec.get("parameters") or spec.get("input_schema") or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []

        proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", prompt)
        sample_text = proper_nouns[-1] if proper_nouns else "sample"

        defaults: dict[str, Any] = {
            "string": sample_text,
            "integer": 1,
            "number": 1.0,
            "boolean": True,
            "array": [],
            "object": {},
        }
        args: dict[str, Any] = {}
        for name in required:
            prop = props.get(name) or {}
            if enum := prop.get("enum"):
                args[name] = enum[0]
            else:
                args[name] = defaults.get(prop.get("type", "string"), sample_text)
        return args

    def _get_response(self, prompt: str, tools: list[dict[str, Any]] | None) -> str:
        """Get appropriate response based on prompt content."""
        for keyword, response in self._responses.items():
            if keyword in prompt:
                return response
        return self._responses["default"]

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
