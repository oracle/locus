# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OpenAI model provider - 100% Pydantic."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from locus.core.events import ModelChunkEvent
from locus.core.messages import Message, ToolCall
from locus.models.base import ModelConfig, ModelResponse


if TYPE_CHECKING:
    import openai


def _decode_tool_arguments(raw: str | None) -> dict[str, Any]:
    """Decode the ``tc.function.arguments`` payload into a dict.

    Most providers send a JSON object string like ``'{"q": "Tokyo"}'``. A few
    (notably some OCI Llama deployments) double-encode it so ``json.loads``
    yields a string that itself parses back to the dict — try once more before
    giving up. Returns ``{}`` on any unrecoverable error.
    """
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(decoded, dict):
        return decoded
    if isinstance(decoded, str):
        try:
            second = json.loads(decoded)
        except json.JSONDecodeError:
            return {}
        if isinstance(second, dict):
            return second
    return {}


class OpenAIConfig(ModelConfig):
    """Configuration for OpenAI models."""

    model: str = "gpt-4o"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    api_key: str | None = Field(default=None, description="OpenAI API key")
    base_url: str | None = Field(default=None, description="Custom API base URL")
    organization: str | None = Field(default=None, description="OpenAI organization ID")

    # Production-safety knobs — match the OCI provider's posture so a
    # transient 429 / 503 / connection drop doesn't immediately kill the
    # agent loop. The openai SDK's defaults are 2 retries / 600s timeout;
    # 3 retries / 120s is a tighter, more agent-friendly default with
    # enough headroom for reasoning + tool-heavy turns where 60s starts
    # cutting things close.
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Retry budget for transient errors (429, 5xx, network).",
    )
    request_timeout: float = Field(
        default=120.0,
        gt=0,
        description="Per-request timeout in seconds.",
    )

    # OpenAI-specific settings
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int | None = None
    stop_sequences: list[str] = Field(default_factory=list)


class OpenAIModel(BaseModel):
    """
    OpenAI model provider.

    Supports GPT-4o, GPT-4, o1, o3 models with streaming and tool calling.

    Example:
        >>> model = OpenAIModel(model="gpt-4o")
        >>> response = await model.complete([Message.user("Hello!")])
    """

    config: OpenAIConfig
    _client: openai.AsyncOpenAI | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def supports_structured_output(self) -> bool:
        """Native ``response_format={"type":"json_schema",...}`` support.

        OpenAI's chat-completions API accepts a JSON-schema response_format
        and guarantees a parseable instance. The agent loop uses this
        property to skip the prompted-JSON fallback when the provider
        ships native structured output.
        """
        return True

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI model."""
        config = OpenAIConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        super().__init__(config=config)

    @property
    def client(self) -> openai.AsyncOpenAI:
        """Get or create the OpenAI client.

        The client is configured with explicit ``max_retries`` and
        ``timeout`` from :class:`OpenAIConfig` so transient errors
        (429, 5xx, network resets) don't kill the agent loop on first
        try. The openai SDK retries with exponential backoff between
        attempts.
        """
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                organization=self.config.organization,
                max_retries=self.config.max_retries,
                timeout=self.config.request_timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the OpenAI client and release resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> OpenAIModel:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - close client."""
        await self.close()

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Locus messages to OpenAI format."""
        openai_messages: list[dict[str, Any]] = []

        for msg in messages:
            openai_messages.append(msg.to_openai_format())

        return openai_messages

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Ensure tools are in OpenAI format."""
        if not tools:
            return None

        # Tools should already be in OpenAI format
        openai_tools = []
        for tool in tools:
            if "type" not in tool:
                # Wrap in function type if not already wrapped
                openai_tools.append(
                    {
                        "type": "function",
                        "function": tool,
                    }
                )
            else:
                openai_tools.append(tool)

        return openai_tools

    @staticmethod
    def _uses_max_completion_tokens(model: str) -> bool:
        """Whether the model requires ``max_completion_tokens`` over ``max_tokens``.

        Detects the o1 / o3 / gpt-5* families. Tolerates a leading
        purely-alphabetic namespace segment so OCI-style model ids
        (``openai.gpt-5``, ``meta.llama-3.3-…``) are treated the same as
        native OpenAI names (``gpt-5.1-chat-latest``). Native ids start
        with a token containing digits/hyphens (``gpt-5``, ``o1-…``) so
        the namespace strip is a no-op for them.
        """
        name = model.lower()
        head, sep, rest = name.partition(".")
        if sep and head.isalpha():
            name = rest
        return any(name.startswith(prefix) for prefix in ("o1", "o3", "gpt-5"))

    @staticmethod
    def _rejects_sampling_params(model: str) -> bool:
        """Whether the model rejects ``temperature`` / ``top_p``.

        OpenAI's ``*-search-preview`` chat-completions models perform their
        own retrieval and refuse caller-supplied sampling controls with a
        400 ``Model incompatible request arguments supplied: temperature,
        top_p`` error. Treat them like reasoning models for the purposes
        of building the request body, even though they still use plain
        ``max_tokens``.
        """
        name = model.lower()
        head, sep, rest = name.partition(".")
        if sep and head.isalpha():
            name = rest
        return "search-preview" in name

    def _parse_response(self, response: Any) -> ModelResponse:
        """Parse OpenAI response to ModelResponse.

        Tolerates providers that return a missing message or null content
        (Gemini does this when the response is filtered or empty).
        """
        choice = response.choices[0]
        msg = getattr(choice, "message", None)

        content = msg.content if msg is not None else None
        tool_calls: list[ToolCall] = []

        if msg is not None and msg.tool_calls:
            for tc in msg.tool_calls:
                arguments = _decode_tool_arguments(tc.function.arguments)
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        message = Message.assistant(content=content, tool_calls=tool_calls)

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

        return ModelResponse(
            message=message,
            usage=usage,
            stop_reason=choice.finish_reason,
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Complete a chat request.

        Args:
            messages: Conversation history
            tools: Tool schemas in OpenAI format
            **kwargs: Additional OpenAI-specific options

        Returns:
            Model response with message and metadata
        """
        openai_messages = self._convert_messages(messages)
        openai_tools = self._convert_tools(tools)

        uses_completion_tokens = self._uses_max_completion_tokens(self.config.model)
        rejects_sampling = self._rejects_sampling_params(self.config.model)

        max_tokens_value = kwargs.get("max_tokens", self.config.max_tokens)

        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": openai_messages,
        }

        # Use appropriate token parameter based on model
        if uses_completion_tokens:
            request_kwargs["max_completion_tokens"] = max_tokens_value
        else:
            request_kwargs["max_tokens"] = max_tokens_value
            if not rejects_sampling:
                request_kwargs["temperature"] = kwargs.get("temperature", self.config.temperature)
                request_kwargs["top_p"] = kwargs.get("top_p", self.config.top_p)
                # Only send penalties when the user customized them. Some
                # providers (Grok) reject the parameter outright, even at
                # zero — server defaults are 0.0 anyway, so omitting the
                # default value is functionally identical for those that
                # accept it.
                freq = kwargs.get("frequency_penalty", self.config.frequency_penalty)
                if freq != 0.0:
                    request_kwargs["frequency_penalty"] = freq
                pres = kwargs.get("presence_penalty", self.config.presence_penalty)
                if pres != 0.0:
                    request_kwargs["presence_penalty"] = pres

        if openai_tools:
            request_kwargs["tools"] = openai_tools

        if self.config.seed is not None:
            request_kwargs["seed"] = self.config.seed

        if self.config.stop_sequences and not uses_completion_tokens:
            request_kwargs["stop"] = self.config.stop_sequences

        # Forward ``response_format`` for structured output. Caller is expected
        # to pass a fully-formed dict (see locus.core.structured.build_response_format).
        response_format = kwargs.get("response_format")
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        response = await self.client.chat.completions.create(**request_kwargs)
        return self._parse_response(response)

    async def ainvoke(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """LangChain-compatible alias — returns Message (AIMessage equivalent)."""
        response = await self.complete(messages, tools=tools, **kwargs)
        return response.message if hasattr(response, "message") else response

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> OpenAIModel:
        """LangChain-compatible bind_tools."""
        bound = self.model_copy()
        object.__setattr__(
            bound,
            "_bound_tools",
            [t.to_openai_schema() if hasattr(t, "to_openai_schema") else t for t in (tools or [])],
        )
        return bound

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        """
        Stream a chat response.

        Args:
            messages: Conversation history
            tools: Tool schemas in OpenAI format
            **kwargs: Additional OpenAI-specific options

        Yields:
            Streaming chunks with content and/or tool calls
        """
        openai_messages = self._convert_messages(messages)
        openai_tools = self._convert_tools(tools)

        uses_completion_tokens = self._uses_max_completion_tokens(self.config.model)
        rejects_sampling = self._rejects_sampling_params(self.config.model)

        max_tokens_value = kwargs.get("max_tokens", self.config.max_tokens)

        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": openai_messages,
            "stream": True,
        }

        # Use appropriate token parameter based on model
        if uses_completion_tokens:
            request_kwargs["max_completion_tokens"] = max_tokens_value
        elif rejects_sampling:
            request_kwargs["max_tokens"] = max_tokens_value
        else:
            request_kwargs["max_tokens"] = max_tokens_value
            request_kwargs["temperature"] = kwargs.get("temperature", self.config.temperature)
            request_kwargs["top_p"] = kwargs.get("top_p", self.config.top_p)
            # See note in complete() — same penalty conditional.
            freq = kwargs.get("frequency_penalty", self.config.frequency_penalty)
            if freq != 0.0:
                request_kwargs["frequency_penalty"] = freq
            pres = kwargs.get("presence_penalty", self.config.presence_penalty)
            if pres != 0.0:
                request_kwargs["presence_penalty"] = pres

        if openai_tools:
            request_kwargs["tools"] = openai_tools

        if self.config.seed is not None:
            request_kwargs["seed"] = self.config.seed

        if self.config.stop_sequences:
            request_kwargs["stop"] = self.config.stop_sequences

        # Forward ``response_format`` for streaming structured output —
        # symmetric with complete(). Caller is expected to pass a fully-
        # formed dict (see locus.core.structured.build_response_format).
        response_format = kwargs.get("response_format")
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        # Track tool calls during streaming
        current_tool_calls: dict[int, dict[str, Any]] = {}

        stream = await self.client.chat.completions.create(**request_kwargs)

        async for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)

            # Some providers (Gemini) emit chunks where ``delta`` is None
            # — skip past content/tool-call handling but still let the
            # finish_reason check below run.
            if delta is None:
                if choice.finish_reason:
                    pass  # fall through to finish-reason block
                else:
                    continue

            # Handle content
            if delta is not None and delta.content:
                yield ModelChunkEvent(content=delta.content)

            # Handle tool calls
            if delta is not None and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }

                    if tc_delta.id:
                        current_tool_calls[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            current_tool_calls[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            current_tool_calls[idx]["arguments"] += tc_delta.function.arguments

            # Check for end of stream
            if choice.finish_reason:
                # Emit any accumulated tool calls
                if current_tool_calls:
                    tool_calls = []
                    for tc_data in current_tool_calls.values():
                        try:
                            arguments = (
                                json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                            )
                        except json.JSONDecodeError:
                            arguments = {}
                        tool_calls.append(
                            ToolCall(
                                id=tc_data["id"],
                                name=tc_data["name"],
                                arguments=arguments,
                            )
                        )
                    yield ModelChunkEvent(tool_calls=tool_calls)

                yield ModelChunkEvent(done=True)
