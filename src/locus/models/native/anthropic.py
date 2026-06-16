# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Anthropic model provider."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from locus.core.events import ModelChunkEvent
from locus.core.messages import Message, Role, ToolCall
from locus.models.base import ModelConfig, ModelResponse


if TYPE_CHECKING:
    import anthropic


# Models that reject `temperature` with `invalid_request_error: temperature is
# deprecated for this model`. Anthropic started doing this with Claude Opus 4.7
# (and presumably extends to later reasoning-flavoured 4.x+ models). Match on
# stable prefixes so we don't need to bump the list every time Anthropic
# publishes a new minor version.
_TEMPERATURE_DEPRECATED_PREFIXES: tuple[str, ...] = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-4-9",
)


def _rejects_temperature(model_id: str) -> bool:
    """Return True if the named Claude model rejects the `temperature` param.

    Public so callers (or wrappers) can pre-flight the same check without
    relying on a 400 round-trip to the API.
    """
    return any(model_id.startswith(p) for p in _TEMPERATURE_DEPRECATED_PREFIXES)


class AnthropicConfig(ModelConfig):
    """Configuration for Anthropic models."""

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    api_key: str | None = Field(default=None, description="Anthropic API key")
    base_url: str | None = Field(default=None, description="Custom API base URL")
    prompt_cache: bool = Field(
        default=False,
        description=(
            "When True, mark the system prompt and tool catalog with "
            "Anthropic's `cache_control: ephemeral` so subsequent turns "
            "reuse the cached input at ~1/10x cost. Default False for "
            "backward compatibility."
        ),
    )

    # Production-safety knobs — same posture as the OpenAI provider.
    # The anthropic SDK retries 5xx + connection errors with exponential
    # backoff between attempts; default of 3 / 120s matches OpenAIConfig
    # and gives reasoning + tool-heavy turns enough headroom.
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Retry budget for transient errors (overloaded, 5xx, network).",
    )
    request_timeout: float = Field(
        default=120.0,
        gt=0,
        description="Per-request timeout in seconds.",
    )


class AnthropicModel(BaseModel):
    """Anthropic model provider.

    Supports Claude 4.6, 4.5, 3.5 models with streaming and tool calling.

    Example:
        >>> model = AnthropicModel(model="claude-sonnet-4-20250514")
        >>> response = await model.complete([Message.user("Hello!")])
    """

    config: AnthropicConfig
    _client: anthropic.AsyncAnthropic | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def supports_structured_output(self) -> bool:
        """Anthropic doesn't ship OpenAI-style ``response_format``.

        The agent loop falls back to the prompted-JSON path with
        post-hoc parsing for Anthropic models.
        """
        return False

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        prompt_cache: bool = False,
        **kwargs: Any,
    ) -> None:
        config = AnthropicConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_cache=prompt_cache,
            **kwargs,
        )
        super().__init__(config=config)

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """Get or create the Anthropic client.

        Configured with explicit ``max_retries`` + ``timeout`` so a
        transient 529 (overloaded) / 5xx / connection reset doesn't
        kill the agent loop on the first try. Retries use exponential
        backoff inside the anthropic SDK.
        """
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                max_retries=self.config.max_retries,
                timeout=self.config.request_timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying httpx client.

        ``Agent.run_sync`` calls this in a ``finally`` block so the
        loop-bound httpx connections are shut down inside the same
        event loop that opened them. Without this, the next
        ``asyncio.run`` invocation closes the prior loop and the
        leftover client's ``__del__`` later tries to ``aclose`` against
        it, raising ``RuntimeError: Event loop is closed``.
        """
        if self._client is not None:
            try:
                await self._client.close()
            finally:
                self._client = None

    async def __aenter__(self) -> AnthropicModel:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert Locus messages to Anthropic format.

        Returns (system_prompt, messages) since Anthropic takes system separately.
        """
        system_prompt: str | None = None
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
                continue

            if msg.role == Role.ASSISTANT:
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                anthropic_messages.append(
                    {"role": "assistant", "content": content or msg.content or ""}
                )

            elif msg.role == Role.TOOL:
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": str(msg.content or ""),
                            }
                        ],
                    }
                )

            elif msg.role == Role.USER:
                anthropic_messages.append({"role": "user", "content": msg.content or ""})

        # Anthropic requires the last message to be a user turn — it does not
        # support assistant-prefill. Strip any trailing assistant messages, but
        # only when the conversation has at least one user message (i.e. it is
        # a real multi-turn exchange, not a lone assistant message being
        # converted for inspection).
        has_user = any(m["role"] == "user" for m in anthropic_messages)
        if has_user:
            while anthropic_messages and anthropic_messages[-1]["role"] == "assistant":
                anthropic_messages.pop()

        return system_prompt, anthropic_messages

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Convert OpenAI-format tools to Anthropic format."""
        if not tools:
            return None

        anthropic_tools = []
        for tool in tools:
            func = tool.get("function", tool)
            anthropic_tools.append(
                {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return anthropic_tools

    _STRUCTURED_TOOL_NAME = "respond_with_schema"

    def _structured_output_tool(self, response_format: dict[str, Any]) -> dict[str, Any]:
        """Translate an OpenAI-style ``response_format`` into an Anthropic tool.

        Anthropic does not support a ``response_format`` parameter; the
        idiomatic way to enforce a JSON schema is to declare a single tool
        whose ``input_schema`` is the desired schema and force the model to
        call it via ``tool_choice``. We name the tool ``respond_with_schema``
        and re-use the underlying schema name as the tool description so the
        model picks up any high-level docstring.
        """
        json_schema = response_format.get("json_schema", {}) or {}
        schema = json_schema.get("schema") or {}
        description = (
            json_schema.get("description")
            or f"Return your final answer as a {json_schema.get('name', 'JSON')} object."
        )
        return {
            "name": self._STRUCTURED_TOOL_NAME,
            "description": description,
            "input_schema": schema or {"type": "object", "properties": {}},
        }

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Complete a chat request.

        Recognises an OpenAI-style ``response_format={"type": "json_schema", ...}``
        kwarg and translates it into Anthropic's tool-use mechanism: a synthetic
        ``respond_with_schema`` tool is appended to the call and ``tool_choice``
        is pinned to it. The tool arguments are then surfaced as the message
        content (canonical JSON) so callers can parse them with
        :func:`locus.core.structured.parse_structured` exactly as they would
        with native ``response_format`` providers.
        """
        import json as _json

        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) or []

        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        # Claude Opus 4.7 (and presumably later 4.x reasoning models) reject
        # `temperature` with `invalid_request_error: temperature is deprecated
        # for this model`. Silently drop the param for those models — locus's
        # own agent runtime_loop always passes `temperature=config.temperature`
        # in `complete_kwargs`, so honouring "caller intent" would still 400
        # every Agent(model="claude-opus-4-7") on the first turn. The
        # wrapper's job here is to keep the agent loop running; callers who
        # need the parameter back can pin to a model that accepts it.
        if not _rejects_temperature(self.config.model):
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        if system_prompt:
            # When prompt-caching is enabled, send the system prompt as a
            # block list with ``cache_control: ephemeral`` so subsequent
            # turns reuse the cached input at ~1/10x cost (Anthropic
            # ephemeral cache TTL is ~5 min).
            if self.config.prompt_cache:
                params["system"] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                params["system"] = system_prompt

        # Structured-output mode: emulate ``response_format`` via tool-use.
        response_format = kwargs.get("response_format")
        structured_mode = (
            isinstance(response_format, dict) and response_format.get("type") == "json_schema"
        )
        if structured_mode:
            assert isinstance(response_format, dict)  # narrowed by structured_mode
            anthropic_tools.append(self._structured_output_tool(response_format))
            params["tool_choice"] = {
                "type": "tool",
                "name": self._STRUCTURED_TOOL_NAME,
            }

        if anthropic_tools:
            # Cache the tool catalog too — it's typically the same across
            # turns and can be large. Anthropic walks the cache_control
            # markers in order; tagging the last tool covers the catalog.
            if self.config.prompt_cache and anthropic_tools:
                anthropic_tools = [
                    *anthropic_tools[:-1],
                    {
                        **anthropic_tools[-1],
                        "cache_control": {"type": "ephemeral"},
                    },
                ]
            params["tools"] = anthropic_tools

        response = await self.client.messages.create(**params)

        # Parse response
        content: str | None = None
        tool_calls: list[ToolCall] = []
        structured_payload: dict[str, Any] | None = None

        for block in response.content:
            if block.type == "text":
                content = (content or "") + block.text
            elif block.type == "tool_use":
                if structured_mode and block.name == self._STRUCTURED_TOOL_NAME:
                    structured_payload = block.input if isinstance(block.input, dict) else {}
                    continue
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        # In structured mode, surface the tool's arguments as the message
        # content so downstream ``parse_structured`` can validate it.
        if structured_mode and structured_payload is not None:
            content = _json.dumps(structured_payload)

        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            }
            # Anthropic returns these only when prompt caching is in play.
            # Surface them on usage so AgentResult.metrics can show
            # cache hits/misses and cost-saved estimates.
            cache_creation = getattr(response.usage, "cache_creation_input_tokens", None)
            cache_read = getattr(response.usage, "cache_read_input_tokens", None)
            if cache_creation is not None:
                usage["cache_creation_input_tokens"] = cache_creation
            if cache_read is not None:
                usage["cache_read_input_tokens"] = cache_read

        return ModelResponse(
            message=Message.assistant(content=content, tool_calls=tool_calls),
            usage=usage,
            stop_reason=response.stop_reason,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        """Stream a chat response."""
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        if system_prompt:
            params["system"] = system_prompt
        if anthropic_tools:
            params["tools"] = anthropic_tools

        async with self.client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield ModelChunkEvent(content=text)

        yield ModelChunkEvent(done=True)
