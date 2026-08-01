# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""OpenAI model provider - 100% Pydantic."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from tulip.core.events import ModelChunkEvent
from tulip.core.messages import Message, ToolCall
from tulip.models.base import ModelConfig, ModelResponse


if TYPE_CHECKING:
    import openai

logger = logging.getLogger(__name__)


def _decode_tool_arguments(raw: str | None) -> dict[str, Any]:
    """Decode the ``tc.function.arguments`` payload into a dict.

    Most providers send a JSON object string like ``'{"q": "Tokyo"}'``. A few
    (notably some non-OpenAI deployments) double-encode it so ``json.loads``
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


#: Long-stable Chat Completions fields, used only if introspecting the
#: ``openai`` package fails.
_FALLBACK_OPENAI_PARAMS = frozenset(
    {
        "logit_bias",
        "logprobs",
        "metadata",
        "n",
        "parallel_tool_calls",
        "reasoning_effort",
        "response_format",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream_options",
        "tool_choice",
        "top_logprobs",
        "user",
    }
)


def _openai_param_names() -> frozenset[str]:
    """Every field the Chat Completions API accepts, per the ``openai`` package.

    Read from the SDK's own request TypedDicts rather than hand-listed, so a
    parameter OpenAI adds is forwardable the day the dependency is bumped —
    a hand-maintained list is a list that goes stale and quietly blocks
    functionality the server already supports.
    """
    try:
        import typing

        from openai.types.chat import completion_create_params as _params

        names: set[str] = set()
        for cls_name in (
            "CompletionCreateParamsBase",
            "CompletionCreateParamsNonStreaming",
            "CompletionCreateParamsStreaming",
        ):
            cls = getattr(_params, cls_name, None)
            if cls is not None:
                names |= set(typing.get_type_hints(cls).keys())
        if names:
            return frozenset(names)
    except (ImportError, AttributeError, TypeError, NameError):  # pragma: no cover
        logger.debug("could not introspect openai request params", exc_info=True)
    # Introspection failed (openai moved its request types). Fall back to the
    # long-stable parameters rather than returning nothing, which would silently
    # block every passthrough instead of just the newest fields.
    return _FALLBACK_OPENAI_PARAMS


#: Parameters this provider owns. Everything else in ``_OPENAI_PARAMS`` is
#: forwarded verbatim from the caller.
_RESERVED_PARAMS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "tools",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
    }
)

_OPENAI_PARAMS = _openai_param_names()


class OpenAIConfig(ModelConfig):
    """Configuration for OpenAI models."""

    model: str = "gpt-4o"
    max_tokens: int = 4096
    # ``None`` means "do not send this parameter" — the server's own default
    # applies. That matters for self-hosted models: vLLM reads temperature /
    # top_p from the model's ``generation_config.json``, and a value we send
    # unasked silently overrides what the model ships as its recommendation.
    temperature: float | None = 0.7
    top_p: float | None = 0.9
    api_key: str | None = Field(default=None, description="OpenAI API key")
    base_url: str | None = Field(default=None, description="Custom API base URL")
    organization: str | None = Field(default=None, description="OpenAI organization ID")

    # Production-safety knobs — keep a resilient posture so a
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
    extra_body: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Provider-specific request fields merged into the request body. "
            "OpenAI-compatible servers accept options the OpenAI schema has no "
            "field for — vLLM's chat_template_kwargs (e.g. enable_thinking), "
            "top_k, min_p, repetition_penalty. Without this they are "
            "unreachable through the SDK."
        ),
    )


class OpenAIModel(BaseModel):
    """
    OpenAI model provider.

    Supports GPT-4o, GPT-4, o1, o3 models with streaming and tool calling.

    Example:
        >>> model = OpenAIModel(model="gpt-4o")
        >>> response = await model.complete([Message.user("Hello!")])
    """

    config: OpenAIConfig

    def _apply_sampling(self, request_kwargs: dict[str, Any], call_kwargs: dict[str, Any]) -> None:
        """Merge sampling parameters into a request, omitting unset ones.

        ``temperature`` / ``top_p`` resolve to ``None`` when the caller wants
        the server to decide — self-hosted models publish their own values in
        ``generation_config.json``, and sending ours unasked overrides the
        model's published recommendation.

        Penalties are sent only when non-zero: some providers (Grok) reject the
        parameter outright even at zero, and zero is the server default anyway.
        """
        temperature = call_kwargs.get("temperature", self.config.temperature)
        if temperature is not None:
            request_kwargs["temperature"] = temperature

        top_p = call_kwargs.get("top_p", self.config.top_p)
        if top_p is not None:
            request_kwargs["top_p"] = top_p

        freq = call_kwargs.get("frequency_penalty", self.config.frequency_penalty)
        if freq != 0.0:
            request_kwargs["frequency_penalty"] = freq

        pres = call_kwargs.get("presence_penalty", self.config.presence_penalty)
        if pres != 0.0:
            request_kwargs["presence_penalty"] = pres

    def _apply_passthrough(
        self, request_kwargs: dict[str, Any], call_kwargs: dict[str, Any]
    ) -> None:
        """Forward any other Chat Completions parameter the caller supplied.

        Without this the provider silently swallows most of the API: an agent
        that needs ``tool_choice`` to force a tool, ``parallel_tool_calls`` to
        serialise them, ``stream_options`` for usage during streaming, or
        ``logprobs`` for confidence has no route to the server and has to drop
        out of the SDK to a raw client. Anything not in the OpenAI schema is
        ignored here — it belongs in ``extra_body``.
        """
        for name, value in call_kwargs.items():
            if name in _RESERVED_PARAMS or name == "extra_body":
                continue
            if name in _OPENAI_PARAMS and name not in request_kwargs:
                request_kwargs[name] = value

    def _apply_extra_body(
        self, request_kwargs: dict[str, Any], call_kwargs: dict[str, Any]
    ) -> None:
        """Merge provider-specific body fields, per-call taking precedence.

        Kept separate from sampling because it must apply to every model —
        including the reasoning models that reject sampling parameters, which
        still accept provider extensions.
        """
        merged: dict[str, Any] = {}
        if self.config.extra_body:
            merged.update(self.config.extra_body)
        per_call = call_kwargs.get("extra_body")
        if per_call:
            merged.update(per_call)
        if merged:
            request_kwargs["extra_body"] = merged

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
        """Convert Tulip messages to OpenAI format.

        A system message after the first position is re-encoded as a user
        note. The agent loop legitimately injects mid-run guidance as system
        messages (grounding replans, repair prompts, iteration nudges), but
        several OpenAI-compatible chat templates accept a system message only
        in first position — vLLM serving Qwen rejects the request outright
        with ``System message must be at the beginning``. That turns a normal
        guided run into a hard 400 partway through, non-deterministically,
        depending on whether the run happened to need guidance.

        The text is preserved and clearly marked, so steering still works
        while the request stays portable.
        """
        openai_messages: list[dict[str, Any]] = []

        for index, msg in enumerate(messages):
            entry = msg.to_openai_format()
            if index > 0 and entry.get("role") == "system":
                entry = {
                    "role": "user",
                    "content": f"[System guidance] {entry.get('content') or ''}",
                }
            openai_messages.append(entry)

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
        purely-alphabetic namespace segment so namespaced model ids
        (``openai.gpt-5``, ``vendor.model-…``) are treated the same as
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
        # Reasoning models (Qwen/DeepSeek via vLLM, o-series, gpt-5)
        # emit their chain of thought in a channel separate from
        # ``content``. The exact field name varies by deployment:
        # ``reasoning_content`` is the OpenAI-compatible / vLLM
        # ``--reasoning-parser qwen`` convention, while some vLLM
        # builds use ``reasoning``. Accept both so the CoT surfaces as
        # ``ThinkEvent.reasoning`` regardless of server variant.
        # The ``isinstance`` guard keeps MagicMock-heavy test stubs
        # from leaking a Mock into the Pydantic ``reasoning`` field.
        reasoning: str | None = None
        if msg is not None:
            for _field in ("reasoning_content", "reasoning"):
                _candidate = getattr(msg, _field, None)
                if isinstance(_candidate, str) and _candidate:
                    reasoning = _candidate
                    break
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

        # ``n>1`` costs the caller tokens for every candidate; keep the extras
        # instead of discarding everything past choices[0].
        candidates: list[Message] = []
        for extra in getattr(response, "choices", [])[1:]:
            extra_msg = getattr(extra, "message", None)
            if extra_msg is None:
                continue
            extra_calls: list[ToolCall] = []
            for tc in getattr(extra_msg, "tool_calls", None) or []:
                extra_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=_decode_tool_arguments(tc.function.arguments),
                    )
                )
            extra_content = getattr(extra_msg, "content", None)
            if not isinstance(extra_content, str):
                extra_content = None
            candidates.append(Message.assistant(content=extra_content, tool_calls=extra_calls))

        # Passed through as the provider shaped it — consumers of logprobs
        # want the raw numbers, not a lossy normalisation.
        logprobs = getattr(choice, "logprobs", None)

        return ModelResponse(
            message=message,
            usage=usage,
            stop_reason=choice.finish_reason,
            reasoning=reasoning,
            logprobs=logprobs,
            candidates=candidates,
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

        max_tokens_value = kwargs.get("max_tokens")
        if max_tokens_value is None:
            max_tokens_value = self.config.max_tokens

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
                self._apply_sampling(request_kwargs, kwargs)

        if openai_tools:
            request_kwargs["tools"] = openai_tools

        if self.config.seed is not None:
            request_kwargs["seed"] = self.config.seed

        if self.config.stop_sequences and not uses_completion_tokens:
            request_kwargs["stop"] = self.config.stop_sequences

        # Forward ``response_format`` for structured output. Caller is expected
        # to pass a fully-formed dict (see tulip.core.structured.build_response_format).
        response_format = kwargs.get("response_format")
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        self._apply_passthrough(request_kwargs, kwargs)
        self._apply_extra_body(request_kwargs, kwargs)

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

        max_tokens_value = kwargs.get("max_tokens")
        if max_tokens_value is None:
            max_tokens_value = self.config.max_tokens

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
            self._apply_sampling(request_kwargs, kwargs)

        if openai_tools:
            request_kwargs["tools"] = openai_tools

        if self.config.seed is not None:
            request_kwargs["seed"] = self.config.seed

        if self.config.stop_sequences:
            request_kwargs["stop"] = self.config.stop_sequences

        # Forward ``response_format`` for streaming structured output —
        # symmetric with complete(). Caller is expected to pass a fully-
        # formed dict (see tulip.core.structured.build_response_format).
        response_format = kwargs.get("response_format")
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        self._apply_passthrough(request_kwargs, kwargs)
        self._apply_extra_body(request_kwargs, kwargs)

        # Track tool calls during streaming
        current_tool_calls: dict[int, dict[str, Any]] = {}

        stream = await self.client.chat.completions.create(**request_kwargs)

        final_usage: dict[str, int] | None = None
        final_stop_reason: str | None = None

        async for chunk in stream:
            # When the caller asks for usage (``stream_options``), it arrives on
            # a trailing chunk that carries no choices — so this has to be read
            # before the empty-choices guard below, which would otherwise drop
            # the only chunk that has it.
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                prompt_tokens = getattr(chunk_usage, "prompt_tokens", None)
                completion_tokens = getattr(chunk_usage, "completion_tokens", None)
                if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                    final_usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }

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

            # Handle reasoning (chain-of-thought) deltas. Qwen / DeepSeek
            # served via vLLM stream their CoT in a channel separate from
            # ``content`` — ``delta.reasoning_content`` with
            # ``--reasoning-parser qwen``, ``delta.reasoning`` on some
            # builds. Accept both, as its own event so consumers can
            # render it distinctly or accumulate it independently.
            # The ``isinstance`` guard keeps MagicMock-heavy test stubs
            # from leaking a Mock into ``ModelChunkEvent.reasoning``.
            reasoning_delta: str | None = None
            if delta is not None:
                for _field in ("reasoning_content", "reasoning"):
                    _candidate = getattr(delta, _field, None)
                    if isinstance(_candidate, str) and _candidate:
                        reasoning_delta = _candidate
                        break
            if reasoning_delta:
                yield ModelChunkEvent(reasoning=reasoning_delta)

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

                if isinstance(choice.finish_reason, str):
                    final_stop_reason = choice.finish_reason

        # Emitted after the loop rather than at ``finish_reason``: the usage
        # chunk arrives *after* the choice that carries the finish reason, so
        # closing early would report a turn we cannot yet meter.
        yield ModelChunkEvent(done=True, usage=final_usage, stop_reason=final_stop_reason)
