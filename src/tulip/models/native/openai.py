# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""OpenAI model provider - 100% Pydantic."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from tulip.core.events import ModelChunkEvent
from tulip.core.messages import Message, Role, ToolCall
from tulip.models.base import ModelConfig, ModelResponse


if TYPE_CHECKING:
    import openai

logger = logging.getLogger(__name__)

#: Chat templates treat a user message that is *only* a tool-response wrapper as
#: part of a multi-step tool exchange rather than as the user's query.
_TOOL_RESPONSE_OPEN = "<tool_response>"
_TOOL_RESPONSE_CLOSE = "</tool_response>"


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


def _strip_model_namespace(name: str) -> str:
    """Drop a leading purely-alphabetic namespace segment.

    Namespaced model ids (``openai.gpt-5``, ``vendor.model-…``) are treated
    the same as native OpenAI names. Native ids start with a token containing
    digits/hyphens (``gpt-5``, ``o1-…``) so the strip is a no-op for them.
    """
    head, sep, rest = name.partition(".")
    if sep and head.isalpha():
        return rest
    return name


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


#: Long-stable Responses API fields, used only if introspecting the
#: ``openai`` package fails.
_FALLBACK_RESPONSES_PARAMS = frozenset(
    {
        "background",
        "include",
        "instructions",
        "max_tool_calls",
        "metadata",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt_cache_key",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "store",
        "stream_options",
        "text",
        "tool_choice",
        "top_logprobs",
        "truncation",
        "user",
    }
)


def _responses_param_names() -> frozenset[str]:
    """Every field the Responses API accepts, per the ``openai`` package.

    Mirrors :func:`_openai_param_names`: read from the SDK's own request
    TypedDicts rather than hand-listed, so a parameter OpenAI adds is
    forwardable the day the dependency is bumped.
    """
    try:
        import typing

        from openai.types.responses import response_create_params as _params

        names: set[str] = set()
        for cls_name in (
            "ResponseCreateParamsBase",
            "ResponseCreateParamsNonStreaming",
            "ResponseCreateParamsStreaming",
        ):
            cls = getattr(_params, cls_name, None)
            if cls is not None:
                names |= set(typing.get_type_hints(cls).keys())
        if names:
            return frozenset(names)
    except (ImportError, AttributeError, TypeError, NameError):  # pragma: no cover
        logger.debug("could not introspect openai responses request params", exc_info=True)
    # Introspection failed (openai moved its request types). Fall back to the
    # long-stable parameters rather than returning nothing, which would
    # silently block every passthrough instead of just the newest fields.
    return _FALLBACK_RESPONSES_PARAMS


#: Parameters the Responses request builder owns or translates from their
#: chat-completions names. Everything else in ``_RESPONSES_PARAMS`` is
#: forwarded verbatim from the caller.
_RESPONSES_RESERVED_PARAMS = frozenset(
    {
        "model",
        "input",
        "messages",
        "stream",
        "tools",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "reasoning_effort",
        "response_format",
    }
)

_RESPONSES_PARAMS = _responses_param_names()

#: Model families served only by the Responses API. GPT-5.6 rejects function
#: tools on chat-completions whenever reasoning is active — the API 400s with
#: "Function tools with reasoning_effort are not supported … use /v1/responses
#: or set reasoning_effort to 'none'" — and disabling reasoning defeats the
#: family's purpose, so ``api="auto"`` routes these to ``/v1/responses``.
_RESPONSES_ONLY_PREFIXES = ("gpt-5.6",)

#: ``Message.metadata`` key under which the Responses path stashes the raw
#: output items of an assistant turn (reasoning items with their
#: ``encrypted_content``, function_call items, message items — in order).
#: On the next turn ``_convert_messages_responses`` replays them verbatim,
#: which is the documented stateless (``store=False``) pattern: reasoning
#: models require the reasoning item that preceded a function call to come
#: back with it, and a reconstruction from ``Message`` fields alone cannot
#: supply that.
RESPONSES_ITEMS_METADATA_KEY = "openai_responses_items"


def _dump_output_item(item: Any) -> dict[str, Any] | None:
    """Serialise a Responses output item for verbatim replay next turn.

    The openai SDK's output items are Pydantic models; ``model_dump`` with
    ``exclude_none`` yields exactly the wire shape the API accepts back as
    input. Returns ``None`` for anything that can't be dumped — the item is
    then simply not replayed rather than poisoning the next request.
    """
    dump = getattr(item, "model_dump", None)
    if not callable(dump):
        return None
    try:
        dumped = dump(mode="json", exclude_none=True)
    except (TypeError, ValueError):
        logger.debug("could not serialise responses output item", exc_info=True)
        return None
    return dumped if isinstance(dumped, dict) else None


def _text_format_from_response_format(response_format: dict[str, Any]) -> dict[str, Any]:
    """Translate a chat-completions ``response_format`` to a Responses ``text`` param.

    Chat nests the schema under ``json_schema``; the Responses API flattens
    it into ``text.format``. ``json_object`` / ``text`` formats carry over
    unchanged.
    """
    if response_format.get("type") == "json_schema":
        inner = response_format.get("json_schema")
        if isinstance(inner, dict):
            fmt: dict[str, Any] = {"type": "json_schema"}
            for key in ("name", "schema", "strict", "description"):
                if key in inner:
                    fmt[key] = inner[key]
            return {"format": fmt}
    return {"format": dict(response_format)}


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
    api: Literal["auto", "responses", "chat_completions"] = Field(
        default="auto",
        description=(
            "Which OpenAI wire API to speak. 'chat_completions' is the classic "
            "/v1/chat/completions path, 'responses' is /v1/responses. 'auto' "
            "(the default) picks chat-completions except for model families "
            "that require the Responses API (gpt-5.6-*) — and only against "
            "api.openai.com itself: a custom base_url points at an "
            "OpenAI-compatible gateway (Together, vLLM, LiteLLM) that does "
            "not serve /v1/responses."
        ),
    )

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

    Supports GPT-4o, GPT-4, o1, o3, gpt-5.x models with streaming and tool
    calling. Speaks both OpenAI wire APIs: chat-completions (the default)
    and the Responses API, selected via ``api=`` on the config or
    automatically for model families that require it (gpt-5.6-*, which
    reject function tools on chat-completions whenever reasoning is on).

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
        self,
        request_kwargs: dict[str, Any],
        call_kwargs: dict[str, Any],
        *,
        allowed: frozenset[str] | None = None,
        reserved: frozenset[str] | None = None,
    ) -> None:
        """Forward any other API parameter the caller supplied.

        Without this the provider silently swallows most of the API: an agent
        that needs ``tool_choice`` to force a tool, ``parallel_tool_calls`` to
        serialise them, ``stream_options`` for usage during streaming, or
        ``logprobs`` for confidence has no route to the server and has to drop
        out of the SDK to a raw client. Anything not in the target API's
        schema is ignored here — it belongs in ``extra_body``.

        Defaults target Chat Completions; the Responses path passes its own
        ``allowed`` / ``reserved`` sets.
        """
        if allowed is None:
            allowed = _OPENAI_PARAMS
        if reserved is None:
            reserved = _RESERVED_PARAMS
        for name, value in call_kwargs.items():
            if name in reserved or name == "extra_body":
                continue
            if name in allowed and name not in request_kwargs:
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

        A request carrying no plain user turn gets one synthesised. The same
        family of chat templates walks the history backwards looking for a
        user message that is not itself a ``<tool_response>`` wrapper, and
        raises ``No user query found in messages.`` when there is none —
        Qwen3's template does this at the top of the render. Sub-calls that
        legitimately omit the user turn (judge/summary/auxiliary passes built
        from ``system + assistant + tool``) therefore 400 on a self-hosted
        Qwen even though the same list is fine on api.openai.com.
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

        return self._ensure_user_turn(openai_messages)

    @classmethod
    def _is_plain_user_turn(cls, entry: dict[str, Any]) -> bool:
        """True when ``entry`` is a user message a chat template counts as the query."""
        if entry.get("role") != "user":
            return False
        content = entry.get("content")
        if not isinstance(content, str):
            # Multi-part (image/audio) content is never a tool-response wrapper.
            return content is not None
        stripped = content.strip()
        if not stripped:
            return False
        return not (
            stripped.startswith(_TOOL_RESPONSE_OPEN) and stripped.endswith(_TOOL_RESPONSE_CLOSE)
        )

    @classmethod
    def _ensure_user_turn(cls, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Guarantee at least one plain user turn, anchoring templates that need one.

        Inserted after a leading system message so the system prompt keeps first
        position — the sibling rule in the same templates.
        """
        if not entries or any(cls._is_plain_user_turn(e) for e in entries):
            return entries

        anchor = {
            "role": "user",
            "content": "[Continue] Continue from the conversation above.",
        }
        at = 1 if entries[0].get("role") == "system" else 0
        return [*entries[:at], anchor, *entries[at:]]

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
        name = _strip_model_namespace(model.lower())
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
        return "search-preview" in _strip_model_namespace(model.lower())

    @staticmethod
    def _requires_responses_api(model: str) -> bool:
        """Whether the model family is served only by the Responses API.

        The GPT-5.6 family (sol / terra / luna) rejects function tools on
        chat-completions whenever reasoning is active: the API 400s with
        "Function tools with reasoning_effort are not supported … To use
        function tools, use /v1/responses or set reasoning_effort to
        'none'". Setting effort to none defeats the family's purpose, so
        these models route to ``/v1/responses`` under ``api="auto"``.
        Tolerates a leading purely-alphabetic namespace segment like the
        other family detectors.
        """
        name = _strip_model_namespace(model.lower())
        return name.startswith(_RESPONSES_ONLY_PREFIXES)

    def _use_responses_api(self) -> bool:
        """Whether this instance's requests go to ``/v1/responses``.

        Explicit config wins in both directions. ``auto`` selects the
        Responses API only for families that require it AND only against
        api.openai.com itself — a custom ``base_url`` points at an
        OpenAI-compatible gateway (Together, vLLM, LiteLLM) that serves
        chat-completions but not ``/v1/responses``, so auto-selection
        must never fire there.
        """
        if self.config.api == "responses":
            return True
        if self.config.api == "chat_completions":
            return False
        return self.config.base_url is None and self._requires_responses_api(self.config.model)

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

    # ------------------------------------------------------------------
    # Responses API (/v1/responses)
    # ------------------------------------------------------------------

    def _convert_tools_responses(
        self, tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Convert tool schemas to the Responses API's flattened shape.

        Chat-completions nests function tools under a ``function`` key; the
        Responses API flattens ``name`` / ``description`` / ``parameters`` to
        the top level. Bare Tulip schemas (no ``type``) flatten directly;
        built-in tools (``web_search`` etc.) and already-flattened entries
        pass through unchanged.
        """
        if not tools:
            return None

        converted: list[dict[str, Any]] = []
        for tool in tools:
            tool_type = tool.get("type")
            function = tool.get("function")
            if tool_type == "function" and isinstance(function, dict):
                flattened: dict[str, Any] = {"type": "function"}
                for key in ("name", "description", "parameters", "strict"):
                    if key in function:
                        flattened[key] = function[key]
                converted.append(flattened)
            elif tool_type is None:
                converted.append({"type": "function", **tool})
            else:
                converted.append(tool)
        return converted

    def _convert_messages_responses(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Tulip messages to Responses API input items.

        An assistant turn that the Responses path itself produced carries its
        raw output items in ``metadata[RESPONSES_ITEMS_METADATA_KEY]`` —
        those are replayed verbatim (reasoning items with their
        ``encrypted_content``, function_call items, message items, in
        order), which is what reasoning models require to continue a
        tool-calling turn statelessly. Assistant turns without that
        annotation (hand-built history, streamed turns) are reconstructed
        from ``content`` / ``tool_calls``; the reconstructed function_call
        items deliberately omit item ``id``s so the server does not try to
        pair them with reasoning items it never received.

        A system message after the first position is re-encoded as a user
        note, exactly like the chat-completions path (see
        :meth:`_convert_messages`), so mid-run guidance behaves the same on
        both transports.
        """
        items: list[dict[str, Any]] = []

        for index, msg in enumerate(messages):
            if msg.role == Role.ASSISTANT:
                raw_items = msg.metadata.get(RESPONSES_ITEMS_METADATA_KEY)
                if isinstance(raw_items, list):
                    replay = [item for item in raw_items if isinstance(item, dict)]
                    if replay:
                        items.extend(replay)
                        continue
                if msg.content:
                    items.append({"role": "assistant", "content": msg.content})
                for tc in msg.tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        }
                    )
            elif msg.role == Role.TOOL:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id or "",
                        "output": msg.content or "",
                    }
                )
            elif msg.role == Role.SYSTEM and index > 0:
                items.append(
                    {
                        "role": "user",
                        "content": f"[System guidance] {msg.content or ''}",
                    }
                )
            else:
                items.append({"role": msg.role.value, "content": msg.content or ""})

        return items

    def _build_responses_request(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        call_kwargs: dict[str, Any],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """Build a ``/v1/responses`` request mirroring the chat request shaping.

        Chat-completions names are translated to their Responses equivalents
        so callers (and the agent loop) don't need to know which transport is
        active:

        - ``max_tokens`` / ``max_completion_tokens`` → ``max_output_tokens``
        - ``reasoning_effort`` → ``reasoning={"effort": …}`` (merged under an
          explicit ``reasoning`` dict, which wins on conflict)
        - ``response_format`` → ``text={"format": …}`` (json_schema flattened)
        - chat-shaped ``tool_choice={"type": "function", "function": {…}}``
          → the flattened Responses shape

        Reasoning stays on: no effort is ever defaulted, so the server's own
        default applies unless the caller asks for something else.

        ``store`` defaults to ``False`` — the transport stays stateless like
        chat-completions and nothing persists server-side. For reasoning
        families, ``include=["reasoning.encrypted_content"]`` is then
        requested so reasoning items can be replayed verbatim on the next
        turn (see :meth:`_convert_messages_responses`).

        Dropped, with no Responses equivalent: ``seed``, ``stop`` /
        ``stop_sequences``, ``frequency_penalty`` / ``presence_penalty``.
        """
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "input": self._convert_messages_responses(messages),
        }
        if stream:
            request_kwargs["stream"] = True

        max_tokens_value = call_kwargs.get("max_tokens")
        if max_tokens_value is None:
            max_tokens_value = call_kwargs.get("max_completion_tokens")
        if max_tokens_value is None:
            max_tokens_value = self.config.max_tokens
        request_kwargs["max_output_tokens"] = max_tokens_value

        reasoning_family = self._uses_max_completion_tokens(self.config.model)
        if not reasoning_family and not self._rejects_sampling_params(self.config.model):
            # Reasoning families reject sampling controls on /v1/responses
            # just as they do on chat-completions. Penalties are chat-only —
            # the Responses API has no such fields — so only temperature /
            # top_p apply here.
            temperature = call_kwargs.get("temperature", self.config.temperature)
            if temperature is not None:
                request_kwargs["temperature"] = temperature
            top_p = call_kwargs.get("top_p", self.config.top_p)
            if top_p is not None:
                request_kwargs["top_p"] = top_p

        responses_tools = self._convert_tools_responses(tools)
        if responses_tools:
            request_kwargs["tools"] = responses_tools

        # Reasoning controls. ``reasoning`` (Responses-native dict) wins;
        # ``reasoning_effort`` (the chat-completions name) merges in so a
        # caller can keep using one spelling across both transports.
        reasoning = call_kwargs.get("reasoning")
        effort = call_kwargs.get("reasoning_effort")
        if isinstance(reasoning, dict):
            merged_reasoning = dict(reasoning)
            if effort is not None:
                merged_reasoning.setdefault("effort", effort)
            request_kwargs["reasoning"] = merged_reasoning
        elif reasoning is not None:
            request_kwargs["reasoning"] = reasoning
        elif effort is not None:
            request_kwargs["reasoning"] = {"effort": effort}

        # Chat-shaped forced tool choice → flattened Responses shape. Other
        # shapes ("auto" / "required" / already-flattened dicts) flow through
        # the passthrough below unchanged.
        tool_choice = call_kwargs.get("tool_choice")
        if (
            isinstance(tool_choice, dict)
            and tool_choice.get("type") == "function"
            and isinstance(tool_choice.get("function"), dict)
        ):
            request_kwargs["tool_choice"] = {
                "type": "function",
                "name": tool_choice["function"].get("name"),
            }

        # Structured output: translate ``response_format`` unless the caller
        # supplied a Responses-native ``text`` param themselves.
        response_format = call_kwargs.get("response_format")
        if response_format is not None and "text" not in call_kwargs:
            request_kwargs["text"] = _text_format_from_response_format(response_format)

        store = call_kwargs.get("store", False)
        request_kwargs["store"] = store
        if store is False and reasoning_family and "include" not in call_kwargs:
            request_kwargs["include"] = ["reasoning.encrypted_content"]

        self._apply_passthrough(
            request_kwargs,
            call_kwargs,
            allowed=_RESPONSES_PARAMS,
            reserved=_RESPONSES_RESERVED_PARAMS,
        )
        self._apply_extra_body(request_kwargs, call_kwargs)

        return request_kwargs

    @staticmethod
    def _responses_usage(response: Any) -> dict[str, int]:
        """Map Responses usage onto the chat-completions key names.

        ``input_tokens`` / ``output_tokens`` land as ``prompt_tokens`` /
        ``completion_tokens`` so metering code sees one shape regardless of
        transport. ``output_tokens`` already includes reasoning tokens —
        which is what the caller is billed for.
        """
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            return {}
        input_tokens = getattr(usage_obj, "input_tokens", None)
        output_tokens = getattr(usage_obj, "output_tokens", None)
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            return {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}
        return {}

    @staticmethod
    def _responses_stop_reason(response: Any, *, has_tool_calls: bool) -> str | None:
        """Map Responses ``status`` onto chat-completions finish reasons.

        The agent loop's termination logic keys on the chat vocabulary
        (``stop`` / ``tool_calls`` / ``length`` / ``content_filter``), so the
        Responses status is translated rather than passed through: a
        completed turn with function calls is ``tool_calls``, an incomplete
        turn that hit ``max_output_tokens`` is ``length``. Unrecognised
        statuses (``failed``, ``cancelled``) surface as themselves.
        """
        status = getattr(response, "status", None)
        if status == "completed":
            return "tool_calls" if has_tool_calls else "stop"
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None)
            if reason == "max_output_tokens":
                return "length"
            return reason if isinstance(reason, str) else "incomplete"
        return status if isinstance(status, str) else None

    def _parse_responses_result(self, response: Any) -> ModelResponse:
        """Parse a Responses API result to ModelResponse.

        Output items map onto the SDK types as follows:

        - ``message`` items: ``output_text`` parts join into
          ``message.content`` (``refusal`` parts stand in when there is no
          text, so a refusal is visible rather than an empty reply);
        - ``function_call`` items: one :class:`ToolCall` each, with
          ``call_id`` as the tool-call id so tool results round-trip;
        - ``reasoning`` items: summary / reasoning texts join into
          ``ModelResponse.reasoning``.

        The raw output items are additionally stashed in the assistant
        message's ``metadata`` (see :data:`RESPONSES_ITEMS_METADATA_KEY`)
        whenever the turn contains more than plain text, so the next turn
        can replay them verbatim.
        """
        text_parts: list[str] = []
        refusal_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_items: list[dict[str, Any]] = []
        needs_replay = False

        for item in getattr(response, "output", None) or []:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for part in getattr(item, "content", None) or []:
                    part_type = getattr(part, "type", None)
                    if part_type == "output_text":
                        text = getattr(part, "text", None)
                        if isinstance(text, str) and text:
                            text_parts.append(text)
                    elif part_type == "refusal":
                        refusal = getattr(part, "refusal", None)
                        if isinstance(refusal, str) and refusal:
                            refusal_parts.append(refusal)
            else:
                needs_replay = True
                if item_type == "function_call":
                    tool_calls.append(
                        ToolCall(
                            id=getattr(item, "call_id", None) or "",
                            name=getattr(item, "name", None) or "",
                            arguments=_decode_tool_arguments(getattr(item, "arguments", None)),
                        )
                    )
                elif item_type == "reasoning":
                    # Reasoning summaries (and, on models that expose it,
                    # raw reasoning text) surface as the chain of thought.
                    for block in (getattr(item, "summary", None) or []) + (
                        getattr(item, "content", None) or []
                    ):
                        text = getattr(block, "text", None)
                        if isinstance(text, str) and text:
                            reasoning_parts.append(text)
            dumped = _dump_output_item(item)
            if dumped is not None:
                raw_items.append(dumped)

        content = "".join(text_parts) or "".join(refusal_parts) or None
        message = Message(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            metadata={RESPONSES_ITEMS_METADATA_KEY: raw_items}
            if needs_replay and raw_items
            else {},
        )

        return ModelResponse(
            message=message,
            usage=self._responses_usage(response),
            stop_reason=self._responses_stop_reason(response, has_tool_calls=bool(tool_calls)),
            reasoning="\n\n".join(reasoning_parts) or None,
        )

    async def _complete_responses(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Complete a request over the Responses API."""
        request_kwargs = self._build_responses_request(messages, tools, kwargs, stream=False)
        response = await self.client.responses.create(**request_kwargs)
        return self._parse_responses_result(response)

    async def _stream_responses(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        """Stream a response over the Responses API.

        Emits the same :class:`ModelChunkEvent` sequence as the
        chat-completions path: content deltas, reasoning deltas, one
        tool-calls chunk once calls are complete, and a terminal ``done``
        chunk carrying usage + stop reason. Usage needs no
        ``stream_options`` opt-in here — the ``response.completed`` event
        always carries it. Server-side tool events (web_search etc.) and
        lifecycle events are ignored.
        """
        request_kwargs = self._build_responses_request(messages, tools, kwargs, stream=True)
        stream = await self.client.responses.create(**request_kwargs)

        # Function calls accumulate per output item id; argument deltas
        # reference the item id, and the ``output_item.done`` payload is
        # authoritative when present.
        tool_calls_by_item: dict[str, dict[str, str]] = {}
        final_usage: dict[str, int] | None = None
        final_stop_reason: str | None = None

        async for event in stream:
            event_type = getattr(event, "type", None)

            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", None)
                if isinstance(delta, str) and delta:
                    yield ModelChunkEvent(content=delta)

            elif event_type in (
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            ):
                delta = getattr(event, "delta", None)
                if isinstance(delta, str) and delta:
                    yield ModelChunkEvent(reasoning=delta)

            elif event_type == "response.output_item.added":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "function_call":
                    item_id = getattr(item, "id", None) or f"item_{len(tool_calls_by_item)}"
                    tool_calls_by_item[item_id] = {
                        "id": getattr(item, "call_id", None) or "",
                        "name": getattr(item, "name", None) or "",
                        "arguments": getattr(item, "arguments", None) or "",
                    }

            elif event_type == "response.function_call_arguments.delta":
                item_id = getattr(event, "item_id", None)
                delta = getattr(event, "delta", None)
                entry = tool_calls_by_item.get(item_id) if isinstance(item_id, str) else None
                if entry is not None and isinstance(delta, str):
                    entry["arguments"] += delta

            elif event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "function_call":
                    item_id = getattr(item, "id", None)
                    entry = tool_calls_by_item.get(item_id) if isinstance(item_id, str) else None
                    if entry is not None:
                        for source, target in (
                            ("call_id", "id"),
                            ("name", "name"),
                            ("arguments", "arguments"),
                        ):
                            value = getattr(item, source, None)
                            if isinstance(value, str) and value:
                                entry[target] = value

            elif event_type in (
                "response.completed",
                "response.incomplete",
                "response.failed",
            ):
                response_obj = getattr(event, "response", None)
                if response_obj is not None:
                    final_usage = self._responses_usage(response_obj) or None
                    final_stop_reason = self._responses_stop_reason(
                        response_obj, has_tool_calls=bool(tool_calls_by_item)
                    )
            # Everything else (created / in_progress / content_part /
            # server-side tool events) carries nothing the chunk stream
            # needs — skip.

        if tool_calls_by_item:
            yield ModelChunkEvent(
                tool_calls=[
                    ToolCall(
                        id=entry["id"],
                        name=entry["name"],
                        arguments=_decode_tool_arguments(entry["arguments"]),
                    )
                    for entry in tool_calls_by_item.values()
                ]
            )

        yield ModelChunkEvent(done=True, usage=final_usage, stop_reason=final_stop_reason)

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
        if self._use_responses_api():
            return await self._complete_responses(messages, tools, **kwargs)

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
        if self._use_responses_api():
            async for event in self._stream_responses(messages, tools, **kwargs):
                yield event
            return

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
