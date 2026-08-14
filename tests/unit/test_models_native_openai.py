# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``tulip.models.native.openai`` (OpenAIModel).

The ``openai`` SDK already supplies test-friendly response objects;
we still need to stub them to control finish_reason, usage, content,
and tool-call deltas. The tests below cover:

- ``_decode_tool_arguments`` (single-encoded, double-encoded, malformed)
- model-name family detection (max_completion_tokens, search-preview)
- ``_parse_response`` for content / tool calls / missing message
- ``complete`` request shaping (max_tokens vs max_completion_tokens,
  reasoning families dropping sampling params, search-preview
  rejecting sampling params, penalty zero-skip, response_format
  forwarding, stop sequences gated by token-param family)
- ``stream`` chunk dispatch incl. tool-call accumulation, malformed
  argument JSON, ``delta is None`` chunks (Gemini emits these)
- async-context-manager + close path
- ``supports_structured_output`` flag
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tulip.core.messages import Message, ToolCall, ToolResult
from tulip.models.native.openai import OpenAIModel, _decode_tool_arguments


# ---------------------------------------------------------------------------
# Lightweight response stubs (mirror openai SDK shape).
# ---------------------------------------------------------------------------


class _Func:
    def __init__(self, name: str = "", arguments: str = "") -> None:
        self.name = name
        self.arguments = arguments


class _ToolCallStub:
    def __init__(self, *, call_id: str = "", name: str = "", arguments: str = "") -> None:
        self.id = call_id
        self.function = _Func(name=name, arguments=arguments)


class _MsgStub:
    def __init__(
        self,
        *,
        content: str | None = "",
        tool_calls: list[_ToolCallStub] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content


class _Usage:
    def __init__(self, prompt: int = 0, completion: int = 0) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _Choice:
    def __init__(
        self,
        *,
        message: _MsgStub | None = None,
        finish_reason: str | None = "stop",
    ) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Response:
    def __init__(
        self,
        *,
        choices: list[_Choice] | None = None,
        usage: _Usage | None = None,
    ) -> None:
        self.choices = choices or [_Choice(message=_MsgStub(content="ok"))]
        self.usage = usage


class _ToolDelta:
    def __init__(
        self,
        *,
        index: int = 0,
        call_id: str | None = "",
        name: str | None = None,
        arguments: str | None = None,
    ) -> None:
        self.index = index
        self.id = call_id

        class _F:
            pass

        if name is None and arguments is None:
            self.function = None
        else:
            f = _F()
            f.name = name  # type: ignore[attr-defined]
            f.arguments = arguments  # type: ignore[attr-defined]
            self.function = f


class _Delta:
    def __init__(
        self,
        *,
        content: str | None = None,
        tool_calls: list[_ToolDelta] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class _ChunkChoice:
    def __init__(
        self,
        *,
        delta: _Delta | None = None,
        finish_reason: str | None = None,
    ) -> None:
        self.delta = delta
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, choices: list[_ChunkChoice]) -> None:
        self.choices = choices


def _stream(chunks: list[_Chunk]) -> AsyncIterator[_Chunk]:
    async def gen() -> AsyncIterator[_Chunk]:
        for c in chunks:
            yield c

    return gen()


def _client_with(
    *,
    response: _Response | None = None,
    stream_chunks: list[_Chunk] | None = None,
) -> AsyncMock:
    """Build a mock ``openai.AsyncOpenAI`` client returning canned data."""
    client = AsyncMock()
    if stream_chunks is not None:
        client.chat.completions.create.return_value = _stream(stream_chunks)
    else:
        client.chat.completions.create.return_value = response or _Response()
    return client


def _model_with(client: AsyncMock, *, model: str = "gpt-4o", **kwargs: Any) -> OpenAIModel:
    m = OpenAIModel(model=model, **kwargs)
    m._client = client
    return m


# ---------------------------------------------------------------------------
# _decode_tool_arguments
# ---------------------------------------------------------------------------


class TestDecodeToolArguments:
    def test_empty_string(self) -> None:
        assert _decode_tool_arguments("") == {}

    def test_valid_json_dict(self) -> None:
        assert _decode_tool_arguments('{"q": "hi"}') == {"q": "hi"}

    def test_double_encoded_string(self) -> None:
        # Some provider deployments occasionally double-encode.
        assert _decode_tool_arguments('"{\\"q\\": \\"hi\\"}"') == {"q": "hi"}

    def test_double_encoded_invalid_inner(self) -> None:
        # Outer parse yields a string, inner parse fails → empty dict.
        assert _decode_tool_arguments('"not json"') == {}

    def test_double_encoded_inner_not_dict(self) -> None:
        # Inner parses successfully but isn't a dict.
        assert _decode_tool_arguments('"42"') == {}

    def test_malformed_json_returns_empty(self) -> None:
        assert _decode_tool_arguments("not json {") == {}

    def test_top_level_array_returns_empty(self) -> None:
        # Top-level JSON array isn't a dict and isn't a string.
        assert _decode_tool_arguments("[1, 2, 3]") == {}


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------


class TestFamilyDetection:
    @pytest.mark.parametrize(
        "model",
        ["o1-preview", "o3-mini", "gpt-5", "gpt-5.1", "gpt-5-codex", "openai.o1"],
    )
    def test_uses_max_completion_tokens_for_reasoning_families(self, model: str) -> None:
        assert OpenAIModel._uses_max_completion_tokens(model) is True

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
    def test_does_not_use_max_completion_tokens_for_classic(self, model: str) -> None:
        assert OpenAIModel._uses_max_completion_tokens(model) is False

    @pytest.mark.parametrize(
        "model",
        ["gpt-4o-search-preview", "openai.gpt-4o-mini-search-preview"],
    )
    def test_search_preview_rejects_sampling(self, model: str) -> None:
        assert OpenAIModel._rejects_sampling_params(model) is True

    def test_classic_model_accepts_sampling(self) -> None:
        assert OpenAIModel._rejects_sampling_params("gpt-4o") is False


# ---------------------------------------------------------------------------
# Construction + capability flag
# ---------------------------------------------------------------------------


class TestBasicProperties:
    def test_supports_structured_output(self) -> None:
        assert OpenAIModel().supports_structured_output is True

    def test_constructor_propagates_overrides(self) -> None:
        m = OpenAIModel(
            model="gpt-4o-mini",
            api_key="sk-test",  # noqa: S106
            base_url="https://api.example.com",
            max_tokens=100,
            temperature=0.0,
        )
        assert m.config.model == "gpt-4o-mini"
        assert m.config.api_key == "sk-test"  # noqa: S105
        assert m.config.max_tokens == 100

    def test_lazy_client_created_with_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _FakeAsyncOpenAI:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        import openai

        monkeypatch.setattr(openai, "AsyncOpenAI", _FakeAsyncOpenAI)
        m = OpenAIModel(api_key="sk-test", organization="org-1")  # noqa: S106
        _ = m.client
        assert captured["api_key"] == "sk-test"  # noqa: S105
        assert captured["organization"] == "org-1"

    def test_close_resets_client(self) -> None:
        client = AsyncMock()
        m = OpenAIModel()
        m._client = client
        import asyncio

        asyncio.run(m.close())
        assert m._client is None
        client.close.assert_called_once()

    def test_close_no_op_when_client_unset(self) -> None:
        m = OpenAIModel()
        import asyncio

        asyncio.run(m.close())
        assert m._client is None


class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_aenter_returns_self(self) -> None:
        m = OpenAIModel()
        async with m as entered:
            assert entered is m

    @pytest.mark.asyncio
    async def test_aexit_closes_client(self) -> None:
        client = AsyncMock()
        m = OpenAIModel()
        m._client = client
        async with m:
            pass
        client.close.assert_called_once()


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_parses_content_and_usage(self) -> None:
        m = OpenAIModel()
        resp = _Response(
            choices=[_Choice(message=_MsgStub(content="hello"))],
            usage=_Usage(prompt=10, completion=5),
        )
        out = m._parse_response(resp)
        assert out.message.content == "hello"
        assert out.usage == {"prompt_tokens": 10, "completion_tokens": 5}
        assert out.stop_reason == "stop"

    def test_parses_tool_calls(self) -> None:
        m = OpenAIModel()
        resp = _Response(
            choices=[
                _Choice(
                    message=_MsgStub(
                        content="",
                        tool_calls=[
                            _ToolCallStub(
                                call_id="call_1",
                                name="search",
                                arguments='{"q": "hi"}',
                            )
                        ],
                    )
                )
            ],
            usage=None,
        )
        out = m._parse_response(resp)
        assert out.message.tool_calls[0].name == "search"
        assert out.message.tool_calls[0].arguments == {"q": "hi"}

    def test_handles_missing_message(self) -> None:
        # Some providers return a choice without ``message``.
        m = OpenAIModel()
        resp = _Response(choices=[_Choice(message=None)])
        out = m._parse_response(resp)
        assert out.message.content is None
        assert out.message.tool_calls == []

    def test_parses_reasoning_content(self) -> None:
        # Qwen/DeepSeek via vLLM put the CoT in ``reasoning_content``.
        m = OpenAIModel()
        resp = _Response(
            choices=[
                _Choice(
                    message=_MsgStub(
                        content="The answer is 42.",
                        reasoning_content="Let me think step by step...",
                    )
                )
            ],
            usage=None,
        )
        out = m._parse_response(resp)
        assert out.message.content == "The answer is 42."
        assert out.reasoning == "Let me think step by step..."

    def test_reasoning_none_when_absent(self) -> None:
        m = OpenAIModel()
        resp = _Response(choices=[_Choice(message=_MsgStub(content="plain"))])
        out = m._parse_response(resp)
        assert out.reasoning is None

    def test_parses_reasoning_field_variant(self) -> None:
        # Some vLLM builds name the CoT channel ``reasoning`` instead of
        # ``reasoning_content`` — accept both.
        class _Msg:
            content = "Answer."
            reasoning = "CoT via reasoning field."
            tool_calls = []

        m = OpenAIModel()
        resp = _Response(choices=[_Choice(message=_Msg())])
        out = m._parse_response(resp)
        assert out.message.content == "Answer."
        assert out.reasoning == "CoT via reasoning field."


# ---------------------------------------------------------------------------
# complete request shaping
# ---------------------------------------------------------------------------


class TestCompleteRequestShaping:
    @pytest.mark.asyncio
    async def test_classic_model_sends_max_tokens(self) -> None:
        client = _client_with()
        m = _model_with(client, model="gpt-4o")
        await m.complete([Message.user("hi")])
        args = client.chat.completions.create.call_args.kwargs
        assert "max_tokens" in args
        assert "max_completion_tokens" not in args
        assert "temperature" in args
        assert "top_p" in args

    @pytest.mark.asyncio
    async def test_reasoning_model_sends_max_completion_tokens_no_sampling(
        self,
    ) -> None:
        client = _client_with()
        m = _model_with(client, model="o1-preview")
        await m.complete([Message.user("hi")])
        args = client.chat.completions.create.call_args.kwargs
        assert "max_completion_tokens" in args
        assert "max_tokens" not in args
        assert "temperature" not in args
        assert "top_p" not in args

    @pytest.mark.asyncio
    async def test_search_preview_drops_sampling(self) -> None:
        client = _client_with()
        m = _model_with(client, model="gpt-4o-search-preview")
        await m.complete([Message.user("hi")])
        args = client.chat.completions.create.call_args.kwargs
        # Search-preview keeps ``max_tokens`` but drops temperature/top_p.
        assert "max_tokens" in args
        assert "temperature" not in args
        assert "top_p" not in args

    @pytest.mark.asyncio
    async def test_zero_penalties_omitted(self) -> None:
        client = _client_with()
        m = _model_with(client)  # defaults: freq=0.0, pres=0.0
        await m.complete([Message.user("hi")])
        args = client.chat.completions.create.call_args.kwargs
        assert "frequency_penalty" not in args
        assert "presence_penalty" not in args

    @pytest.mark.asyncio
    async def test_nonzero_penalties_forwarded(self) -> None:
        client = _client_with()
        m = _model_with(client)
        await m.complete([Message.user("hi")], frequency_penalty=0.5, presence_penalty=0.3)
        args = client.chat.completions.create.call_args.kwargs
        assert args["frequency_penalty"] == 0.5
        assert args["presence_penalty"] == 0.3

    @pytest.mark.asyncio
    async def test_stop_sequences_only_for_classic(self) -> None:
        client = _client_with()
        m = _model_with(client, model="gpt-4o", stop_sequences=["END"])
        await m.complete([Message.user("hi")])
        assert client.chat.completions.create.call_args.kwargs["stop"] == ["END"]

    @pytest.mark.asyncio
    async def test_stop_sequences_skipped_for_reasoning_family(self) -> None:
        client = _client_with()
        m = _model_with(client, model="o1-preview", stop_sequences=["END"])
        await m.complete([Message.user("hi")])
        assert "stop" not in client.chat.completions.create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_seed_propagated(self) -> None:
        client = _client_with()
        m = _model_with(client, seed=42)
        await m.complete([Message.user("hi")])
        assert client.chat.completions.create.call_args.kwargs["seed"] == 42

    @pytest.mark.asyncio
    async def test_response_format_propagated(self) -> None:
        client = _client_with()
        m = _model_with(client)
        rf = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
        await m.complete([Message.user("hi")], response_format=rf)
        assert client.chat.completions.create.call_args.kwargs["response_format"] == rf

    @pytest.mark.asyncio
    async def test_tools_wrapped_in_function_envelope(self) -> None:
        client = _client_with()
        m = _model_with(client)
        await m.complete([Message.user("hi")], tools=[{"name": "search", "parameters": {}}])
        args = client.chat.completions.create.call_args.kwargs
        assert args["tools"][0]["type"] == "function"

    @pytest.mark.asyncio
    async def test_tools_with_existing_type_passed_through(self) -> None:
        client = _client_with()
        m = _model_with(client)
        already = [{"type": "function", "function": {"name": "search"}}]
        await m.complete([Message.user("hi")], tools=already)
        assert client.chat.completions.create.call_args.kwargs["tools"] == already


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------


class TestStream:
    @pytest.mark.asyncio
    async def test_yields_content_chunks(self) -> None:
        chunks = [
            _Chunk(choices=[_ChunkChoice(delta=_Delta(content="Hello "))]),
            _Chunk(choices=[_ChunkChoice(delta=_Delta(content="world"))]),
            _Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")]),
        ]
        client = _client_with(stream_chunks=chunks)
        m = _model_with(client)
        events = [ev async for ev in m.stream([Message.user("hi")])]
        contents = [ev.content for ev in events if ev.content]
        assert contents == ["Hello ", "world"]
        assert any(ev.done for ev in events)

    @pytest.mark.asyncio
    async def test_accumulates_tool_call_deltas(self) -> None:
        # A complete tool call streamed as multiple deltas.
        chunks = [
            _Chunk(
                choices=[
                    _ChunkChoice(
                        delta=_Delta(
                            tool_calls=[_ToolDelta(index=0, call_id="call_1", name="search")]
                        )
                    )
                ]
            ),
            _Chunk(
                choices=[
                    _ChunkChoice(delta=_Delta(tool_calls=[_ToolDelta(index=0, arguments='{"q":')]))
                ]
            ),
            _Chunk(
                choices=[
                    _ChunkChoice(delta=_Delta(tool_calls=[_ToolDelta(index=0, arguments=' "hi"}')]))
                ]
            ),
            _Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="tool_calls")]),
        ]
        client = _client_with(stream_chunks=chunks)
        m = _model_with(client)
        events = [ev async for ev in m.stream([Message.user("hi")])]
        tool_call_events = [ev for ev in events if ev.tool_calls]
        assert len(tool_call_events) == 1
        tc = tool_call_events[0].tool_calls[0]
        assert tc.name == "search"
        assert tc.arguments == {"q": "hi"}

    @pytest.mark.asyncio
    async def test_malformed_tool_arguments_become_empty(self) -> None:
        chunks = [
            _Chunk(
                choices=[
                    _ChunkChoice(
                        delta=_Delta(
                            tool_calls=[
                                _ToolDelta(
                                    index=0,
                                    call_id="c",
                                    name="x",
                                    arguments="{not json",
                                )
                            ]
                        )
                    )
                ]
            ),
            _Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="tool_calls")]),
        ]
        client = _client_with(stream_chunks=chunks)
        m = _model_with(client)
        events = [ev async for ev in m.stream([Message.user("hi")])]
        tool_call_events = [ev for ev in events if ev.tool_calls]
        assert tool_call_events[0].tool_calls[0].arguments == {}

    @pytest.mark.asyncio
    async def test_handles_none_delta_chunks(self) -> None:
        # Some providers (Gemini) emit chunks where delta is None.
        chunks = [
            _Chunk(choices=[_ChunkChoice(delta=None, finish_reason=None)]),
            _Chunk(choices=[_ChunkChoice(delta=_Delta(content="hi"))]),
            _Chunk(choices=[_ChunkChoice(delta=None, finish_reason="stop")]),
        ]
        client = _client_with(stream_chunks=chunks)
        m = _model_with(client)
        events = [ev async for ev in m.stream([Message.user("hi")])]
        assert any(ev.content == "hi" for ev in events)
        assert any(ev.done for ev in events)

    @pytest.mark.asyncio
    async def test_skips_chunks_with_no_choices(self) -> None:
        chunks = [
            _Chunk(choices=[]),  # No choices — skip.
            _Chunk(choices=[_ChunkChoice(delta=_Delta(content="ok"))]),
            _Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")]),
        ]
        client = _client_with(stream_chunks=chunks)
        m = _model_with(client)
        events = [ev async for ev in m.stream([Message.user("hi")])]
        assert any(ev.content == "ok" for ev in events)

    @pytest.mark.asyncio
    async def test_yields_reasoning_deltas(self) -> None:
        # Qwen/DeepSeek via vLLM stream CoT in delta.reasoning_content.
        chunks = [
            _Chunk(choices=[_ChunkChoice(delta=_Delta(reasoning_content="Let me "))]),
            _Chunk(choices=[_ChunkChoice(delta=_Delta(reasoning_content="think."))]),
            _Chunk(choices=[_ChunkChoice(delta=_Delta(content="Final answer"))]),
            _Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")]),
        ]
        client = _client_with(stream_chunks=chunks)
        m = _model_with(client)
        events = [ev async for ev in m.stream([Message.user("hi")])]
        reasoning = [ev.reasoning for ev in events if ev.reasoning]
        assert reasoning == ["Let me ", "think."]
        # Content stays in its own channel.
        assert [ev.content for ev in events if ev.content] == ["Final answer"]

    @pytest.mark.asyncio
    async def test_reasoning_and_content_not_mixed(self) -> None:
        # A reasoning chunk must never be mislabelled as content.
        chunks = [
            _Chunk(choices=[_ChunkChoice(delta=_Delta(reasoning_content="CoT"))]),
            _Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")]),
        ]
        client = _client_with(stream_chunks=chunks)
        m = _model_with(client)
        events = [ev async for ev in m.stream([Message.user("hi")])]
        assert not any(ev.content == "CoT" for ev in events)
        assert [ev.reasoning for ev in events if ev.reasoning] == ["CoT"]

    @pytest.mark.asyncio
    async def test_yields_reasoning_via_reasoning_field(self) -> None:
        # vLLM builds that use ``delta.reasoning`` (not
        # ``reasoning_content``) must surface the CoT the same way.
        chunks = [
            _Chunk(choices=[_ChunkChoice(delta=_Delta(reasoning_content="CoT via parser"))]),
            _Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")]),
        ]
        client = _client_with(stream_chunks=chunks)
        m = _model_with(client)
        events = [ev async for ev in m.stream([Message.user("hi")])]
        assert [ev.reasoning for ev in events if ev.reasoning] == ["CoT via parser"]

        # Same behaviour for the ``delta.reasoning`` variant.
        class _ReasoningDelta(_Delta):
            def __init__(self, reasoning: str) -> None:
                self.content = None
                self.tool_calls = None
                self.reasoning_content = None
                self.reasoning = reasoning

        chunks = [
            _Chunk(choices=[_ChunkChoice(delta=_ReasoningDelta("CoT via field"))]),
            _Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")]),
        ]
        client = _client_with(stream_chunks=chunks)
        m = _model_with(client)
        events = [ev async for ev in m.stream([Message.user("hi")])]
        assert [ev.reasoning for ev in events if ev.reasoning] == ["CoT via field"]


# ---------------------------------------------------------------------------
# Stream request shaping (mirrors complete shaping)
# ---------------------------------------------------------------------------


class TestStreamRequestShaping:
    @pytest.mark.asyncio
    async def test_reasoning_family_omits_sampling_in_stream(self) -> None:
        client = _client_with(
            stream_chunks=[_Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")])]
        )
        m = _model_with(client, model="o1-preview")
        async for _ in m.stream([Message.user("hi")]):
            pass
        args = client.chat.completions.create.call_args.kwargs
        assert "temperature" not in args
        assert args["stream"] is True
        assert args["max_completion_tokens"] == m.config.max_tokens

    @pytest.mark.asyncio
    async def test_search_preview_keeps_max_tokens_drops_sampling(self) -> None:
        client = _client_with(
            stream_chunks=[_Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")])]
        )
        m = _model_with(client, model="gpt-4o-search-preview")
        async for _ in m.stream([Message.user("hi")]):
            pass
        args = client.chat.completions.create.call_args.kwargs
        assert "max_tokens" in args
        assert "temperature" not in args

    @pytest.mark.asyncio
    async def test_stream_response_format_propagated(self) -> None:
        client = _client_with(
            stream_chunks=[_Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")])]
        )
        m = _model_with(client)
        rf = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
        async for _ in m.stream([Message.user("hi")], response_format=rf):
            pass
        assert client.chat.completions.create.call_args.kwargs["response_format"] == rf

    @pytest.mark.asyncio
    async def test_stream_zero_penalties_omitted(self) -> None:
        client = _client_with(
            stream_chunks=[_Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")])]
        )
        m = _model_with(client)
        async for _ in m.stream([Message.user("hi")]):
            pass
        args = client.chat.completions.create.call_args.kwargs
        assert "frequency_penalty" not in args
        assert "presence_penalty" not in args

    @pytest.mark.asyncio
    async def test_stream_seed_and_stop(self) -> None:
        client = _client_with(
            stream_chunks=[_Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")])]
        )
        m = _model_with(client, seed=7, stop_sequences=["END"])
        async for _ in m.stream([Message.user("hi")]):
            pass
        args = client.chat.completions.create.call_args.kwargs
        assert args["seed"] == 7
        assert args["stop"] == ["END"]

    @pytest.mark.asyncio
    async def test_stream_tools_wrapped(self) -> None:
        client = _client_with(
            stream_chunks=[_Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")])]
        )
        m = _model_with(client)
        async for _ in m.stream([Message.user("hi")], tools=[{"name": "search", "parameters": {}}]):
            pass
        assert client.chat.completions.create.call_args.kwargs["tools"][0]["type"] == "function"


# ---------------------------------------------------------------------------
# Sampling params: None means "let the server decide"
# ---------------------------------------------------------------------------


class TestSamplingOmission:
    """Self-hosted models publish sampling defaults in ``generation_config.json``.
    Sending ours unasked overrides the model's own recommendation, so ``None``
    must leave the parameter out of the request entirely."""

    @pytest.mark.asyncio
    async def test_none_temperature_is_omitted(self) -> None:
        client = _client_with()
        m = _model_with(client, temperature=None)
        await m.complete([Message.user("hi")])
        args = client.chat.completions.create.call_args.kwargs
        assert "temperature" not in args
        assert args["top_p"] == 0.9  # unset params still send their default

    @pytest.mark.asyncio
    async def test_none_top_p_is_omitted(self) -> None:
        client = _client_with()
        m = _model_with(client, top_p=None)
        await m.complete([Message.user("hi")])
        args = client.chat.completions.create.call_args.kwargs
        assert "top_p" not in args
        assert args["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_both_omitted_leaves_server_defaults(self) -> None:
        client = _client_with()
        m = _model_with(client, temperature=None, top_p=None)
        await m.complete([Message.user("hi")])
        args = client.chat.completions.create.call_args.kwargs
        assert "temperature" not in args
        assert "top_p" not in args

    @pytest.mark.asyncio
    async def test_defaults_are_unchanged(self) -> None:
        """Existing callers must see identical behaviour."""
        client = _client_with()
        m = _model_with(client)
        await m.complete([Message.user("hi")])
        args = client.chat.completions.create.call_args.kwargs
        assert args["temperature"] == 0.7
        assert args["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_per_call_override_wins(self) -> None:
        client = _client_with()
        m = _model_with(client, temperature=0.7)
        await m.complete([Message.user("hi")], temperature=None)
        assert "temperature" not in client.chat.completions.create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_stream_omits_none_sampling(self) -> None:
        client = _client_with(
            stream_chunks=[_Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")])]
        )
        m = _model_with(client, temperature=None, top_p=None)
        async for _ in m.stream([Message.user("hi")]):
            pass
        args = client.chat.completions.create.call_args.kwargs
        assert "temperature" not in args
        assert "top_p" not in args


# ---------------------------------------------------------------------------
# extra_body: provider-specific request fields
# ---------------------------------------------------------------------------


class TestExtraBody:
    """OpenAI-compatible servers accept fields the OpenAI schema has no slot
    for — vLLM's ``chat_template_kwargs``, ``top_k``, ``min_p``. Without a
    passthrough they are unreachable through the SDK."""

    @pytest.mark.asyncio
    async def test_config_extra_body_is_forwarded(self) -> None:
        client = _client_with()
        m = _model_with(client, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        await m.complete([Message.user("hi")])
        args = client.chat.completions.create.call_args.kwargs
        assert args["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    @pytest.mark.asyncio
    async def test_absent_by_default(self) -> None:
        client = _client_with()
        m = _model_with(client)
        await m.complete([Message.user("hi")])
        assert "extra_body" not in client.chat.completions.create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_per_call_merges_over_config(self) -> None:
        client = _client_with()
        m = _model_with(client, extra_body={"top_k": 20, "min_p": 0.0})
        await m.complete([Message.user("hi")], extra_body={"min_p": 0.05})
        body = client.chat.completions.create.call_args.kwargs["extra_body"]
        assert body == {"top_k": 20, "min_p": 0.05}

    @pytest.mark.asyncio
    async def test_forwarded_for_reasoning_models_too(self) -> None:
        """o-series reject sampling params but still accept provider extensions."""
        client = _client_with()
        m = _model_with(client, model="o1-mini", extra_body={"top_k": 20})
        await m.complete([Message.user("hi")])
        args = client.chat.completions.create.call_args.kwargs
        assert args["extra_body"] == {"top_k": 20}
        assert "temperature" not in args

    @pytest.mark.asyncio
    async def test_stream_forwards_extra_body(self) -> None:
        client = _client_with(
            stream_chunks=[_Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")])]
        )
        m = _model_with(client, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        async for _ in m.stream([Message.user("hi")]):
            pass
        args = client.chat.completions.create.call_args.kwargs
        assert args["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


# ---------------------------------------------------------------------------
# Full Chat Completions surface: don't clog what the server supports
# ---------------------------------------------------------------------------


class TestParameterPassthrough:
    """The provider previously read six keys out of **kwargs and dropped the
    rest, so most of the API was unreachable and callers had to leave the SDK
    to get at it."""

    @pytest.mark.asyncio
    async def test_tool_choice_reaches_the_server(self) -> None:
        client = _client_with()
        m = _model_with(client)
        await m.complete(
            [Message.user("hi")],
            tools=[{"name": "search", "parameters": {}}],
            tool_choice={"type": "function", "function": {"name": "search"}},
        )
        args = client.chat.completions.create.call_args.kwargs
        assert args["tool_choice"] == {"type": "function", "function": {"name": "search"}}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("parallel_tool_calls", False),
            ("logprobs", True),
            ("top_logprobs", 5),
            ("n", 3),
            ("user", "u-1"),
            ("logit_bias", {"123": -100}),
            ("reasoning_effort", "high"),
            ("service_tier", "auto"),
            ("metadata", {"run": "abc"}),
            ("store", True),
        ],
    )
    async def test_documented_params_are_forwarded(self, name: str, value: Any) -> None:
        client = _client_with()
        m = _model_with(client)
        await m.complete([Message.user("hi")], **{name: value})
        assert client.chat.completions.create.call_args.kwargs[name] == value

    @pytest.mark.asyncio
    async def test_unknown_keys_are_not_forwarded(self) -> None:
        """A non-OpenAI key must not reach the API — that belongs in extra_body."""
        client = _client_with()
        m = _model_with(client)
        await m.complete([Message.user("hi")], not_a_real_openai_param=1)
        assert "not_a_real_openai_param" not in client.chat.completions.create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_provider_owned_params_are_not_overridden(self) -> None:
        client = _client_with()
        m = _model_with(client, model="gpt-4o")
        await m.complete([Message.user("hi")], model="sneaky", stream=True, max_tokens=99)
        args = client.chat.completions.create.call_args.kwargs
        assert args["model"] == "gpt-4o", "provider owns the model name"
        assert args.get("stream") is not True, "complete() must not be flipped to streaming"
        # max_tokens keeps the family-aware path, not a second passthrough copy
        assert args["max_tokens"] == 99

    @pytest.mark.asyncio
    async def test_stream_options_forwarded_when_streaming(self) -> None:
        client = _client_with(
            stream_chunks=[_Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")])]
        )
        m = _model_with(client)
        async for _ in m.stream([Message.user("hi")], stream_options={"include_usage": True}):
            pass
        args = client.chat.completions.create.call_args.kwargs
        assert args["stream_options"] == {"include_usage": True}
        assert args["stream"] is True

    def test_param_set_is_read_from_the_openai_package(self) -> None:
        """Hand-maintained lists go stale; this one must track the dependency."""
        from tulip.models.native.openai import _OPENAI_PARAMS

        for expected in ("tool_choice", "parallel_tool_calls", "logprobs", "n", "seed"):
            assert expected in _OPENAI_PARAMS


# ---------------------------------------------------------------------------
# ModelResponse carries logprobs and extra candidates (#53)
# ---------------------------------------------------------------------------


class _UsageChunk:
    """Trailing chunk carrying usage and no choices (stream_options)."""

    def __init__(self, prompt: int, completion: int) -> None:
        self.choices: list[Any] = []
        self.usage = _Usage(prompt, completion)


class TestResponseExtras:
    @pytest.mark.asyncio
    async def test_extra_candidates_are_kept(self) -> None:
        """n>1 is paid for; the extra choices must not be discarded."""
        client = _client_with(
            response=_Response(
                choices=[
                    _Choice(message=_MsgStub(content="first")),
                    _Choice(message=_MsgStub(content="second")),
                    _Choice(message=_MsgStub(content="third")),
                ]
            )
        )
        m = _model_with(client)
        resp = await m.complete([Message.user("hi")], n=3)
        assert resp.message.content == "first"
        assert [c.content for c in resp.candidates] == ["second", "third"]

    @pytest.mark.asyncio
    async def test_single_choice_leaves_candidates_empty(self) -> None:
        client = _client_with()
        m = _model_with(client)
        resp = await m.complete([Message.user("hi")])
        assert resp.candidates == []

    @pytest.mark.asyncio
    async def test_logprobs_are_passed_through(self) -> None:
        client = _client_with()
        client.chat.completions.create.return_value.choices[0].logprobs = {"content": [1, 2]}
        m = _model_with(client)
        resp = await m.complete([Message.user("hi")], logprobs=True)
        assert resp.logprobs == {"content": [1, 2]}


# ---------------------------------------------------------------------------
# Streaming exposes usage and stop_reason (#54)
# ---------------------------------------------------------------------------


class TestStreamTermination:
    @pytest.mark.asyncio
    async def test_usage_chunk_is_not_dropped(self) -> None:
        """The usage chunk carries no choices — it must survive the guard."""
        client = _client_with(
            stream_chunks=[
                _Chunk(choices=[_ChunkChoice(delta=_Delta(content="hi"), finish_reason=None)]),
                _Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")]),
                _UsageChunk(11, 5),
            ]
        )
        m = _model_with(client)
        events = [ev async for ev in m.stream([Message.user("hi")])]

        final = events[-1]
        assert final.done is True
        assert final.usage == {"prompt_tokens": 11, "completion_tokens": 5}
        assert final.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_length_truncation_is_reported(self) -> None:
        client = _client_with(
            stream_chunks=[_Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="length")])]
        )
        m = _model_with(client)
        events = [ev async for ev in m.stream([Message.user("hi")])]
        assert events[-1].stop_reason == "length"

    @pytest.mark.asyncio
    async def test_done_is_emitted_exactly_once(self) -> None:
        client = _client_with(
            stream_chunks=[
                _Chunk(choices=[_ChunkChoice(delta=_Delta(content="a"), finish_reason=None)]),
                _Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")]),
                _UsageChunk(1, 1),
            ]
        )
        m = _model_with(client)
        events = [ev async for ev in m.stream([Message.user("hi")])]
        assert sum(1 for e in events if e.done) == 1


# ---------------------------------------------------------------------------
# Mid-run system messages must stay portable
# ---------------------------------------------------------------------------


class TestSystemMessagePosition:
    """The loop injects guidance as system messages mid-run; vLLM serving Qwen
    rejects those outright with 'System message must be at the beginning'."""

    @pytest.mark.asyncio
    async def test_leading_system_message_is_untouched(self) -> None:
        client = _client_with()
        m = _model_with(client)
        await m.complete([Message.system("you are helpful"), Message.user("hi")])
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert sent[0]["role"] == "system"
        assert sent[0]["content"] == "you are helpful"

    @pytest.mark.asyncio
    async def test_later_system_message_becomes_a_user_note(self) -> None:
        client = _client_with()
        m = _model_with(client)
        await m.complete(
            [
                Message.system("you are helpful"),
                Message.user("hi"),
                Message.system("[Grounding Check Failed] try again"),
            ]
        )
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert [x["role"] for x in sent] == ["system", "user", "user"]
        # the guidance survives, just re-encoded
        assert "[Grounding Check Failed] try again" in sent[-1]["content"]
        assert sent[-1]["content"].startswith("[System guidance]")

    @pytest.mark.asyncio
    async def test_stream_applies_the_same_normalisation(self) -> None:
        client = _client_with(
            stream_chunks=[_Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")])]
        )
        m = _model_with(client)
        async for _ in m.stream([Message.user("hi"), Message.system("mid-run guidance")]):
            pass
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert all(x["role"] != "system" for x in sent[1:])


# ---------------------------------------------------------------------------
# Every request must carry a user turn
# ---------------------------------------------------------------------------


class TestUserTurnIsAlwaysPresent:
    """Qwen-family chat templates scan backwards for a user message that is not
    a ``<tool_response>`` wrapper and raise ``No user query found in messages.``
    when there is none. Sub-calls built from ``system + assistant + tool`` (judge,
    summary, auxiliary passes) otherwise 400 on a self-hosted Qwen while working
    fine against api.openai.com."""

    ANCHOR = "[Continue] Continue from the conversation above."

    @pytest.mark.asyncio
    async def test_system_assistant_tool_gets_a_user_turn(self) -> None:
        client = _client_with()
        m = _model_with(client)
        await m.complete(
            [
                Message.system("you are helpful"),
                Message.assistant(
                    "", tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})]
                ),
                Message.tool(ToolResult(tool_call_id="c1", name="read_file", content="contents")),
            ]
        )
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert [x["role"] for x in sent] == ["system", "user", "assistant", "tool"]
        # the system prompt keeps first position — the sibling template rule
        assert sent[0]["content"] == "you are helpful"
        assert sent[1]["content"] == self.ANCHOR

    @pytest.mark.asyncio
    async def test_a_real_user_turn_is_not_duplicated(self) -> None:
        client = _client_with()
        m = _model_with(client)
        await m.complete([Message.system("you are helpful"), Message.user("hi")])
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert [x["role"] for x in sent] == ["system", "user"]
        assert sent[1]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_tool_response_wrapper_does_not_count_as_the_query(self) -> None:
        """The template ignores such a message, so the guard must too."""
        client = _client_with()
        m = _model_with(client)
        await m.complete(
            [
                Message.system("you are helpful"),
                Message.user("<tool_response>file listed</tool_response>"),
            ]
        )
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert [x["role"] for x in sent] == ["system", "user", "user"]
        assert sent[1]["content"] == self.ANCHOR

    @pytest.mark.asyncio
    async def test_anchor_goes_first_when_there_is_no_system_message(self) -> None:
        client = _client_with()
        m = _model_with(client)
        await m.complete([Message.assistant("hi")])
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert [x["role"] for x in sent] == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_empty_message_list_is_left_alone(self) -> None:
        client = _client_with()
        m = _model_with(client)
        await m.complete([])
        assert client.chat.completions.create.call_args.kwargs["messages"] == []

    @pytest.mark.asyncio
    async def test_stream_applies_the_same_guard(self) -> None:
        client = _client_with(
            stream_chunks=[_Chunk(choices=[_ChunkChoice(delta=_Delta(), finish_reason="stop")])]
        )
        m = _model_with(client)
        async for _ in m.stream([Message.system("s"), Message.assistant("hi")]):
            pass
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert any(x["role"] == "user" for x in sent)


class TestResponseExtrasEdgeCases:
    """Edge cases in candidate parsing and the param-set fallback."""

    @pytest.mark.asyncio
    async def test_candidate_without_message_is_skipped(self) -> None:
        """A provider can return a filtered/empty choice with no message."""
        client = _client_with(
            response=_Response(
                choices=[
                    _Choice(message=_MsgStub(content="first")),
                    _Choice(message=None),
                    _Choice(message=_MsgStub(content="third")),
                ]
            )
        )
        m = _model_with(client)
        resp = await m.complete([Message.user("hi")], n=3)
        assert [c.content for c in resp.candidates] == ["third"]

    @pytest.mark.asyncio
    async def test_candidate_tool_calls_are_decoded(self) -> None:
        client = _client_with(
            response=_Response(
                choices=[
                    _Choice(message=_MsgStub(content="first")),
                    _Choice(
                        message=_MsgStub(
                            content=None,
                            tool_calls=[
                                _ToolCallStub(call_id="c1", name="search", arguments='{"q":"x"}')
                            ],
                        )
                    ),
                ]
            )
        )
        m = _model_with(client)
        resp = await m.complete([Message.user("hi")], n=2)
        assert len(resp.candidates) == 1
        call = resp.candidates[0].tool_calls[0]
        assert (call.name, call.arguments) == ("search", {"q": "x"})

    @pytest.mark.asyncio
    async def test_candidate_non_string_content_becomes_none(self) -> None:
        """Mock-heavy stubs (and some providers) hand back non-str content."""
        client = _client_with(
            response=_Response(
                choices=[
                    _Choice(message=_MsgStub(content="first")),
                    _Choice(message=_MsgStub(content=object())),  # type: ignore[arg-type]
                ]
            )
        )
        m = _model_with(client)
        resp = await m.complete([Message.user("hi")], n=2)
        assert resp.candidates[0].content is None

    def test_param_names_fall_back_when_introspection_fails(self, monkeypatch) -> None:
        """A stale hand-list would block passthrough entirely; the fallback
        keeps the long-stable parameters reachable."""
        import typing as _typing

        import tulip.models.native.openai as mod

        def boom(*_a, **_k):
            raise AttributeError("openai moved its request types")

        monkeypatch.setattr(_typing, "get_type_hints", boom)

        names = mod._openai_param_names()
        assert "tool_choice" in names
        assert names == mod._FALLBACK_OPENAI_PARAMS
