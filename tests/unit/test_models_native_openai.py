# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.models.native.openai`` (OpenAIModel).

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

from locus.core.messages import Message
from locus.models.native.openai import OpenAIModel, _decode_tool_arguments


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
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


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
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls


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
        # OCI Llama deployments occasionally double-encode.
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
