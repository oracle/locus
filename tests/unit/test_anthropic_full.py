# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage tests for ``locus.models.native.anthropic``.

The existing ``test_anthropic_prompt_caching.py`` covers the cache-control
wiring. This file targets the rest:

- ``supports_structured_output`` property is stably False
- ``client`` property lazily constructs the SDK client
- ``_convert_messages`` for assistant + tool_calls + tool result roles
- ``_convert_tools`` for the empty / non-empty cases
- ``_structured_output_tool`` shape
- ``complete()`` structured-mode tool_choice + payload extraction
- ``complete()`` parses regular tool_use blocks into ``ToolCall``
- ``stream()`` yields chunks then a ``done=True`` terminator

The Anthropic SDK is mocked end-to-end so the test never touches the
real client.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


pytest.importorskip("anthropic")

from locus.core.messages import Message, Role, ToolCall  # noqa: E402
from locus.models.native.anthropic import AnthropicModel  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _block(**kw: Any) -> SimpleNamespace:
    return SimpleNamespace(**kw)


_SENTINEL = object()


def _make_response(
    *,
    blocks: list[Any] | None = None,
    stop_reason: str = "end_turn",
    usage: Any = _SENTINEL,
) -> SimpleNamespace:
    if usage is _SENTINEL:
        usage = SimpleNamespace(input_tokens=1, output_tokens=1)
    return SimpleNamespace(
        content=blocks or [_block(type="text", text="hi")],
        usage=usage,
        stop_reason=stop_reason,
    )


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_supports_structured_output_is_false(self) -> None:
        m = AnthropicModel(api_key="sk-test")  # noqa: S106
        assert m.supports_structured_output is False


# ---------------------------------------------------------------------------
# Client property
# ---------------------------------------------------------------------------


class TestClientLazyInit:
    def test_client_constructs_async_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import anthropic as _real_anthropic

        captured: dict[str, Any] = {}

        class _FakeClient:
            def __init__(self, **kw: Any) -> None:
                captured.update(kw)

        monkeypatch.setattr(_real_anthropic, "AsyncAnthropic", _FakeClient)
        m = AnthropicModel(api_key="sk-test", base_url="https://x.example")  # noqa: S106
        client1 = m.client
        client2 = m.client
        assert client1 is client2  # cached
        assert isinstance(client1, _FakeClient)
        assert captured["api_key"] == "sk-test"  # noqa: S105
        assert captured["base_url"] == "https://x.example"


# ---------------------------------------------------------------------------
# _convert_messages
# ---------------------------------------------------------------------------


class TestConvertMessages:
    def test_extracts_system_prompt(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        sys_prompt, msgs = m._convert_messages([Message.system("be terse"), Message.user("hi")])
        assert sys_prompt == "be terse"
        assert msgs[0]["role"] == "user"

    def test_assistant_with_text_only(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        _, msgs = m._convert_messages([Message.assistant(content="answer text")])
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"][0] == {"type": "text", "text": "answer text"}

    def test_assistant_with_tool_calls(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        tc = ToolCall(id="t1", name="search", arguments={"q": "x"})
        _, msgs = m._convert_messages([Message.assistant(content=None, tool_calls=[tc])])
        # First (and only) content block should be a tool_use entry
        block = msgs[0]["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "t1"
        assert block["name"] == "search"
        assert block["input"] == {"q": "x"}

    def test_assistant_with_text_and_tool_calls(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        tc = ToolCall(id="t2", name="run", arguments={})
        _, msgs = m._convert_messages([Message.assistant(content="reasoning", tool_calls=[tc])])
        # Two blocks: text + tool_use
        types = [b["type"] for b in msgs[0]["content"]]
        assert types == ["text", "tool_use"]

    def test_tool_role_becomes_user_tool_result(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        tool_msg = Message(role=Role.TOOL, content="output", tool_call_id="t9")
        _, msgs = m._convert_messages([tool_msg])
        assert msgs[0]["role"] == "user"
        block = msgs[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "t9"
        assert block["content"] == "output"

    def test_tool_role_without_id_becomes_empty_string(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        tool_msg = Message(role=Role.TOOL, content="output")
        _, msgs = m._convert_messages([tool_msg])
        block = msgs[0]["content"][0]
        assert block["tool_use_id"] == ""


# ---------------------------------------------------------------------------
# _convert_tools
# ---------------------------------------------------------------------------


class TestConvertTools:
    def test_none_returns_none(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        assert m._convert_tools(None) is None

    def test_empty_list_returns_none(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        assert m._convert_tools([]) is None

    def test_openai_function_format_translated(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ]
        out = m._convert_tools(tools)
        assert out is not None
        assert out[0]["name"] == "search"
        assert out[0]["description"] == "search the web"
        assert out[0]["input_schema"]["properties"]["q"]["type"] == "string"

    def test_top_level_format_translated(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        tools = [{"name": "ping", "description": "", "parameters": {}}]
        out = m._convert_tools(tools)
        assert out is not None
        assert out[0]["name"] == "ping"
        # Default empty schema when parameters is empty dict
        assert out[0]["input_schema"] == {}


# ---------------------------------------------------------------------------
# _structured_output_tool
# ---------------------------------------------------------------------------


class TestStructuredOutputTool:
    def test_with_named_schema(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "Answer",
                "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        }
        tool = m._structured_output_tool(rf)
        assert tool["name"] == "respond_with_schema"
        assert "Answer" in tool["description"]
        assert tool["input_schema"]["properties"]["x"]["type"] == "string"

    def test_falls_back_to_empty_schema(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        tool = m._structured_output_tool({"type": "json_schema"})
        assert tool["input_schema"] == {"type": "object", "properties": {}}

    def test_uses_explicit_description(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "X",
                "schema": {"type": "object"},
                "description": "specific override",
            },
        }
        tool = m._structured_output_tool(rf)
        assert tool["description"] == "specific override"


# ---------------------------------------------------------------------------
# complete() — full SDK mock
# ---------------------------------------------------------------------------


def _patch_client(model: AnthropicModel, response: Any) -> MagicMock:
    """Install a mock SDK client on ``model`` and return the mock."""
    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=response)
    model._client = mock_client  # type: ignore[assignment]
    return mock_client


class TestComplete:
    @pytest.mark.asyncio
    async def test_parses_text_only_response(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        _patch_client(m, _make_response(blocks=[_block(type="text", text="hello")]))
        resp = await m.complete([Message.user("hi")])
        assert resp.message.content == "hello"
        assert resp.message.tool_calls == []
        assert resp.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_parses_tool_use_block_into_tool_calls(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        blocks = [
            _block(type="text", text="thinking..."),
            _block(type="tool_use", id="t1", name="search", input={"q": "x"}),
        ]
        _patch_client(m, _make_response(blocks=blocks))
        resp = await m.complete([Message.user("hi")])
        assert len(resp.message.tool_calls) == 1
        assert resp.message.tool_calls[0].name == "search"
        assert resp.message.tool_calls[0].arguments == {"q": "x"}

    @pytest.mark.asyncio
    async def test_tool_use_with_non_dict_input_becomes_empty(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        blocks = [_block(type="tool_use", id="t2", name="run", input="not-a-dict")]
        _patch_client(m, _make_response(blocks=blocks))
        resp = await m.complete([Message.user("hi")])
        assert resp.message.tool_calls[0].arguments == {}

    @pytest.mark.asyncio
    async def test_structured_mode_pins_tool_choice(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        # Response contains a tool_use block named ``respond_with_schema``.
        blocks = [
            _block(
                type="tool_use",
                id="s1",
                name="respond_with_schema",
                input={"answer": 42},
            ),
        ]
        client = _patch_client(m, _make_response(blocks=blocks))
        resp = await m.complete(
            [Message.user("how many?")],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "Answer",
                    "schema": {"type": "object", "properties": {"answer": {"type": "integer"}}},
                },
            },
        )
        # The structured payload becomes the canonical-JSON content.
        assert resp.message.content is not None
        assert '"answer"' in resp.message.content
        assert "42" in resp.message.content
        # And the request was sent with tool_choice pinned to our synthetic tool.
        sent_params = client.messages.create.call_args.kwargs
        assert sent_params["tool_choice"]["type"] == "tool"
        assert sent_params["tool_choice"]["name"] == "respond_with_schema"

    @pytest.mark.asyncio
    async def test_structured_mode_with_non_dict_input_yields_empty_payload(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        blocks = [
            _block(type="tool_use", id="s2", name="respond_with_schema", input="bad"),
        ]
        _patch_client(m, _make_response(blocks=blocks))
        resp = await m.complete(
            [Message.user("hi")],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "X", "schema": {"type": "object"}},
            },
        )
        assert resp.message.content == "{}"

    @pytest.mark.asyncio
    async def test_usage_without_cache_fields(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        _patch_client(m, _make_response(usage=usage))
        resp = await m.complete([Message.user("hi")])
        assert resp.usage["prompt_tokens"] == 10
        assert resp.usage["completion_tokens"] == 5
        assert "cache_creation_input_tokens" not in resp.usage

    @pytest.mark.asyncio
    async def test_no_usage_returns_empty_usage_dict(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106
        _patch_client(m, _make_response(usage=None))
        resp = await m.complete([Message.user("hi")])
        assert resp.usage == {}


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_yields_chunks_then_done(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106

        # Build an async-context-manager that exposes ``text_stream`` as an
        # async iterator.
        class _FakeTextStream:
            def __init__(self, parts: list[str]) -> None:
                self._parts = parts

            def __aiter__(self) -> _FakeTextStream:
                self._iter = iter(self._parts)
                return self

            async def __anext__(self) -> str:
                try:
                    return next(self._iter)
                except StopIteration as e:
                    raise StopAsyncIteration from e

        class _FakeStreamCM:
            def __init__(self, parts: list[str]) -> None:
                self.text_stream = _FakeTextStream(parts)

            async def __aenter__(self) -> _FakeStreamCM:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

        client = MagicMock()
        client.messages = MagicMock()
        client.messages.stream = MagicMock(return_value=_FakeStreamCM(["foo", "bar"]))
        m._client = client  # type: ignore[assignment]

        events: list[Any] = []
        async for ev in m.stream([Message.user("hi")]):
            events.append(ev)
        # foo, bar, done
        assert len(events) == 3
        assert events[0].content == "foo"
        assert events[1].content == "bar"
        assert events[2].done is True

    @pytest.mark.asyncio
    async def test_stream_with_tools_and_system(self) -> None:
        m = AnthropicModel(api_key="sk-x")  # noqa: S106

        class _FakeTextStream:
            def __aiter__(self) -> _FakeTextStream:
                return self

            async def __anext__(self) -> str:
                raise StopAsyncIteration

        class _FakeStreamCM:
            text_stream = _FakeTextStream()

            async def __aenter__(self) -> _FakeStreamCM:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

        client = MagicMock()
        client.messages = MagicMock()
        captured: dict[str, Any] = {}

        def _make_stream(**kw: Any) -> _FakeStreamCM:
            captured.update(kw)
            return _FakeStreamCM()

        client.messages.stream = _make_stream
        m._client = client  # type: ignore[assignment]

        events = []
        async for ev in m.stream(
            [Message.system("be brief"), Message.user("hi")],
            tools=[{"name": "x", "description": "", "parameters": {}}],
        ):
            events.append(ev)
        # Check both branches were taken
        assert captured["system"] == "be brief"
        assert "tools" in captured
        # Just the done event
        assert events[-1].done is True
