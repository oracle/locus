# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.models.native.ollama`` (OllamaModel).

The Ollama provider talks to a local server. These tests stub the
``ollama.AsyncClient`` so we never make a network call. Coverage:

- config defaults
- message + tool format conversion
- ``complete`` response parsing for both ``ollama.Message`` objects
  and the legacy dict shape
- tool-call deserialisation (string-encoded JSON arguments + dict)
- ``stream`` async iteration yielding ``ModelChunkEvent``
- ``supports_structured_output`` capability flag
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from locus.core.messages import Message, ToolCall
from locus.models.native.ollama import OllamaConfig, OllamaModel


# ---------------------------------------------------------------------------
# Stub Ollama client.
# ---------------------------------------------------------------------------


class _StubOllamaClient:
    """Minimal stand-in for ``ollama.AsyncClient``."""

    def __init__(self, *, response: Any = None, stream: list[Any] | None = None) -> None:
        self._response = response
        self._stream = stream or []
        self.last_call: dict[str, Any] = {}

    async def chat(self, **kwargs: Any) -> Any:
        self.last_call = kwargs
        if kwargs.get("stream"):

            async def gen() -> AsyncIterator[Any]:
                for chunk in self._stream:
                    yield chunk

            return gen()
        return self._response


def _model_with(client: _StubOllamaClient) -> OllamaModel:
    """Build an OllamaModel and inject the stub client."""
    model = OllamaModel(model="llama3.3")
    model._client = client
    return model


# ---------------------------------------------------------------------------
# Config + capability flag
# ---------------------------------------------------------------------------


class TestOllamaConfig:
    def test_defaults(self) -> None:
        cfg = OllamaConfig()
        assert cfg.model == "llama3.3"
        assert cfg.base_url == "http://localhost:11434"
        assert cfg.max_tokens == 4096

    def test_constructor_propagates_overrides(self) -> None:
        m = OllamaModel(model="phi4", base_url="http://hostX:1234", max_tokens=10, temperature=0.1)
        assert m.config.model == "phi4"
        assert m.config.base_url == "http://hostX:1234"
        assert m.config.max_tokens == 10

    def test_does_not_support_structured_output(self) -> None:
        # Ollama doesn't ship OpenAI-style ``response_format``; the agent
        # falls back to prompted-JSON for these models.
        assert OllamaModel().supports_structured_output is False


# ---------------------------------------------------------------------------
# Lazy client init
# ---------------------------------------------------------------------------


class TestClientImport:
    def test_missing_ollama_package_raises_clear_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Sub the import to raise.
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "ollama":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        model = OllamaModel()
        with pytest.raises(ImportError, match="ollama package required"):
            _ = model.client


# ---------------------------------------------------------------------------
# Message + tool conversion
# ---------------------------------------------------------------------------


class TestConversions:
    def test_convert_messages_simple(self) -> None:
        m = OllamaModel()
        out = m._convert_messages([Message.user("hi")])
        assert out == [{"role": "user", "content": "hi"}]

    def test_convert_messages_with_tool_calls(self) -> None:
        m = OllamaModel()
        msg = Message.assistant(
            "",
            tool_calls=[ToolCall(id="t1", name="search", arguments={"q": "x"})],
        )
        out = m._convert_messages([msg])
        assert out[0]["tool_calls"][0]["function"]["name"] == "search"

    def test_convert_tools_wraps_in_function_envelope(self) -> None:
        m = OllamaModel()
        out = m._convert_tools([{"name": "search", "parameters": {}}])
        assert out is not None
        assert out[0] == {
            "type": "function",
            "function": {"name": "search", "parameters": {}},
        }

    def test_convert_tools_passes_already_typed_tools_through(self) -> None:
        m = OllamaModel()
        already = [{"type": "function", "function": {"name": "search"}}]
        out = m._convert_tools(already)
        assert out == already

    def test_convert_tools_none_returns_none(self) -> None:
        assert OllamaModel()._convert_tools(None) is None

    def test_convert_tools_empty_returns_none(self) -> None:
        assert OllamaModel()._convert_tools([]) is None


# ---------------------------------------------------------------------------
# complete() — response parsing
# ---------------------------------------------------------------------------


class _OllamaMessage:
    """Mimic ``ollama.Message`` — has ``content`` + ``tool_calls`` attrs."""

    def __init__(self, content: str = "", tool_calls: list[dict[str, Any]] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class TestComplete:
    @pytest.mark.asyncio
    async def test_parses_ollama_message_object(self) -> None:
        client = _StubOllamaClient(response={"message": _OllamaMessage(content="hello world")})
        m = _model_with(client)
        resp = await m.complete([Message.user("hi")])
        assert resp.message.content == "hello world"

    @pytest.mark.asyncio
    async def test_parses_legacy_dict_shape(self) -> None:
        client = _StubOllamaClient(response={"message": {"content": "from dict"}})
        m = _model_with(client)
        resp = await m.complete([Message.user("hi")])
        assert resp.message.content == "from dict"

    @pytest.mark.asyncio
    async def test_parses_tool_calls_with_dict_arguments(self) -> None:
        client = _StubOllamaClient(
            response={
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "search", "arguments": {"q": "x"}}}],
                }
            }
        )
        m = _model_with(client)
        resp = await m.complete([Message.user("hi")])
        assert len(resp.message.tool_calls) == 1
        assert resp.message.tool_calls[0].name == "search"
        assert resp.message.tool_calls[0].arguments == {"q": "x"}

    @pytest.mark.asyncio
    async def test_parses_tool_calls_with_json_string_arguments(self) -> None:
        # Some Ollama builds return arguments as a JSON-encoded string.
        client = _StubOllamaClient(
            response={
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search",
                                "arguments": json.dumps({"q": "x"}),
                            }
                        }
                    ],
                }
            }
        )
        m = _model_with(client)
        resp = await m.complete([Message.user("hi")])
        assert resp.message.tool_calls[0].arguments == {"q": "x"}

    @pytest.mark.asyncio
    async def test_parses_tool_calls_with_invalid_json_string_arguments(self) -> None:
        # Malformed JSON → empty dict, no crash.
        client = _StubOllamaClient(
            response={
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "search", "arguments": "{not json"}}],
                }
            }
        )
        m = _model_with(client)
        resp = await m.complete([Message.user("hi")])
        assert resp.message.tool_calls[0].arguments == {}

    @pytest.mark.asyncio
    async def test_includes_usage_when_response_has_token_counts(self) -> None:
        client = _StubOllamaClient(
            response={
                "message": {"content": "ok"},
                "prompt_eval_count": 10,
                "eval_count": 25,
                "done": True,
            }
        )
        m = _model_with(client)
        resp = await m.complete([Message.user("hi")])
        assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 25}
        assert resp.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_no_usage_when_token_counts_missing(self) -> None:
        client = _StubOllamaClient(response={"message": {"content": "ok"}})
        m = _model_with(client)
        resp = await m.complete([Message.user("hi")])
        assert resp.usage == {}

    @pytest.mark.asyncio
    async def test_propagates_tool_param_to_client(self) -> None:
        client = _StubOllamaClient(response={"message": {"content": "ok"}})
        m = _model_with(client)
        await m.complete(
            [Message.user("hi")],
            tools=[{"name": "search", "parameters": {}}],
        )
        assert "tools" in client.last_call

    @pytest.mark.asyncio
    async def test_kwargs_override_config_temperature(self) -> None:
        client = _StubOllamaClient(response={"message": {"content": "ok"}})
        m = _model_with(client)
        await m.complete([Message.user("hi")], temperature=0.0, max_tokens=100)
        assert client.last_call["options"]["temperature"] == 0.0
        assert client.last_call["options"]["num_predict"] == 100


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


class TestStream:
    @pytest.mark.asyncio
    async def test_yields_chunk_events_and_done(self) -> None:
        client = _StubOllamaClient(
            stream=[
                {"message": {"content": "Greet"}},
                {"message": {"content": "ings"}},
                {"done": True, "message": {"content": ""}},
            ]
        )
        m = _model_with(client)
        events = [ev async for ev in m.stream([Message.user("hi")])]
        # Two content chunks + one ``done=True`` terminator.
        contents = [ev.content for ev in events if ev.content]
        assert contents == ["Greet", "ings"]
        assert any(ev.done for ev in events)

    @pytest.mark.asyncio
    async def test_skips_empty_content_chunks(self) -> None:
        client = _StubOllamaClient(
            stream=[
                {"message": {"content": ""}},
                {"message": {"content": "ok"}},
                {"done": True, "message": {"content": ""}},
            ]
        )
        m = _model_with(client)
        events = [ev async for ev in m.stream([Message.user("hi")])]
        # The empty leading chunk emits no event; only "ok" + done.
        non_done_events = [ev for ev in events if not ev.done]
        assert all(ev.content for ev in non_done_events)

    @pytest.mark.asyncio
    async def test_emits_done_when_stream_exhausted_without_done_flag(self) -> None:
        # If the upstream stream ends without ever sending ``done=True``,
        # the wrapper still emits a closing ``ModelChunkEvent(done=True)``.
        client = _StubOllamaClient(stream=[{"message": {"content": "partial"}}])
        m = _model_with(client)
        events = [ev async for ev in m.stream([Message.user("hi")])]
        # Exactly one done event at the end.
        assert events[-1].done is True
