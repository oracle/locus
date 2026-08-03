# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Responses API path of ``tulip.models.native.openai``.

The GPT-5.6 family rejects function tools on ``/v1/chat/completions``
whenever reasoning is active, so ``OpenAIModel`` speaks ``/v1/responses``
for it. The tests below cover:

- API selection (``api="auto"|"responses"|"chat_completions"``, the
  gpt-5.6 known-prefix set, and the custom-``base_url`` guard that keeps
  auto-selection away from OpenAI-compatible gateways)
- message → input-item conversion, including the verbatim replay of raw
  output items stashed in ``Message.metadata`` (the stateless reasoning
  round trip) and its reconstruction fallback
- tool-schema flattening to the Responses shape
- request shaping (``max_output_tokens``, sampling gates, ``reasoning`` /
  ``reasoning_effort`` merging, ``response_format`` → ``text``,
  ``tool_choice`` translation, ``store`` / ``include`` defaults,
  passthrough + ``extra_body``)
- result parsing (output items → content / tool calls / reasoning,
  usage + stop_reason mapped onto the chat-completions vocabulary)
- streaming (content / reasoning deltas, tool-call accumulation across
  events, terminal usage + stop reason, unknown events ignored)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from tulip.core.messages import Message, ToolCall, ToolResult
from tulip.models.native.openai import (
    RESPONSES_ITEMS_METADATA_KEY,
    OpenAIModel,
    _dump_output_item,
    _strip_model_namespace,
    _text_format_from_response_format,
)


# ---------------------------------------------------------------------------
# Lightweight response stubs (mirror the openai SDK's Responses shapes).
# ---------------------------------------------------------------------------


class _Obj:
    """Attribute bag mirroring openai SDK model objects (no ``model_dump``)."""

    def __init__(self, **attrs: Any) -> None:
        for key, value in attrs.items():
            setattr(self, key, value)


class _Item(_Obj):
    """Output item with a ``model_dump``, like the SDK's Pydantic items."""

    def __init__(self, dump: dict[str, Any] | None = None, **attrs: Any) -> None:
        super().__init__(**attrs)
        self._dump = dump if dump is not None else {"type": attrs.get("type")}

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return self._dump


def _text_item(text: str = "ok", *, item_id: str = "msg_1") -> _Item:
    return _Item(
        dump={
            "type": "message",
            "id": item_id,
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        },
        type="message",
        id=item_id,
        role="assistant",
        content=[_Obj(type="output_text", text=text, annotations=[])],
        status="completed",
    )


def _fn_call_item(
    *,
    item_id: str = "fc_1",
    call_id: str = "call_1",
    name: str = "get_weather",
    arguments: str = '{"city": "Tokyo"}',
) -> _Item:
    return _Item(
        dump={
            "type": "function_call",
            "id": item_id,
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        },
        type="function_call",
        id=item_id,
        call_id=call_id,
        name=name,
        arguments=arguments,
        status="completed",
    )


def _reasoning_item(
    *,
    item_id: str = "rs_1",
    summary: tuple[str, ...] = ("Thinking it through.",),
    content: tuple[str, ...] = (),
    encrypted: str = "gAAAA-opaque",
) -> _Item:
    return _Item(
        dump={
            "type": "reasoning",
            "id": item_id,
            "encrypted_content": encrypted,
            "summary": [{"type": "summary_text", "text": t} for t in summary],
        },
        type="reasoning",
        id=item_id,
        summary=[_Obj(type="summary_text", text=t) for t in summary],
        content=[_Obj(type="reasoning_text", text=t) for t in content],
        encrypted_content=encrypted,
    )


def _resp(
    *,
    output: list[Any] | None = None,
    usage: Any | None = None,
    status: str | None = "completed",
    incomplete_reason: str | None = None,
) -> _Obj:
    return _Obj(
        output=output if output is not None else [_text_item()],
        usage=usage,
        status=status,
        incomplete_details=(_Obj(reason=incomplete_reason) if incomplete_reason else None),
    )


def _events_stream(events: list[Any]) -> AsyncIterator[Any]:
    async def gen() -> AsyncIterator[Any]:
        for event in events:
            yield event

    return gen()


def _responses_client(
    *,
    response: Any | None = None,
    stream_events: list[Any] | None = None,
) -> AsyncMock:
    """Build a mock ``openai.AsyncOpenAI`` client canned for ``/v1/responses``."""
    client = AsyncMock()
    if stream_events is not None:
        client.responses.create.return_value = _events_stream(stream_events)
    else:
        client.responses.create.return_value = response if response is not None else _resp()
    return client


def _chat_response() -> _Obj:
    msg = _Obj(content="ok", tool_calls=None, reasoning_content=None)
    return _Obj(
        choices=[_Obj(message=msg, finish_reason="stop", logprobs=None)],
        usage=None,
    )


def _model_with(client: AsyncMock, *, model: str = "gpt-5.6-sol", **kwargs: Any) -> OpenAIModel:
    m = OpenAIModel(model=model, **kwargs)
    m._client = client
    return m


# ---------------------------------------------------------------------------
# API selection
# ---------------------------------------------------------------------------


class TestApiSelection:
    @pytest.mark.parametrize(
        "model",
        ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6", "openai.gpt-5.6-sol"],
    )
    def test_gpt_5_6_family_requires_responses(self, model: str) -> None:
        assert OpenAIModel._requires_responses_api(model) is True

    @pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5", "gpt-5-codex", "gpt-4o", "o3-mini"])
    def test_other_families_do_not_require_responses(self, model: str) -> None:
        assert OpenAIModel._requires_responses_api(model) is False

    def test_strip_model_namespace(self) -> None:
        assert _strip_model_namespace("openai.gpt-5.6-sol") == "gpt-5.6-sol"
        assert _strip_model_namespace("gpt-5.6-sol") == "gpt-5.6-sol"

    @pytest.mark.asyncio
    async def test_auto_selects_responses_for_gpt_5_6(self) -> None:
        client = _responses_client()
        m = _model_with(client, model="gpt-5.6-sol")
        await m.complete([Message.user("hi")])
        client.responses.create.assert_called_once()
        client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_never_fires_for_custom_base_url(self) -> None:
        # Together / vLLM / LiteLLM serve chat-completions, not /v1/responses.
        client = AsyncMock()
        client.chat.completions.create.return_value = _chat_response()
        m = _model_with(
            client,
            model="gpt-5.6-sol",
            base_url="https://api.together.xyz/v1",
        )
        await m.complete([Message.user("hi")])
        client.chat.completions.create.assert_called_once()
        client.responses.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_responses_wins_over_base_url(self) -> None:
        client = _responses_client()
        m = _model_with(
            client, model="gpt-4o", base_url="https://gw.example.com/v1", api="responses"
        )
        await m.complete([Message.user("hi")])
        client.responses.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_chat_completions_wins_over_family(self) -> None:
        client = AsyncMock()
        client.chat.completions.create.return_value = _chat_response()
        m = _model_with(client, model="gpt-5.6-sol", api="chat_completions")
        await m.complete([Message.user("hi")])
        client.chat.completions.create.assert_called_once()
        client.responses.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_keeps_classic_models_on_chat(self) -> None:
        client = AsyncMock()
        client.chat.completions.create.return_value = _chat_response()
        m = _model_with(client, model="gpt-4o")
        await m.complete([Message.user("hi")])
        client.chat.completions.create.assert_called_once()
        client.responses.create.assert_not_called()

    def test_invalid_api_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OpenAIModel(model="gpt-4o", api="respones")

    @pytest.mark.asyncio
    async def test_stream_auto_selects_responses(self) -> None:
        events = [_Obj(type="response.completed", response=_resp())]
        client = _responses_client(stream_events=events)
        m = _model_with(client, model="gpt-5.6-sol")
        async for _ in m.stream([Message.user("hi")]):
            pass
        args = client.responses.create.call_args.kwargs
        assert args["stream"] is True
        client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Message → input-item conversion
# ---------------------------------------------------------------------------


class TestConvertMessagesResponses:
    def test_system_and_user_become_role_items(self) -> None:
        m = OpenAIModel(model="gpt-5.6-sol")
        items = m._convert_messages_responses([Message.system("be helpful"), Message.user("hi")])
        assert items == [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]

    def test_mid_run_system_becomes_user_note(self) -> None:
        m = OpenAIModel(model="gpt-5.6-sol")
        items = m._convert_messages_responses([Message.user("hi"), Message.system("try again")])
        assert items[1]["role"] == "user"
        assert items[1]["content"].startswith("[System guidance]")
        assert "try again" in items[1]["content"]

    def test_assistant_reconstruction_without_metadata(self) -> None:
        m = OpenAIModel(model="gpt-5.6-sol")
        msg = Message.assistant(
            content="checking",
            tool_calls=[ToolCall(id="call_9", name="get_weather", arguments={"city": "Tokyo"})],
        )
        items = m._convert_messages_responses([msg])
        assert items[0] == {"role": "assistant", "content": "checking"}
        assert items[1]["type"] == "function_call"
        assert items[1]["call_id"] == "call_9"
        assert items[1]["name"] == "get_weather"
        assert items[1]["arguments"] == '{"city": "Tokyo"}'
        # Reconstructed calls must NOT carry an item id — the server would
        # try to pair it with a reasoning item it never received.
        assert "id" not in items[1]

    def test_assistant_tool_calls_without_content(self) -> None:
        m = OpenAIModel(model="gpt-5.6-sol")
        msg = Message.assistant(tool_calls=[ToolCall(id="call_3", name="probe", arguments={})])
        items = m._convert_messages_responses([msg])
        assert items == [
            {"type": "function_call", "call_id": "call_3", "name": "probe", "arguments": "{}"}
        ]

    def test_tool_result_becomes_function_call_output(self) -> None:
        m = OpenAIModel(model="gpt-5.6-sol")
        msg = Message.tool(
            ToolResult(tool_call_id="call_9", name="get_weather", content='{"temp": 21}')
        )
        items = m._convert_messages_responses([msg])
        assert items == [
            {"type": "function_call_output", "call_id": "call_9", "output": '{"temp": 21}'}
        ]

    def test_metadata_items_are_replayed_verbatim(self) -> None:
        m = OpenAIModel(model="gpt-5.6-sol")
        raw = [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "t"},
        ]
        msg = Message(
            role="assistant",
            tool_calls=[ToolCall(id="call_1", name="t", arguments={})],
            metadata={RESPONSES_ITEMS_METADATA_KEY: raw},
        )
        items = m._convert_messages_responses([msg])
        # Verbatim replay — no reconstructed items alongside.
        assert items == raw

    def test_metadata_non_dict_entries_are_filtered(self) -> None:
        m = OpenAIModel(model="gpt-5.6-sol")
        raw = ["bogus", {"type": "function_call", "call_id": "call_1", "name": "t"}]
        msg = Message(
            role="assistant",
            metadata={RESPONSES_ITEMS_METADATA_KEY: raw},
        )
        items = m._convert_messages_responses([msg])
        assert items == [{"type": "function_call", "call_id": "call_1", "name": "t"}]

    def test_metadata_all_non_dict_falls_back_to_reconstruction(self) -> None:
        m = OpenAIModel(model="gpt-5.6-sol")
        msg = Message(
            role="assistant",
            content="hello",
            metadata={RESPONSES_ITEMS_METADATA_KEY: ["bogus"]},
        )
        items = m._convert_messages_responses([msg])
        assert items == [{"role": "assistant", "content": "hello"}]

    def test_metadata_non_list_ignored(self) -> None:
        m = OpenAIModel(model="gpt-5.6-sol")
        msg = Message(
            role="assistant",
            content="hello",
            metadata={RESPONSES_ITEMS_METADATA_KEY: "not-a-list"},
        )
        items = m._convert_messages_responses([msg])
        assert items == [{"role": "assistant", "content": "hello"}]


# ---------------------------------------------------------------------------
# Tool-schema conversion
# ---------------------------------------------------------------------------


class TestConvertToolsResponses:
    def test_none_and_empty(self) -> None:
        m = OpenAIModel(model="gpt-5.6-sol")
        assert m._convert_tools_responses(None) is None
        assert m._convert_tools_responses([]) is None

    def test_bare_schema_is_flattened(self) -> None:
        m = OpenAIModel(model="gpt-5.6-sol")
        out = m._convert_tools_responses(
            [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        )
        assert out == [
            {
                "type": "function",
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object"},
            }
        ]

    def test_chat_wrapped_schema_is_flattened(self) -> None:
        m = OpenAIModel(model="gpt-5.6-sol")
        out = m._convert_tools_responses(
            [
                {
                    "type": "function",
                    "function": {"name": "search", "parameters": {}, "strict": True},
                }
            ]
        )
        assert out == [{"type": "function", "name": "search", "parameters": {}, "strict": True}]

    def test_builtin_tool_passes_through(self) -> None:
        m = OpenAIModel(model="gpt-5.6-sol")
        out = m._convert_tools_responses([{"type": "web_search"}])
        assert out == [{"type": "web_search"}]

    def test_already_flattened_function_passes_through(self) -> None:
        m = OpenAIModel(model="gpt-5.6-sol")
        already = [{"type": "function", "name": "search", "parameters": {}}]
        assert m._convert_tools_responses(already) == already


# ---------------------------------------------------------------------------
# Request shaping
# ---------------------------------------------------------------------------


class TestResponsesRequestShaping:
    @pytest.mark.asyncio
    async def test_reasoning_family_defaults(self) -> None:
        client = _responses_client()
        m = _model_with(client, model="gpt-5.6-sol")
        await m.complete([Message.user("hi")])
        args = client.responses.create.call_args.kwargs
        assert args["max_output_tokens"] == m.config.max_tokens
        assert "max_tokens" not in args
        # Reasoning families reject sampling controls.
        assert "temperature" not in args
        assert "top_p" not in args
        # Stateless by default, with the encrypted-reasoning round trip on.
        assert args["store"] is False
        assert args["include"] == ["reasoning.encrypted_content"]
        # Reasoning stays on server defaults — never forced off.
        assert "reasoning" not in args

    @pytest.mark.asyncio
    async def test_classic_model_gets_sampling_no_include(self) -> None:
        client = _responses_client()
        m = _model_with(client, model="gpt-4o", api="responses")
        await m.complete([Message.user("hi")])
        args = client.responses.create.call_args.kwargs
        assert args["temperature"] == m.config.temperature
        assert args["top_p"] == m.config.top_p
        assert "include" not in args

    @pytest.mark.asyncio
    async def test_null_sampling_config_omitted(self) -> None:
        # ``None`` means "let the server decide" — same as the chat path.
        client = _responses_client()
        m = _model_with(client, model="gpt-4o", api="responses", temperature=None, top_p=None)
        await m.complete([Message.user("hi")])
        args = client.responses.create.call_args.kwargs
        assert "temperature" not in args
        assert "top_p" not in args

    @pytest.mark.asyncio
    async def test_reasoning_dict_alone_forwarded(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        await m.complete([Message.user("hi")], reasoning={"effort": "minimal"})
        assert client.responses.create.call_args.kwargs["reasoning"] == {"effort": "minimal"}

    @pytest.mark.asyncio
    async def test_search_preview_drops_sampling(self) -> None:
        client = _responses_client()
        m = _model_with(client, model="gpt-4o-search-preview", api="responses")
        await m.complete([Message.user("hi")])
        args = client.responses.create.call_args.kwargs
        assert "temperature" not in args
        assert "top_p" not in args

    @pytest.mark.asyncio
    async def test_max_tokens_kwarg_wins(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        await m.complete([Message.user("hi")], max_tokens=512)
        assert client.responses.create.call_args.kwargs["max_output_tokens"] == 512

    @pytest.mark.asyncio
    async def test_max_completion_tokens_kwarg_accepted(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        await m.complete([Message.user("hi")], max_completion_tokens=256)
        assert client.responses.create.call_args.kwargs["max_output_tokens"] == 256

    @pytest.mark.asyncio
    async def test_tools_flattened_into_request(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        await m.complete(
            [Message.user("hi")],
            tools=[{"name": "get_weather", "parameters": {"type": "object"}}],
        )
        tools = client.responses.create.call_args.kwargs["tools"]
        assert tools == [
            {"type": "function", "name": "get_weather", "parameters": {"type": "object"}}
        ]

    @pytest.mark.asyncio
    async def test_reasoning_effort_translated(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        await m.complete([Message.user("hi")], reasoning_effort="high")
        args = client.responses.create.call_args.kwargs
        assert args["reasoning"] == {"effort": "high"}
        assert "reasoning_effort" not in args

    @pytest.mark.asyncio
    async def test_reasoning_dict_merges_effort(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        await m.complete(
            [Message.user("hi")],
            reasoning={"summary": "auto"},
            reasoning_effort="low",
        )
        assert client.responses.create.call_args.kwargs["reasoning"] == {
            "summary": "auto",
            "effort": "low",
        }

    @pytest.mark.asyncio
    async def test_reasoning_dict_effort_wins_over_alias(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        await m.complete(
            [Message.user("hi")],
            reasoning={"effort": "high"},
            reasoning_effort="low",
        )
        assert client.responses.create.call_args.kwargs["reasoning"] == {"effort": "high"}

    @pytest.mark.asyncio
    async def test_non_dict_reasoning_forwarded_as_is(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        marker = object()
        await m.complete([Message.user("hi")], reasoning=marker)
        assert client.responses.create.call_args.kwargs["reasoning"] is marker

    @pytest.mark.asyncio
    async def test_chat_shaped_tool_choice_translated(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        await m.complete(
            [Message.user("hi")],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        assert client.responses.create.call_args.kwargs["tool_choice"] == {
            "type": "function",
            "name": "get_weather",
        }

    @pytest.mark.asyncio
    async def test_string_tool_choice_passes_through(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        await m.complete([Message.user("hi")], tool_choice="required")
        assert client.responses.create.call_args.kwargs["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_response_format_json_schema_becomes_text(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        rf = {
            "type": "json_schema",
            "json_schema": {"name": "Out", "schema": {"type": "object"}, "strict": True},
        }
        await m.complete([Message.user("hi")], response_format=rf)
        assert client.responses.create.call_args.kwargs["text"] == {
            "format": {
                "type": "json_schema",
                "name": "Out",
                "schema": {"type": "object"},
                "strict": True,
            }
        }

    @pytest.mark.asyncio
    async def test_response_format_json_object_carries_over(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        await m.complete([Message.user("hi")], response_format={"type": "json_object"})
        assert client.responses.create.call_args.kwargs["text"] == {
            "format": {"type": "json_object"}
        }

    @pytest.mark.asyncio
    async def test_caller_text_param_wins_over_response_format(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        await m.complete(
            [Message.user("hi")],
            response_format={"type": "json_object"},
            text={"format": {"type": "text"}},
        )
        assert client.responses.create.call_args.kwargs["text"] == {"format": {"type": "text"}}

    @pytest.mark.asyncio
    async def test_store_true_disables_include(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        await m.complete([Message.user("hi")], store=True)
        args = client.responses.create.call_args.kwargs
        assert args["store"] is True
        assert "include" not in args

    @pytest.mark.asyncio
    async def test_caller_include_not_overridden(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        await m.complete([Message.user("hi")], include=["message.output_text.logprobs"])
        assert client.responses.create.call_args.kwargs["include"] == [
            "message.output_text.logprobs"
        ]

    @pytest.mark.asyncio
    async def test_passthrough_forwards_responses_params(self) -> None:
        client = _responses_client()
        m = _model_with(client)
        await m.complete([Message.user("hi")], parallel_tool_calls=False, junk_param=1)
        args = client.responses.create.call_args.kwargs
        assert args["parallel_tool_calls"] is False
        assert "junk_param" not in args

    @pytest.mark.asyncio
    async def test_penalties_not_forwarded(self) -> None:
        # The Responses API has no penalty fields — chat-only names are
        # reserved so they never leak into the request.
        client = _responses_client()
        m = _model_with(client, model="gpt-4o", api="responses")
        await m.complete([Message.user("hi")], frequency_penalty=0.5, presence_penalty=0.5)
        args = client.responses.create.call_args.kwargs
        assert "frequency_penalty" not in args
        assert "presence_penalty" not in args

    @pytest.mark.asyncio
    async def test_seed_and_stop_sequences_dropped(self) -> None:
        # Neither has a Responses equivalent.
        client = _responses_client()
        m = _model_with(client, seed=42, stop_sequences=["END"])
        await m.complete([Message.user("hi")])
        args = client.responses.create.call_args.kwargs
        assert "seed" not in args
        assert "stop" not in args

    @pytest.mark.asyncio
    async def test_extra_body_merged(self) -> None:
        client = _responses_client()
        m = _model_with(client, extra_body={"a": 1})
        await m.complete([Message.user("hi")], extra_body={"b": 2})
        assert client.responses.create.call_args.kwargs["extra_body"] == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


class TestParseResponsesResult:
    @pytest.mark.asyncio
    async def test_text_usage_and_stop(self) -> None:
        client = _responses_client(
            response=_resp(
                output=[_text_item("Hello!")],
                usage=_Obj(input_tokens=12, output_tokens=7),
            )
        )
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.message.content == "Hello!"
        assert out.usage == {"prompt_tokens": 12, "completion_tokens": 7}
        assert out.stop_reason == "stop"
        # A plain text turn reconstructs faithfully — no replay metadata.
        assert RESPONSES_ITEMS_METADATA_KEY not in out.message.metadata

    @pytest.mark.asyncio
    async def test_multiple_text_parts_join(self) -> None:
        item = _Item(
            type="message",
            content=[
                _Obj(type="output_text", text="Hello, "),
                _Obj(type="output_text", text="world."),
            ],
        )
        client = _responses_client(response=_resp(output=[item]))
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.message.content == "Hello, world."

    @pytest.mark.asyncio
    async def test_tool_call_mapped_with_call_id(self) -> None:
        client = _responses_client(response=_resp(output=[_reasoning_item(), _fn_call_item()]))
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert len(out.message.tool_calls) == 1
        tc = out.message.tool_calls[0]
        assert tc.id == "call_1"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "Tokyo"}
        assert out.stop_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_raw_items_stashed_for_replay(self) -> None:
        reasoning = _reasoning_item()
        fn_call = _fn_call_item()
        client = _responses_client(response=_resp(output=[reasoning, fn_call]))
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        stash = out.message.metadata[RESPONSES_ITEMS_METADATA_KEY]
        assert stash == [reasoning.model_dump(), fn_call.model_dump()]
        # And the stash converts back verbatim next turn.
        items = m._convert_messages_responses([out.message])
        assert items == stash

    @pytest.mark.asyncio
    async def test_reasoning_summary_and_content_surface(self) -> None:
        client = _responses_client(
            response=_resp(
                output=[
                    _reasoning_item(summary=("First.",), content=("Raw thought.",)),
                    _text_item("Answer."),
                ]
            )
        )
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.reasoning == "First.\n\nRaw thought."
        assert out.message.content == "Answer."

    @pytest.mark.asyncio
    async def test_refusal_stands_in_for_content(self) -> None:
        item = _Item(type="message", content=[_Obj(type="refusal", refusal="I can't do that.")])
        client = _responses_client(response=_resp(output=[item]))
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.message.content == "I can't do that."

    @pytest.mark.asyncio
    async def test_text_wins_over_refusal(self) -> None:
        item = _Item(
            type="message",
            content=[
                _Obj(type="refusal", refusal="nope"),
                _Obj(type="output_text", text="actually fine"),
            ],
        )
        client = _responses_client(response=_resp(output=[item]))
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.message.content == "actually fine"

    @pytest.mark.asyncio
    async def test_malformed_message_parts_skipped(self) -> None:
        item = _Item(
            type="message",
            content=[
                _Obj(type="unknown_part"),
                _Obj(type="output_text", text=None),
                _Obj(type="refusal", refusal=None),
            ],
        )
        client = _responses_client(response=_resp(output=[item]))
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.message.content is None

    @pytest.mark.asyncio
    async def test_empty_reasoning_blocks_skipped(self) -> None:
        item = _Item(
            type="reasoning",
            summary=[_Obj(type="summary_text", text=""), _Obj(type="summary_text", text=None)],
            content=[],
        )
        client = _responses_client(response=_resp(output=[item, _text_item("done")]))
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.reasoning is None

    @pytest.mark.asyncio
    async def test_incomplete_max_output_tokens_is_length(self) -> None:
        client = _responses_client(
            response=_resp(status="incomplete", incomplete_reason="max_output_tokens")
        )
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.stop_reason == "length"

    @pytest.mark.asyncio
    async def test_incomplete_other_reason_passes_through(self) -> None:
        client = _responses_client(
            response=_resp(status="incomplete", incomplete_reason="content_filter")
        )
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.stop_reason == "content_filter"

    @pytest.mark.asyncio
    async def test_incomplete_without_reason(self) -> None:
        client = _responses_client(response=_resp(status="incomplete"))
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.stop_reason == "incomplete"

    @pytest.mark.asyncio
    async def test_failed_status_passes_through(self) -> None:
        client = _responses_client(response=_resp(status="failed"))
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.stop_reason == "failed"

    @pytest.mark.asyncio
    async def test_missing_status_and_usage(self) -> None:
        client = _responses_client(response=_resp(status=None, usage=None))
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.stop_reason is None
        assert out.usage == {}

    @pytest.mark.asyncio
    async def test_non_int_usage_ignored(self) -> None:
        client = _responses_client(response=_resp(usage=_Obj(input_tokens=None, output_tokens="7")))
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.usage == {}

    @pytest.mark.asyncio
    async def test_empty_output(self) -> None:
        client = _responses_client(response=_resp(output=[]))
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.message.content is None
        assert out.message.tool_calls == []
        assert out.message.metadata == {}

    @pytest.mark.asyncio
    async def test_unknown_item_type_kept_for_replay(self) -> None:
        # Server-side tool items (web_search_call etc.) don't map to
        # ToolCall but must replay so the transcript stays faithful.
        ws = _Item(
            dump={"type": "web_search_call", "id": "ws_1"},
            type="web_search_call",
            id="ws_1",
        )
        client = _responses_client(response=_resp(output=[ws, _text_item("found it")]))
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.message.tool_calls == []
        assert out.message.metadata[RESPONSES_ITEMS_METADATA_KEY] == [
            {"type": "web_search_call", "id": "ws_1"},
            _text_item("found it").model_dump(),
        ]

    @pytest.mark.asyncio
    async def test_undumpable_item_skipped_in_stash_but_still_parsed(self) -> None:
        # An item without model_dump still parses (tool call surfaces);
        # it just isn't replayable.
        fn_call = _Obj(
            type="function_call",
            call_id="call_2",
            name="probe",
            arguments="{}",
        )
        client = _responses_client(response=_resp(output=[fn_call]))
        m = _model_with(client)
        out = await m.complete([Message.user("hi")])
        assert out.message.tool_calls[0].name == "probe"
        assert RESPONSES_ITEMS_METADATA_KEY not in out.message.metadata


class TestDumpOutputItem:
    def test_no_model_dump(self) -> None:
        assert _dump_output_item(_Obj(type="reasoning")) is None

    def test_model_dump_raises(self) -> None:
        class _Bad:
            def model_dump(self, **_: Any) -> dict[str, Any]:
                raise TypeError("unsupported")

        assert _dump_output_item(_Bad()) is None

    def test_model_dump_non_dict(self) -> None:
        class _Odd:
            def model_dump(self, **_: Any) -> Any:
                return ["not", "a", "dict"]

        assert _dump_output_item(_Odd()) is None

    def test_model_dump_dict(self) -> None:
        assert _dump_output_item(_fn_call_item()) == _fn_call_item().model_dump()


class TestTextFormatTranslation:
    def test_json_schema_flattened(self) -> None:
        rf = {
            "type": "json_schema",
            "json_schema": {"name": "X", "schema": {}, "strict": False, "description": "d"},
        }
        assert _text_format_from_response_format(rf) == {
            "format": {
                "type": "json_schema",
                "name": "X",
                "schema": {},
                "strict": False,
                "description": "d",
            }
        }

    def test_json_schema_with_malformed_inner(self) -> None:
        rf = {"type": "json_schema", "json_schema": "oops"}
        assert _text_format_from_response_format(rf) == {"format": rf}

    def test_other_formats_carry_over(self) -> None:
        assert _text_format_from_response_format({"type": "text"}) == {"format": {"type": "text"}}


class TestResponsesParamIntrospection:
    def test_param_names_include_responses_fields(self) -> None:
        import tulip.models.native.openai as mod

        names = mod._responses_param_names()
        assert {"store", "include", "reasoning", "text"} <= names

    def test_param_names_fall_back_when_introspection_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import typing as _typing

        import tulip.models.native.openai as mod

        def boom(*_a: Any, **_k: Any) -> Any:
            raise AttributeError("openai moved its request types")

        monkeypatch.setattr(_typing, "get_type_hints", boom)

        names = mod._responses_param_names()
        assert names == mod._FALLBACK_RESPONSES_PARAMS


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreamResponses:
    @pytest.mark.asyncio
    async def test_content_and_reasoning_deltas(self) -> None:
        events = [
            _Obj(type="response.created", response=None),
            _Obj(type="response.reasoning_summary_text.delta", delta="Think "),
            _Obj(type="response.reasoning_text.delta", delta="harder."),
            _Obj(type="response.output_text.delta", delta="Hello "),
            _Obj(type="response.output_text.delta", delta="world"),
            _Obj(type="response.output_text.delta", delta=""),
            _Obj(
                type="response.completed",
                response=_resp(usage=_Obj(input_tokens=3, output_tokens=2)),
            ),
        ]
        client = _responses_client(stream_events=events)
        m = _model_with(client)
        chunks = [ev async for ev in m.stream([Message.user("hi")])]
        assert [c.content for c in chunks if c.content] == ["Hello ", "world"]
        assert [c.reasoning for c in chunks if c.reasoning] == ["Think ", "harder."]
        done = chunks[-1]
        assert done.done is True
        assert done.usage == {"prompt_tokens": 3, "completion_tokens": 2}
        assert done.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_tool_call_accumulation(self) -> None:
        added = _Obj(
            type="response.output_item.added",
            item=_Obj(
                type="function_call", id="fc_1", call_id="call_1", name="get_weather", arguments=""
            ),
        )
        done_item = _Obj(
            type="response.output_item.done",
            item=_Obj(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="get_weather",
                arguments='{"city": "Tokyo"}',
            ),
        )
        events = [
            added,
            _Obj(type="response.function_call_arguments.delta", item_id="fc_1", delta='{"city"'),
            _Obj(type="response.function_call_arguments.delta", item_id="fc_1", delta=': "Tokyo"}'),
            done_item,
            _Obj(
                type="response.completed",
                response=_resp(usage=_Obj(input_tokens=9, output_tokens=4)),
            ),
        ]
        client = _responses_client(stream_events=events)
        m = _model_with(client)
        chunks = [ev async for ev in m.stream([Message.user("hi")])]
        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1
        tc = tool_chunks[0].tool_calls[0]
        assert (tc.id, tc.name, tc.arguments) == ("call_1", "get_weather", {"city": "Tokyo"})
        assert chunks[-1].stop_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_arguments_from_deltas_when_done_item_lacks_them(self) -> None:
        events = [
            _Obj(
                type="response.output_item.added",
                item=_Obj(
                    type="function_call", id="fc_1", call_id="call_1", name="t", arguments=None
                ),
            ),
            _Obj(type="response.function_call_arguments.delta", item_id="fc_1", delta='{"q": 1}'),
            _Obj(
                type="response.output_item.done",
                item=_Obj(type="function_call", id="fc_1", call_id=None, name=None, arguments=None),
            ),
            _Obj(type="response.completed", response=_resp()),
        ]
        client = _responses_client(stream_events=events)
        m = _model_with(client)
        chunks = [ev async for ev in m.stream([Message.user("hi")])]
        tc = next(c for c in chunks if c.tool_calls).tool_calls[0]
        assert tc.arguments == {"q": 1}

    @pytest.mark.asyncio
    async def test_malformed_streamed_arguments_become_empty(self) -> None:
        events = [
            _Obj(
                type="response.output_item.added",
                item=_Obj(
                    type="function_call", id="fc_1", call_id="c", name="x", arguments="{not json"
                ),
            ),
            _Obj(type="response.completed", response=_resp()),
        ]
        client = _responses_client(stream_events=events)
        m = _model_with(client)
        chunks = [ev async for ev in m.stream([Message.user("hi")])]
        assert next(c for c in chunks if c.tool_calls).tool_calls[0].arguments == {}

    @pytest.mark.asyncio
    async def test_orphan_and_non_function_events_ignored(self) -> None:
        events = [
            # Argument delta for an item never added — ignored.
            _Obj(type="response.function_call_arguments.delta", item_id="ghost", delta="{}"),
            # Non-function item added/done — ignored.
            _Obj(type="response.output_item.added", item=_Obj(type="reasoning", id="rs_1")),
            _Obj(type="response.output_item.done", item=_Obj(type="reasoning", id="rs_1")),
            # Done for a function item never added — ignored.
            _Obj(
                type="response.output_item.done",
                item=_Obj(type="function_call", id="ghost2", call_id="c", name="n", arguments="{}"),
            ),
            _Obj(type="response.completed", response=_resp()),
        ]
        client = _responses_client(stream_events=events)
        m = _model_with(client)
        chunks = [ev async for ev in m.stream([Message.user("hi")])]
        assert not any(c.tool_calls for c in chunks)
        assert chunks[-1].stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_added_item_without_id_gets_fallback_key(self) -> None:
        events = [
            _Obj(
                type="response.output_item.added",
                item=_Obj(
                    type="function_call", id=None, call_id="call_7", name="t", arguments="{}"
                ),
            ),
            _Obj(type="response.completed", response=_resp()),
        ]
        client = _responses_client(stream_events=events)
        m = _model_with(client)
        chunks = [ev async for ev in m.stream([Message.user("hi")])]
        tc = next(c for c in chunks if c.tool_calls).tool_calls[0]
        assert tc.id == "call_7"

    @pytest.mark.asyncio
    async def test_non_string_deltas_ignored(self) -> None:
        events = [
            _Obj(type="response.output_text.delta", delta=None),
            _Obj(type="response.reasoning_summary_text.delta", delta=None),
            _Obj(type="response.completed", response=_resp()),
        ]
        client = _responses_client(stream_events=events)
        m = _model_with(client)
        chunks = [ev async for ev in m.stream([Message.user("hi")])]
        assert not any(c.content for c in chunks)
        assert not any(c.reasoning for c in chunks)

    @pytest.mark.asyncio
    async def test_incomplete_terminal_maps_to_length(self) -> None:
        events = [
            _Obj(type="response.output_text.delta", delta="truncat"),
            _Obj(
                type="response.incomplete",
                response=_resp(status="incomplete", incomplete_reason="max_output_tokens"),
            ),
        ]
        client = _responses_client(stream_events=events)
        m = _model_with(client)
        chunks = [ev async for ev in m.stream([Message.user("hi")])]
        assert chunks[-1].stop_reason == "length"

    @pytest.mark.asyncio
    async def test_no_terminal_event_still_yields_done(self) -> None:
        events = [_Obj(type="response.output_text.delta", delta="hi")]
        client = _responses_client(stream_events=events)
        m = _model_with(client)
        chunks = [ev async for ev in m.stream([Message.user("hi")])]
        done = chunks[-1]
        assert done.done is True
        assert done.usage is None
        assert done.stop_reason is None

    @pytest.mark.asyncio
    async def test_terminal_event_without_response_ignored(self) -> None:
        events = [_Obj(type="response.failed", response=None)]
        client = _responses_client(stream_events=events)
        m = _model_with(client)
        chunks = [ev async for ev in m.stream([Message.user("hi")])]
        assert chunks[-1].stop_reason is None
