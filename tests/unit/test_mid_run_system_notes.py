# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Mid-run system notes never replace the agent's instructions, on any adapter.

The agent loop appends system-role messages partway through a run — the
iteration-limit notice, grounding and verification reminders, the final-answer
nudge — and places a recalled-memory block right after the instructions. The
Anthropic adapter used to send the *last* system message as ``system``, so the
first such note silently replaced the agent's real instructions; Bedrock
hoisted every note into ``system``, away from the point in the run it refers
to.

Every native adapter is held to the same contract here:

- the leading system messages (instructions, then memory) are the system
  prompt, in order;
- a later system message stays in the conversation at its position, as
  user-role guidance;
- roles alternate where the wire format demands it, and each tool result
  still directly answers the tool call it belongs to.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tulip.core.messages import Message, ToolCall, ToolResult


INSTRUCTIONS = "You are the release agent. Never push to main."
MEMORY = "<memory>The user prefers small PRs.</memory>"
NOTE = "[Verification Reminder] You modified files/data. Verify before completing."


def _tool_turn() -> list[Message]:
    """An assistant turn with two parallel tool calls, then both results."""
    return [
        Message.assistant(
            "Reading, then writing.",
            tool_calls=[
                ToolCall(id="t1", name="read", arguments={"path": "a.py"}),
                ToolCall(id="t2", name="write", arguments={"path": "a.py"}),
            ],
        ),
        Message.tool(ToolResult(tool_call_id="t1", name="read", content="print(1)")),
        Message.tool(ToolResult(tool_call_id="t2", name="write", content="ok")),
    ]


def _note_after_tool_results() -> list[Message]:
    return [
        Message.system(INSTRUCTIONS),
        Message.system(MEMORY),
        Message.user("Fix the bug."),
        *_tool_turn(),
        Message.system(NOTE),
    ]


def _note_after_final_text() -> list[Message]:
    """The final-answer / grounding nudge follows an assistant text turn."""
    return [
        Message.system(INSTRUCTIONS),
        Message.user("Fix the bug."),
        *_tool_turn(),
        Message.assistant("Done."),
        Message.system(NOTE),
    ]


def _note_between_call_and_results() -> list[Message]:
    """A note landing between a tool call and its results must not split them."""
    turn = _tool_turn()
    return [
        Message.system(INSTRUCTIONS),
        Message.user("Fix the bug."),
        turn[0],
        Message.system(NOTE),
        *turn[1:],
    ]


# ------------------------------------------------------------------ anthropic


def _anthropic(*, prompt_cache: bool = False) -> tuple[Any, AsyncMock]:
    pytest.importorskip("anthropic")
    from types import SimpleNamespace  # noqa: PLC0415

    from tulip.models.native.anthropic import AnthropicModel  # noqa: PLC0415

    model = AnthropicModel(model="claude-sonnet-4-6", api_key="sk-test", prompt_cache=prompt_cache)
    client = AsyncMock()
    client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hi")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
        )
    )
    model._client = client
    return model, client


async def _anthropic_request(messages: list[Message], **kw: Any) -> dict[str, Any]:
    model, client = _anthropic(**kw)
    await model.complete(messages)
    sent: dict[str, Any] = client.messages.create.call_args.kwargs
    return sent


def _assert_valid_anthropic_turns(turns: list[dict[str, Any]]) -> None:
    """Roles alternate starting with user; every tool_use is answered next, first."""
    assert turns[0]["role"] == "user"
    for prev, cur in pairwise(turns):
        assert prev["role"] != cur["role"], f"consecutive {cur['role']} turns"
    for i, turn in enumerate(turns):
        if turn["role"] != "assistant" or isinstance(turn["content"], str):
            continue
        uses = [b["id"] for b in turn["content"] if b["type"] == "tool_use"]
        if not uses:
            continue
        nxt = turns[i + 1]["content"]
        assert isinstance(nxt, list)
        leading = [b["tool_use_id"] for b in nxt[: len(uses)] if b["type"] == "tool_result"]
        assert leading == uses


def _anthropic_text(turn: dict[str, Any]) -> list[str]:
    content = turn["content"]
    if isinstance(content, str):
        return [content]
    return [b["text"] for b in content if b["type"] == "text"]


@pytest.mark.asyncio
async def test_anthropic_keeps_instructions_when_a_note_follows_tool_results() -> None:
    sent = await _anthropic_request(_note_after_tool_results())

    assert sent["system"] == f"{INSTRUCTIONS}\n\n{MEMORY}"
    turns = sent["messages"]
    _assert_valid_anthropic_turns(turns)
    last = turns[-1]["content"]
    assert [b["type"] for b in last] == ["tool_result", "tool_result", "text"]
    assert last[-1]["text"] == f"<system-note>\n{NOTE}\n</system-note>"
    assert all(NOTE not in str(t) for t in turns[:-1])


@pytest.mark.asyncio
async def test_anthropic_note_after_final_text_becomes_the_next_user_turn() -> None:
    sent = await _anthropic_request(_note_after_final_text())

    assert sent["system"] == INSTRUCTIONS
    turns = sent["messages"]
    _assert_valid_anthropic_turns(turns)
    assert turns[-2] == {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
    assert turns[-1]["role"] == "user"
    assert _anthropic_text(turns[-1]) == [f"<system-note>\n{NOTE}\n</system-note>"]


@pytest.mark.asyncio
async def test_anthropic_note_never_splits_a_tool_call_from_its_results() -> None:
    sent = await _anthropic_request(_note_between_call_and_results())

    assert sent["system"] == INSTRUCTIONS
    turns = sent["messages"]
    _assert_valid_anthropic_turns(turns)
    assert [b["type"] for b in turns[-1]["content"]] == ["tool_result", "tool_result", "text"]


@pytest.mark.asyncio
async def test_anthropic_note_before_a_user_message_joins_that_turn() -> None:
    sent = await _anthropic_request(
        [
            Message.system(INSTRUCTIONS),
            Message.user("first"),
            Message.assistant("ok"),
            Message.system(NOTE),
            Message.user("second"),
        ]
    )

    turns = sent["messages"]
    _assert_valid_anthropic_turns(turns)
    assert _anthropic_text(turns[-1]) == [f"<system-note>\n{NOTE}\n</system-note>", "second"]


@pytest.mark.asyncio
async def test_anthropic_prompt_cache_marks_the_instructions_and_the_whole_prompt() -> None:
    sent = await _anthropic_request(_note_after_tool_results(), prompt_cache=True)

    assert sent["system"] == [
        {"type": "text", "text": INSTRUCTIONS, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": MEMORY, "cache_control": {"type": "ephemeral"}},
    ]
    assert "cache_control" not in str(sent["messages"])


def test_anthropic_convert_messages_still_returns_the_joined_prompt() -> None:
    model, _ = _anthropic()
    system, turns = model._convert_messages(_note_after_tool_results())
    assert system == f"{INSTRUCTIONS}\n\n{MEMORY}"
    _assert_valid_anthropic_turns(turns)


# -------------------------------------------------------------------- bedrock


def _bedrock_params(messages: list[Message]) -> dict[str, Any]:
    pytest.importorskip("boto3", reason="bedrock extra not installed")
    from tulip.models.native.bedrock import BedrockModel  # noqa: PLC0415

    model = BedrockModel(
        model="us.amazon.nova-micro-v1:0",
        region="us-east-1",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",  # noqa: S106
    )
    params: dict[str, Any] = model._params(messages, None)
    return params


def _assert_valid_converse_turns(turns: list[dict[str, Any]]) -> None:
    assert turns[0]["role"] == "user"
    for prev, cur in pairwise(turns):
        assert prev["role"] != cur["role"], f"consecutive {cur['role']} turns"
    for i, turn in enumerate(turns):
        uses = [b["toolUse"]["toolUseId"] for b in turn["content"] if "toolUse" in b]
        if not uses:
            continue
        nxt = turns[i + 1]["content"]
        assert [b["toolResult"]["toolUseId"] for b in nxt[: len(uses)]] == uses


def test_bedrock_keeps_instructions_when_a_note_follows_tool_results() -> None:
    params = _bedrock_params(_note_after_tool_results())

    assert params["system"] == [{"text": INSTRUCTIONS}, {"text": MEMORY}]
    turns = params["messages"]
    _assert_valid_converse_turns(turns)
    last = turns[-1]["content"]
    assert ["toolResult" in b for b in last] == [True, True, False]
    assert last[-1] == {"text": f"<system-note>\n{NOTE}\n</system-note>"}


def test_bedrock_note_after_final_text_becomes_the_next_user_turn() -> None:
    params = _bedrock_params(_note_after_final_text())

    assert params["system"] == [{"text": INSTRUCTIONS}]
    turns = params["messages"]
    _assert_valid_converse_turns(turns)
    assert turns[-1] == {
        "role": "user",
        "content": [{"text": f"<system-note>\n{NOTE}\n</system-note>"}],
    }


def test_bedrock_note_never_splits_a_tool_call_from_its_results() -> None:
    params = _bedrock_params(_note_between_call_and_results())

    _assert_valid_converse_turns(params["messages"])
    assert params["system"] == [{"text": INSTRUCTIONS}]


# ---------------------------------------------------- openai (and azure/gemini)


def _openai_model(model: str = "gpt-4o") -> Any:
    pytest.importorskip("openai", reason="openai extra not installed")
    from tulip.models.native.openai import OpenAIModel  # noqa: PLC0415

    return OpenAIModel(model=model, api_key="sk-test")


def _assert_tool_messages_follow_their_call(entries: list[dict[str, Any]]) -> None:
    for i, entry in enumerate(entries):
        calls = [c["id"] for c in entry.get("tool_calls") or []]
        if calls:
            answered = [e.get("tool_call_id") for e in entries[i + 1 : i + 1 + len(calls)]]
            assert answered == calls


def test_openai_chat_keeps_one_leading_system_message_and_notes_in_place() -> None:
    entries = _openai_model()._convert_messages(_note_after_tool_results())

    assert entries[0] == {"role": "system", "content": f"{INSTRUCTIONS}\n\n{MEMORY}"}
    assert [e["role"] for e in entries] == ["system", "user", "assistant", "tool", "tool", "user"]
    assert entries[-1]["content"] == f"[System guidance] {NOTE}"
    _assert_tool_messages_follow_their_call(entries)


def test_openai_responses_keeps_one_leading_system_item_and_notes_in_place() -> None:
    items = _openai_model("gpt-5.6-sol")._convert_messages_responses(_note_after_tool_results())

    assert items[0] == {"role": "system", "content": f"{INSTRUCTIONS}\n\n{MEMORY}"}
    assert sum(1 for i in items if i.get("role") == "system") == 1
    assert items[-1] == {"role": "user", "content": f"[System guidance] {NOTE}"}
    outputs = [i["call_id"] for i in items if i.get("type") == "function_call_output"]
    assert outputs == ["t1", "t2"]


def test_azure_inherits_the_same_mapping() -> None:
    pytest.importorskip("openai", reason="openai extra not installed")
    from tulip.models.native.azure import AzureOpenAIModel  # noqa: PLC0415

    model = AzureOpenAIModel(
        model="gpt-4o", endpoint="https://my-resource.openai.azure.com", api_key="k"
    )
    entries = model._convert_messages(_note_after_tool_results())
    assert entries[0]["content"] == f"{INSTRUCTIONS}\n\n{MEMORY}"
    assert entries[-1]["content"] == f"[System guidance] {NOTE}"
