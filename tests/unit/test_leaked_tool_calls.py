# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Leaked tool-call markup: a model's own call format, arriving as text.

The failure these guard against: DeepSeek V4 through OpenRouter answered with
``<｜DSML｜tool_calls>…`` in the message body and no structured call, the
loop took the markup for the final answer, and the run ended with the edit
never made. Each format is checked against the shape its publisher (or the
vLLM parser for it) writes, and against the ways it must *not* be read:
markup followed by more text, a call to an unknown tool, arguments the tool
does not declare.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tulip import Agent
from tulip.agent.leaked_tool_calls import (
    LEAKED_TOOL_CALL_FORMATS,
    match_leaked_tool_calls,
    parse_leaked_tool_calls,
    unfinished_leaked_tool_call,
)
from tulip.core.messages import Message
from tulip.models.base import ModelResponse
from tulip.models.profiles import PROFILES_ENV, profile_for
from tulip.testing import ScriptedModel, text, tool_call
from tulip.tools import tool
from tulip.tools.registry import ToolRegistry


ran: list[tuple[str, dict[str, Any]]] = []


@tool
def edit(path: str, old_string: str, new_string: str) -> str:
    """Replace ``old_string`` with ``new_string`` in a file."""
    ran.append(("edit", {"path": path, "old_string": old_string, "new_string": new_string}))
    return "edited"


@tool
def read(path: str, limit: int = 0) -> str:
    """Read a file."""
    ran.append(("read", {"path": path, "limit": limit}))
    return "contents"


@tool
def todo_write(todos: list[dict[str, Any]]) -> str:
    """Replace the todo list."""
    ran.append(("todo_write", {"todos": todos}))
    return "ok"


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    ran.clear()
    monkeypatch.delenv(PROFILES_ENV, raising=False)


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_many([edit, read, todo_write])
    return reg


def _calls(body: str, registry: ToolRegistry, *formats: str) -> list[tuple[str, dict[str, Any]]]:
    calls, _ = parse_leaked_tool_calls(body, registry, formats or LEAKED_TOOL_CALL_FORMATS)
    return [(c.name, c.arguments) for c in calls]


#: The reel-cache-generation reply, completed: DeepSeek V4's DSML with the
#: full-width bar (U+FF5C) and no structured call beside it.
DSML_EDIT = (
    "<｜DSML｜tool_calls>\n"
    '<｜DSML｜invoke name="edit">\n'
    '<｜DSML｜parameter name="path" string="true">server/seg_cache.py</｜DSML｜parameter>\n'
    '<｜DSML｜parameter name="old_string" string="true">def segment_path(key):\n'
    "    return CACHE / key</｜DSML｜parameter>\n"
    '<｜DSML｜parameter name="new_string" string="true">def segment_path(key: str) -> Path:\n'
    '    return CACHE / f"{key}.ts"</｜DSML｜parameter>\n'
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls>"
)

EDIT_ARGS = {
    "path": "server/seg_cache.py",
    "old_string": "def segment_path(key):\n    return CACHE / key",
    "new_string": 'def segment_path(key: str) -> Path:\n    return CACHE / f"{key}.ts"',
}


class TestDsml:
    def test_the_reported_reply(self, registry: ToolRegistry) -> None:
        assert _calls(DSML_EDIT, registry, "dsml") == [("edit", EDIT_ARGS)]

    def test_v32_function_calls_wrapper_and_several_invokes(self, registry: ToolRegistry) -> None:
        body = (
            "<｜DSML｜function_calls>\n"
            '<｜DSML｜invoke name="read">\n'
            '<｜DSML｜parameter name="path" string="true">a.py</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="limit" string="false">40</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            '<｜DSML｜invoke name="todo_write">\n'
            '<｜DSML｜parameter name="todos" string="false">[{"content": "x"}]</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜function_calls><｜end▁of▁sentence｜>"
        )
        assert _calls(body, registry, "dsml") == [
            ("read", {"path": "a.py", "limit": 40}),
            ("todo_write", {"todos": [{"content": "x"}]}),
        ]

    def test_ascii_bars(self, registry: ToolRegistry) -> None:
        body = DSML_EDIT.replace("｜", "|")
        assert _calls(body, registry, "dsml") == [("edit", EDIT_ARGS)]

    def test_leading_prose_is_kept_apart(self, registry: ToolRegistry) -> None:
        body = "Now I'll make the edit.\n\n" + DSML_EDIT
        calls, prose = parse_leaked_tool_calls(body, registry, ["dsml"])
        assert [c.name for c in calls] == ["edit"]
        assert prose == "Now I'll make the edit."

    def test_cut_off_after_the_last_parameter(self, registry: ToolRegistry) -> None:
        body = DSML_EDIT.removesuffix("</｜DSML｜invoke>\n</｜DSML｜tool_calls>")
        assert _calls(body, registry, "dsml") == [("edit", EDIT_ARGS)]
        body = DSML_EDIT.removesuffix("</｜DSML｜tool_calls>")
        assert _calls(body, registry, "dsml") == [("edit", EDIT_ARGS)]

    def test_an_unclosed_parameter_is_not_a_call(self, registry: ToolRegistry) -> None:
        body = DSML_EDIT.split(
            '</｜DSML｜parameter>\n<｜DSML｜parameter name="new_string"', maxsplit=1
        )[0]
        body += '\n<｜DSML｜parameter name="new_string" string="true">def segment_pa'
        assert _calls(body, registry, "dsml") == []

    def test_prose_after_the_block_is_not_a_call(self, registry: ToolRegistry) -> None:
        assert _calls(DSML_EDIT + "\n\nThat should fix it.", registry, "dsml") == []

    def test_unknown_tool(self, registry: ToolRegistry) -> None:
        assert _calls(DSML_EDIT.replace('name="edit"', 'name="rm_rf"'), registry, "dsml") == []

    def test_undeclared_argument(self, registry: ToolRegistry) -> None:
        body = DSML_EDIT.replace('name="path"', 'name="file"')
        assert _calls(body, registry, "dsml") == []

    def test_missing_required_argument(self, registry: ToolRegistry) -> None:
        body = DSML_EDIT.replace(
            '<｜DSML｜parameter name="path" string="true">server/seg_cache.py</｜DSML｜parameter>\n',
            "",
        )
        assert _calls(body, registry, "dsml") == []

    def test_string_false_must_be_json(self, registry: ToolRegistry) -> None:
        body = DSML_EDIT.replace(
            '<｜DSML｜parameter name="path" string="true">',
            '<｜DSML｜parameter name="path" string="false">',
        )
        assert _calls(body, registry, "dsml") == []

    def test_duplicate_parameter(self, registry: ToolRegistry) -> None:
        dup = '<｜DSML｜parameter name="path" string="true">b.py</｜DSML｜parameter>\n'
        body = DSML_EDIT.replace("</｜DSML｜invoke>", dup + "</｜DSML｜invoke>")
        assert _calls(body, registry, "dsml") == []

    def test_one_bad_call_rejects_the_block(self, registry: ToolRegistry) -> None:
        bad = '<｜DSML｜invoke name="rm_rf">\n</｜DSML｜invoke>\n'
        body = DSML_EDIT.replace("</｜DSML｜tool_calls>", bad + "</｜DSML｜tool_calls>")
        assert _calls(body, registry, "dsml") == []

    @pytest.mark.parametrize(
        "body",
        [
            "<｜DSML｜tool_calls>\n</｜DSML｜tool_calls>",
            "<｜DSML｜tool_calls>\nnot an invoke",
            '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="read">\nstray text',
        ],
    )
    def test_malformed_blocks(self, registry: ToolRegistry, body: str) -> None:
        assert _calls(body, registry, "dsml") == []


class TestDeepseekLegacy:
    def test_v3_function_and_json_fence(self, registry: ToolRegistry) -> None:
        body = (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>read\n"
            '```json\n{"path": "a.py"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
            "<｜end▁of▁sentence｜>"
        )
        assert _calls(body, registry, "deepseek") == [("read", {"path": "a.py"})]

    def test_v31_name_and_json(self, registry: ToolRegistry) -> None:
        body = (
            "Reading both.<｜tool▁calls▁begin｜>"
            '<｜tool▁call▁begin｜>read<｜tool▁sep｜>{"path": "a.py"}<｜tool▁call▁end｜>'
            '<｜tool▁call▁begin｜>read<｜tool▁sep｜>{"path": "b.py"}<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
        )
        found = match_leaked_tool_calls(body, registry, ["deepseek"])
        assert found is not None
        assert found.format == "deepseek"
        assert found.prose == "Reading both."
        assert [c.arguments["path"] for c in found.calls] == ["a.py", "b.py"]

    def test_cut_off_before_the_section_end(self, registry: ToolRegistry) -> None:
        body = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>read<｜tool▁sep｜>{"path": "a.py"}<｜tool▁call▁end｜>'
        assert _calls(body, registry, "deepseek") == [("read", {"path": "a.py"})]

    def test_bad_json_rejects_the_block(self, registry: ToolRegistry) -> None:
        body = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>read<｜tool▁sep｜>{path: a.py}<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        assert _calls(body, registry, "deepseek") == []

    def test_no_call_in_the_section(self, registry: ToolRegistry) -> None:
        assert _calls("<｜tool▁calls▁begin｜><｜tool▁calls▁end｜>", registry, "deepseek") == []


class TestHermes:
    def test_one_and_several_tags(self, registry: ToolRegistry) -> None:
        one = '<tool_call>\n{"name": "read", "arguments": {"path": "a.py"}}\n</tool_call>'
        assert _calls(one, registry, "hermes") == [("read", {"path": "a.py"})]
        two = one + "\n" + one.replace("a.py", "b.py")
        assert [a["path"] for _, a in _calls(two, registry, "hermes")] == ["a.py", "b.py"]

    def test_double_encoded_arguments(self, registry: ToolRegistry) -> None:
        body = '<tool_call>{"name": "read", "arguments": "{\\"path\\": \\"a.py\\"}"}</tool_call>'
        assert _calls(body, registry, "hermes") == [("read", {"path": "a.py"})]

    @pytest.mark.parametrize(
        "inner",
        [
            "not json",
            '["read"]',
            '{"name": "read", "arguments": [1]}',
            '{"name": "read", "arguments": "not json"}',
        ],
    )
    def test_not_a_call(self, registry: ToolRegistry, inner: str) -> None:
        assert _calls(f"<tool_call>{inner}</tool_call>", registry, "hermes") == []

    def test_a_tag_mentioned_in_prose_is_not_a_call(self, registry: ToolRegistry) -> None:
        body = (
            'Qwen writes <tool_call>{"name": "read", "arguments": {"path": "a.py"}}'
            "</tool_call> when it calls a tool."
        )
        assert _calls(body, registry) == []


class TestQwenXml:
    def test_values_follow_the_schema(self, registry: ToolRegistry) -> None:
        body = (
            "<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n"
            "<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>"
        )
        assert _calls(body, registry, "qwen_xml") == [("read", {"path": "a.py", "limit": 20})]

    def test_a_string_value_stays_text(self, registry: ToolRegistry) -> None:
        body = (
            "<tool_call>\n<function=read>\n<parameter=path>\n42\n</parameter>\n"
            "<parameter=limit>\nmany\n</parameter>\n</function>\n</tool_call>"
        )
        # "42" is a path (declared string); "many" does not decode, so the
        # tool's own validation decides.
        assert _calls(body, registry, "qwen_xml") == [("read", {"path": "42", "limit": "many"})]

    def test_stray_text_inside_the_function(self, registry: ToolRegistry) -> None:
        body = (
            "<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n"
            "and more\n</function>\n</tool_call>"
        )
        assert _calls(body, registry, "qwen_xml") == []

    def test_not_a_function(self, registry: ToolRegistry) -> None:
        assert _calls("<tool_call>read a.py</tool_call>", registry, "qwen_xml") == []

    def test_qwen_family_reads_both_its_shapes(self, registry: ToolRegistry) -> None:
        formats = profile_for("vllm:Qwen/Qwen3-Coder-30B").leaked_tool_call_formats
        xml = "<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>"
        hermes = '<tool_call>{"name": "read", "arguments": {"path": "a.py"}}</tool_call>'
        assert _calls(xml, registry, *formats) == [("read", {"path": "a.py"})]
        assert _calls(hermes, registry, *formats) == [("read", {"path": "a.py"})]


class TestKimi:
    def test_section(self, registry: ToolRegistry) -> None:
        body = (
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.read:0"
            '<|tool_call_argument_begin|>{"path": "a.py"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        assert _calls(body, registry, "kimi") == [("read", {"path": "a.py"})]

    def test_singular_section_and_unclosed(self, registry: ToolRegistry) -> None:
        body = (
            "<|tool_call_section_begin|><|tool_call_begin|>functions.read:3"
            '<|tool_call_argument_begin|>{"path": "a.py"}<|tool_call_end|>'
        )
        assert _calls(body, registry, "kimi") == [("read", {"path": "a.py"})]

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            ('"{\\"path\\": \\"a.py\\"}"', [("read", {"path": "a.py"})]),
            ('"not json"', []),
            ("[1, 2]", []),
        ],
    )
    def test_arguments_must_be_an_object(
        self, registry: ToolRegistry, arguments: str, expected: list[Any]
    ) -> None:
        body = (
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.read:0"
            f"<|tool_call_argument_begin|>{arguments}<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        assert _calls(body, registry, "kimi") == expected

    def test_empty_section(self, registry: ToolRegistry) -> None:
        body = "<|tool_calls_section_begin|><|tool_calls_section_end|>"
        assert _calls(body, registry, "kimi") == []


class TestGlm:
    def test_pairs_follow_the_schema(self, registry: ToolRegistry) -> None:
        body = (
            "<tool_call>read\n<arg_key>path</arg_key>\n<arg_value>a.py</arg_value>\n"
            "<arg_key>limit</arg_key>\n<arg_value>5</arg_value>\n</tool_call>"
        )
        assert _calls(body, registry, "glm") == [("read", {"path": "a.py", "limit": 5})]

    def test_stray_text_between_pairs(self, registry: ToolRegistry) -> None:
        body = (
            "<tool_call>read\n<arg_key>path</arg_key><arg_value>a.py</arg_value>\noops</tool_call>"
        )
        assert _calls(body, registry, "glm") == []

    def test_no_name(self, registry: ToolRegistry) -> None:
        assert _calls("<tool_call><arg_key>path</arg_key></tool_call>", registry, "glm") == []


class TestGuards:
    def test_no_text_or_no_tools(self, registry: ToolRegistry) -> None:
        assert parse_leaked_tool_calls(None, registry, ["dsml"]) == ([], None)
        assert parse_leaked_tool_calls(DSML_EDIT, None, ["dsml"]) == ([], None)
        assert parse_leaked_tool_calls(DSML_EDIT, ToolRegistry(), ["dsml"]) == ([], None)

    def test_only_the_named_formats(self, registry: ToolRegistry) -> None:
        assert _calls(DSML_EDIT, registry, "hermes", "kimi") == []

    def test_unknown_format_names_are_skipped(self, registry: ToolRegistry) -> None:
        assert _calls(DSML_EDIT, registry, "from-the-future", "dsml") == [("edit", EDIT_ARGS)]

    def test_names_resolve_like_the_text_parser(self, registry: ToolRegistry) -> None:
        body = DSML_EDIT.replace('name="edit"', 'name="Todo-Write"').split("<｜DSML｜parameter")[0]
        body += '<｜DSML｜parameter name="todos" string="false">[]</｜DSML｜parameter>\n'
        assert _calls(body, registry, "dsml") == [("todo_write", {"todos": []})]

    def test_prose_only(self, registry: ToolRegistry) -> None:
        assert _calls("I edited server/seg_cache.py.", registry) == []


class TestProfiles:
    @pytest.mark.parametrize(
        ("model", "family", "formats"),
        [
            ("openrouter:deepseek/deepseek-v4-pro", "deepseek", ("dsml", "deepseek")),
            ("vllm:Qwen/Qwen3-Coder-30B", "qwen", ("hermes", "qwen_xml")),
            ("openrouter:moonshotai/kimi-k2", "kimi", ("kimi",)),
            ("openrouter:z-ai/glm-4.6", "glm", ("glm",)),
            ("anthropic:claude-sonnet-5-5", "claude", ()),
            ("openai:gpt-5.5", "gpt", ()),
        ],
    )
    def test_family_defaults(self, model: str, family: str, formats: tuple[str, ...]) -> None:
        profile = profile_for(model)
        assert profile.family == family
        assert profile.leaked_tool_call_formats == formats

    def test_an_override_file_sets_them(self, tmp_path: Path) -> None:
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps({"my-finetune": {"leaked_tool_call_formats": ["hermes"]}}))
        assert profile_for("vllm:my-finetune", overrides=path).leaked_tool_call_formats == (
            "hermes",
        )

    def test_an_unknown_format_in_an_override_is_refused(self) -> None:
        with pytest.raises(ValueError, match="leaked_tool_call_formats"):
            profile_for("x", overrides={"x": {"leaked_tool_call_formats": ["morse"]}})


def _deepseek_model(*turns: Any) -> ScriptedModel:
    model = ScriptedModel(list(turns))
    model.model = "deepseek/deepseek-v4-pro"  # type: ignore[attr-defined]
    return model


class TestLoop:
    @pytest.mark.asyncio
    async def test_the_reported_reply_runs_the_edit(self) -> None:
        model = _deepseek_model(text(DSML_EDIT), text("Done: the cache path is typed."))
        agent = Agent(model=model, tools=[edit, read])

        result = await agent.arun("type the cache path")

        assert ran == [("edit", EDIT_ARGS)]
        assert result.text == "Done: the cache path is typed."
        assert model.call_count == 2
        # The model sees a structured call and its result, not its own markup.
        assistant = [m for m in model.received_messages[1] if m.role.value == "assistant"]
        assert assistant[-1].tool_calls[0].name == "edit"
        assert not assistant[-1].content

    @pytest.mark.asyncio
    async def test_leading_prose_stays_in_the_conversation(self) -> None:
        model = _deepseek_model(text("Making the edit.\n" + DSML_EDIT), text("Done."))
        agent = Agent(model=model, tools=[edit])

        await agent.arun("type the cache path")

        assistant = [m for m in model.received_messages[1] if m.role.value == "assistant"]
        assert assistant[-1].content == "Making the edit."

    @pytest.mark.asyncio
    async def test_a_model_without_leaked_formats_ends_on_it(self) -> None:
        model = ScriptedModel([text(DSML_EDIT)])
        model.model = "openai:gpt-5.5"  # type: ignore[attr-defined]
        agent = Agent(model=model, tools=[edit])

        result = await agent.arun("type the cache path")

        assert ran == []
        assert result.text == DSML_EDIT

    @pytest.mark.asyncio
    async def test_the_config_names_formats_for_any_model(self) -> None:
        model = ScriptedModel([text(DSML_EDIT), text("Done.")])
        agent = Agent(model=model, tools=[edit], leaked_tool_call_formats=["dsml"])

        await agent.arun("type the cache path")

        assert ran == [("edit", EDIT_ARGS)]

    @pytest.mark.parametrize(
        "kwargs", [{"leaked_tool_call_formats": []}, {"text_tool_calls": "off"}]
    )
    @pytest.mark.asyncio
    async def test_off(self, kwargs: dict[str, Any]) -> None:
        model = _deepseek_model(text(DSML_EDIT))
        agent = Agent(model=model, tools=[edit], **kwargs)

        await agent.arun("type the cache path")

        assert ran == []

    @pytest.mark.asyncio
    async def test_an_unreadable_profile_turns_recovery_off(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        broken = tmp_path / "profiles.json"
        broken.write_text("not json")
        monkeypatch.setenv(PROFILES_ENV, str(broken))
        model = _deepseek_model(text(DSML_EDIT))
        agent = Agent(model=model, tools=[edit])

        result = await agent.arun("type the cache path")

        assert ran == []
        assert result.text == DSML_EDIT

    @pytest.mark.asyncio
    async def test_a_resumed_turn_recovers_too(self) -> None:
        from tulip.core.messages import Message
        from tulip.core.state import AgentState

        model = _deepseek_model(text(DSML_EDIT), text("Done."))
        agent = Agent(model=model, tools=[edit])
        agent._initialize()
        state = AgentState(messages=(Message.user("type the cache path"),))

        events = [e async for e in agent._run_from_state(state, "type the cache path", None, None)]

        assert ran == [("edit", EDIT_ARGS)]
        assert type(events[-1]).__name__ == "TerminateEvent"


#: DSML_EDIT cut off by the output-token limit inside its last value.
DSML_CUT = DSML_EDIT.split("    return CACHE / f", maxsplit=1)[0]

#: The benchmark reply (gw-run-summary): two reads, the second with typed
#: ``string="false"`` values.
DSML_READS = (
    "<｜DSML｜tool_calls>\n"
    '<｜DSML｜invoke name="read">\n'
    '<｜DSML｜parameter name="path" string="true">pyproject.toml</｜DSML｜parameter>\n'
    "</｜DSML｜invoke>\n"
    '<｜DSML｜invoke name="read">\n'
    '<｜DSML｜parameter name="path" string="true">src/run.py</｜DSML｜parameter>\n'
    '<｜DSML｜parameter name="limit" string="false">50</｜DSML｜parameter>\n'
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls>"
)
READS = [
    ("read", {"path": "pyproject.toml", "limit": 0}),
    ("read", {"path": "src/run.py", "limit": 50}),
]


def _empty(reasoning: str | None = None) -> ModelResponse:
    """A reply with no body and no call, its reasoning in the separate channel."""
    return ModelResponse(
        message=Message.assistant(content=""),
        usage={"prompt_tokens": 1, "completion_tokens": 1},
        stop_reason="stop",
        reasoning=reasoning,
    )


def _notes(messages: list[Message]) -> list[str]:
    """The loop's own notes: system messages and automated user-role ones."""
    return [m.content or "" for m in messages if m.role.value in ("system", "user")]


class TestUnfinished:
    def test_a_block_cut_off_inside_a_value(self) -> None:
        found = unfinished_leaked_tool_call("Writing it now.\n" + DSML_CUT, ["dsml"])
        assert found is not None
        assert (found.format, found.prose) == ("dsml", "Writing it now.")

    def test_a_closed_block_is_finished(self) -> None:
        bad = DSML_EDIT.replace('name="edit"', 'name="nope"')
        assert unfinished_leaked_tool_call(bad, ["dsml"]) is None

    def test_the_last_block_counts(self) -> None:
        assert unfinished_leaked_tool_call(DSML_EDIT + "\n" + DSML_CUT, ["dsml"]) is not None

    @pytest.mark.parametrize(
        ("body", "fmt"),
        [
            ("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>read<｜tool▁sep｜>{", "deepseek"),
            ("<|tool_calls_section_begin|><|tool_call_begin|>functions.read:0", "kimi"),
            ('Reading.\n<tool_call>{"name": "read", "argu', "hermes"),
        ],
    )
    def test_other_formats(self, body: str, fmt: str) -> None:
        found = unfinished_leaked_tool_call(body, [fmt])
        assert found is not None
        assert found.format == fmt

    def test_a_tag_quoted_in_a_sentence_is_not_a_call(self) -> None:
        body = "Qwen writes calls as <tool_call> followed by JSON."
        assert unfinished_leaked_tool_call(body, ["hermes", "qwen_xml", "glm"]) is None

    def test_guards(self) -> None:
        assert unfinished_leaked_tool_call(None, ["dsml"]) is None
        assert unfinished_leaked_tool_call(DSML_CUT, ["hermes", "from-the-future"]) is None
        assert unfinished_leaked_tool_call("Done.", ["dsml"]) is None


class TestLoopAroundEmptyReplies:
    """Where the benchmark runs lost their calls: replies that looked empty.

    DeepSeek V4 through a router left its call in the reasoning channel, so
    the reply had no body and was sent back; the next one too, and the loop
    asked for a final answer with the tools taken away. With nothing to call,
    the model wrote its next read as DSML, and that was the run's answer —
    after a completion check sent it back three times, three more.
    """

    @pytest.mark.asyncio
    async def test_a_call_ending_the_reasoning_of_an_empty_reply_runs(self) -> None:
        model = _deepseek_model(_empty("I need the build file.\n" + DSML_READS), text("Done."))
        agent = Agent(model=model, tools=[edit, read])

        result = await agent.arun("what builds this?")

        assert ran == READS
        assert result.text == "Done."
        assistant = [m for m in model.received_messages[1] if m.role.value == "assistant"]
        assert [c.name for c in assistant[-1].tool_calls] == ["read", "read"]
        assert not assistant[-1].content

    @pytest.mark.asyncio
    async def test_reasoning_that_only_mentions_a_call_is_no_call(self) -> None:
        model = _deepseek_model(_empty(DSML_READS + "\nThen I will answer."), text("Done."))
        agent = Agent(model=model, tools=[read])

        result = await agent.arun("what builds this?")

        assert ran == []
        assert result.text == "Done."

    @pytest.mark.asyncio
    async def test_a_call_answering_the_final_answer_request_runs(self) -> None:
        model = _deepseek_model(
            tool_call("read", path="README.md"),
            _empty(),
            _empty(),
            text(DSML_READS),
            text("Done."),
        )
        agent = Agent(model=model, tools=[edit, read])

        result = await agent.arun("what builds this?")

        # The fourth call is the no-tools request for an answer; its reply
        # is the next tool step, and the run goes on with the tools.
        assert model.offered_tools[3] == []
        assert model.offered_tools[4] == ["edit", "read"]
        assert ran == [("read", {"path": "README.md", "limit": 0}), *READS]
        assert result.text == "Done."
        assert not any("[Final answer requested]" in n for n in _notes(model.received_messages[4]))

    @pytest.mark.asyncio
    async def test_an_answer_to_the_final_answer_request_still_ends_the_run(self) -> None:
        model = _deepseek_model(
            tool_call("read", path="README.md"), _empty(), _empty(), text("It is hatch.")
        )
        agent = Agent(model=model, tools=[read])

        result = await agent.arun("what builds this?")

        assert result.text == "It is hatch."
        assert model.call_count == 4

    @pytest.mark.asyncio
    async def test_a_failed_final_answer_request_falls_back(self) -> None:
        model = _deepseek_model(tool_call("read", path="README.md"), _empty(), _empty())
        agent = Agent(model=model, tools=[read])

        result = await agent.arun("what builds this?")

        # The script has run out, so the no-tools request raises.
        assert result.text
        assert "DSML" not in result.text

    @pytest.mark.asyncio
    async def test_a_cut_off_call_answering_the_final_answer_request_is_sent_back(self) -> None:
        model = _deepseek_model(
            tool_call("read", path="README.md"),
            _empty(),
            _empty(),
            text(DSML_CUT, stop_reason="length"),
            text(DSML_EDIT),
            text("Done."),
        )
        agent = Agent(model=model, tools=[edit, read])

        result = await agent.arun("type the cache path")

        assert ran[-1] == ("edit", EDIT_ARGS)
        assert result.text == "Done."
        notes = _notes(model.received_messages[4])
        assert any("[Unfinished tool call" in n for n in notes)
        assert not any("[Final answer requested]" in n for n in notes)


class TestLoopAroundCutOffCalls:
    @pytest.mark.asyncio
    async def test_a_cut_off_call_is_asked_for_again(self) -> None:
        model = _deepseek_model(
            text("Typing it.\n" + DSML_CUT, stop_reason="length"), text(DSML_EDIT), text("Done.")
        )
        agent = Agent(model=model, tools=[edit])

        result = await agent.arun("type the cache path")

        assert ran == [("edit", EDIT_ARGS)]
        assert result.text == "Done."
        second = model.received_messages[1]
        assert any("[Unfinished tool call" in n for n in _notes(second))
        # The model keeps its prose, not half a call in its own markup.
        assistant = [m for m in second if m.role.value == "assistant"]
        assert assistant[-1].content == "Typing it."
        assert not any("DSML" in (m.content or "") for m in second)

    @pytest.mark.asyncio
    async def test_a_cut_off_call_with_no_prose_leaves_no_message(self) -> None:
        model = _deepseek_model(text(DSML_CUT, stop_reason="length"), text("Done."))
        agent = Agent(model=model, tools=[edit])

        await agent.arun("type the cache path")

        assert not [m for m in model.received_messages[1] if m.role.value == "assistant"]

    @pytest.mark.asyncio
    async def test_the_sends_are_capped(self) -> None:
        model = _deepseek_model(text(DSML_CUT, stop_reason="length"))
        model._repeat_last = True
        agent = Agent(model=model, tools=[edit])

        result = await agent.arun("type the cache path")

        assert model.call_count == 3
        assert ran == []
        assert result.text == DSML_CUT

    @pytest.mark.asyncio
    async def test_the_cap_counts_since_the_last_call(self) -> None:
        model = _deepseek_model(
            text(DSML_CUT),
            text(DSML_CUT),
            text(DSML_EDIT),
            text(DSML_CUT),
            text(DSML_EDIT),
            text("Done."),
        )
        agent = Agent(model=model, tools=[edit])

        result = await agent.arun("type the cache path")

        assert ran == [("edit", EDIT_ARGS), ("edit", EDIT_ARGS)]
        assert result.text == "Done."

    @pytest.mark.asyncio
    async def test_off_means_off(self) -> None:
        model = _deepseek_model(text(DSML_CUT))
        agent = Agent(model=model, tools=[edit], text_tool_calls="off")

        result = await agent.arun("type the cache path")

        assert model.call_count == 1
        assert result.text == DSML_CUT

    @pytest.mark.asyncio
    async def test_a_resumed_turn_sends_it_back_too(self) -> None:
        from tulip.core.state import AgentState

        model = _deepseek_model(text(DSML_CUT), text(DSML_EDIT), text("Done."))
        agent = Agent(model=model, tools=[edit])
        agent._initialize()
        state = AgentState(messages=(Message.user("type the cache path"),))

        events = [e async for e in agent._run_from_state(state, "type the cache path", None, None)]

        assert ran == [("edit", EDIT_ARGS)]
        assert getattr(events[-1], "final_message", None) == "Done."

    @pytest.mark.asyncio
    async def test_a_call_is_no_iteration_limit_summary(self) -> None:
        model = _deepseek_model(tool_call("read", path="a.py"), text(DSML_READS))
        agent = Agent(model=model, tools=[read], max_iterations=1)

        result = await agent.arun("what builds this?")

        assert "DSML" not in (result.text or "")

    @pytest.mark.asyncio
    async def test_with_recovery_off_the_iteration_limit_summary_is_kept(self) -> None:
        model = _deepseek_model(tool_call("read", path="a.py"), text(DSML_READS))
        agent = Agent(model=model, tools=[read], max_iterations=1, text_tool_calls="off")

        result = await agent.arun("what builds this?")

        assert result.text == DSML_READS
