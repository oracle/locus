# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Per-transport guard that N>1 wire-format tool_calls are normalized
into ``ModelResponse.message.tool_calls`` of length N.

Companion to ``tests/integration/test_concurrent_tools_models.py``: the
live matrix proves end-to-end parallelism on transports where the LLM
chose to fan out. Where the LLM declined this file deterministically
pins the transport's normalization independently of live model
behavior, so a regression in any transport's ``parse_response`` is
caught regardless of model whims.

Covers:

* ``OpenAIModel._parse_response`` (the OpenAI-compat wire)
"""

from __future__ import annotations


# =============================================================================
# OpenAIModel — OpenAI-compat wire
# =============================================================================


class _Func:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCallStub:
    def __init__(self, *, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = _Func(name=name, arguments=arguments)


class _MsgStub:
    def __init__(self, *, content: str | None, tool_calls: list[_ToolCallStub]) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, *, message: _MsgStub, finish_reason: str = "tool_calls") -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, *, choices: list[_Choice]) -> None:
        self.choices = choices
        self.usage = None


def test_openai_parse_response_preserves_n_parallel_tool_calls() -> None:
    """OpenAI wire returns 3 ``tool_calls``; normalised list must have length 3
    with ids/names/args intact."""
    from tulip.models.native.openai import OpenAIModel

    m = OpenAIModel()
    resp = _Response(
        choices=[
            _Choice(
                message=_MsgStub(
                    content=None,
                    tool_calls=[
                        _ToolCallStub(call_id="c0", name="lookup", arguments='{"topic": "a"}'),
                        _ToolCallStub(call_id="c1", name="lookup", arguments='{"topic": "b"}'),
                        _ToolCallStub(call_id="c2", name="lookup", arguments='{"topic": "c"}'),
                    ],
                )
            )
        ]
    )

    out = m._parse_response(resp)

    assert len(out.message.tool_calls) == 3
    assert [tc.id for tc in out.message.tool_calls] == ["c0", "c1", "c2"]
    assert {tc.name for tc in out.message.tool_calls} == {"lookup"}
    assert [tc.arguments for tc in out.message.tool_calls] == [
        {"topic": "a"},
        {"topic": "b"},
        {"topic": "c"},
    ]
