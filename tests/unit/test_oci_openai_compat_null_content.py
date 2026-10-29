# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OCI V1 transport: a tool-call-only assistant turn is sent with ``""``.

OpenAI accepts ``"content": null`` on an assistant message that only made
tool calls; OCI's endpoint rejects it (``400 Invalid type for
'messages[n].content': expected an object, but got null``). The history is
replayed on every turn after a tool call, so one such message ends the run.
"""

from __future__ import annotations

import pytest

pytest.importorskip("oci")

from locus.core.messages import Message, Role, ToolCall  # noqa: E402
from locus.models.providers.oci.openai_compat import (  # noqa: E402
    OCIChatCompletionsModel,
    blank_null_assistant_content,
)


def _model(name: str) -> OCIChatCompletionsModel:
    return OCIChatCompletionsModel(
        model=name,
        auth_type="instance_principal",
        compartment_id="ocid1.compartment.oc1..test",
    )


def _tool_call_turn() -> Message:
    return Message.assistant(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="add", arguments={"a": 1})],
    )


class TestBlankNullAssistantContent:
    def test_tool_call_only_turn_gets_empty_text(self) -> None:
        out = blank_null_assistant_content([Message.user("Q"), _tool_call_turn()])
        assert out[1].role is Role.ASSISTANT
        assert out[1].content == ""
        assert out[1].tool_calls and out[1].tool_calls[0].name == "add"

    def test_text_turns_and_other_roles_are_untouched(self) -> None:
        msgs = [
            Message.system("S"),
            Message.user("Q"),
            Message.assistant("A"),
            Message(role=Role.TOOL, content=None, tool_call_id="call_1"),
        ]
        assert blank_null_assistant_content(msgs) is msgs

    def test_input_is_not_mutated(self) -> None:
        turn = _tool_call_turn()
        blank_null_assistant_content([turn])
        assert turn.content is None


class TestPrepareMessagesBlanksForEveryVendor:
    @pytest.mark.parametrize(
        "name",
        ["openai.gpt-4o", "meta.llama-3.3-70b-instruct", "google.gemini-2.5-flash", "xai.grok-4"],
    )
    def test_assistant_null_content_becomes_empty(self, name: str) -> None:
        m = _model(name)
        out = m._prepare_messages_for_oci([Message.user("Q"), _tool_call_turn()], None)
        assistant = next(x for x in out if x.role is Role.ASSISTANT)
        assert assistant.content == ""
