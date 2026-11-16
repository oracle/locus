# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A trace recorded from notebook 60 on a hosted model validates against the conventions.

``fixtures/semconv_trace_notebook60.json`` is what ``TelemetryHook`` emitted,
through the in-memory exporter, for the notebook's ticket reply on an OCI
Generative AI model: one ``invoke_agent`` span, two iterations, a ``chat``
span per model call with the provider's token counts, and the
``execute_tool`` span of the lookup. It is the reference a change to the
hook's names or attributes has to keep valid.
"""

from __future__ import annotations

import pathlib

from locus.observability import semconv as sc


FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "semconv_trace_notebook60.json"


def _spans() -> list[sc.SpanRecord]:
    return sc.load_spans(FIXTURE.read_text())


def test_recorded_trace_follows_the_conventions() -> None:
    spans = _spans()
    assert sc.validate_spans(spans) == []


def test_recorded_trace_shape() -> None:
    spans = _spans()
    by_name: dict[str, list[sc.SpanRecord]] = {}
    for span in spans:
        by_name.setdefault(span.name, []).append(span)

    assert set(by_name) == {
        "invoke_agent support-desk",
        sc.ITERATION_SPAN_NAME,
        "chat openai.gpt-4.1",
        "execute_tool lookup_order",
    }
    (root,) = by_name["invoke_agent support-desk"]
    assert root.parent_span_id is None
    assert root.kind == "INTERNAL"
    assert root.attributes[sc.GEN_AI_AGENT_NAME] == "support-desk"
    assert root.attributes[sc.GEN_AI_REQUEST_MODEL] == "openai.gpt-4.1"

    chats = by_name["chat openai.gpt-4.1"]
    assert len(chats) == 2
    for chat in chats:
        assert chat.kind == "CLIENT"
        assert chat.attributes[sc.GEN_AI_PROVIDER_NAME] == sc.PROVIDER_OCI_GENERATIVE_AI
        assert chat.attributes[sc.GEN_AI_USAGE_INPUT_TOKENS] > 0
        assert chat.attributes[sc.GEN_AI_USAGE_OUTPUT_TOKENS] > 0
    finish = sorted(tuple(c.attributes[sc.GEN_AI_RESPONSE_FINISH_REASONS]) for c in chats)
    assert finish == [("stop",), ("tool_calls",)]

    # The run's totals are the sum of its model calls.
    assert root.attributes[sc.GEN_AI_USAGE_INPUT_TOKENS] == sum(
        c.attributes[sc.GEN_AI_USAGE_INPUT_TOKENS] for c in chats
    )
    assert root.attributes[sc.GEN_AI_USAGE_OUTPUT_TOKENS] == sum(
        c.attributes[sc.GEN_AI_USAGE_OUTPUT_TOKENS] for c in chats
    )

    (tool,) = by_name["execute_tool lookup_order"]
    assert tool.attributes[sc.GEN_AI_TOOL_NAME] == "lookup_order"
    assert tool.attributes[sc.GEN_AI_TOOL_TYPE] == "function"
    assert tool.attributes[sc.GEN_AI_TOOL_DESCRIPTION].startswith("Look up an order")
    assert tool.attributes[sc.GEN_AI_TOOL_CALL_ID]
    assert tool.status == "OK"

    # Every chat and tool span sits under an iteration, every iteration under the root.
    ids = {s.span_id: s for s in spans}
    for span in chats + [tool]:
        parent = ids[span.parent_span_id]
        assert parent.name == sc.ITERATION_SPAN_NAME
        assert ids[parent.parent_span_id] is root
