# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage gap fills for ``locus.hooks.builtin.guardrails.OutputFilterHook``.

The existing ``test_guardrails.py`` covers the happy paths. These tests
hit the remaining branches:

- ``priority`` / ``name`` properties
- ``on_after_model_call`` early-exit on empty content
- ``topic_policy`` rejection path
- ``ContentPolicy`` block-via-keyword path (no regex match in the
  default categories — exercise via custom keywords)
- PII redaction with the ``no PII present`` fast path
"""

from __future__ import annotations

import pytest

from locus.core.messages import Message
from locus.hooks.builtin.guardrails import (
    ContentPolicy,
    OutputFilterHook,
    TopicPolicy,
)
from locus.hooks.provider import AfterModelCallEvent, HookPriority
from locus.models.base import ModelResponse


def _event(content: str) -> AfterModelCallEvent:
    return AfterModelCallEvent(
        response=ModelResponse(message=Message.assistant(content)),
        messages=[],
    )


class TestOutputFilterHookProperties:
    def test_default_priority(self) -> None:
        # OutputFilterHook bumps SECURITY_DEFAULT by +5 so it runs
        # after the GuardrailsHook in the same security band.
        hook = OutputFilterHook()
        assert hook.priority == HookPriority.SECURITY_DEFAULT + 5

    def test_explicit_priority(self) -> None:
        hook = OutputFilterHook(priority=42)
        assert hook.priority == 42

    def test_name(self) -> None:
        assert OutputFilterHook().name == "OutputFilterHook"


class TestEmptyContent:
    @pytest.mark.asyncio
    async def test_empty_content_short_circuits(self) -> None:
        # No content → no checks run, no violations recorded.
        hook = OutputFilterHook(content_policy=ContentPolicy())
        event = _event("")
        await hook.on_after_model_call(event)
        assert hook.violations == []
        # Response is unchanged.
        assert event.response.message.content == ""


class TestTopicPolicy:
    @pytest.mark.asyncio
    async def test_blocks_topic_keyword_match(self) -> None:
        # TopicPolicy blocks any topic whose keyword shows up in the text.
        topic = TopicPolicy(
            blocked_topics={"weapons"},
            keywords={"weapons": ["gun", "rifle"]},
        )
        hook = OutputFilterHook(topic_policy=topic)
        event = _event("Choose your gun carefully.")
        await hook.on_after_model_call(event)
        assert any(v.startswith("topic_policy:") for v in hook.violations)
        # The response is replaced with a topic-block message.
        assert "outside my allowed topics" in event.response.message.content

    @pytest.mark.asyncio
    async def test_topic_policy_passes_when_no_keyword_match(self) -> None:
        topic = TopicPolicy(
            blocked_topics={"weapons"},
            keywords={"weapons": ["gun"]},
        )
        hook = OutputFilterHook(topic_policy=topic)
        event = _event("Quarterly finance report — revenue grew.")
        await hook.on_after_model_call(event)
        assert not any(v.startswith("topic_policy:") for v in hook.violations)


class TestPIIRedactionFastPath:
    @pytest.mark.asyncio
    async def test_no_pii_keeps_response_intact(self) -> None:
        # Output without PII goes through the redactor but the
        # ``redacted != content`` branch is False — no replacement.
        hook = OutputFilterHook(redact_pii=True)
        event = _event("This is a benign sentence with no PII at all.")
        original = event.response.message.content
        await hook.on_after_model_call(event)
        assert event.response.message.content == original
        assert "pii_redacted" not in hook.violations
