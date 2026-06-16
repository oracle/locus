# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration test for prompt-cache breakpoint helpers (D.2).

Verifies that the cache-control marker survives the round-trip
through Locus's message-handling layer alongside the model metadata
registry — the typical end-to-end path a user adopts.
"""

from __future__ import annotations

from locus.core.messages import Message
from locus.models.caching import (
    CACHE_CONTROL_KEY,
    is_cache_breakpoint,
    mark_cache_breakpoint,
)
from locus.models.metadata import metadata_for


class TestCacheGatedOnMetadata:
    def test_anthropic_model_supports_caching(self) -> None:
        meta = metadata_for("claude-opus-4")
        assert meta is not None
        assert meta.supports_prompt_caching is True

    def test_user_pattern_skips_marker_when_unsupported(self) -> None:
        # Idiomatic user code: only mark when the metadata registry
        # confirms the model supports caching. When it doesn't, the
        # message is left unmarked.
        meta = metadata_for("meta.llama-3.3-70b-instruct")
        assert meta is not None
        assert meta.supports_prompt_caching is False

        msg = Message.system("You are a helpful assistant.")
        if meta.supports_prompt_caching:
            msg = mark_cache_breakpoint(msg)

        assert is_cache_breakpoint(msg) is False

    def test_user_pattern_marks_when_supported(self) -> None:
        meta = metadata_for("claude-opus-4")
        assert meta is not None
        msg = Message.system("You are a helpful assistant.")
        if meta.supports_prompt_caching:
            msg = mark_cache_breakpoint(msg)

        assert is_cache_breakpoint(msg) is True


class TestRoundTripThroughSerialization:
    def test_marker_survives_pydantic_dump_and_validate(self) -> None:
        marked = mark_cache_breakpoint(Message.system("x"))
        blob = marked.model_dump_json()
        restored = Message.model_validate_json(blob)
        assert is_cache_breakpoint(restored) is True
        assert restored.metadata[CACHE_CONTROL_KEY] == {"type": "ephemeral"}


class TestPreservesOtherMessageFields:
    def test_assistant_with_tool_calls(self) -> None:
        from locus.core.messages import ToolCall

        original = Message.assistant(
            content="calling tool",
            tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "hello"})],
        )
        marked = mark_cache_breakpoint(original)
        assert marked.tool_calls == original.tool_calls
        assert marked.content == original.content
        assert marked.role == original.role
        assert is_cache_breakpoint(marked)
