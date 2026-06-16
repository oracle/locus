# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for prompt-cache breakpoint helpers in ``locus.models.caching``."""

from __future__ import annotations

from locus.core.messages import Message
from locus.models.caching import (
    CACHE_CONTROL_KEY,
    is_cache_breakpoint,
    mark_cache_breakpoint,
)


class TestMarkCacheBreakpoint:
    def test_returns_new_instance(self) -> None:
        original = Message.system("hello")
        marked = mark_cache_breakpoint(original)
        assert marked is not original
        # Original remains unmodified (frozen).
        assert original.metadata == {}

    def test_metadata_populated(self) -> None:
        marked = mark_cache_breakpoint(Message.system("hello"))
        assert marked.metadata.get(CACHE_CONTROL_KEY) == {"type": "ephemeral"}

    def test_preserves_other_metadata(self) -> None:
        msg = Message(
            role="user",
            content="hi",
            metadata={"user_tag": "v1"},
        )
        marked = mark_cache_breakpoint(msg)
        assert marked.metadata["user_tag"] == "v1"
        assert marked.metadata[CACHE_CONTROL_KEY] == {"type": "ephemeral"}

    def test_preserves_body(self) -> None:
        original = Message.assistant("tool_call_test")
        marked = mark_cache_breakpoint(original)
        assert marked.role == original.role
        assert marked.content == original.content


class TestIsCacheBreakpoint:
    def test_true_for_marked(self) -> None:
        assert is_cache_breakpoint(mark_cache_breakpoint(Message.system("x")))

    def test_false_for_unmarked(self) -> None:
        assert not is_cache_breakpoint(Message.system("x"))

    def test_false_when_metadata_has_other_keys(self) -> None:
        msg = Message(role="user", content="hi", metadata={"other": "thing"})
        assert not is_cache_breakpoint(msg)

    def test_false_when_cache_control_malformed(self) -> None:
        msg = Message(role="user", content="hi", metadata={CACHE_CONTROL_KEY: "bad"})
        assert not is_cache_breakpoint(msg)
