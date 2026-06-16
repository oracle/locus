# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for ``locus.tools.result_storage.ToolResultStore``."""

from __future__ import annotations

import pytest

from locus.core.messages import ToolResult
from locus.tools.result_storage import (
    REFERENCE_MARKER,
    ToolResultStore,
    extract_reference_key,
)


def _result(content: str, *, name: str = "fetch") -> ToolResult:
    return ToolResult(tool_call_id="call-1", name=name, content=content)


@pytest.fixture
def backing_store() -> dict[str, str]:
    return {}


@pytest.fixture
def store(backing_store: dict[str, str]) -> ToolResultStore:
    return ToolResultStore(
        save=lambda k, v: backing_store.__setitem__(k, v),
        load=backing_store.get,
        threshold_chars=100,
        preview_chars=20,
    )


# ---------------------------------------------------------------------------
# Under / over threshold.
# ---------------------------------------------------------------------------


class TestOffloadBehaviour:
    def test_small_result_passes_through(
        self, store: ToolResultStore, backing_store: dict[str, str]
    ) -> None:
        original = _result("hello")
        out = store.maybe_offload(original, run_id="r1", iteration=0)
        assert out is original
        assert backing_store == {}

    def test_large_result_offloaded(
        self, store: ToolResultStore, backing_store: dict[str, str]
    ) -> None:
        big = "x" * 500
        original = _result(big, name="fetch")
        out = store.maybe_offload(original, run_id="r1", iteration=3)
        assert out is not original
        # Full content preserved in the backing store.
        assert len(backing_store) == 1
        stored_key, stored_value = next(iter(backing_store.items()))
        assert stored_value == big
        # Replacement content carries marker + length + key + preview.
        assert REFERENCE_MARKER in (out.content or "")
        assert f"key={stored_key}" in (out.content or "")
        assert "500 chars" in (out.content or "")
        preview_len = 20
        assert "x" * preview_len in (out.content or "")
        # Metadata preserved.
        assert out.tool_call_id == "call-1"
        assert out.name == "fetch"

    def test_replacement_is_much_shorter(self, store: ToolResultStore) -> None:
        big = "y" * 50_000
        out = store.maybe_offload(_result(big), run_id="r", iteration=0)
        assert out.content is not None
        assert len(out.content) < 1_000  # marker + key + preview only


# ---------------------------------------------------------------------------
# Key recovery.
# ---------------------------------------------------------------------------


class TestReferenceKey:
    def test_extract_from_replacement(
        self, store: ToolResultStore, backing_store: dict[str, str]
    ) -> None:
        big = "z" * 400
        out = store.maybe_offload(_result(big), run_id="r1", iteration=5)
        key = extract_reference_key(out.content or "")
        assert key is not None
        assert key in backing_store
        assert store.load(key) == big

    def test_extract_missing_marker_returns_none(self) -> None:
        assert extract_reference_key("no marker here") is None

    def test_extract_empty_returns_none(self) -> None:
        assert extract_reference_key("") is None

    def test_load_unknown_key_returns_none(self, store: ToolResultStore) -> None:
        assert store.load("locus:result:missing:0:tool") is None


# ---------------------------------------------------------------------------
# Key formatting — run_id / tool sanitisation.
# ---------------------------------------------------------------------------


class TestKeyFormat:
    def test_colon_slash_sanitised(self, store: ToolResultStore) -> None:
        big = "a" * 200
        out = store.maybe_offload(
            _result(big, name="foo/bar:baz"),
            run_id="run:one/two",
            iteration=2,
        )
        key = extract_reference_key(out.content or "")
        assert key is not None
        # Key shouldn't be broken apart by extract_reference_key's
        # whitespace stop — colons and slashes would, so the
        # implementation replaces them.
        assert ":" in key  # the locus:result:...:N prefix uses colons
        # But no embedded user colon / slash should survive in
        # the ID segments (we asserted those get replaced with _).
        user_segments = key.split(":")[2:]  # drop 'locus:result'
        assert not any("/" in seg for seg in user_segments)


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


class TestValidation:
    def test_threshold_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            ToolResultStore(save=lambda k, v: None, load=lambda k: None, threshold_chars=0)

    def test_preview_not_larger_than_threshold(self) -> None:
        with pytest.raises(ValueError):
            ToolResultStore(
                save=lambda k, v: None,
                load=lambda k: None,
                threshold_chars=100,
                preview_chars=200,
            )

    def test_preview_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            ToolResultStore(
                save=lambda k, v: None,
                load=lambda k: None,
                threshold_chars=100,
                preview_chars=-1,
            )


# ---------------------------------------------------------------------------
# Round-trip through a fake in-memory backing store.
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_save_then_load(self, store: ToolResultStore, backing_store: dict[str, str]) -> None:
        big = "payload " * 200
        out = store.maybe_offload(_result(big), run_id="r", iteration=0)
        key = extract_reference_key(out.content or "")
        assert key is not None
        loaded = store.load(key)
        assert loaded == big
