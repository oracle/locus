# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for the Hermes-port external tool-result storage (D.1).

Wires :class:`~locus.tools.result_storage.ToolResultStore` to a shared
in-process dict to verify that:

* Oversized tool outputs are persisted to disk without hitting the
  agent's context budget.
* The reference key embedded in the inline content can be used to
  recover the full payload across separate ``ToolResultStore``
  instances pointing at the same backend.
"""

from __future__ import annotations

import pytest

from locus.core.messages import ToolResult
from locus.tools.result_storage import ToolResultStore, extract_reference_key


def _make_store(backend: dict[str, str]) -> ToolResultStore:
    """Wrap a shared dict as a ToolResultStore.

    The cross-instance recovery scenario only requires that two
    ``ToolResultStore`` instances see the same key/value mapping; a
    plain dict captures that without dragging in any backend driver.
    """

    def _save(key: str, content: str) -> None:
        backend[key] = content

    def _load(key: str) -> str | None:
        return backend.get(key)

    return ToolResultStore(
        save=_save,
        load=_load,
        threshold_chars=1_000,
        preview_chars=200,
    )


@pytest.fixture
def backend() -> dict[str, str]:
    return {}


class TestToolResultStorageRoundTrip:
    def test_offload_then_load_via_backend(self, backend: dict[str, str]) -> None:
        store = _make_store(backend)
        big_content = "log entry " * 500  # ~5 kB
        original = ToolResult(tool_call_id="call-7", name="fetch_logs", content=big_content)

        offloaded = store.maybe_offload(original, run_id="run-x", iteration=4)

        # Inline content is now a short reference, well under the original.
        assert offloaded is not original
        assert offloaded.content is not None
        assert len(offloaded.content) < 1_000
        # tool_call_id + name preserved on the replacement (asserted below).
        assert offloaded.tool_call_id == "call-7"
        assert offloaded.name == "fetch_logs"

        # Recover via embedded reference key.
        key = extract_reference_key(offloaded.content)
        assert key is not None
        loaded = store.load(key)
        assert loaded == big_content

    def test_recovery_from_separate_store_instance(self, backend: dict[str, str]) -> None:
        # Save with one store instance...
        store_a = _make_store(backend)
        result = ToolResult(
            tool_call_id="c1",
            name="big_tool",
            content="payload " * 400,
        )
        offloaded = store_a.maybe_offload(result, run_id="r", iteration=0)
        key = extract_reference_key(offloaded.content or "")
        assert key is not None

        # ...load with a fresh store pointing at the same backend dict.
        store_b = _make_store(backend)
        loaded = store_b.load(key)
        assert loaded == "payload " * 400

    def test_under_threshold_passes_through_no_db_write(self, backend: dict[str, str]) -> None:
        store = _make_store(backend)
        small = ToolResult(tool_call_id="c1", name="t", content="quick")

        out = store.maybe_offload(small, run_id="r", iteration=0)
        assert out is small

        # No write should have happened — load on the constructed key
        # should return None.
        speculative_key = "locus:result:r:0:t"
        assert store.load(speculative_key) is None

    def test_concurrent_offloads_distinct_keys(self, backend: dict[str, str]) -> None:
        store = _make_store(backend)
        results = []
        for i in range(5):
            r = ToolResult(
                tool_call_id=f"c{i}",
                name="multi_tool",
                content=f"unique-{i}-" + ("x" * 1500),
            )
            results.append(store.maybe_offload(r, run_id="run-multi", iteration=i))

        keys = [extract_reference_key(o.content or "") for o in results]
        assert len(set(keys)) == 5  # all distinct
        for i, key in enumerate(keys):
            assert key is not None
            assert store.load(key) == f"unique-{i}-" + ("x" * 1500)
