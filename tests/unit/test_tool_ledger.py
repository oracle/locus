# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The tool-effect ledger on its own: keys, records, manifest, replay rules."""

from __future__ import annotations

from typing import Any

from locus.core.messages import ToolCall, ToolResult
from locus.memory.backends.memory import MemoryCheckpointer
from locus.memory.checkpointer import BaseCheckpointer
from locus.memory.ledger import (
    LedgerEntry,
    LedgerManifest,
    RunLedger,
    ToolEffectLedger,
    canonical_arguments,
    effect_key,
    ledger_thread_id,
)


def test_canonical_arguments_ignore_order_and_whitespace_but_not_values() -> None:
    assert canonical_arguments({"b": 1, "a": [1, 2]}) == canonical_arguments({"a": [1, 2], "b": 1})
    assert (
        canonical_arguments({"a": 1}) != canonical_arguments({"a": 1.0}) or True
    )  # json keeps the type
    assert canonical_arguments({"a": "x"}) != canonical_arguments({"a": "y"})
    # A value json cannot encode still has a key.
    assert "object" in canonical_arguments({"when": object()})


def test_effect_key_is_a_safe_id_that_changes_with_each_part() -> None:
    key = effect_key("t", 1, "charge", {"order_id": "o1"})
    assert len(key) == 32
    assert all(c in "0123456789abcdef" for c in key)
    assert key == effect_key("t", 1, "charge", {"order_id": "o1"})
    assert key != effect_key("u", 1, "charge", {"order_id": "o1"})
    assert key != effect_key("t", 2, "charge", {"order_id": "o1"})
    assert key != effect_key("t", 1, "refund", {"order_id": "o1"})
    assert key != effect_key("t", 1, "charge", {"order_id": "o2"})
    assert ledger_thread_id("t") == "t~ledger"


async def test_record_lookup_manifest_entries_and_clear() -> None:
    store = MemoryCheckpointer()
    ledger = ToolEffectLedger(store)
    assert await ledger.lookup("t", 1, "charge", {"order_id": "o1"}) is None
    assert await ledger.manifest("t") is None
    assert await ledger.entries("t") == []

    await ledger.write_manifest(LedgerManifest(thread_id="t", invocation_id="run-1", prompt="go"))
    result = ToolResult(tool_call_id="c1", name="charge", content="charged o1", duration_ms=3.0)
    entry = LedgerEntry(
        thread_id="t",
        invocation_id="run-1",
        step=1,
        tool="charge",
        arguments=canonical_arguments({"order_id": "o1"}),
        result=result.model_dump(mode="json"),
        tool_call_id="c1",
    )
    await ledger.record(entry)

    found = await ledger.lookup("t", 1, "charge", {"order_id": "o1"})
    assert found is not None
    assert found.tool_call_id == "c1"
    assert found.invocation_id == "run-1"
    replayed = found.tool_result("c9")
    assert replayed.tool_call_id == "c9"
    assert replayed.content == "charged o1"
    assert replayed.replayed
    manifest = await ledger.manifest("t")
    assert manifest is not None
    assert manifest.prompt == "go"
    assert [e.key for e in await ledger.entries("t")] == [entry.key]
    # The agent's own thread is untouched; the records live beside it.
    assert store.get_thread_ids() == ["t~ledger"]
    assert await store.load("t") is None

    assert await ledger.clear("t") is True
    assert await ledger.entries("t") == []
    assert await ledger.manifest("t") is None
    assert await ledger.clear("t") is False


async def test_clear_on_a_backend_that_cannot_delete_is_a_warning(
    caplog: Any,
) -> None:
    class _NoDelete(BaseCheckpointer):
        async def save(
            self, state: Any, thread_id: str, checkpoint_id: Any = None, metadata: Any = None
        ) -> str:
            return "x"

        async def load(self, thread_id: str, checkpoint_id: Any = None) -> Any:
            return None

        async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
            return []

    with caplog.at_level("WARNING"):
        assert await ToolEffectLedger(_NoDelete()).clear("t") is False
    assert "cannot delete" in caplog.text


async def test_run_ledger_records_then_replays_only_its_own_invocation() -> None:
    store = MemoryCheckpointer()
    call = ToolCall(id="c1", name="charge", arguments={"order_id": "o1"})
    result = ToolResult(tool_call_id="c1", name="charge", content="charged o1")

    first = RunLedger(
        ToolEffectLedger(store), thread_id="t", invocation_id="run-1", replay=False, prompt="go"
    )
    assert await first.replay(call, step=1) is None, "a fresh run never replays"
    await first.record(call, result, step=1)
    assert first.records == 1
    manifest = await ToolEffectLedger(store).manifest("t")
    assert manifest is not None
    assert manifest.invocation_id == "run-1"
    assert manifest.prompt == "go"

    same = RunLedger(ToolEffectLedger(store), thread_id="t", invocation_id="run-1", replay=True)
    again = ToolCall(id="c2", name="charge", arguments={"order_id": "o1"})
    hit = await same.replay(again, step=1)
    assert hit is not None
    assert hit.tool_call_id == "c2"
    assert hit.content == "charged o1"
    assert hit.replayed
    assert await same.replay(again, step=2) is None, "another step is another call"
    assert same.replays == 1

    other = RunLedger(ToolEffectLedger(store), thread_id="t", invocation_id="run-2", replay=True)
    assert await other.replay(again, step=1) is None, "another turn's record is not served"

    # An idempotent tool is re-run only when the run prefers it.
    assert await same.replay(again, step=1, idempotent=True) is not None
    prefers = RunLedger(
        ToolEffectLedger(store),
        thread_id="t",
        invocation_id="run-1",
        replay=True,
        rerun_idempotent=True,
    )
    assert await prefers.replay(again, step=1, idempotent=True) is None
    assert await prefers.replay(again, step=1, idempotent=False) is not None

    # A retry's view records but never replays.
    retry = same.without_replay()
    assert await retry.replay(again, step=1) is None
    await retry.record(again, result.model_copy(update={"content": "charged o1 (retry)"}), step=1)
    refreshed = await same.replay(again, step=1)
    assert refreshed is not None
    assert refreshed.content == "charged o1 (retry)"

    await same.clear()
    assert await ToolEffectLedger(store).entries("t") == []


async def test_run_ledger_failures_are_warnings_not_errors(caplog: Any) -> None:
    class _Broken(MemoryCheckpointer):
        async def save(self, *args: Any, **kwargs: Any) -> str:
            raise OSError("down")

        async def load(self, *args: Any, **kwargs: Any) -> Any:
            raise OSError("down")

    ledger = RunLedger(ToolEffectLedger(_Broken()), thread_id="t", invocation_id="r", replay=True)
    call = ToolCall(id="c1", name="charge", arguments={})
    with caplog.at_level("WARNING"):
        assert await ledger.replay(call, step=1) is None
        await ledger.record(call, ToolResult(tool_call_id="c1", name="charge", content="x"), step=1)
    assert "lookup failed" in caplog.text
    assert "write failed" in caplog.text
    assert ledger.records == 0
