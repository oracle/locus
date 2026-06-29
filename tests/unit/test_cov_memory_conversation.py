# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for ``tulip.memory.conversation.SummarizingManager``.

Covers the system-message and summary-cache arms of both the sync
``apply`` and async ``async_apply`` paths.
"""

from __future__ import annotations

from typing import Any

from tulip.core.messages import Message, Role
from tulip.memory.conversation import SummarizingManager


def _user(content: str) -> Message:
    return Message(role=Role.USER, content=content)


def _system(content: str = "sys") -> Message:
    return Message(role=Role.SYSTEM, content=content)


# ---------------------------------------------------------------------------
# Sync apply
# ---------------------------------------------------------------------------


def test_apply_under_threshold_keeps_system_message() -> None:
    mgr = SummarizingManager(threshold=10, keep_recent=3)
    msgs = [_system(), _user("hi"), Message(role=Role.ASSISTANT, content="yo")]
    out = mgr.apply(msgs)
    assert out[0].role == Role.SYSTEM
    assert len(out) == 3


def test_generate_summary_uses_fn_then_serves_from_cache() -> None:
    def _fn(_msgs: list[Message]) -> str:
        return "ignored-by-sync-path"

    mgr = SummarizingManager(threshold=3, keep_recent=1, summarize_fn=_fn)
    msgs = [_user(f"m{i}") for i in range(10)]

    out1 = mgr.apply(msgs)
    summary1 = next(m for m in out1 if m.role == Role.SYSTEM)
    assert "summarized" in (summary1.content or "")

    # Second call hits the summary cache for the identical prefix.
    out2 = mgr.apply(msgs)
    summary2 = next(m for m in out2 if m.role == Role.SYSTEM)
    assert summary2.content == summary1.content


# ---------------------------------------------------------------------------
# Async apply
# ---------------------------------------------------------------------------


async def test_async_apply_empty_returns_empty() -> None:
    assert await SummarizingManager().async_apply([]) == []


async def test_async_apply_under_threshold_keeps_system_message() -> None:
    mgr = SummarizingManager(threshold=10, keep_recent=3)
    out = await mgr.async_apply([_system(), _user("hi")])
    assert out[0].role == Role.SYSTEM
    assert len(out) == 2


async def test_async_apply_over_threshold_summarizes_with_system() -> None:
    mgr = SummarizingManager(threshold=3, keep_recent=1)  # no summarize_fn
    msgs: list[Message] = [_system()] + [_user(f"u{i}") for i in range(10)]
    out = await mgr.async_apply(msgs)
    assert out[0].role == Role.SYSTEM
    assert any("Summary of previous conversation" in (m.content or "") for m in out)


async def test_async_apply_uses_async_summarize_fn() -> None:
    captured: dict[str, Any] = {}

    async def _async_summarize(to_summarize: list[Message]) -> str:
        captured["count"] = len(to_summarize)
        return "ASYNC-SUMMARY-TEXT"

    mgr = SummarizingManager(threshold=3, keep_recent=1, summarize_fn=_async_summarize)
    msgs: list[Message] = [_system()] + [_user(f"u{i}") for i in range(10)]
    out = await mgr.async_apply(msgs)
    assert captured["count"] == 9
    assert any("ASYNC-SUMMARY-TEXT" in (m.content or "") for m in out)
