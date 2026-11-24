# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``SummarizingManager`` calls the ``summarize_fn`` it was given, on both paths.

A plain function writes the summary through ``apply()`` and through
``async_apply()``. An async one is awaited by ``async_apply()``; ``apply()``
is synchronous and cannot await it, so it keeps a placeholder and says so
once per manager with a ``LocusWarning`` that names ``async_apply()``.
"""

from __future__ import annotations

import functools
import warnings

import pytest

from locus.core.messages import Message, Role
from locus.core.warnings import LocusWarning
from locus.memory.conversation import SummarizingManager


PLACEHOLDER = "Conversation with 4 messages summarized."


def _thread(n: int = 5) -> list[Message]:
    return [Message(role=Role.USER, content=f"m{i}") for i in range(n)]


def _summary(messages: list[Message]) -> str:
    return next(m.content or "" for m in messages if m.role == Role.SYSTEM)


def test_apply_calls_a_sync_summarize_fn_once_with_the_messages_to_summarise() -> None:
    seen: list[list[Message]] = []

    def summarise(messages: list[Message]) -> str:
        seen.append(messages)
        return "USER SUMMARY"

    manager = SummarizingManager(threshold=3, keep_recent=1, summarize_fn=summarise)
    thread = _thread()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = manager.apply(thread)

    assert len(seen) == 1
    assert seen[0] == thread[:4]
    assert _summary(out) == "[Summary of previous conversation (4 messages)]:\nUSER SUMMARY"
    assert out[-1] == thread[-1]


async def test_async_apply_calls_a_sync_summarize_fn_too() -> None:
    calls: list[int] = []

    def summarise(messages: list[Message]) -> str:
        calls.append(len(messages))
        return "USER SUMMARY"

    manager = SummarizingManager(threshold=3, keep_recent=1, summarize_fn=summarise)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = await manager.async_apply(_thread())

    assert calls == [4]
    assert _summary(out).endswith("USER SUMMARY")


async def test_an_async_summarize_fn_warns_once_on_apply_and_is_used_by_async_apply() -> None:
    calls: list[int] = []

    async def summarise(messages: list[Message]) -> str:
        calls.append(len(messages))
        return "ASYNC SUMMARY"

    manager = SummarizingManager(threshold=3, keep_recent=1, summarize_fn=summarise)

    with pytest.warns(LocusWarning, match=r"async_apply\(\)") as caught:
        out = manager.apply(_thread())
    assert len(caught) == 1
    assert caught[0].filename == __file__, "the warning points at the caller of apply()"
    assert _summary(out).endswith(PLACEHOLDER)
    assert calls == []

    # Once per manager: a second thread over the threshold does not warn again.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = manager.apply(_thread(6))
    assert _summary(again).endswith("Conversation with 5 messages summarized.")

    # The same manager on the async path awaits the summariser.
    out = await manager.async_apply(_thread())
    assert calls == [4]
    assert _summary(out).endswith("ASYNC SUMMARY")


def test_each_manager_warns_for_itself() -> None:
    async def summarise(messages: list[Message]) -> str:
        return "ASYNC SUMMARY"

    for _ in range(2):
        manager = SummarizingManager(threshold=3, keep_recent=1, summarize_fn=summarise)
        with pytest.warns(LocusWarning, match="placeholder"):
            manager.apply(_thread())


class _AsyncCallable:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, messages: list[Message]) -> str:
        self.calls += 1
        return "FROM AN OBJECT"


async def test_an_object_with_an_async_call_and_a_partial_count_as_async() -> None:
    async def summarise(prefix: str, messages: list[Message]) -> str:
        return f"{prefix} {len(messages)}"

    for summarize_fn, expected in (
        (_AsyncCallable(), "FROM AN OBJECT"),
        (functools.partial(summarise, "PARTIAL"), "PARTIAL 4"),
    ):
        manager = SummarizingManager(threshold=3, keep_recent=1, summarize_fn=summarize_fn)
        with pytest.warns(LocusWarning, match=r"async_apply\(\)"):
            assert _summary(manager.apply(_thread())).endswith(PLACEHOLDER)
        assert _summary(await manager.async_apply(_thread())).endswith(expected)


def test_a_plain_callable_that_returns_a_coroutine_gets_the_placeholder_not_its_repr() -> None:
    async def summarise(messages: list[Message]) -> str:
        return "NEVER AWAITED"

    manager = SummarizingManager(
        threshold=3,
        keep_recent=1,
        summarize_fn=lambda messages: summarise(messages),  # noqa: PLW0108 - the case under test
    )

    with warnings.catch_warnings():
        # "coroutine ... was never awaited" would surface here as an error.
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.warns(LocusWarning, match="placeholder"):
            out = manager.apply(_thread())

    assert _summary(out).endswith(PLACEHOLDER)
    assert "coroutine" not in _summary(out)


def test_a_sync_summary_is_cached_for_the_same_messages() -> None:
    calls: list[int] = []

    def summarise(messages: list[Message]) -> str:
        calls.append(len(messages))
        return f"SUMMARY {len(calls)}"

    manager = SummarizingManager(threshold=3, keep_recent=1, summarize_fn=summarise)
    thread = _thread()

    assert _summary(manager.apply(thread)).endswith("SUMMARY 1")
    assert _summary(manager.apply(thread)).endswith("SUMMARY 1")
    assert calls == [4]
    assert _summary(manager.apply(_thread(7))).endswith("SUMMARY 2")
    assert calls == [4, 6]


def test_a_sync_summarize_fn_that_raises_propagates_and_nothing_is_cached() -> None:
    attempts: list[int] = []

    def summarise(messages: list[Message]) -> str:
        attempts.append(len(messages))
        if len(attempts) == 1:
            raise RuntimeError("model unavailable")
        return "SECOND TRY"

    manager = SummarizingManager(threshold=3, keep_recent=1, summarize_fn=summarise)
    thread = _thread()

    with pytest.raises(RuntimeError, match="model unavailable"):
        manager.apply(thread)
    assert _summary(manager.apply(thread)).endswith("SECOND TRY")
