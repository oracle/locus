# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for ``tulip.memory.compactor.LLMCompactor`` internals.

Covers the early-return arms of ``async_apply`` and the small helper
methods (``_split_system``, ``_grow_tail``, ``_compact_without_llm``
overlap trimming) plus the module-level ``_is_async`` predicate.
"""

from __future__ import annotations

from typing import Any

from tulip.core.messages import Message, Role
from tulip.memory.compactor import LLMCompactor, _is_async


def _user(content: str) -> Message:
    return Message(role=Role.USER, content=content)


def _system(content: str) -> Message:
    return Message(role=Role.SYSTEM, content=content)


async def _summarize(middle: list[Message], prev: str | None) -> str:
    return "SUMMARY"


# ---------------------------------------------------------------------------
# async_apply early returns
# ---------------------------------------------------------------------------


async def test_async_apply_empty_returns_empty() -> None:
    assert await LLMCompactor().async_apply([]) == []


async def test_async_apply_without_summarize_fn_falls_back() -> None:
    c = LLMCompactor(
        context_length=100,
        trigger_fraction=0.5,
        tool_output_ttl_turns=0,
        summarize_fn=None,
    )
    msgs = [_user("x" * 200) for _ in range(4)]  # ~50 tokens each, over trigger
    out = await c.async_apply(msgs)
    assert out
    assert not any("REFERENCE ONLY" in (m.content or "") for m in out)


async def test_async_apply_rest_smaller_than_head_turns() -> None:
    c = LLMCompactor(
        context_length=100,
        trigger_fraction=0.5,
        head_turns=5,
        tool_output_ttl_turns=0,
        summarize_fn=_summarize,
    )
    msgs = [_system("S" * 40), _user("A" * 400)]
    out = await c.async_apply(msgs)
    # Only system + the single non-system message remain; no summary inserted.
    assert [m.role for m in out] == [Role.SYSTEM, Role.USER]


async def test_async_apply_empty_middle_returns_head_and_tail() -> None:
    calls: list[int] = []

    async def _spy(middle: list[Message], prev: str | None) -> str:
        calls.append(len(middle))
        return "SUMMARY"

    c = LLMCompactor(
        context_length=100,
        trigger_fraction=0.5,
        head_turns=1,
        tail_token_fraction=0.9,
        tool_output_ttl_turns=0,
        summarize_fn=_spy,
    )
    msgs = [_system("S" * 400), _user("A" * 400), _user("b"), _user("c")]
    out = await c.async_apply(msgs)
    # Middle was empty, so summarize_fn was never called.
    assert calls == []
    assert not any("REFERENCE ONLY" in (m.content or "") for m in out)
    assert len(out) == 4


# ---------------------------------------------------------------------------
# _split_system
# ---------------------------------------------------------------------------


def test_split_system_without_preserve_returns_none_system() -> None:
    c = LLMCompactor(preserve_system=False)
    system, rest = c._split_system([_system("s"), _user("u")])
    assert system is None
    assert len(rest) == 2


def test_split_system_when_no_leading_system_message() -> None:
    c = LLMCompactor()
    system, rest = c._split_system([_user("u")])
    assert system is None
    assert len(rest) == 1


# ---------------------------------------------------------------------------
# _grow_tail
# ---------------------------------------------------------------------------


def test_grow_tail_zero_budget_returns_empty() -> None:
    c = LLMCompactor(context_length=1, tail_token_fraction=0.5)  # budget = int(0.5) = 0
    assert c._grow_tail([_user("a"), _user("b")]) == []


# ---------------------------------------------------------------------------
# _compact_without_llm overlap trimming (sync apply path)
# ---------------------------------------------------------------------------


def test_compact_without_llm_trims_head_tail_overlap() -> None:
    c = LLMCompactor(
        context_length=100,
        trigger_fraction=0.8,
        head_turns=3,
        tail_token_fraction=0.5,
        tool_output_ttl_turns=0,
    )
    msgs = [_user("x" * 100) for _ in range(4)]  # 25 tokens each, total over trigger
    out = c.apply(msgs)
    # Overlap between head (3) and tail (2) is trimmed → no duplicated messages.
    assert len(out) == len(msgs)
    assert [id(m) for m in out] == [id(m) for m in msgs]


# ---------------------------------------------------------------------------
# module-level helper
# ---------------------------------------------------------------------------


def test_is_async_predicate() -> None:
    async def _coro() -> None: ...

    def _plain() -> None: ...

    assert _is_async(_coro) is True
    assert _is_async(_plain) is False


def test_is_async_with_non_callable() -> None:
    value: Any = 123
    assert _is_async(value) is False
