# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end integration test for the LLM compactor (C.3).

Exercises :class:`~locus.memory.compactor.LLMCompactor` against the
session-scoped ``model`` fixture in ``conftest.py`` (OCI GenAI or
OpenAI, depending on environment). The fixture auto-skips when no
model service is configured.

The test feeds a synthetic 200-message conversation, summarises the
middle via the live model, and asserts the head / tail / summary
shape that the compactor promises.
"""

from __future__ import annotations

from typing import Any

import pytest

from locus.core.messages import Message, Role
from locus.memory.compactor import LLMCompactor


pytestmark = pytest.mark.integration


@pytest.fixture
def long_conversation() -> list[Message]:
    """Build a synthetic 200-message conversation with a system anchor."""
    msgs: list[Message] = [
        Message.system(
            "You are an assistant helping with a multi-day debugging session "
            "on a distributed billing system. Never invent component names."
        )
    ]
    for i in range(100):
        msgs.append(
            Message.user(
                f"Step {i}: I'm checking node-{i % 5} and it returns "
                f"latency {i * 13 % 200}ms with payload {'x' * 80}"
            )
        )
        msgs.append(
            Message.assistant(
                f"Acknowledged step {i}. The latency on node-{i % 5} is "
                f"trending up; here are some observations: {'.' * 80}"
            )
        )
    return msgs


def _build_summarize_fn(model: Any) -> Any:
    """Return an async summarize_fn that uses ``model.generate``."""
    from locus.core.messages import Message as Msg

    async def _summarise(middle: list[Message], previous: str | None) -> str:
        instructions = (
            "Summarise the conversation excerpt below in three sections: "
            "Resolved, Pending, Remaining work. Be concrete and brief."
        )
        if previous:
            instructions += f"\n\nPrior summary (reuse where applicable):\n{previous}"

        # Render the middle as a single user message so the model can read it
        # without us reconstructing message-by-message API translation.
        rendered = "\n".join(
            f"[{m.role.value}] {(m.content or '')[:400]}" for m in middle if m.content
        )
        prompt = [
            Msg.system(instructions),
            Msg.user(f"Conversation excerpt:\n{rendered}"),
        ]
        response = await model.complete(prompt)
        out = response.message.content if response.message else ""
        return (out or "").strip() or "[empty summary]"

    return _summarise


# ---------------------------------------------------------------------------
# End-to-end happy path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compaction_with_real_model(model: Any, long_conversation: list[Message]) -> None:
    if model is None:
        pytest.skip("no model service configured (OCI / OpenAI env vars)")

    compactor = LLMCompactor(
        # Force compaction: the synthetic conversation is well under any real
        # model context window, so we shrink the budget to trigger.
        context_length=8_000,
        trigger_fraction=0.2,
        head_turns=2,
        tail_token_fraction=0.3,
        tool_output_ttl_turns=0,
        summarize_fn=_build_summarize_fn(model),
    )

    out = await compactor.async_apply(long_conversation)

    # System prompt preserved verbatim at index 0.
    assert out[0].role == Role.SYSTEM
    assert out[0].content is not None
    assert "distributed billing system" in out[0].content

    # Summary message inserted at index 1 with the handoff preamble.
    assert out[1].role == Role.SYSTEM
    assert out[1].content is not None
    assert "REFERENCE ONLY" in out[1].content
    # The summary text itself is non-trivial (real model produced something).
    assert len(out[1].content) > len("[CONTEXT COMPACTION — REFERENCE ONLY]") + 50

    # Head preserved (first two non-system messages from the original).
    head_a, head_b = out[2], out[3]
    assert head_a.content is not None
    assert head_a.content.startswith("Step 0:")
    assert head_b.content is not None
    assert head_b.content.startswith("Acknowledged step 0")

    # Tail includes the most recent message.
    last = out[-1]
    assert last.content is not None
    assert "step 99" in last.content.lower() or "Acknowledged step 99" in last.content

    # Compaction shrank the message count meaningfully.
    assert len(out) < len(long_conversation) // 2


# ---------------------------------------------------------------------------
# Iterative compaction: the previous summary is forwarded.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iterative_compaction_carries_previous_summary(
    model: Any, long_conversation: list[Message]
) -> None:
    if model is None:
        pytest.skip("no model service configured")

    seen_previous: list[str | None] = []

    base_summarise = _build_summarize_fn(model)

    async def _wrapped(middle: list[Message], previous: str | None) -> str:
        seen_previous.append(previous)
        return await base_summarise(middle, previous)

    compactor = LLMCompactor(
        context_length=8_000,
        trigger_fraction=0.2,
        head_turns=2,
        tail_token_fraction=0.3,
        tool_output_ttl_turns=0,
        summarize_fn=_wrapped,
    )

    out1 = await compactor.async_apply(long_conversation)

    # Append more turns and recompact.
    more: list[Message] = list(out1)
    for i in range(40):
        more.append(Message.user(f"Follow-up {i}: " + ("y" * 200)))
        more.append(Message.assistant(f"Note {i}: " + ("z" * 200)))
    await compactor.async_apply(more)

    assert len(seen_previous) == 2
    assert seen_previous[0] is None
    assert seen_previous[1] is not None
    assert isinstance(seen_previous[1], str)
    assert len(seen_previous[1]) > 0
