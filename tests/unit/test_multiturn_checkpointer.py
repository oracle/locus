# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Multi-turn regression test: checkpointed state + new user message must
re-enter the model.

Regression: the loop used to read `should_terminate` at the start of every
run, and the old implementation returned `(True, "no_tools")` whenever
`iteration > 0` and any assistant message existed anywhere in history.
A checkpointed conversation resumes with `iteration > 0` by design, so the
check fired before the Think node ever ran — the model was never called for
any turn after the first.
"""

from __future__ import annotations

from typing import Any

import pytest

from locus.agent.agent import Agent
from locus.core.messages import Message
from locus.memory.backends.memory import MemoryCheckpointer
from locus.models.base import ModelResponse


class RecordingModel:
    """Fake model that records every `complete` call and returns canned replies."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append(list(messages))
        if not self._replies:
            raise AssertionError("RecordingModel ran out of canned replies")
        reply = self._replies.pop(0)
        return ModelResponse(
            message=Message.assistant(reply),
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            stop_reason="complete",
        )


class TestCheckpointedMultiTurn:
    def test_second_turn_invokes_model_when_first_turn_had_no_tool_calls(self):
        """A second run_sync on the same thread must actually call the model.

        Without the fix, `state.should_terminate` fires `no_tools` at the top
        of turn 2 because the checkpoint carries `iteration=1` and the prior
        assistant reply has no tool calls. That skips the Think node, leaves
        the new user message unanswered, and returns a stale message.
        """
        model = RecordingModel(
            replies=[
                "Hi there!",  # turn 1 reply
                "I'm a helpful assistant.",  # turn 2 reply
            ]
        )
        checkpointer = MemoryCheckpointer()
        agent = Agent(
            model=model,
            system_prompt="You are a helpful assistant.",
            checkpointer=checkpointer,
            max_iterations=5,
        )

        # Turn 1
        r1 = agent.run_sync("hi", thread_id="conv-1")
        assert len(model.calls) == 1, "model must be called on turn 1"
        assert r1.message == "Hi there!"

        # Turn 2 — same thread, new user message. The bug caused the loop
        # to terminate before Think ran, so model.calls stayed at 1.
        r2 = agent.run_sync("whats your job", thread_id="conv-1")
        assert len(model.calls) == 2, (
            f"model must be called on turn 2 (regression: got {len(model.calls)} "
            f"call(s) across both turns, meaning turn 2 short-circuited)"
        )
        assert r2.message == "I'm a helpful assistant."

        # Second call should see the full rolling history, including the
        # freshly appended user message.
        turn2_messages = model.calls[1]
        roles = [m.role.value for m in turn2_messages]
        contents = [(m.content or "") for m in turn2_messages]
        assert roles[-1] == "user"
        assert contents[-1] == "whats your job"
        assert "hi" in contents, "turn 1 user message should remain in history"

    def test_three_turn_conversation_all_calls_model(self):
        """Every turn in a longer conversation must call the model exactly once."""
        model = RecordingModel(replies=["A1", "A2", "A3"])
        agent = Agent(
            model=model,
            system_prompt="Be brief.",
            checkpointer=MemoryCheckpointer(),
            max_iterations=5,
        )

        agent.run_sync("u1", thread_id="conv-2")
        agent.run_sync("u2", thread_id="conv-2")
        agent.run_sync("u3", thread_id="conv-2")

        assert len(model.calls) == 3
        # Last call should have accumulated all three user messages.
        contents = [(m.content or "") for m in model.calls[-1]]
        for expected in ("u1", "u2", "u3"):
            assert expected in contents, f"turn 3 context missing prior message {expected!r}"

    def test_different_threads_are_independent(self):
        """Regression check: the fix must not leak termination state across threads."""
        model = RecordingModel(replies=["A1a", "A1b", "A2a", "A2b"])
        agent = Agent(
            model=model,
            system_prompt="Be brief.",
            checkpointer=MemoryCheckpointer(),
            max_iterations=5,
        )

        agent.run_sync("hi", thread_id="thread-A")
        agent.run_sync("hi", thread_id="thread-B")
        agent.run_sync("follow up", thread_id="thread-A")
        agent.run_sync("follow up", thread_id="thread-B")

        assert len(model.calls) == 4


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
