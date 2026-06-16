# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end integration tests for MemoryManager with a real model.

Exercises the full session lifecycle:
  - Session 1: agent runs, on_session_end extracts memories from the
    conversation and persists them to an InMemoryStore.
  - Session 2: agent runs, on_session_start injects the persisted
    memories into the system prompt before the first model call.

The tests use the session-scoped ``model`` fixture from conftest.py
(OCI GenAI or OpenAI depending on environment) and auto-skip when no
model service is configured.
"""

from __future__ import annotations

import pytest

from locus.agent import Agent, AgentConfig
from locus.core.messages import Message, Role
from locus.memory.manager import LLMMemoryManager, Memory, MemoryType
from locus.memory.store import InMemoryStore


pytestmark = [pytest.mark.integration, pytest.mark.requires_model]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_extract_fn(memories: list[Memory]):
    """Return a deterministic extract_fn that always yields ``memories``."""

    async def _fn(messages):
        return memories

    return _fn


def _has_memory_block(messages) -> bool:
    return any(
        m.role == Role.SYSTEM and "[Long-term Memory]" in (m.content or "") for m in messages
    )


# ---------------------------------------------------------------------------
# Test: on_session_start injects memories before first model call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memories_injected_into_live_agent(model):
    """Memories stored in the backing store appear in the agent's context."""
    store = InMemoryStore()
    manager = LLMMemoryManager(store=store)

    # Pre-populate the store with a known fact.
    known_memory = Memory(
        type=MemoryType.USER,
        key="preferred_language",
        content="The user prefers Python over JavaScript.",
        metadata={},
    )
    await manager.save([known_memory])

    agent = Agent(
        config=AgentConfig(
            model=model,
            system_prompt=(
                "You are a helpful assistant. When asked what you know about "
                "the user, recite EXACTLY what appears in your [Long-term Memory] "
                "block — word for word."
            ),
            memory_manager=manager,
            max_iterations=2,
            max_tokens=256,
        )
    )

    result = agent.run_sync("What do you know about my programming language preference?")
    reply = (result.message or "").lower()

    # The agent should surface the injected memory fact.
    assert "python" in reply, (
        f"Expected 'python' in reply from injected memory, got: {result.message!r}"
    )


# ---------------------------------------------------------------------------
# Test: on_session_end extracts and persists memories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_end_persists_memories(model):
    """on_session_end fires after agent run and writes memories to the store."""
    store = InMemoryStore()

    custom_memory = Memory(
        type=MemoryType.FEEDBACK,
        key="no_mocks",
        content="Never mock the database in tests.",
        metadata={},
    )
    manager = LLMMemoryManager(
        store=store,
        extract_fn=_make_extract_fn([custom_memory]),
    )

    agent = Agent(
        config=AgentConfig(
            model=model,
            system_prompt="You are a helpful assistant.",
            memory_manager=manager,
            max_iterations=2,
            max_tokens=128,
        )
    )

    agent.run_sync("How should I approach database testing?")

    # After the run, the extracted memory should be in the store.
    retrieved = await store.get(("locus_memory", "feedback"), "no_mocks")
    assert retrieved is not None, "Memory was not persisted after session end"
    assert "Never mock the database" in retrieved.get("content", "")


# ---------------------------------------------------------------------------
# Test: full cross-session cycle with real model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_cross_session_memory_cycle(model):
    """Session 1 teaches the agent a fact; Session 2 recalls it from memory."""
    store = InMemoryStore()

    # What we want to persist after session 1.
    session1_memories = [
        Memory(
            type=MemoryType.USER,
            key="role",
            content="The user is a senior Python engineer specialising in distributed systems.",
            metadata={},
        ),
        Memory(
            type=MemoryType.FEEDBACK,
            key="terse_style",
            content="Keep answers short — the user dislikes verbose explanations.",
            metadata={},
        ),
    ]

    manager = LLMMemoryManager(
        store=store,
        extract_fn=_make_extract_fn(session1_memories),
    )

    # --- Session 1 ---
    agent1 = Agent(
        config=AgentConfig(
            model=model,
            system_prompt="You are a helpful assistant.",
            memory_manager=manager,
            max_iterations=2,
            max_tokens=128,
        )
    )
    agent1.run_sync("I'm a senior Python engineer. I prefer short answers. What's the CAP theorem?")

    # Verify memories were extracted.
    retrieved = await manager.retrieve()
    assert any(m.key == "role" for m in retrieved), "role memory not extracted"
    assert any(m.key == "terse_style" for m in retrieved), "terse_style memory not extracted"

    # --- Session 2 (fresh agent, same store) ---
    manager2 = LLMMemoryManager(
        store=store,
        extract_fn=_make_extract_fn([]),  # nothing new to learn
    )

    agent2 = Agent(
        config=AgentConfig(
            model=model,
            system_prompt=(
                "You are a helpful assistant. When asked about the user, "
                "describe what your [Long-term Memory] block says about them."
            ),
            memory_manager=manager2,
            max_iterations=2,
            max_tokens=256,
        )
    )
    result = agent2.run_sync("Who am I? What do you know about my style preferences?")
    reply = (result.message or "").lower()

    # The second agent should mention what was stored from session 1.
    assert "python" in reply or "distributed" in reply or "senior" in reply, (
        f"Session 2 should recall user facts from session 1, got: {result.message!r}"
    )


# ---------------------------------------------------------------------------
# Test: NoopMemoryManager wired to agent — no injection, no extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_noop_manager_with_live_agent(model):
    """NoopMemoryManager wires cleanly — agent runs normally, nothing stored."""
    from locus.memory.manager import NoopMemoryManager

    manager = NoopMemoryManager()

    agent = Agent(
        config=AgentConfig(
            model=model,
            system_prompt="You are a helpful assistant.",
            memory_manager=manager,
            max_iterations=2,
            max_tokens=64,
        )
    )
    result = agent.run_sync("Say hello.")
    assert result.message, "Agent with NoopMemoryManager should still produce output"


# ---------------------------------------------------------------------------
# Test: memory injection appears in state before model call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_block_in_state_before_run(model):
    """Verify _inject_memories_into_state produces the expected shape."""
    from locus.core.state import AgentState

    store = InMemoryStore()
    manager = LLMMemoryManager(store=store)

    memories = [
        Memory(type=MemoryType.USER, key="k1", content="Fact one.", metadata={}),
        Memory(type=MemoryType.REFERENCE, key="k2", content="Linear INGEST.", metadata={}),
    ]
    await manager.save(memories)

    state = AgentState(
        messages=(Message(role=Role.SYSTEM, content="System prompt."),),
    )
    new_state = await manager.on_session_start(state)

    msgs = list(new_state.messages)
    assert msgs[0].content == "System prompt.", "Original system prompt must stay first"
    memory_msgs = [
        m for m in msgs if m.role == Role.SYSTEM and "[Long-term Memory]" in (m.content or "")
    ]
    assert len(memory_msgs) == 1, "Exactly one memory block should be injected"
    assert "USER [k1]:" in memory_msgs[0].content
    assert "REFERENCE [k2]:" in memory_msgs[0].content
