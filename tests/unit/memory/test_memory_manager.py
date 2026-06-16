# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""End-to-end tests for the long-term memory manager."""

from __future__ import annotations

import pytest

from locus.core.messages import Message, Role
from locus.core.state import AgentState
from locus.memory.manager import (
    LLMMemoryManager,
    Memory,
    MemoryType,
    NoopMemoryManager,
    _format_memory_block,
    _heuristic_extract,
    _inject_memories_into_state,
)
from locus.memory.store import InMemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def sample_memories() -> list[Memory]:
    return [
        Memory(
            type=MemoryType.USER,
            key="role",
            content="Senior Python engineer, new to React.",
            metadata={},
        ),
        Memory(
            type=MemoryType.FEEDBACK,
            key="no_db_mocks",
            content="Never mock the database in tests. Why: prior mock/prod divergence.",
            metadata={},
        ),
        Memory(
            type=MemoryType.PROJECT,
            key="auth_rewrite",
            content="Auth rewrite driven by compliance, not tech debt.",
            metadata={},
        ),
        Memory(
            type=MemoryType.REFERENCE,
            key="linear_pipeline",
            content="Pipeline bugs tracked in Linear project 'INGEST'.",
            metadata={},
        ),
    ]


@pytest.fixture
def simple_state() -> AgentState:
    return AgentState(
        messages=(Message(role=Role.SYSTEM, content="You are a helpful assistant."),),
    )


@pytest.fixture
def conversation_state() -> AgentState:
    return AgentState(
        messages=(
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(
                role=Role.USER,
                content="I'm a senior Python engineer new to React. I need help with hooks.",
            ),
            Message(role=Role.ASSISTANT, content="Happy to help! What specifically about hooks?"),
            Message(
                role=Role.USER,
                content="Don't use class components — I only want functional components.",
            ),
            Message(role=Role.ASSISTANT, content="Understood, functional components only."),
        ),
    )


# ---------------------------------------------------------------------------
# NoopMemoryManager
# ---------------------------------------------------------------------------


class TestNoopMemoryManager:
    @pytest.mark.asyncio
    async def test_extract_returns_empty(self, conversation_state):
        mgr = NoopMemoryManager()
        result = await mgr.extract(list(conversation_state.messages))
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_returns_empty(self):
        mgr = NoopMemoryManager()
        result = await mgr.retrieve()
        assert result == []

    @pytest.mark.asyncio
    async def test_save_is_silent(self, sample_memories):
        mgr = NoopMemoryManager()
        await mgr.save(sample_memories)  # should not raise

    @pytest.mark.asyncio
    async def test_on_session_start_returns_state_unchanged(self, simple_state):
        mgr = NoopMemoryManager()
        result = await mgr.on_session_start(simple_state)
        assert result.messages == simple_state.messages

    @pytest.mark.asyncio
    async def test_on_session_end_is_silent(self, conversation_state):
        mgr = NoopMemoryManager()
        await mgr.on_session_end(conversation_state)  # should not raise


# ---------------------------------------------------------------------------
# Memory dataclass
# ---------------------------------------------------------------------------


class TestMemory:
    def test_roundtrip(self, sample_memories):
        for m in sample_memories:
            d = m.to_store_value()
            restored = Memory.from_store_value(d)
            assert restored.type == m.type
            assert restored.key == m.key
            assert restored.content == m.content

    def test_all_types_roundtrip(self):
        for t in MemoryType:
            m = Memory(type=t, key="k", content="c", metadata={"x": 1})
            assert Memory.from_store_value(m.to_store_value()).type == t


# ---------------------------------------------------------------------------
# Formatting / injection helpers
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_format_block_contains_labels(self, sample_memories):
        block = _format_memory_block(sample_memories)
        assert "[Long-term Memory]" in block
        assert "USER [role]:" in block
        assert "FEEDBACK [no_db_mocks]:" in block
        assert "PROJECT [auth_rewrite]:" in block
        assert "REFERENCE [linear_pipeline]:" in block

    def test_inject_after_system_prompt(self, simple_state, sample_memories):
        new_state = _inject_memories_into_state(simple_state, sample_memories)
        msgs = list(new_state.messages)
        assert len(msgs) == 2
        assert msgs[0].role == Role.SYSTEM  # original system prompt stays first
        assert msgs[1].role == Role.SYSTEM  # memory block injected at position 1
        assert "[Long-term Memory]" in msgs[1].content

    def test_inject_without_system_prompt(self, sample_memories):
        state = AgentState(
            messages=(Message(role=Role.USER, content="Hello"),),
        )
        new_state = _inject_memories_into_state(state, sample_memories)
        msgs = list(new_state.messages)
        assert len(msgs) == 2
        assert msgs[0].role == Role.SYSTEM  # memory block goes first
        assert "[Long-term Memory]" in msgs[0].content

    def test_inject_empty_messages(self, sample_memories):
        state = AgentState(messages=())
        new_state = _inject_memories_into_state(state, sample_memories)
        assert len(new_state.messages) == 1
        assert "[Long-term Memory]" in new_state.messages[0].content


# ---------------------------------------------------------------------------
# Heuristic extractor
# ---------------------------------------------------------------------------


class TestHeuristicExtract:
    def test_detects_correction(self):
        msgs = [
            Message(role=Role.USER, content="Don't use class components."),
        ]
        memories = _heuristic_extract(msgs)
        assert any(m.type == MemoryType.FEEDBACK for m in memories)

    def test_detects_user_role(self):
        msgs = [
            Message(role=Role.USER, content="I'm a senior Python engineer."),
        ]
        memories = _heuristic_extract(msgs)
        assert any(m.type == MemoryType.USER for m in memories)

    def test_detects_project_context(self):
        msgs = [
            Message(
                role=Role.USER,
                content="We're working on a compliance-driven rewrite with a deadline next week.",
            ),
        ]
        memories = _heuristic_extract(msgs)
        assert any(m.type == MemoryType.PROJECT for m in memories)

    def test_detects_reference(self):
        msgs = [
            Message(
                role=Role.USER,
                content="Check the Grafana dashboard at https://grafana.internal/d/api-latency.",
            ),
        ]
        memories = _heuristic_extract(msgs)
        assert any(m.type == MemoryType.REFERENCE for m in memories)

    def test_ignores_assistant_messages_for_user_role(self):
        msgs = [
            Message(role=Role.ASSISTANT, content="I'm a helpful assistant."),
        ]
        memories = _heuristic_extract(msgs)
        # "I'm a" in assistant message should not produce a USER memory
        assert not any(m.type == MemoryType.USER for m in memories)

    def test_empty_messages(self):
        assert _heuristic_extract([]) == []


# ---------------------------------------------------------------------------
# LLMMemoryManager — save / retrieve cycle
# ---------------------------------------------------------------------------


class TestLLMMemoryManagerStore:
    @pytest.mark.asyncio
    async def test_save_and_retrieve_round_trip(self, store, sample_memories):
        mgr = LLMMemoryManager(store=store)
        await mgr.save(sample_memories)
        retrieved = await mgr.retrieve()
        assert len(retrieved) == len(sample_memories)
        keys = {m.key for m in retrieved}
        assert "role" in keys
        assert "no_db_mocks" in keys
        assert "auth_rewrite" in keys
        assert "linear_pipeline" in keys

    @pytest.mark.asyncio
    async def test_save_upserts_by_key(self, store):
        mgr = LLMMemoryManager(store=store)
        m1 = Memory(type=MemoryType.USER, key="role", content="Junior engineer.")
        m2 = Memory(type=MemoryType.USER, key="role", content="Senior engineer.")
        await mgr.save([m1])
        await mgr.save([m2])
        retrieved = await mgr.retrieve()
        role_memories = [m for m in retrieved if m.key == "role"]
        assert len(role_memories) == 1
        assert role_memories[0].content == "Senior engineer."

    @pytest.mark.asyncio
    async def test_retrieve_empty_store(self, store):
        mgr = LLMMemoryManager(store=store)
        result = await mgr.retrieve()
        assert result == []

    @pytest.mark.asyncio
    async def test_namespace_prefix_isolates_tenants(self):
        store = InMemoryStore()
        mgr_a = LLMMemoryManager(store=store, namespace_prefix=("tenant_a",))
        mgr_b = LLMMemoryManager(store=store, namespace_prefix=("tenant_b",))

        await mgr_a.save([Memory(type=MemoryType.USER, key="role", content="Tenant A user.")])
        await mgr_b.save([Memory(type=MemoryType.USER, key="role", content="Tenant B user.")])

        a_memories = await mgr_a.retrieve()
        b_memories = await mgr_b.retrieve()

        assert a_memories[0].content == "Tenant A user."
        assert b_memories[0].content == "Tenant B user."


# ---------------------------------------------------------------------------
# LLMMemoryManager — extraction
# ---------------------------------------------------------------------------


class TestLLMMemoryManagerExtraction:
    @pytest.mark.asyncio
    async def test_uses_custom_extract_fn(self, store, conversation_state):
        custom_memory = Memory(
            type=MemoryType.FEEDBACK,
            key="functional_only",
            content="Use only functional React components.",
            metadata={},
        )

        async def my_extractor(messages):
            return [custom_memory]

        mgr = LLMMemoryManager(store=store, extract_fn=my_extractor)
        memories = await mgr.extract(list(conversation_state.messages))
        assert len(memories) == 1
        assert memories[0].key == "functional_only"

    @pytest.mark.asyncio
    async def test_falls_back_to_heuristic_without_extract_fn(self, store, conversation_state):
        mgr = LLMMemoryManager(store=store)
        memories = await mgr.extract(list(conversation_state.messages))
        # "I'm a senior Python engineer" and "Don't use class components" should trigger
        assert len(memories) > 0

    @pytest.mark.asyncio
    async def test_extract_fn_errors_propagate(self, store):
        async def bad_extractor(messages):
            raise ValueError("LLM call failed")

        mgr = LLMMemoryManager(store=store, extract_fn=bad_extractor)
        with pytest.raises(ValueError, match="LLM call failed"):
            await mgr.extract([Message(role=Role.USER, content="Hello")])


# ---------------------------------------------------------------------------
# LLMMemoryManager — full session lifecycle
# ---------------------------------------------------------------------------


class TestLLMMemoryManagerSessionLifecycle:
    @pytest.mark.asyncio
    async def test_on_session_end_saves_extracted_memories(self, store, conversation_state):
        extracted = [
            Memory(type=MemoryType.USER, key="role", content="Senior Python engineer."),
        ]

        async def my_extractor(messages):
            return extracted

        mgr = LLMMemoryManager(store=store, extract_fn=my_extractor)
        await mgr.on_session_end(conversation_state)

        retrieved = await mgr.retrieve()
        assert any(m.key == "role" for m in retrieved)

    @pytest.mark.asyncio
    async def test_on_session_start_injects_memories(self, store, simple_state, sample_memories):
        mgr = LLMMemoryManager(store=store)
        await mgr.save(sample_memories)

        new_state = await mgr.on_session_start(simple_state)
        msgs = list(new_state.messages)

        # A memory block message should be inserted
        memory_msgs = [
            m for m in msgs if m.role == Role.SYSTEM and "[Long-term Memory]" in (m.content or "")
        ]
        assert len(memory_msgs) == 1
        assert "USER [role]:" in memory_msgs[0].content

    @pytest.mark.asyncio
    async def test_on_session_start_no_op_when_store_empty(self, store, simple_state):
        mgr = LLMMemoryManager(store=store)
        new_state = await mgr.on_session_start(simple_state)
        # No memories → state unchanged
        assert new_state.messages == simple_state.messages

    @pytest.mark.asyncio
    async def test_full_cross_session_cycle(self):
        """Session 1 extracts; Session 2 gets memories injected."""
        store = InMemoryStore()

        session1_msgs = [
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(
                role=Role.USER,
                content="I'm a senior Python engineer and I prefer integration tests — don't mock the database.",
            ),
            Message(role=Role.ASSISTANT, content="Noted, I'll use real database connections."),
        ]

        remembered: list[Memory] = []

        async def extractor(messages):
            # Simulates an LLM extracting the correction and user fact.
            return [
                Memory(
                    type=MemoryType.USER,
                    key="role",
                    content="Senior Python engineer.",
                    metadata={},
                ),
                Memory(
                    type=MemoryType.FEEDBACK,
                    key="no_db_mocks",
                    content="Never mock the database. Why: user requested it.",
                    metadata={},
                ),
            ]

        mgr = LLMMemoryManager(store=store, extract_fn=extractor)

        # --- Session 1: agent runs, memories extracted at end ---
        session1_state = AgentState(messages=tuple(session1_msgs))
        await mgr.on_session_end(session1_state)

        # --- Session 2: new session, memories injected at start ---
        session2_state = AgentState(
            messages=(Message(role=Role.SYSTEM, content="You are a helpful assistant."),),
        )
        injected_state = await mgr.on_session_start(session2_state)

        msgs = list(injected_state.messages)
        memory_block = next(
            m for m in msgs if m.role == Role.SYSTEM and "[Long-term Memory]" in (m.content or "")
        )

        assert "USER [role]:" in memory_block.content
        assert "FEEDBACK [no_db_mocks]:" in memory_block.content
        assert "Senior Python engineer" in memory_block.content
        assert "Never mock the database" in memory_block.content


# ---------------------------------------------------------------------------
# AgentConfig integration
# ---------------------------------------------------------------------------


class TestAgentConfigIntegration:
    def test_memory_manager_accepted_in_config(self, store):
        from locus.agent.config import AgentConfig

        mgr = LLMMemoryManager(store=store)
        config = AgentConfig(model="openai:gpt-4o", memory_manager=mgr)
        assert config.memory_manager is mgr

    def test_memory_manager_defaults_to_none(self):
        from locus.agent.config import AgentConfig

        config = AgentConfig(model="openai:gpt-4o")
        assert config.memory_manager is None

    def test_noop_manager_accepted(self):
        from locus.agent.config import AgentConfig

        config = AgentConfig(model="openai:gpt-4o", memory_manager=NoopMemoryManager())
        assert isinstance(config.memory_manager, NoopMemoryManager)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_noop_repr(self):
        assert repr(NoopMemoryManager()) == "NoopMemoryManager()"

    def test_llm_repr(self, store):
        mgr = LLMMemoryManager(store=store, namespace_prefix=("users", "u1"))
        r = repr(mgr)
        assert "LLMMemoryManager" in r
        assert "InMemoryStore" in r
        assert "('users', 'u1')" in r
