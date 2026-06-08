# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Integration tests for checkpointer adapters with Agent."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

from tulip.core.messages import Message, Role
from tulip.core.state import AgentState
from tulip.memory.backends import (
    MemoryCheckpointer,
    StorageBackendAdapter,
)


pytestmark = pytest.mark.integration


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_state() -> AgentState:
    """Create a sample agent state for testing."""
    state = AgentState(
        agent_id="test-agent",
        max_iterations=10,
        confidence=0.5,
        metadata={"key": "value"},
    )
    state = state.with_message(Message(role=Role.USER, content="Hello"))
    state = state.with_message(Message(role=Role.ASSISTANT, content="Hi there!"))
    return state


# =============================================================================
# StorageBackendAdapter Tests
# =============================================================================


class _FakeStorageBackend:
    """In-memory storage backend exposing the simple save/load dict API."""

    def __init__(self) -> None:
        self._data: dict[str, dict] = {}

    async def save(self, thread_id: str, data: dict) -> None:
        self._data[thread_id] = data

    async def load(self, thread_id: str) -> dict | None:
        return self._data.get(thread_id)

    async def delete(self, thread_id: str) -> bool:
        return self._data.pop(thread_id, None) is not None

    async def exists(self, thread_id: str) -> bool:
        return thread_id in self._data

    async def list_threads(
        self, limit: int = 100, offset: int = 0, pattern: str = "%"
    ) -> list[str]:
        return list(self._data.keys())[offset : offset + limit]


class TestStorageBackendAdapter:
    """Test StorageBackendAdapter using an in-memory storage backend."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with in-memory storage backend."""
        return StorageBackendAdapter(_FakeStorageBackend())

    @pytest.mark.asyncio
    async def test_save_and_load(self, adapter, sample_state):
        """Save and load state through adapter."""
        # Save state
        checkpoint_id = await adapter.save(sample_state, "thread-1")
        assert checkpoint_id is not None

        # Load state
        loaded = await adapter.load("thread-1")
        assert loaded is not None
        assert loaded.agent_id == sample_state.agent_id
        assert len(loaded.messages) == len(sample_state.messages)
        assert loaded.confidence == sample_state.confidence

    @pytest.mark.asyncio
    async def test_load_specific_checkpoint(self, adapter, sample_state):
        """Load a specific checkpoint by ID."""
        # Save multiple checkpoints
        cp1 = await adapter.save(sample_state, "thread-1")

        state2 = sample_state.with_confidence(0.8)
        cp2 = await adapter.save(state2, "thread-1")

        # Load specific checkpoint
        loaded1 = await adapter.load("thread-1", cp1)
        assert loaded1.confidence == 0.5

        loaded2 = await adapter.load("thread-1", cp2)
        assert loaded2.confidence == 0.8

        # Load latest (should be cp2)
        latest = await adapter.load("thread-1")
        assert latest.confidence == 0.8

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, adapter, sample_state):
        """List checkpoints for a thread."""
        # Save multiple checkpoints
        await adapter.save(sample_state, "thread-1", "cp-1")
        await adapter.save(sample_state, "thread-1", "cp-2")
        await adapter.save(sample_state, "thread-1", "cp-3")

        # List checkpoints
        checkpoints = await adapter.list_checkpoints("thread-1")
        assert len(checkpoints) == 3
        assert "cp-1" in checkpoints
        assert "cp-2" in checkpoints
        assert "cp-3" in checkpoints

    @pytest.mark.asyncio
    async def test_delete_specific_checkpoint(self, adapter, sample_state):
        """Delete a specific checkpoint."""
        await adapter.save(sample_state, "thread-1", "cp-1")
        await adapter.save(sample_state, "thread-1", "cp-2")

        # Delete cp-1
        result = await adapter.delete("thread-1", "cp-1")
        assert result is True

        # cp-1 should be gone, cp-2 should exist
        assert not await adapter.exists("thread-1", "cp-1")
        assert await adapter.exists("thread-1", "cp-2")

    @pytest.mark.asyncio
    async def test_delete_all_checkpoints(self, adapter, sample_state):
        """Delete all checkpoints for a thread."""
        await adapter.save(sample_state, "thread-1", "cp-1")
        await adapter.save(sample_state, "thread-1", "cp-2")

        # Delete all
        result = await adapter.delete("thread-1")
        assert result is True

        # All should be gone
        assert not await adapter.exists("thread-1")

    @pytest.mark.asyncio
    async def test_exists(self, adapter, sample_state):
        """Check checkpoint existence."""
        assert not await adapter.exists("thread-1")

        await adapter.save(sample_state, "thread-1", "cp-1")

        assert await adapter.exists("thread-1")
        assert await adapter.exists("thread-1", "cp-1")
        assert not await adapter.exists("thread-1", "cp-nonexistent")


# =============================================================================
# Factory Function Tests
# =============================================================================


# =============================================================================
# Agent Integration Tests
# =============================================================================


class TestAgentWithCheckpointer:
    """Test Agent with various checkpointer backends."""

    @pytest.mark.asyncio
    async def test_agent_with_memory_checkpointer(self):
        """Agent with MemoryCheckpointer."""
        from unittest.mock import AsyncMock, MagicMock

        from tulip import Agent
        from tulip.models.base import ModelResponse

        # Create mock model
        mock_model = MagicMock()
        mock_response = ModelResponse(
            message=Message(
                role=Role.ASSISTANT,
                content="The answer is 4.",
                tool_calls=[],
            ),
            usage={"total_tokens": 100},
            raw={},
        )
        mock_model.complete = AsyncMock(return_value=mock_response)

        # Create checkpointer
        checkpointer = MemoryCheckpointer()

        # Create agent
        agent = Agent(
            model=mock_model,
            system_prompt="You are helpful.",
            checkpointer=checkpointer,
            max_iterations=5,
        )

        # Run agent
        result = agent.run_sync("What is 2+2?", thread_id="test-thread")

        assert result.success
        assert "4" in result.message

        # Check state was saved
        assert await checkpointer.exists("test-thread")
        loaded = await checkpointer.load("test-thread")
        assert loaded is not None

    @pytest.mark.asyncio
    async def test_agent_resumes_from_checkpoint(self):
        """Agent resumes conversation from checkpoint."""
        from unittest.mock import AsyncMock, MagicMock

        from tulip import Agent
        from tulip.models.base import ModelResponse

        checkpointer = MemoryCheckpointer()

        def create_mock_model(response_text: str):
            mock_model = MagicMock()
            mock_response = ModelResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content=response_text,
                    tool_calls=[],
                ),
                usage={"total_tokens": 50},
                raw={},
            )
            mock_model.complete = AsyncMock(return_value=mock_response)
            return mock_model

        # First conversation
        agent1 = Agent(
            model=create_mock_model("Hello! My name is Assistant."),
            system_prompt="You are helpful.",
            checkpointer=checkpointer,
            max_iterations=5,
        )
        result1 = agent1.run_sync("Hi, what's your name?", thread_id="resume-thread")
        assert "Assistant" in result1.message

        # Verify checkpoint was saved
        assert await checkpointer.exists("resume-thread")

        # Load the checkpoint and verify state
        loaded_state = await checkpointer.load("resume-thread")
        assert loaded_state is not None
        # Should have: system, user ("Hi..."), assistant ("Hello!...")
        assert len(loaded_state.messages) >= 2

    @pytest.mark.asyncio
    async def test_agent_with_auto_checkpoint(self):
        """Agent with auto-checkpoint every N iterations."""
        from unittest.mock import AsyncMock, MagicMock

        from tulip import Agent
        from tulip.core.messages import ToolCall
        from tulip.tools import tool

        checkpointer = MemoryCheckpointer()

        # Create mock model that makes tool calls
        mock_model = MagicMock()
        call_count = [0]

        def make_response():
            call_count[0] += 1
            if call_count[0] <= 2:
                # First two calls: use a tool
                msg = Message(
                    role=Role.ASSISTANT,
                    content="Let me calculate that.",
                    tool_calls=[
                        ToolCall(id=f"call-{call_count[0]}", name="add", arguments={"a": 1, "b": 2})
                    ],
                )
            else:
                # Final call: return result
                msg = Message(
                    role=Role.ASSISTANT,
                    content="The answer is 3.",
                )
            mock_response = MagicMock()
            mock_response.message = msg
            mock_response.usage = {"total_tokens": 50}
            return mock_response

        mock_model.complete = AsyncMock(side_effect=lambda **kwargs: make_response())

        @tool
        async def add(a: int, b: int) -> str:
            """Add two numbers."""
            return str(a + b)

        # Create agent with auto-checkpoint
        agent = Agent(
            model=mock_model,
            tools=[add],
            system_prompt="You are a calculator.",
            checkpointer=checkpointer,
            checkpoint_every_n_iterations=1,  # Checkpoint every iteration
            max_iterations=5,
        )

        # Run agent
        events = []
        async for event in agent.run("Add 1 and 2", thread_id="auto-thread"):
            events.append(event)

        # Verify checkpoint exists
        assert await checkpointer.exists("auto-thread")


# =============================================================================
# Redis Backend Tests (requires Redis)
# =============================================================================


@pytest.mark.requires_redis
class TestRedisAdapter:
    """Test Redis checkpointer adapter."""

    @pytest.fixture
    async def adapter(self):
        from tulip.memory.backends import redis_checkpointer

        adapter = redis_checkpointer(
            url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            prefix="tulip:test:adapter:",
        )
        yield adapter
        # Cleanup
        await adapter.delete("redis-thread")
        await adapter.close()

    @pytest.mark.asyncio
    async def test_redis_adapter_roundtrip(self, adapter, sample_state):
        """Redis adapter save/load roundtrip."""
        checkpoint_id = await adapter.save(sample_state, "redis-thread")
        assert checkpoint_id is not None

        loaded = await adapter.load("redis-thread")
        assert loaded is not None
        assert loaded.agent_id == sample_state.agent_id


# =============================================================================
# PostgreSQL Backend Tests (requires PostgreSQL)
# =============================================================================


@pytest.mark.requires_postgres
class TestPostgreSQLAdapter:
    """Test PostgreSQL checkpointer adapter."""

    @pytest.fixture
    async def adapter(self):
        from tulip.memory.backends import postgresql_checkpointer

        adapter = postgresql_checkpointer(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "tulip_test"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )
        yield adapter
        await adapter.delete("pg-thread")
        await adapter.close()

    @pytest.mark.asyncio
    async def test_postgresql_adapter_roundtrip(self, adapter, sample_state):
        """PostgreSQL adapter save/load roundtrip."""
        checkpoint_id = await adapter.save(sample_state, "pg-thread")
        assert checkpoint_id is not None

        loaded = await adapter.load("pg-thread")
        assert loaded is not None
        assert loaded.agent_id == sample_state.agent_id


# =============================================================================
# MySQL Backend Tests (requires MySQL)
# =============================================================================


@pytest.mark.requires_mysql
class TestMySQLAdapter:
    """Test MySQL checkpointer adapter."""

    def _mysql_adapter(self, table_name: str):
        from tulip.memory.backends import mysql_checkpointer

        return mysql_checkpointer(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            database=os.getenv("MYSQL_DB", "tulip_test"),
            user=os.getenv("MYSQL_USER", "tulip"),
            password=os.getenv("MYSQL_PASSWORD", "tulip"),
            table_name=table_name,
        )

    @pytest.fixture
    async def adapter(self):
        adapter = self._mysql_adapter("test_adapter_checkpoints")
        yield adapter
        await adapter.delete("mysql-thread")
        await adapter.close()

    async def _drop_adapter_table(self, adapter) -> None:
        backend = adapter._backend
        pool = await backend._get_pool()
        async with pool.acquire() as conn:
            async with await conn.cursor() as cur:
                await cur.execute(f"DROP TABLE IF EXISTS {backend._quoted_table_name}")  # noqa: S608
            await conn.commit()

    @pytest.mark.asyncio
    async def test_mysql_adapter_roundtrip(self, adapter, sample_state):
        """MySQL adapter save/load roundtrip."""
        checkpoint_id = await adapter.save(sample_state, "mysql-thread")
        assert checkpoint_id is not None

        loaded = await adapter.load("mysql-thread")
        assert loaded is not None
        assert loaded.agent_id == sample_state.agent_id

    @pytest.mark.asyncio
    async def test_mysql_adapter_concurrent_same_thread_cold_start(self, sample_state):
        """Concurrent first use creates a fresh table/index without duplicate-index races."""
        adapter = self._mysql_adapter(f"test_adapter_cold_start_{uuid4().hex[:12]}")
        thread_id = "mysql-cold-start-thread"

        async def save_checkpoint(i: int) -> str:
            state = sample_state.model_copy(update={"agent_id": f"{sample_state.agent_id}-{i}"})
            return await adapter.save(state, thread_id)

        try:
            checkpoint_ids = await asyncio.gather(*(save_checkpoint(i) for i in range(20)))

            assert len(set(checkpoint_ids)) == 20
            assert await adapter.load(thread_id) is not None
        finally:
            await self._drop_adapter_table(adapter)
            await adapter.close()


# =============================================================================
# OpenSearch Backend Tests (requires OpenSearch)
# =============================================================================


@pytest.mark.requires_opensearch
class TestOpenSearchAdapter:
    """Test OpenSearch checkpointer adapter."""

    @pytest.fixture
    async def adapter(self):
        from tulip.memory.backends import opensearch_checkpointer

        hosts_env = os.getenv("OPENSEARCH_HOSTS") or os.getenv("OPENSEARCH_HOST", "localhost:9200")
        hosts = [h.strip() for h in hosts_env.split(",")]
        adapter = opensearch_checkpointer(
            hosts=hosts,
            index_name="tulip-test-adapter",
            username=os.getenv("OPENSEARCH_USER"),
            password=os.getenv("OPENSEARCH_PASSWORD"),
            use_ssl=os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true",
            verify_certs=os.getenv("OPENSEARCH_VERIFY_CERTS", "true").lower() == "true",
        )
        yield adapter
        await adapter.delete("os-thread")
        await adapter.close()

    @pytest.mark.asyncio
    async def test_opensearch_adapter_roundtrip(self, adapter, sample_state):
        """OpenSearch adapter save/load roundtrip."""
        checkpoint_id = await adapter.save(sample_state, "os-thread")
        assert checkpoint_id is not None

        await asyncio.sleep(1)  # Wait for indexing

        loaded = await adapter.load("os-thread")
        assert loaded is not None
        assert loaded.agent_id == sample_state.agent_id
