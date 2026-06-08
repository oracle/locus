# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Integration tests for checkpoint backends."""

from __future__ import annotations

import asyncio
import os

import pytest

from tulip.core.messages import Message, Role
from tulip.core.state import AgentState


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


@pytest.fixture
def sample_data(sample_state: AgentState) -> dict:
    """Convert state to checkpoint data."""
    return sample_state.to_checkpoint()


# =============================================================================
# MemoryCheckpointer Tests
# =============================================================================


class TestMemoryCheckpointer:
    """Test in-memory checkpoint backend."""

    @pytest.fixture
    def backend(self):
        from tulip.memory.backends import MemoryCheckpointer

        return MemoryCheckpointer()

    @pytest.mark.asyncio
    async def test_save_and_load(self, backend, sample_state):
        """Save and load state."""
        checkpoint_id = await backend.save(sample_state, "thread-1")
        assert checkpoint_id is not None

        loaded = await backend.load("thread-1")
        assert loaded is not None
        assert loaded.agent_id == sample_state.agent_id
        assert len(loaded.messages) == len(sample_state.messages)

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, backend, sample_state):
        """List available checkpoints."""
        await backend.save(sample_state, "thread-1", "cp-1")
        await backend.save(sample_state, "thread-1", "cp-2")
        await backend.save(sample_state, "thread-1", "cp-3")

        checkpoints = await backend.list_checkpoints("thread-1")
        assert len(checkpoints) == 3

    @pytest.mark.asyncio
    async def test_delete(self, backend, sample_state):
        """Delete checkpoints."""
        await backend.save(sample_state, "thread-1")

        assert await backend.exists("thread-1")

        deleted = await backend.delete("thread-1")
        assert deleted

        assert not await backend.exists("thread-1")

    @pytest.mark.asyncio
    async def test_multiple_threads(self, backend, sample_state):
        """Handle multiple threads."""
        await backend.save(sample_state, "thread-1")
        await backend.save(sample_state.with_confidence(0.8), "thread-2")

        state1 = await backend.load("thread-1")
        state2 = await backend.load("thread-2")

        assert state1.confidence == 0.5
        assert state2.confidence == 0.8


# =============================================================================
# RedisBackend Tests (requires Redis)
# =============================================================================


@pytest.mark.requires_redis
class TestRedisBackend:
    """Test Redis checkpoint backend."""

    @pytest.fixture
    async def backend(self):
        from tulip.memory.backends import RedisBackend

        backend = RedisBackend(
            url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            prefix="tulip:test:",
        )
        yield backend
        # Cleanup
        threads = await backend.list_threads()
        for t in threads:
            await backend.delete(t)
        await backend.close()

    @pytest.mark.asyncio
    async def test_save_and_load(self, backend, sample_data):
        """Save and load data."""
        await backend.save("thread-1", sample_data)

        loaded = await backend.load("thread-1")
        assert loaded is not None
        assert loaded["agent_id"] == sample_data["agent_id"]

    @pytest.mark.asyncio
    async def test_exists(self, backend, sample_data):
        """Check existence."""
        assert not await backend.exists("thread-1")

        await backend.save("thread-1", sample_data)

        assert await backend.exists("thread-1")

    @pytest.mark.asyncio
    async def test_delete(self, backend, sample_data):
        """Delete checkpoint."""
        await backend.save("thread-1", sample_data)
        assert await backend.exists("thread-1")

        deleted = await backend.delete("thread-1")
        assert deleted

        assert not await backend.exists("thread-1")

    @pytest.mark.asyncio
    async def test_list_threads(self, backend, sample_data):
        """List thread IDs."""
        await backend.save("test-thread-1", sample_data)
        await backend.save("test-thread-2", sample_data)

        threads = await backend.list_threads(pattern="test-*")
        assert len(threads) >= 2


# =============================================================================
# PostgreSQLBackend Tests (requires PostgreSQL)
# =============================================================================


@pytest.mark.requires_postgres
class TestPostgreSQLBackend:
    """Test PostgreSQL checkpoint backend."""

    @pytest.fixture
    async def backend(self):
        from tulip.memory.backends import PostgreSQLBackend

        backend = PostgreSQLBackend(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "tulip_test"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            table_name="test_checkpoints",
        )
        yield backend
        # Cleanup
        threads = await backend.list_threads()
        for t in threads:
            await backend.delete(t)
        await backend.close()

    @pytest.mark.asyncio
    async def test_save_and_load(self, backend, sample_data):
        """Save and load data."""
        checkpoint_id = await backend.save("thread-1", sample_data)
        assert checkpoint_id is not None

        loaded = await backend.load("thread-1")
        assert loaded is not None
        assert loaded["agent_id"] == sample_data["agent_id"]

    @pytest.mark.asyncio
    async def test_metadata_storage(self, backend, sample_data):
        """Save and query by metadata."""
        await backend.save(
            "thread-1",
            sample_data,
            metadata={"user_id": "user-123", "session": "abc"},
        )
        await backend.save(
            "thread-2",
            sample_data,
            metadata={"user_id": "user-123", "session": "def"},
        )
        await backend.save(
            "thread-3",
            sample_data,
            metadata={"user_id": "user-456", "session": "ghi"},
        )

        results = await backend.query_by_metadata("user_id", "user-123")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_data(self, backend, sample_data):
        """Search by data field."""
        await backend.save("thread-1", sample_data)

        modified = {**sample_data, "agent_id": "special-agent"}
        await backend.save("thread-2", modified)

        results = await backend.search_data("agent_id", "special-agent")
        assert len(results) == 1
        assert results[0]["thread_id"] == "thread-2"

    @pytest.mark.asyncio
    async def test_count(self, backend, sample_data):
        """Count checkpoints."""
        await backend.save("thread-1", sample_data)
        await backend.save("thread-2", sample_data)

        count = await backend.count()
        assert count >= 2


# =============================================================================
# MySQLBackend Tests (requires MySQL)
# =============================================================================


@pytest.mark.requires_mysql
class TestMySQLBackend:
    """Test MySQL checkpoint backend."""

    @pytest.fixture
    async def backend(self):
        from tulip.memory.backends import MySQLBackend

        backend = MySQLBackend(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            database=os.getenv("MYSQL_DB", "tulip_test"),
            user=os.getenv("MYSQL_USER", "tulip"),
            password=os.getenv("MYSQL_PASSWORD", "tulip"),
            table_name="test_checkpoints",
            min_pool_size=0,
            max_pool_size=5,
        )
        yield backend
        # Cleanup
        threads = await backend.list_threads()
        for t in threads:
            await backend.delete(t)
        await backend.close()

    @pytest.mark.asyncio
    async def test_save_and_load(self, backend, sample_data):
        """Save and load data."""
        checkpoint_id = await backend.save("thread-1", sample_data)
        assert checkpoint_id is not None

        loaded = await backend.load("thread-1")
        assert loaded is not None
        assert loaded["agent_id"] == sample_data["agent_id"]

    @pytest.mark.asyncio
    async def test_metadata_storage(self, backend, sample_data):
        """Save and query by metadata."""
        await backend.save(
            "thread-1",
            sample_data,
            metadata={"user_id": "user-123", "session": "abc"},
        )
        await backend.save(
            "thread-2",
            sample_data,
            metadata={"user_id": "user-123", "session": "def"},
        )
        await backend.save(
            "thread-3",
            sample_data,
            metadata={"user_id": "user-456", "session": "ghi"},
        )

        results = await backend.query_by_metadata("user_id", "user-123")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_data(self, backend, sample_data):
        """Search by data field."""
        await backend.save("thread-1", sample_data)

        modified = {**sample_data, "agent_id": "special-agent"}
        await backend.save("thread-2", modified)

        results = await backend.search_data("agent_id", "special-agent")
        assert len(results) == 1
        assert results[0]["thread_id"] == "thread-2"

    @pytest.mark.asyncio
    async def test_count(self, backend, sample_data):
        """Count checkpoints."""
        await backend.save("thread-1", sample_data)
        await backend.save("thread-2", sample_data)

        count = await backend.count()
        assert count >= 2

    @pytest.mark.asyncio
    async def test_delete(self, backend, sample_data):
        """Delete saved checkpoint."""
        await backend.save("delete-thread", sample_data)

        assert await backend.exists("delete-thread")
        assert await backend.delete("delete-thread")
        assert not await backend.exists("delete-thread")
        assert await backend.load("delete-thread") is None

    @pytest.mark.asyncio
    async def test_list_threads(self, backend, sample_data):
        """List saved thread IDs."""
        await backend.save("mysql-list-1", sample_data)
        await backend.save("mysql-list-2", sample_data)

        threads = await backend.list_threads(pattern="mysql-list-%")

        assert {"mysql-list-1", "mysql-list-2"}.issubset(set(threads))

    @pytest.mark.asyncio
    async def test_vacuum(self, backend, sample_data):
        """Remove stale checkpoints."""
        await backend.save("vacuum-thread", sample_data)

        deleted = await backend.vacuum(older_than_days=-1)

        assert deleted >= 1
        assert await backend.load("vacuum-thread") is None

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, backend, sample_data):
        """Handle concurrent writes across multiple threads."""
        thread_ids = [f"mysql-concurrent-{i}" for i in range(20)]

        async def save_thread(thread_id: str) -> str:
            payload = {**sample_data, "agent_id": thread_id}
            return await backend.save(thread_id, payload)

        checkpoint_ids = await asyncio.gather(*(save_thread(t) for t in thread_ids))

        assert len(set(checkpoint_ids)) == len(thread_ids)
        loaded = await asyncio.gather(*(backend.load(t) for t in thread_ids))
        assert {item["agent_id"] for item in loaded if item is not None} == set(thread_ids)

    @pytest.mark.asyncio
    async def test_concurrent_writes_same_thread_cold_start(self):
        """Concurrent saves to one thread on a brand-new table must not race
        table/index creation.

        Regression for the ``_ensure_table`` cold-start race: MySQL has no
        ``CREATE INDEX IF NOT EXISTS``, so a separate check-then-CREATE INDEX
        let concurrent first-use saves all issue ``CREATE INDEX`` and the
        losers crashed with ``1061 (Duplicate key name)``. A fresh table name
        per run guarantees the cold path is exercised.
        """
        import uuid

        from tulip.memory.backends import mysql_checkpointer

        ck = mysql_checkpointer(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            database=os.getenv("MYSQL_DB", "tulip_test"),
            user=os.getenv("MYSQL_USER", "tulip"),
            password=os.getenv("MYSQL_PASSWORD", "tulip"),
            table_name="test_cold_" + uuid.uuid4().hex[:8],
        )
        state = AgentState(agent_id="x", max_iterations=1).with_message(
            Message(role=Role.USER, content="hi")
        )
        try:
            await asyncio.gather(
                *(ck.save(state, "hot", checkpoint_id=f"cp-{i}") for i in range(20))
            )
            assert await ck.exists("hot")
            assert await ck.load("hot") is not None
        finally:
            await ck.delete("hot")
            await ck.close()


# =============================================================================
# OpenSearchBackend Tests (requires OpenSearch)
# =============================================================================


@pytest.mark.requires_opensearch
class TestOpenSearchBackend:
    """Test OpenSearch checkpoint backend."""

    @pytest.fixture
    async def backend(self):
        from tulip.memory.backends import OpenSearchBackend

        hosts_env = os.getenv("OPENSEARCH_HOSTS") or os.getenv("OPENSEARCH_HOST", "localhost:9200")
        hosts = [h.strip() for h in hosts_env.split(",")]
        backend = OpenSearchBackend(
            hosts=hosts,
            index_name="tulip-test-checkpoints",
            username=os.getenv("OPENSEARCH_USER"),
            password=os.getenv("OPENSEARCH_PASSWORD"),
            use_ssl=os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true",
            verify_certs=os.getenv("OPENSEARCH_VERIFY_CERTS", "true").lower() == "true",
        )
        yield backend
        # Cleanup
        threads = await backend.list_threads()
        for t in threads:
            await backend.delete(t)
        await backend.close()

    @pytest.mark.asyncio
    async def test_save_and_load(self, backend, sample_data):
        """Save and load data."""
        await backend.save("thread-1", sample_data)

        loaded = await backend.load("thread-1")
        assert loaded is not None
        assert loaded["agent_id"] == sample_data["agent_id"]

    @pytest.mark.asyncio
    async def test_search(self, backend, sample_data):
        """Full-text search."""
        await backend.save("thread-1", sample_data)

        # Give OpenSearch time to index
        await asyncio.sleep(1)

        results = await backend.search("test-agent")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_metadata_query(self, backend, sample_data):
        """Query by metadata."""
        await backend.save(
            "thread-1",
            sample_data,
            metadata={"category": "support"},
        )
        await backend.save(
            "thread-2",
            sample_data,
            metadata={"category": "sales"},
        )

        await asyncio.sleep(1)

        results = await backend.get_by_metadata("category", "support")
        assert len(results) >= 1


# =============================================================================
# Cross-Backend Compatibility Tests
# =============================================================================


class TestBackendCompatibility:
    """Test that all backends produce compatible data."""

    @pytest.mark.asyncio
    async def test_state_roundtrip_memory(self, sample_state):
        """State survives memory backend roundtrip."""
        from tulip.memory.backends import MemoryCheckpointer

        backend = MemoryCheckpointer()
        await backend.save(sample_state, "thread-1")

        loaded = await backend.load("thread-1")
        assert loaded is not None
        # Compare key fields (frozenset ordering may differ)
        assert loaded.agent_id == sample_state.agent_id
        assert loaded.confidence == sample_state.confidence
        assert len(loaded.messages) == len(sample_state.messages)
        assert set(loaded.terminal_tools) == set(sample_state.terminal_tools)
