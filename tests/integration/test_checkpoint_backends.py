# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for checkpoint backends."""

from __future__ import annotations

import asyncio
import os

import pytest

from locus.core.messages import Message, Role
from locus.core.state import AgentState


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
        from locus.memory.backends import MemoryCheckpointer

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
        from locus.memory.backends import RedisBackend

        backend = RedisBackend(
            url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            prefix="locus:test:",
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
        from locus.memory.backends import PostgreSQLBackend

        backend = PostgreSQLBackend(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "locus_test"),
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
        from locus.memory.backends import MySQLBackend

        backend = MySQLBackend(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            database=os.getenv("MYSQL_DB", "locus_test"),
            user=os.getenv("MYSQL_USER", "locus"),
            password=os.getenv("MYSQL_PASSWORD", "locus"),
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

        from locus.memory.backends import mysql_checkpointer

        ck = mysql_checkpointer(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            database=os.getenv("MYSQL_DB", "locus_test"),
            user=os.getenv("MYSQL_USER", "locus"),
            password=os.getenv("MYSQL_PASSWORD", "locus"),
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
        from locus.memory.backends import OpenSearchBackend

        hosts_env = os.getenv("OPENSEARCH_HOSTS") or os.getenv("OPENSEARCH_HOST", "localhost:9200")
        hosts = [h.strip() for h in hosts_env.split(",")]
        backend = OpenSearchBackend(
            hosts=hosts,
            index_name="locus-test-checkpoints",
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
# OCIBucketBackend Tests (requires OCI)
# =============================================================================


@pytest.mark.requires_oci_bucket
class TestOCIBucketBackend:
    """Test OCI Object Storage checkpointer (native BaseCheckpointer)."""

    @pytest.fixture
    async def backend(self, oci_bucket_config):
        from locus.memory.backends import OCIBucketBackend

        backend = OCIBucketBackend(
            bucket_name=oci_bucket_config["bucket_name"],
            namespace=oci_bucket_config["namespace"],
            prefix=f"{oci_bucket_config['prefix']}checkpoints/",
            profile_name=oci_bucket_config["profile_name"],
            auth_type=oci_bucket_config["auth_type"],
            region=oci_bucket_config["region"],
        )
        yield backend
        # Cleanup every thread the test produced.
        threads = await backend.list_threads(limit=1000)
        for t in threads:
            await backend.delete(t)

    @pytest.mark.asyncio
    async def test_save_and_load(self, backend, sample_state):
        """Round-trip an AgentState through the native interface."""
        checkpoint_id = await backend.save(sample_state, "thread-1")
        assert checkpoint_id

        loaded = await backend.load("thread-1")
        assert loaded is not None
        assert loaded.agent_id == sample_state.agent_id
        assert len(loaded.messages) == len(sample_state.messages)

    @pytest.mark.asyncio
    async def test_exists(self, backend, sample_state):
        """``exists`` follows the ``_latest`` pointer."""
        assert not await backend.exists("thread-nonexistent")

        await backend.save(sample_state, "thread-1")
        assert await backend.exists("thread-1")

    @pytest.mark.asyncio
    async def test_list_checkpoints_newest_first(self, backend, sample_state):
        """Multiple checkpoints per thread are listed newest-first."""
        cp1 = await backend.save(sample_state, "thread-1")
        await asyncio.sleep(1.1)  # Object Storage timestamps are second-granularity.
        cp2 = await backend.save(sample_state, "thread-1")

        checkpoints = await backend.list_checkpoints("thread-1")
        assert checkpoints[:2] == [cp2, cp1]

    @pytest.mark.asyncio
    async def test_load_specific_checkpoint(self, backend, sample_state):
        """Loading by checkpoint_id returns that exact checkpoint."""
        cp1 = await backend.save(sample_state, "thread-1")
        other_state = sample_state.with_message(Message(role=Role.USER, content="second turn"))
        await backend.save(other_state, "thread-1")

        loaded_first = await backend.load("thread-1", cp1)
        assert loaded_first is not None
        assert len(loaded_first.messages) == len(sample_state.messages)

    @pytest.mark.asyncio
    async def test_list_threads(self, backend, sample_state):
        """List thread IDs via prefix-delimiter listing."""
        await backend.save(sample_state, "oci-thread-a")
        await backend.save(sample_state, "oci-thread-b")

        threads = await backend.list_threads()
        assert "oci-thread-a" in threads
        assert "oci-thread-b" in threads

    @pytest.mark.asyncio
    async def test_metadata(self, backend, sample_state):
        """Metadata persists alongside the checkpoint."""
        await backend.save(sample_state, "thread-1", metadata={"user": "test-user"})

        meta = await backend.get_metadata("thread-1")
        assert meta is not None
        assert meta["metadata"]["user"] == "test-user"

    @pytest.mark.asyncio
    async def test_list_with_metadata(self, backend, sample_state):
        """``list_with_metadata`` returns per-thread latest metadata."""
        await backend.save(sample_state, "thread-1", metadata={"priority": "high"})
        await backend.save(sample_state, "thread-2", metadata={"priority": "low"})

        results = await backend.list_with_metadata()
        by_thread = {r["thread_id"]: r["metadata"].get("priority") for r in results}
        assert by_thread.get("thread-1") == "high"
        assert by_thread.get("thread-2") == "low"

    @pytest.mark.asyncio
    async def test_delete_single_checkpoint(self, backend, sample_state):
        """Deleting a specific checkpoint leaves siblings intact."""
        cp1 = await backend.save(sample_state, "thread-1")
        cp2 = await backend.save(sample_state, "thread-1")

        await backend.delete("thread-1", cp1)
        remaining = await backend.list_checkpoints("thread-1")
        assert cp1 not in remaining
        assert cp2 in remaining

    @pytest.mark.asyncio
    async def test_delete_entire_thread(self, backend, sample_state):
        """Deleting without a checkpoint_id wipes the whole thread."""
        await backend.save(sample_state, "thread-1")
        await backend.save(sample_state, "thread-1")

        assert await backend.delete("thread-1")
        assert not await backend.exists("thread-1")
        assert await backend.list_checkpoints("thread-1") == []

    @pytest.mark.asyncio
    async def test_copy_thread_branching(self, backend, sample_state):
        """Branching: source checkpoints are copied to dest."""
        await backend.save(sample_state, "source-thread")
        await backend.save(sample_state, "source-thread")

        assert await backend.copy_thread("source-thread", "dest-thread")
        source_ids = set(await backend.list_checkpoints("source-thread"))
        dest_ids = set(await backend.list_checkpoints("dest-thread"))
        assert source_ids == dest_ids
        assert await backend.exists("dest-thread")

    @pytest.mark.asyncio
    async def test_capabilities_advertised(self, backend):
        """The backend advertises the capabilities it implements."""
        caps = backend.capabilities
        assert caps.list_threads
        assert caps.list_with_metadata
        assert caps.metadata_query
        assert caps.branching
        assert caps.vacuum
        assert caps.persistent_checkpoint_ids


# =============================================================================
# Cross-Backend Compatibility Tests
# =============================================================================


class TestBackendCompatibility:
    """Test that all backends produce compatible data."""

    @pytest.mark.asyncio
    async def test_state_roundtrip_memory(self, sample_state):
        """State survives memory backend roundtrip."""
        from locus.memory.backends import MemoryCheckpointer

        backend = MemoryCheckpointer()
        await backend.save(sample_state, "thread-1")

        loaded = await backend.load("thread-1")
        assert loaded is not None
        # Compare key fields (frozenset ordering may differ)
        assert loaded.agent_id == sample_state.agent_id
        assert loaded.confidence == sample_state.confidence
        assert len(loaded.messages) == len(sample_state.messages)
        assert set(loaded.terminal_tools) == set(sample_state.terminal_tools)
