# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for delta checkpointer."""

from datetime import UTC, datetime

import pytest

from locus.core.state import AgentState
from locus.memory.delta import (
    CheckpointMetadata,
    DeltaCheckpoint,
    DeltaCheckpointer,
    InMemoryDeltaStorage,
)


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating checkpoint metadata."""
        meta = CheckpointMetadata(
            checkpoint_id="cp1",
            thread_id="thread1",
        )
        assert meta.checkpoint_id == "cp1"
        assert meta.thread_id == "thread1"
        assert meta.parent_id is None
        assert meta.is_full is True
        assert meta.chain_depth == 0

    def test_create_delta_metadata(self):
        """Test creating delta checkpoint metadata."""
        meta = CheckpointMetadata(
            checkpoint_id="cp2",
            thread_id="thread1",
            parent_id="cp1",
            is_full=False,
            chain_depth=1,
        )
        assert meta.parent_id == "cp1"
        assert meta.is_full is False
        assert meta.chain_depth == 1

    def test_metadata_with_size(self):
        """Test metadata with size information."""
        meta = CheckpointMetadata(
            checkpoint_id="cp1",
            thread_id="thread1",
            size_bytes=1000,
            compressed_size_bytes=300,
        )
        assert meta.size_bytes == 1000
        assert meta.compressed_size_bytes == 300


class TestDeltaCheckpoint:
    """Tests for DeltaCheckpoint dataclass."""

    def test_create_checkpoint(self):
        """Test creating delta checkpoint."""
        meta = CheckpointMetadata(checkpoint_id="cp1", thread_id="t1")
        checkpoint = DeltaCheckpoint(
            metadata=meta,
            data=b"compressed data",
            is_delta=False,
        )
        assert checkpoint.metadata is meta
        assert checkpoint.data == b"compressed data"
        assert checkpoint.is_delta is False

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        meta = CheckpointMetadata(
            checkpoint_id="cp1",
            thread_id="t1",
            size_bytes=1000,
            compressed_size_bytes=200,
        )
        checkpoint = DeltaCheckpoint(metadata=meta, data=b"data")
        assert checkpoint.compression_ratio == 5.0

    def test_compression_ratio_zero_compressed(self):
        """Test compression ratio when compressed size is 0."""
        meta = CheckpointMetadata(
            checkpoint_id="cp1",
            thread_id="t1",
            size_bytes=1000,
            compressed_size_bytes=0,
        )
        checkpoint = DeltaCheckpoint(metadata=meta, data=b"")
        assert checkpoint.compression_ratio == 1.0


class TestInMemoryDeltaStorage:
    """Tests for InMemoryDeltaStorage."""

    @pytest.fixture
    def storage(self):
        """Create in-memory storage."""
        return InMemoryDeltaStorage()

    @pytest.fixture
    def sample_checkpoint(self):
        """Create sample checkpoint."""
        meta = CheckpointMetadata(
            checkpoint_id="cp1",
            thread_id="thread1",
            created_at=datetime.now(UTC),
        )
        return DeltaCheckpoint(metadata=meta, data=b"test data")

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, storage, sample_checkpoint):
        """Test storing and retrieving checkpoint."""
        await storage.store("thread1", "cp1", sample_checkpoint)
        retrieved = await storage.retrieve("thread1", "cp1")

        assert retrieved is not None
        assert retrieved.metadata.checkpoint_id == "cp1"
        assert retrieved.data == b"test data"

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent(self, storage):
        """Test retrieving nonexistent checkpoint."""
        result = await storage.retrieve("thread1", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_thread(self, storage):
        """Test retrieving from nonexistent thread."""
        result = await storage.retrieve("nonexistent", "cp1")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, storage):
        """Test listing checkpoints."""
        for i in range(3):
            meta = CheckpointMetadata(
                checkpoint_id=f"cp{i}",
                thread_id="thread1",
                created_at=datetime(2024, 1, i + 1, tzinfo=UTC),
            )
            checkpoint = DeltaCheckpoint(metadata=meta, data=b"data")
            await storage.store("thread1", f"cp{i}", checkpoint)

        result = await storage.list_checkpoints("thread1")

        assert len(result) == 3
        # Should be sorted by created_at descending
        assert result[0].checkpoint_id == "cp2"

    @pytest.mark.asyncio
    async def test_list_checkpoints_empty_thread(self, storage):
        """Test listing checkpoints for empty thread."""
        result = await storage.list_checkpoints("nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_list_checkpoints_with_limit(self, storage):
        """Test listing checkpoints with limit."""
        for i in range(10):
            meta = CheckpointMetadata(
                checkpoint_id=f"cp{i}",
                thread_id="thread1",
                created_at=datetime(2024, 1, i + 1, tzinfo=UTC),
            )
            checkpoint = DeltaCheckpoint(metadata=meta, data=b"data")
            await storage.store("thread1", f"cp{i}", checkpoint)

        result = await storage.list_checkpoints("thread1", limit=5)
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_delete_specific_checkpoint(self, storage, sample_checkpoint):
        """Test deleting specific checkpoint."""
        await storage.store("thread1", "cp1", sample_checkpoint)
        result = await storage.delete("thread1", "cp1")

        assert result is True
        retrieved = await storage.retrieve("thread1", "cp1")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_all_checkpoints(self, storage):
        """Test deleting all checkpoints for a thread."""
        meta1 = CheckpointMetadata(checkpoint_id="cp1", thread_id="thread1")
        meta2 = CheckpointMetadata(checkpoint_id="cp2", thread_id="thread1")
        await storage.store("thread1", "cp1", DeltaCheckpoint(metadata=meta1, data=b"1"))
        await storage.store("thread1", "cp2", DeltaCheckpoint(metadata=meta2, data=b"2"))

        result = await storage.delete("thread1")

        assert result is True
        assert await storage.list_checkpoints("thread1") == []

    @pytest.mark.asyncio
    async def test_delete_nonexistent_thread(self, storage):
        """Test deleting from nonexistent thread."""
        result = await storage.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent_checkpoint(self, storage, sample_checkpoint):
        """Test deleting nonexistent checkpoint."""
        await storage.store("thread1", "cp1", sample_checkpoint)
        result = await storage.delete("thread1", "nonexistent")
        assert result is False


class TestDeltaCheckpointerInit:
    """Tests for DeltaCheckpointer initialization."""

    def test_default_init(self):
        """Test default initialization."""
        cp = DeltaCheckpointer()
        assert isinstance(cp.storage, InMemoryDeltaStorage)
        assert cp.max_chain_depth == 5
        assert cp.compression_level == 6

    def test_custom_init(self):
        """Test custom initialization."""
        storage = InMemoryDeltaStorage()
        cp = DeltaCheckpointer(
            storage=storage,
            max_chain_depth=10,
            compression_level=9,
        )
        assert cp.storage is storage
        assert cp.max_chain_depth == 10
        assert cp.compression_level == 9


class TestDeltaCheckpointerSaveLoad:
    """Tests for save and load operations."""

    @pytest.fixture
    def checkpointer(self):
        """Create checkpointer."""
        return DeltaCheckpointer(max_chain_depth=3)

    @pytest.fixture
    def mock_state(self):
        """Create mock agent state."""
        state = AgentState(
            run_id="test-run",
            messages=[],
            iteration=1,
        )
        return state

    @pytest.mark.asyncio
    async def test_save_first_checkpoint(self, checkpointer, mock_state):
        """Test saving first checkpoint creates full checkpoint."""
        checkpoint_id = await checkpointer.save(mock_state, "thread1")

        assert checkpoint_id is not None
        checkpoints = await checkpointer.storage.list_checkpoints("thread1")
        assert len(checkpoints) == 1
        assert checkpoints[0].is_full is True

    @pytest.mark.asyncio
    async def test_save_with_custom_id(self, checkpointer, mock_state):
        """Test saving with custom checkpoint ID."""
        checkpoint_id = await checkpointer.save(mock_state, "thread1", checkpoint_id="my-id")

        assert checkpoint_id == "my-id"

    @pytest.mark.asyncio
    async def test_save_multiple_creates_deltas(self, checkpointer, mock_state):
        """Test saving multiple times creates delta checkpoints."""
        await checkpointer.save(mock_state, "thread1", "cp1")

        # Second save should create delta (create new state since AgentState is frozen)
        mock_state2 = AgentState(run_id="test-run-2", messages=[], iteration=2)
        await checkpointer.save(mock_state2, "thread1", "cp2")

        checkpoints = await checkpointer.storage.list_checkpoints("thread1")
        assert len(checkpoints) == 2

    @pytest.mark.asyncio
    async def test_load_latest(self, checkpointer, mock_state):
        """Test loading latest checkpoint."""
        await checkpointer.save(mock_state, "thread1")
        loaded = await checkpointer.load("thread1")

        assert loaded is not None
        assert loaded.run_id == "test-run"

    @pytest.mark.asyncio
    async def test_load_specific_checkpoint(self, checkpointer, mock_state):
        """Test loading specific checkpoint."""
        cp_id = await checkpointer.save(mock_state, "thread1", "cp1")
        loaded = await checkpointer.load("thread1", "cp1")

        assert loaded is not None
        assert loaded.run_id == "test-run"

    @pytest.mark.asyncio
    async def test_load_nonexistent_thread(self, checkpointer):
        """Test loading from nonexistent thread."""
        loaded = await checkpointer.load("nonexistent")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_load_from_cache(self, checkpointer, mock_state):
        """Test loading from cache."""
        await checkpointer.save(mock_state, "thread1", "cp1")

        # Second load should use cache
        loaded = await checkpointer.load("thread1", "cp1")
        assert loaded is not None


class TestDeltaCheckpointerListDelete:
    """Tests for list and delete operations."""

    @pytest.fixture
    def checkpointer(self):
        """Create checkpointer."""
        return DeltaCheckpointer()

    @pytest.fixture
    def mock_state(self):
        """Create mock state."""
        return AgentState(run_id="test", messages=[])

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, checkpointer, mock_state):
        """Test listing checkpoints."""
        await checkpointer.save(mock_state, "thread1", "cp1")
        await checkpointer.save(mock_state, "thread1", "cp2")

        result = await checkpointer.list_checkpoints("thread1")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_checkpoints_empty(self, checkpointer):
        """Test listing from empty thread."""
        result = await checkpointer.list_checkpoints("thread1")
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, checkpointer, mock_state):
        """Test deleting checkpoint."""
        await checkpointer.save(mock_state, "thread1", "cp1")
        result = await checkpointer.delete("thread1", "cp1")

        assert result is True
        loaded = await checkpointer.load("thread1", "cp1")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_all_checkpoints(self, checkpointer, mock_state):
        """Test deleting all checkpoints."""
        await checkpointer.save(mock_state, "thread1", "cp1")
        await checkpointer.save(mock_state, "thread1", "cp2")

        result = await checkpointer.delete("thread1")
        assert result is True

        checkpoints = await checkpointer.list_checkpoints("thread1")
        assert checkpoints == []

    def test_repr(self, checkpointer):
        """Test repr."""
        repr_str = repr(checkpointer)
        assert "DeltaCheckpointer" in repr_str
        assert "max_chain_depth=5" in repr_str
        assert "compression_level=6" in repr_str


class TestDeltaCheckpointerChainDepth:
    """Tests for chain depth limiting."""

    @pytest.mark.asyncio
    async def test_full_checkpoint_at_max_depth(self):
        """Test that full checkpoint is created at max chain depth."""
        checkpointer = DeltaCheckpointer(max_chain_depth=2)
        state = AgentState(run_id="test", messages=[])

        # Save multiple times
        await checkpointer.save(state, "thread1", "cp1")  # Full
        await checkpointer.save(state, "thread1", "cp2")  # Delta, depth 1
        await checkpointer.save(state, "thread1", "cp3")  # Delta, depth 2
        await checkpointer.save(state, "thread1", "cp4")  # Should be full again

        # Get latest checkpoint
        checkpoints = await checkpointer.storage.list_checkpoints("thread1", limit=1)
        # The checkpoint at max depth should trigger a new full checkpoint
        assert len(checkpoints) == 1
