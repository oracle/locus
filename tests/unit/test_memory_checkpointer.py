# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for in-memory checkpointer."""

import pytest

from locus.core.protocols import CheckpointerCapabilities
from locus.core.state import AgentState
from locus.memory.backends.memory import MemoryCheckpointer


class TestMemoryCheckpointerInit:
    """Tests for MemoryCheckpointer initialization."""

    def test_create_checkpointer(self):
        """Test creating a memory checkpointer."""
        checkpointer = MemoryCheckpointer()
        assert checkpointer._storage == {}

    def test_capabilities(self):
        """Test capabilities property."""
        checkpointer = MemoryCheckpointer()
        caps = checkpointer.capabilities

        assert isinstance(caps, CheckpointerCapabilities)
        assert caps.list_threads is True
        assert caps.persistent_checkpoint_ids is True
        assert caps.search is False

    def test_repr_empty(self):
        """Test repr with no checkpoints."""
        checkpointer = MemoryCheckpointer()
        repr_str = repr(checkpointer)
        assert "MemoryCheckpointer" in repr_str
        assert "threads=0" in repr_str
        assert "checkpoints=0" in repr_str


class TestMemoryCheckpointerSave:
    """Tests for save operations."""

    @pytest.fixture
    def checkpointer(self):
        """Create checkpointer for testing."""
        return MemoryCheckpointer()

    @pytest.fixture
    def state(self):
        """Create test state."""
        return AgentState()

    @pytest.mark.asyncio
    async def test_save_generates_id(self, checkpointer, state):
        """Test that save generates checkpoint ID."""
        checkpoint_id = await checkpointer.save(state, "thread1")

        assert checkpoint_id is not None
        assert len(checkpoint_id) == 32  # UUID hex

    @pytest.mark.asyncio
    async def test_save_with_specific_id(self, checkpointer, state):
        """Test saving with specific checkpoint ID."""
        checkpoint_id = await checkpointer.save(state, "thread1", checkpoint_id="my-checkpoint")

        assert checkpoint_id == "my-checkpoint"

    @pytest.mark.asyncio
    async def test_save_creates_thread(self, checkpointer, state):
        """Test that save creates thread storage."""
        await checkpointer.save(state, "new-thread")

        assert "new-thread" in checkpointer._storage

    @pytest.mark.asyncio
    async def test_save_with_metadata(self, checkpointer, state):
        """Test saving with metadata."""
        metadata = {"version": "1.0", "user": "test"}
        checkpoint_id = await checkpointer.save(state, "thread1", metadata=metadata)

        # Verify metadata was stored
        stored_data = checkpointer._storage["thread1"][checkpoint_id]
        assert stored_data[2] == metadata

    @pytest.mark.asyncio
    async def test_save_multiple_checkpoints(self, checkpointer, state):
        """Test saving multiple checkpoints to same thread."""
        id1 = await checkpointer.save(state, "thread1")
        id2 = await checkpointer.save(state, "thread1")
        id3 = await checkpointer.save(state, "thread1")

        assert len(checkpointer._storage["thread1"]) == 3
        assert id1 != id2 != id3


class TestMemoryCheckpointerLoad:
    """Tests for load operations."""

    @pytest.fixture
    def checkpointer(self):
        """Create checkpointer for testing."""
        return MemoryCheckpointer()

    @pytest.fixture
    def state(self):
        """Create test state."""
        state = AgentState()
        for _ in range(5):
            state = state.next_iteration()
        return state

    @pytest.mark.asyncio
    async def test_load_nonexistent_thread(self, checkpointer):
        """Test loading from nonexistent thread."""
        result = await checkpointer.load("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_empty_thread(self, checkpointer):
        """Test loading from empty thread."""
        checkpointer._storage["empty"] = {}
        result = await checkpointer.load("empty")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_latest_checkpoint(self, checkpointer, state):
        """Test loading latest checkpoint."""
        await checkpointer.save(state, "thread1")
        state2 = state.next_iteration()
        await checkpointer.save(state2, "thread1")

        loaded = await checkpointer.load("thread1")

        assert loaded is not None
        assert loaded.iteration == 6

    @pytest.mark.asyncio
    async def test_load_specific_checkpoint(self, checkpointer, state):
        """Test loading specific checkpoint."""
        id1 = await checkpointer.save(state, "thread1")
        state2 = state.next_iteration()
        await checkpointer.save(state2, "thread1")

        loaded = await checkpointer.load("thread1", checkpoint_id=id1)

        assert loaded is not None
        assert loaded.iteration == 5

    @pytest.mark.asyncio
    async def test_load_nonexistent_checkpoint(self, checkpointer, state):
        """Test loading nonexistent checkpoint ID."""
        await checkpointer.save(state, "thread1")

        result = await checkpointer.load("thread1", checkpoint_id="nonexistent")

        assert result is None


class TestMemoryCheckpointerListCheckpoints:
    """Tests for list_checkpoints operations."""

    @pytest.fixture
    def checkpointer(self):
        """Create checkpointer for testing."""
        return MemoryCheckpointer()

    @pytest.mark.asyncio
    async def test_list_empty_thread(self, checkpointer):
        """Test listing checkpoints for nonexistent thread."""
        result = await checkpointer.list_checkpoints("nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, checkpointer):
        """Test listing checkpoints."""
        state = AgentState()
        id1 = await checkpointer.save(state, "thread1", checkpoint_id="cp1")
        id2 = await checkpointer.save(state, "thread1", checkpoint_id="cp2")
        id3 = await checkpointer.save(state, "thread1", checkpoint_id="cp3")

        result = await checkpointer.list_checkpoints("thread1")

        assert len(result) == 3
        # Should be newest first
        assert result[0] == "cp3"

    @pytest.mark.asyncio
    async def test_list_checkpoints_with_limit(self, checkpointer):
        """Test listing checkpoints with limit."""
        state = AgentState()
        for _ in range(10):
            await checkpointer.save(state, "thread1")

        result = await checkpointer.list_checkpoints("thread1", limit=5)

        assert len(result) == 5


class TestMemoryCheckpointerDelete:
    """Tests for delete operations."""

    @pytest.fixture
    def checkpointer(self):
        """Create checkpointer for testing."""
        return MemoryCheckpointer()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_thread(self, checkpointer):
        """Test deleting from nonexistent thread."""
        result = await checkpointer.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_all_checkpoints(self, checkpointer):
        """Test deleting all checkpoints for a thread."""
        state = AgentState()
        await checkpointer.save(state, "thread1")
        await checkpointer.save(state, "thread1")

        result = await checkpointer.delete("thread1")

        assert result is True
        assert "thread1" not in checkpointer._storage

    @pytest.mark.asyncio
    async def test_delete_specific_checkpoint(self, checkpointer):
        """Test deleting a specific checkpoint."""
        state = AgentState()
        id1 = await checkpointer.save(state, "thread1")
        id2 = await checkpointer.save(state, "thread1")

        result = await checkpointer.delete("thread1", checkpoint_id=id1)

        assert result is True
        assert id1 not in checkpointer._storage["thread1"]
        assert id2 in checkpointer._storage["thread1"]

    @pytest.mark.asyncio
    async def test_delete_nonexistent_checkpoint(self, checkpointer):
        """Test deleting nonexistent checkpoint."""
        state = AgentState()
        await checkpointer.save(state, "thread1")

        result = await checkpointer.delete("thread1", checkpoint_id="nonexistent")

        assert result is False


class TestMemoryCheckpointerClear:
    """Tests for clear operations."""

    def test_clear(self):
        """Test clearing all data."""
        checkpointer = MemoryCheckpointer()
        checkpointer._storage["thread1"] = {"cp1": ({"data": "test"}, None, {})}
        checkpointer._storage["thread2"] = {"cp2": ({"data": "test"}, None, {})}

        checkpointer.clear()

        assert checkpointer._storage == {}


class TestMemoryCheckpointerThreads:
    """Tests for thread listing operations."""

    @pytest.fixture
    def checkpointer(self):
        """Create checkpointer for testing."""
        return MemoryCheckpointer()

    def test_get_thread_ids(self, checkpointer):
        """Test getting thread IDs."""
        state = AgentState()
        checkpointer._storage["thread1"] = {}
        checkpointer._storage["thread2"] = {}

        result = checkpointer.get_thread_ids()

        assert set(result) == {"thread1", "thread2"}

    @pytest.mark.asyncio
    async def test_list_threads(self, checkpointer):
        """Test listing threads."""
        state = AgentState()
        await checkpointer.save(state, "thread-a")
        await checkpointer.save(state, "thread-b")
        await checkpointer.save(state, "other")

        result = await checkpointer.list_threads()

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_list_threads_with_limit(self, checkpointer):
        """Test listing threads with limit."""
        state = AgentState()
        for i in range(10):
            await checkpointer.save(state, f"thread-{i}")

        result = await checkpointer.list_threads(limit=5)

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_list_threads_with_pattern(self, checkpointer):
        """Test listing threads with pattern filter."""
        state = AgentState()
        await checkpointer.save(state, "user-1")
        await checkpointer.save(state, "user-2")
        await checkpointer.save(state, "session-1")

        result = await checkpointer.list_threads(pattern="user-*")

        assert len(result) == 2
        assert all(t.startswith("user-") for t in result)


class TestMemoryCheckpointerCount:
    """Tests for checkpoint counting."""

    @pytest.fixture
    def checkpointer(self):
        """Create checkpointer for testing."""
        return MemoryCheckpointer()

    def test_count_all_checkpoints(self, checkpointer):
        """Test counting all checkpoints."""
        checkpointer._storage["t1"] = {"cp1": None, "cp2": None}
        checkpointer._storage["t2"] = {"cp3": None}

        count = checkpointer.get_checkpoint_count()

        assert count == 3

    def test_count_thread_checkpoints(self, checkpointer):
        """Test counting checkpoints for specific thread."""
        checkpointer._storage["t1"] = {"cp1": None, "cp2": None}
        checkpointer._storage["t2"] = {"cp3": None}

        count = checkpointer.get_checkpoint_count("t1")

        assert count == 2

    def test_count_nonexistent_thread(self, checkpointer):
        """Test counting checkpoints for nonexistent thread."""
        count = checkpointer.get_checkpoint_count("nonexistent")
        assert count == 0

    def test_repr_with_data(self, checkpointer):
        """Test repr with checkpoints."""
        checkpointer._storage["t1"] = {"cp1": None, "cp2": None}
        checkpointer._storage["t2"] = {"cp3": None}

        repr_str = repr(checkpointer)

        assert "threads=2" in repr_str
        assert "checkpoints=3" in repr_str
