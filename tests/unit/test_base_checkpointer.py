# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for base checkpointer module."""

from unittest.mock import MagicMock

import pytest

from locus.core.protocols import CheckpointerCapabilities
from locus.memory.checkpointer import BaseCheckpointer


class MinimalCheckpointer(BaseCheckpointer):
    """Minimal implementation using base class defaults."""

    async def save(self, state, thread_id, checkpoint_id=None, metadata=None):
        return "cp1"

    async def load(self, thread_id, checkpoint_id=None):
        return None

    async def list_checkpoints(self, thread_id, limit=10):
        return []


class MockCheckpointer(BaseCheckpointer):
    """Concrete implementation for testing."""

    def __init__(self, **kwargs):
        self._capabilities = CheckpointerCapabilities(**kwargs)
        self.saved_states = {}
        self.checkpoints = {}

    @property
    def capabilities(self):
        return self._capabilities

    async def save(self, state, thread_id, checkpoint_id=None, metadata=None):
        checkpoint_id = checkpoint_id or "cp1"
        self.saved_states[(thread_id, checkpoint_id)] = state
        if thread_id not in self.checkpoints:
            self.checkpoints[thread_id] = []
        self.checkpoints[thread_id].append(checkpoint_id)
        return checkpoint_id

    async def load(self, thread_id, checkpoint_id=None):
        if checkpoint_id is None:
            cps = self.checkpoints.get(thread_id, [])
            if not cps:
                return None
            checkpoint_id = cps[-1]
        return self.saved_states.get((thread_id, checkpoint_id))

    async def list_checkpoints(self, thread_id, limit=10):
        cps = self.checkpoints.get(thread_id, [])
        return cps[:limit]


class TestCheckpointerCapabilities:
    """Tests for CheckpointerCapabilities."""

    def test_default_capabilities(self):
        """Test default capabilities are all False."""
        caps = CheckpointerCapabilities()
        assert caps.search is False
        assert caps.metadata_query is False
        assert caps.vacuum is False
        assert caps.branching is False
        assert caps.ttl is False
        assert caps.list_threads is False

    def test_base_checkpointer_default_capabilities(self):
        """Test BaseCheckpointer default capabilities property."""
        cp = MinimalCheckpointer()
        caps = cp.capabilities
        assert isinstance(caps, CheckpointerCapabilities)
        assert caps.search is False

    def test_custom_capabilities(self):
        """Test setting custom capabilities."""
        caps = CheckpointerCapabilities(
            search=True,
            metadata_query=True,
            list_threads=True,
        )
        assert caps.search is True
        assert caps.metadata_query is True
        assert caps.list_threads is True
        assert caps.vacuum is False


class TestBaseCheckpointer:
    """Tests for BaseCheckpointer."""

    @pytest.fixture
    def checkpointer(self):
        """Create a mock checkpointer."""
        return MockCheckpointer()

    @pytest.mark.asyncio
    async def test_save_and_load(self, checkpointer):
        """Test basic save and load."""
        mock_state = MagicMock()
        cp_id = await checkpointer.save(mock_state, "thread1")

        loaded = await checkpointer.load("thread1", cp_id)
        assert loaded is mock_state

    @pytest.mark.asyncio
    async def test_load_latest(self, checkpointer):
        """Test loading latest checkpoint."""
        state1 = MagicMock()
        state2 = MagicMock()

        await checkpointer.save(state1, "thread1", "cp1")
        await checkpointer.save(state2, "thread1", "cp2")

        loaded = await checkpointer.load("thread1")  # No checkpoint_id
        assert loaded is state2

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, checkpointer):
        """Test loading nonexistent thread."""
        loaded = await checkpointer.load("nonexistent")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, checkpointer):
        """Test listing checkpoints."""
        await checkpointer.save(MagicMock(), "thread1", "cp1")
        await checkpointer.save(MagicMock(), "thread1", "cp2")

        cps = await checkpointer.list_checkpoints("thread1")
        assert len(cps) == 2
        assert "cp1" in cps
        assert "cp2" in cps

    @pytest.mark.asyncio
    async def test_exists_with_checkpoint(self, checkpointer):
        """Test exists with specific checkpoint."""
        state = MagicMock()
        await checkpointer.save(state, "thread1", "cp1")

        exists = await checkpointer.exists("thread1", "cp1")
        assert exists is True

        exists = await checkpointer.exists("thread1", "nonexistent")
        assert exists is False

    @pytest.mark.asyncio
    async def test_exists_without_checkpoint(self, checkpointer):
        """Test exists without specific checkpoint."""
        await checkpointer.save(MagicMock(), "thread1", "cp1")

        exists = await checkpointer.exists("thread1")
        assert exists is True

        exists = await checkpointer.exists("nonexistent")
        assert exists is False

    @pytest.mark.asyncio
    async def test_delete_not_implemented(self, checkpointer):
        """Test delete raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await checkpointer.delete("thread1")

    @pytest.mark.asyncio
    async def test_close(self, checkpointer):
        """Test close does nothing by default."""
        await checkpointer.close()  # Should not raise

    def test_repr(self, checkpointer):
        """Test string representation."""
        repr_str = repr(checkpointer)
        assert "MockCheckpointer" in repr_str

    @pytest.mark.asyncio
    async def test_search_without_capability(self, checkpointer):
        """Test search without capability raises error."""
        with pytest.raises(NotImplementedError, match="does not support"):
            await checkpointer.search("query")

    @pytest.mark.asyncio
    async def test_query_by_metadata_without_capability(self, checkpointer):
        """Test query_by_metadata without capability raises error."""
        with pytest.raises(NotImplementedError):
            await checkpointer.query_by_metadata("key", "value")

    @pytest.mark.asyncio
    async def test_get_metadata_without_capability(self, checkpointer):
        """Test get_metadata without capability raises error."""
        with pytest.raises(NotImplementedError):
            await checkpointer.get_metadata("thread1")

    @pytest.mark.asyncio
    async def test_vacuum_without_capability(self, checkpointer):
        """Test vacuum without capability raises error."""
        with pytest.raises(NotImplementedError):
            await checkpointer.vacuum()

    @pytest.mark.asyncio
    async def test_copy_thread_without_capability(self, checkpointer):
        """Test copy_thread without capability raises error."""
        with pytest.raises(NotImplementedError):
            await checkpointer.copy_thread("src", "dest")

    @pytest.mark.asyncio
    async def test_list_threads_without_capability(self, checkpointer):
        """Test list_threads without capability raises error."""
        with pytest.raises(NotImplementedError):
            await checkpointer.list_threads()

    @pytest.mark.asyncio
    async def test_list_with_metadata_without_capability(self, checkpointer):
        """Test list_with_metadata without capability raises error."""
        with pytest.raises(NotImplementedError):
            await checkpointer.list_with_metadata()

    def test_require_capability_raises(self, checkpointer):
        """Test _require_capability raises for missing capability."""
        with pytest.raises(NotImplementedError, match="does not support"):
            checkpointer._require_capability("search")
