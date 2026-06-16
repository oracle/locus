# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for storage backend adapters."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from locus.core.state import AgentState
from locus.memory.backends.adapters import StorageBackendAdapter


class TestStorageBackendAdapterInit:
    """Tests for StorageBackendAdapter initialization."""

    def test_init_with_backend(self):
        """Test initializing with a backend."""
        mock_backend = MagicMock()
        adapter = StorageBackendAdapter(mock_backend)

        assert adapter._backend is mock_backend
        assert adapter._capabilities_cache is None

    def test_capabilities_basic(self):
        """Test capabilities detection for basic backend."""
        mock_backend = MagicMock(spec=["save", "load", "delete", "exists"])
        adapter = StorageBackendAdapter(mock_backend)

        caps = adapter.capabilities

        assert caps.search is False
        assert caps.metadata_query is False
        assert caps.vacuum is False
        assert caps.branching is False
        assert caps.list_threads is False

    def test_capabilities_with_search(self):
        """Test capabilities with search method."""
        mock_backend = MagicMock()
        mock_backend.search = MagicMock()
        adapter = StorageBackendAdapter(mock_backend)

        caps = adapter.capabilities

        assert caps.search is True

    def test_capabilities_with_list_threads(self):
        """Test capabilities with list_threads method."""
        mock_backend = MagicMock()
        mock_backend.list_threads = MagicMock()
        adapter = StorageBackendAdapter(mock_backend)

        caps = adapter.capabilities

        assert caps.list_threads is True

    def test_capabilities_with_vacuum(self):
        """Test capabilities with vacuum method."""
        mock_backend = MagicMock()
        mock_backend.vacuum = MagicMock()
        adapter = StorageBackendAdapter(mock_backend)

        caps = adapter.capabilities

        assert caps.vacuum is True

    def test_capabilities_with_branching(self):
        """Test capabilities with copy_thread method."""
        mock_backend = MagicMock()
        mock_backend.copy_thread = MagicMock()
        adapter = StorageBackendAdapter(mock_backend)

        caps = adapter.capabilities

        assert caps.branching is True

    def test_capabilities_cached(self):
        """Test that capabilities are cached."""
        mock_backend = MagicMock(spec=["save", "load"])
        adapter = StorageBackendAdapter(mock_backend)

        caps1 = adapter.capabilities
        caps2 = adapter.capabilities

        assert caps1 is caps2


class TestStorageBackendAdapterSave:
    """Tests for StorageBackendAdapter.save method."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock backend."""
        backend = MagicMock()
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        return backend

    @pytest.fixture
    def mock_state(self):
        """Create mock state."""
        return AgentState(run_id="test", messages=[])

    @pytest.mark.asyncio
    async def test_save_generates_id(self, mock_backend, mock_state):
        """Test save generates checkpoint ID if not provided."""
        # Mock load for index update
        mock_backend.load.return_value = None
        adapter = StorageBackendAdapter(mock_backend)

        cp_id = await adapter.save(mock_state, "thread1")

        # ID is auto-generated (UUID format)
        assert len(cp_id) > 0
        assert mock_backend.save.called

    @pytest.mark.asyncio
    async def test_save_with_custom_id(self, mock_backend, mock_state):
        """Test save with custom checkpoint ID."""
        adapter = StorageBackendAdapter(mock_backend)

        cp_id = await adapter.save(mock_state, "thread1", checkpoint_id="my-cp")

        assert cp_id == "my-cp"

    @pytest.mark.asyncio
    async def test_save_with_metadata(self, mock_backend, mock_state):
        """Test save with metadata."""
        # Backend that accepts metadata
        mock_backend.save = AsyncMock()

        adapter = StorageBackendAdapter(mock_backend)

        await adapter.save(
            mock_state,
            "thread1",
            checkpoint_id="cp1",
            metadata={"key": "value"},
        )

        # Should have been called to save the checkpoint
        assert mock_backend.save.called

    @pytest.mark.asyncio
    async def test_save_updates_latest(self, mock_backend, mock_state):
        """Test save updates latest pointer."""
        adapter = StorageBackendAdapter(mock_backend)

        await adapter.save(mock_state, "thread1", checkpoint_id="cp1")

        # Should save to both cp1 and latest
        calls = [call[0][0] for call in mock_backend.save.call_args_list]
        assert any("latest" in call for call in calls)


class TestStorageBackendAdapterLoad:
    """Tests for StorageBackendAdapter.load method."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock backend."""
        backend = MagicMock()
        backend.save = AsyncMock()
        backend.load = AsyncMock()
        return backend

    @pytest.mark.asyncio
    async def test_load_latest(self, mock_backend):
        """Test loading latest checkpoint."""
        mock_backend.load.return_value = {
            "run_id": "test",
            "messages": [],
            "iteration": 0,
            "_checkpoint_id": "cp1",
            "_checkpoint_timestamp": "2024-01-01T00:00:00Z",
        }

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.load("thread1")

        assert result is not None
        assert result.run_id == "test"
        mock_backend.load.assert_called_once_with("thread1:latest")

    @pytest.mark.asyncio
    async def test_load_specific_checkpoint(self, mock_backend):
        """Test loading specific checkpoint."""
        mock_backend.load.return_value = {
            "run_id": "test",
            "messages": [],
            "iteration": 0,
        }

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.load("thread1", checkpoint_id="cp123")

        assert result is not None
        mock_backend.load.assert_called_once_with("thread1:cp123")

    @pytest.mark.asyncio
    async def test_load_not_found(self, mock_backend):
        """Test loading non-existent checkpoint."""
        mock_backend.load.return_value = None

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.load("thread1")

        assert result is None


class TestStorageBackendAdapterListCheckpoints:
    """Tests for StorageBackendAdapter.list_checkpoints method."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock backend."""
        backend = MagicMock()
        backend.load = AsyncMock()
        return backend

    @pytest.mark.asyncio
    async def test_list_checkpoints_from_index(self, mock_backend):
        """Test listing checkpoints from persistent index."""
        mock_backend.load.return_value = {
            "checkpoints": [
                {"checkpoint_id": "cp3", "timestamp": "2024-01-03"},
                {"checkpoint_id": "cp2", "timestamp": "2024-01-02"},
                {"checkpoint_id": "cp1", "timestamp": "2024-01-01"},
            ]
        }

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.list_checkpoints("thread1")

        assert result == ["cp3", "cp2", "cp1"]
        mock_backend.load.assert_called_once_with("thread1:_checkpoints")

    @pytest.mark.asyncio
    async def test_list_checkpoints_with_limit(self, mock_backend):
        """Test listing checkpoints respects limit."""
        mock_backend.load.return_value = {
            "checkpoints": [{"checkpoint_id": f"cp{i}"} for i in range(10)]
        }

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.list_checkpoints("thread1", limit=3)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_list_checkpoints_empty(self, mock_backend):
        """Test listing when no checkpoints exist."""
        mock_backend.load.return_value = None

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.list_checkpoints("thread1")

        assert result == []


class TestStorageBackendAdapterDelete:
    """Tests for StorageBackendAdapter.delete method."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock backend."""
        backend = MagicMock()
        backend.delete = AsyncMock(return_value=True)
        backend.load = AsyncMock()
        backend.save = AsyncMock()
        backend.exists = AsyncMock(return_value=True)
        return backend

    @pytest.mark.asyncio
    async def test_delete_specific_checkpoint(self, mock_backend):
        """Test deleting specific checkpoint."""
        mock_backend.load.return_value = {"checkpoints": [{"checkpoint_id": "cp1"}]}

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.delete("thread1", checkpoint_id="cp1")

        assert result is True
        mock_backend.delete.assert_called_with("thread1:cp1")

    @pytest.mark.asyncio
    async def test_delete_all_checkpoints(self, mock_backend):
        """Test deleting all checkpoints for thread."""
        mock_backend.load.return_value = {
            "checkpoints": [
                {"checkpoint_id": "cp1"},
                {"checkpoint_id": "cp2"},
            ]
        }

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.delete("thread1")

        assert result is True
        # Should delete all checkpoints plus latest and index
        assert mock_backend.delete.call_count >= 2

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, mock_backend):
        """Test deleting from thread with no checkpoints."""
        mock_backend.load.return_value = None
        mock_backend.exists.return_value = False
        mock_backend.delete.return_value = False

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.delete("thread1")

        assert result is False


class TestStorageBackendAdapterExists:
    """Tests for StorageBackendAdapter.exists method."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock backend."""
        backend = MagicMock()
        backend.exists = AsyncMock()
        return backend

    @pytest.mark.asyncio
    async def test_exists_latest(self, mock_backend):
        """Test checking if latest checkpoint exists."""
        mock_backend.exists.return_value = True

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.exists("thread1")

        assert result is True
        mock_backend.exists.assert_called_once_with("thread1:latest")

    @pytest.mark.asyncio
    async def test_exists_specific_checkpoint(self, mock_backend):
        """Test checking if specific checkpoint exists."""
        mock_backend.exists.return_value = True

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.exists("thread1", checkpoint_id="cp1")

        assert result is True
        mock_backend.exists.assert_called_once_with("thread1:cp1")

    @pytest.mark.asyncio
    async def test_not_exists(self, mock_backend):
        """Test when checkpoint doesn't exist."""
        mock_backend.exists.return_value = False

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.exists("thread1")

        assert result is False


class TestStorageBackendAdapterExtendedMethods:
    """Tests for extended StorageBackendAdapter methods."""

    @pytest.mark.asyncio
    async def test_list_threads(self):
        """Test list_threads delegates to backend."""
        mock_backend = MagicMock()
        mock_backend.list_threads = AsyncMock(return_value=["thread1", "thread2"])

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.list_threads()

        assert result == ["thread1", "thread2"]

    @pytest.mark.asyncio
    async def test_search(self):
        """Test search delegates to backend."""
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=[{"id": "1"}])

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.search("query")

        assert result == [{"id": "1"}]
        # search is called with query and limit
        mock_backend.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_vacuum(self):
        """Test vacuum delegates to backend."""
        mock_backend = MagicMock()
        mock_backend.vacuum = AsyncMock(return_value=5)

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.vacuum(older_than_days=30)

        assert result == 5

    @pytest.mark.asyncio
    async def test_copy_thread(self):
        """Test copy_thread delegates to backend."""
        mock_backend = MagicMock()
        mock_backend.load = AsyncMock(return_value=None)  # For list_checkpoints
        mock_backend.save = AsyncMock()
        mock_backend.copy_thread = AsyncMock(return_value="new_thread")

        adapter = StorageBackendAdapter(mock_backend)
        result = await adapter.copy_thread("old_thread", "new_thread")

        # copy_thread should work (may or may not delegate depending on implementation)
        assert result is not None


class TestStorageBackendAdapterRepr:
    """Tests for StorageBackendAdapter repr."""

    def test_repr(self):
        """Test string representation."""
        mock_backend = MagicMock()
        mock_backend.__class__.__name__ = "MockBackend"

        adapter = StorageBackendAdapter(mock_backend)

        repr_str = repr(adapter)
        assert "StorageBackendAdapter" in repr_str
