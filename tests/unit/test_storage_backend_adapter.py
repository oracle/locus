# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for StorageBackendAdapter."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from locus.memory.backends.adapters import StorageBackendAdapter


class TestStorageBackendAdapter:
    """Tests for StorageBackendAdapter."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock storage backend."""
        backend = MagicMock(spec=["save", "load", "delete", "exists"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)
        return backend

    @pytest.fixture
    def adapter(self, mock_backend):
        """Create an adapter with mock backend."""
        return StorageBackendAdapter(mock_backend)

    @pytest.fixture
    def mock_state(self):
        """Create a mock agent state."""
        state = MagicMock()
        state.to_checkpoint = MagicMock(
            return_value={
                "run_id": "test-run",
                "messages": [],
                "iteration": 0,
            }
        )
        return state

    def test_create_adapter(self, mock_backend):
        """Test creating adapter."""
        adapter = StorageBackendAdapter(mock_backend)
        assert adapter._backend is mock_backend

    def test_capabilities_basic(self, mock_backend):
        """Test basic capabilities detection."""
        adapter = StorageBackendAdapter(mock_backend)
        caps = adapter.capabilities

        # Default backend has no special methods
        assert caps.search is False
        assert caps.metadata_query is False
        assert caps.vacuum is False
        assert caps.branching is False
        assert caps.list_threads is False
        assert caps.persistent_checkpoint_ids is True

    def test_capabilities_with_search(self, mock_backend):
        """Test capabilities when backend has search."""
        mock_backend.search = AsyncMock()
        adapter = StorageBackendAdapter(mock_backend)

        assert adapter.capabilities.search is True

    def test_capabilities_with_metadata(self, mock_backend):
        """Test capabilities when backend has metadata query."""
        mock_backend.get_metadata = AsyncMock()
        adapter = StorageBackendAdapter(mock_backend)

        assert adapter.capabilities.metadata_query is True

    def test_capabilities_with_vacuum(self, mock_backend):
        """Test capabilities when backend has vacuum."""
        mock_backend.vacuum = AsyncMock()
        adapter = StorageBackendAdapter(mock_backend)

        assert adapter.capabilities.vacuum is True

    def test_capabilities_with_branching(self, mock_backend):
        """Test capabilities when backend has copy_thread."""
        mock_backend.copy_thread = AsyncMock()
        adapter = StorageBackendAdapter(mock_backend)

        assert adapter.capabilities.branching is True

    def test_capabilities_with_list_threads(self, mock_backend):
        """Test capabilities when backend has list_threads."""
        mock_backend.list_threads = AsyncMock()
        adapter = StorageBackendAdapter(mock_backend)

        assert adapter.capabilities.list_threads is True

    def test_capabilities_with_ttl(self, mock_backend):
        """Test capabilities when backend has TTL config."""
        mock_backend.config = MagicMock()
        mock_backend.config.ttl_seconds = 3600
        adapter = StorageBackendAdapter(mock_backend)

        assert adapter.capabilities.ttl is True

    def test_capabilities_cached(self, mock_backend):
        """Test that capabilities are cached."""
        adapter = StorageBackendAdapter(mock_backend)

        caps1 = adapter.capabilities
        caps2 = adapter.capabilities

        assert caps1 is caps2

    @pytest.mark.asyncio
    async def test_save(self, adapter, mock_backend, mock_state):
        """Test saving state."""
        mock_backend.load.return_value = None  # No existing index

        checkpoint_id = await adapter.save(mock_state, "thread1")

        assert checkpoint_id is not None
        assert mock_backend.save.called

    @pytest.mark.asyncio
    async def test_save_with_checkpoint_id(self, adapter, mock_backend, mock_state):
        """Test saving state with specific checkpoint ID."""
        mock_backend.load.return_value = None

        checkpoint_id = await adapter.save(mock_state, "thread1", "cp123")

        assert checkpoint_id == "cp123"

    @pytest.mark.asyncio
    async def test_save_with_metadata(self, adapter, mock_backend, mock_state):
        """Test saving state with metadata."""
        mock_backend.load.return_value = None

        await adapter.save(mock_state, "thread1", metadata={"key": "value"})

        # Index should contain metadata
        calls = mock_backend.save.call_args_list
        # Last call should be index update
        assert len(calls) >= 1

    @pytest.mark.asyncio
    async def test_load_latest(self, adapter, mock_backend):
        """Test loading latest checkpoint."""
        mock_backend.load.return_value = {
            "run_id": "test-run",
            "messages": [],
            "iteration": 0,
            "_checkpoint_id": "cp1",
            "_checkpoint_timestamp": datetime.now(UTC).isoformat(),
        }

        with patch("locus.core.state.AgentState") as mock_agent_state:
            mock_agent_state.from_checkpoint.return_value = MagicMock()
            state = await adapter.load("thread1")

            assert state is not None
            mock_backend.load.assert_called_with("thread1:latest")

    @pytest.mark.asyncio
    async def test_load_specific_checkpoint(self, adapter, mock_backend):
        """Test loading specific checkpoint."""
        mock_backend.load.return_value = {
            "run_id": "test-run",
            "messages": [],
            "iteration": 0,
        }

        with patch("locus.core.state.AgentState") as mock_agent_state:
            mock_agent_state.from_checkpoint.return_value = MagicMock()
            await adapter.load("thread1", "cp123")

            mock_backend.load.assert_called_with("thread1:cp123")

    @pytest.mark.asyncio
    async def test_load_not_found(self, adapter, mock_backend):
        """Test loading non-existent checkpoint."""
        mock_backend.load.return_value = None

        state = await adapter.load("thread1")

        assert state is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, adapter, mock_backend):
        """Test listing checkpoints."""
        mock_backend.load.return_value = {
            "checkpoints": [
                {"checkpoint_id": "cp3", "timestamp": "2024-01-03T00:00:00"},
                {"checkpoint_id": "cp2", "timestamp": "2024-01-02T00:00:00"},
                {"checkpoint_id": "cp1", "timestamp": "2024-01-01T00:00:00"},
            ]
        }

        checkpoints = await adapter.list_checkpoints("thread1")

        assert checkpoints == ["cp3", "cp2", "cp1"]
        mock_backend.load.assert_called_with("thread1:_checkpoints")

    @pytest.mark.asyncio
    async def test_list_checkpoints_empty(self, adapter, mock_backend):
        """Test listing checkpoints when none exist."""
        mock_backend.load.return_value = None

        checkpoints = await adapter.list_checkpoints("thread1")

        assert checkpoints == []

    @pytest.mark.asyncio
    async def test_list_checkpoints_with_limit(self, adapter, mock_backend):
        """Test listing checkpoints with limit."""
        mock_backend.load.return_value = {
            "checkpoints": [{"checkpoint_id": f"cp{i}"} for i in range(20)]
        }

        checkpoints = await adapter.list_checkpoints("thread1", limit=5)

        assert len(checkpoints) == 5

    @pytest.mark.asyncio
    async def test_delete_specific_checkpoint(self, adapter, mock_backend):
        """Test deleting specific checkpoint."""
        mock_backend.load.return_value = {
            "checkpoints": [
                {"checkpoint_id": "cp1"},
                {"checkpoint_id": "cp2"},
            ]
        }

        result = await adapter.delete("thread1", "cp1")

        assert result is True
        mock_backend.delete.assert_called_with("thread1:cp1")

    @pytest.mark.asyncio
    async def test_delete_all_checkpoints(self, adapter, mock_backend):
        """Test deleting all checkpoints for thread."""
        # Setup: index with checkpoints
        mock_backend.load.return_value = {
            "checkpoints": [
                {"checkpoint_id": "cp1"},
                {"checkpoint_id": "cp2"},
            ]
        }
        mock_backend.exists.return_value = True

        result = await adapter.delete("thread1")

        assert result is True

    @pytest.mark.asyncio
    async def test_exists(self, adapter, mock_backend):
        """Test checking existence."""
        mock_backend.exists.return_value = True

        result = await adapter.exists("thread1", "cp1")

        assert result is True
        mock_backend.exists.assert_called_with("thread1:cp1")

    @pytest.mark.asyncio
    async def test_exists_latest(self, adapter, mock_backend):
        """Test checking existence of latest."""
        mock_backend.exists.return_value = True

        result = await adapter.exists("thread1")

        assert result is True
        mock_backend.exists.assert_called_with("thread1:latest")

    @pytest.mark.asyncio
    async def test_search_requires_capability(self, adapter):
        """Test search requires capability."""
        with pytest.raises(NotImplementedError):
            await adapter.search("query")

    @pytest.mark.asyncio
    async def test_search_with_capability(self, mock_backend):
        """Test search with capable backend."""
        mock_backend.search = AsyncMock(return_value=[{"id": "1"}])
        adapter = StorageBackendAdapter(mock_backend)

        results = await adapter.search("query", limit=5)

        assert results == [{"id": "1"}]
        mock_backend.search.assert_called_with("query", limit=5)

    @pytest.mark.asyncio
    async def test_vacuum_requires_capability(self, adapter):
        """Test vacuum requires capability."""
        with pytest.raises(NotImplementedError):
            await adapter.vacuum()

    @pytest.mark.asyncio
    async def test_vacuum_with_capability(self, mock_backend):
        """Test vacuum with capable backend."""
        mock_backend.vacuum = AsyncMock(return_value=10)
        adapter = StorageBackendAdapter(mock_backend)

        count = await adapter.vacuum(older_than_days=30)

        assert count == 10
        mock_backend.vacuum.assert_called_with(30)

    @pytest.mark.asyncio
    async def test_close_with_closable_backend(self, mock_backend):
        """Test close when backend supports it."""
        mock_backend.close = AsyncMock()
        adapter = StorageBackendAdapter(mock_backend)

        await adapter.close()

        mock_backend.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_closable_backend(self, mock_backend):
        """Test close when backend doesn't support it."""
        # Create backend without close method
        backend = MagicMock(spec=["save", "load", "delete", "exists"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)
        adapter = StorageBackendAdapter(backend)

        await adapter.close()  # Should not raise

    def test_repr(self, adapter):
        """Test string representation."""
        repr_str = repr(adapter)
        assert "StorageBackendAdapter" in repr_str

    @pytest.mark.asyncio
    async def test_get_metadata_from_backend(self, mock_backend):
        """Test get_metadata when backend supports it."""
        mock_backend.get_metadata = AsyncMock(return_value={"key": "value"})
        adapter = StorageBackendAdapter(mock_backend)

        meta = await adapter.get_metadata("thread1", "cp1")

        assert meta == {"key": "value"}
        mock_backend.get_metadata.assert_called_with("thread1:cp1")

    @pytest.mark.asyncio
    async def test_get_metadata_from_index(self):
        """Test get_metadata from checkpoint index."""
        # Create backend without get_metadata method
        backend = MagicMock(spec=["save", "load", "delete", "exists"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(
            return_value={
                "checkpoints": [
                    {"checkpoint_id": "cp1", "metadata": {"key": "value"}},
                ]
            }
        )
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)
        adapter = StorageBackendAdapter(backend)

        meta = await adapter.get_metadata("thread1", "cp1")

        assert meta["checkpoint_id"] == "cp1"

    @pytest.mark.asyncio
    async def test_get_metadata_latest(self):
        """Test get_metadata for latest checkpoint."""
        # Create backend without get_metadata method
        backend = MagicMock(spec=["save", "load", "delete", "exists"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(
            return_value={
                "checkpoints": [
                    {"checkpoint_id": "cp2", "metadata": {}},
                    {"checkpoint_id": "cp1", "metadata": {}},
                ]
            }
        )
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)
        adapter = StorageBackendAdapter(backend)

        meta = await adapter.get_metadata("thread1")

        # Should return first (latest) checkpoint
        assert meta["checkpoint_id"] == "cp2"

    @pytest.mark.asyncio
    async def test_query_by_metadata_requires_capability(self):
        """Test query_by_metadata requires capability."""
        # Create backend without query_by_metadata
        backend = MagicMock(spec=["save", "load", "delete", "exists"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)
        adapter = StorageBackendAdapter(backend)

        with pytest.raises(NotImplementedError):
            await adapter.query_by_metadata("key", "value")

    @pytest.mark.asyncio
    async def test_list_threads_requires_capability(self):
        """Test list_threads requires capability."""
        # Create backend without list_threads
        backend = MagicMock(spec=["save", "load", "delete", "exists"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)
        adapter = StorageBackendAdapter(backend)

        with pytest.raises(NotImplementedError):
            await adapter.list_threads()

    @pytest.mark.asyncio
    async def test_list_with_metadata_requires_capability(self):
        """Test list_with_metadata requires capability."""
        # Create backend without list_with_metadata
        backend = MagicMock(spec=["save", "load", "delete", "exists"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)
        adapter = StorageBackendAdapter(backend)

        with pytest.raises(NotImplementedError):
            await adapter.list_with_metadata()

    @pytest.mark.asyncio
    async def test_copy_thread_requires_capability(self):
        """Test copy_thread requires branching capability."""
        # Create backend without copy_thread
        backend = MagicMock(spec=["save", "load", "delete", "exists"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)
        adapter = StorageBackendAdapter(backend)

        with pytest.raises(NotImplementedError):
            await adapter.copy_thread("src", "dest")


class TestStorageBackendAdapterExtendedMethods:
    """Tests for extended adapter methods."""

    @pytest.mark.asyncio
    async def test_query_by_metadata_with_get_by_metadata(self):
        """Test query_by_metadata uses get_by_metadata fallback."""
        backend = MagicMock(spec=["save", "load", "delete", "exists", "get_by_metadata"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)
        backend.get_by_metadata = AsyncMock(return_value=[{"key": "value"}])
        # Make get_metadata available for capability detection
        backend.get_metadata = AsyncMock()

        adapter = StorageBackendAdapter(backend)
        result = await adapter.query_by_metadata("key", "value")

        backend.get_by_metadata.assert_called_once()
        assert result == [{"key": "value"}]

    @pytest.mark.asyncio
    async def test_query_by_metadata_no_method(self):
        """Test query_by_metadata raises when no method available."""
        backend = MagicMock(spec=["save", "load", "delete", "exists", "get_metadata"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)
        backend.get_metadata = AsyncMock()

        adapter = StorageBackendAdapter(backend)

        with pytest.raises(NotImplementedError):
            await adapter.query_by_metadata("key", "value")

    @pytest.mark.asyncio
    async def test_get_metadata_not_found(self):
        """Test get_metadata returns None for missing checkpoint."""
        backend = MagicMock(spec=["save", "load", "delete", "exists"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)

        adapter = StorageBackendAdapter(backend)
        result = await adapter.get_metadata("thread1")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_metadata_checkpoint_id_not_found(self):
        """Test get_metadata returns None for missing checkpoint ID."""
        backend = MagicMock(spec=["save", "load", "delete", "exists"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(
            return_value={
                "checkpoints": [
                    {"checkpoint_id": "cp1", "metadata": {}},
                ]
            }
        )
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)

        adapter = StorageBackendAdapter(backend)
        result = await adapter.get_metadata("thread1", "nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_copy_thread_empty_source(self):
        """Test copy_thread returns False for empty source thread."""
        backend = MagicMock(spec=["save", "load", "delete", "exists", "copy_thread"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)
        backend.copy_thread = AsyncMock()

        adapter = StorageBackendAdapter(backend)
        result = await adapter.copy_thread("src", "dest")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_threads_with_pattern(self):
        """Test list_threads applies pattern filter."""
        backend = MagicMock(spec=["save", "load", "delete", "exists", "list_threads"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)
        backend.list_threads = AsyncMock(return_value=["user_1", "user_2", "admin_1"])

        adapter = StorageBackendAdapter(backend)
        result = await adapter.list_threads(pattern="user_*")

        assert result == ["user_1", "user_2"]
        assert "admin_1" not in result

    @pytest.mark.asyncio
    async def test_list_threads_without_limit_param(self):
        """Test list_threads when backend doesn't have limit parameter."""
        from unittest.mock import AsyncMock, MagicMock

        # Create backend with list_threads that has no limit param
        backend = MagicMock()
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)

        # Create list_threads without limit parameter
        async def list_threads_no_params():
            return ["thread1", "thread2", "thread3"]

        backend.list_threads = list_threads_no_params

        adapter = StorageBackendAdapter(backend)
        result = await adapter.list_threads(limit=2)

        assert len(result) <= 2

    @pytest.mark.asyncio
    async def test_list_with_metadata_delegates(self):
        """Test list_with_metadata delegates to backend."""
        backend = MagicMock(spec=["save", "load", "delete", "exists", "list_with_metadata"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)
        backend.list_with_metadata = AsyncMock(
            return_value=[
                {"thread_id": "t1", "count": 5},
            ]
        )

        adapter = StorageBackendAdapter(backend)
        result = await adapter.list_with_metadata(limit=10)

        backend.list_with_metadata.assert_called_once_with(limit=10)
        assert result == [{"thread_id": "t1", "count": 5}]

    @pytest.mark.asyncio
    async def test_close_delegates(self):
        """Test close delegates to backend."""
        backend = MagicMock(spec=["save", "load", "delete", "exists", "close"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)
        backend.close = AsyncMock()

        adapter = StorageBackendAdapter(backend)
        await adapter.close()

        backend.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_close_method(self):
        """Test close does nothing when backend has no close method."""
        backend = MagicMock(spec=["save", "load", "delete", "exists"])
        backend.save = AsyncMock()
        backend.load = AsyncMock(return_value=None)
        backend.delete = AsyncMock(return_value=True)
        backend.exists = AsyncMock(return_value=False)

        adapter = StorageBackendAdapter(backend)
        await adapter.close()  # Should not raise

    def test_repr(self):
        """Test string representation."""
        backend = MagicMock()
        adapter = StorageBackendAdapter(backend)

        repr_str = repr(adapter)
        assert "StorageBackendAdapter" in repr_str
