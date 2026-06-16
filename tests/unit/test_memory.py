# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tests for memory and checkpointing modules."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from locus.core.messages import Message, Role
from locus.core.state import AgentState
from locus.memory import (
    CheckpointMetadata,
    DeltaCheckpoint,
    DeltaCheckpointer,
    InMemoryDeltaStorage,
    NullManager,
    SlidingWindowManager,
    SummarizingManager,
)
from locus.memory.backends import FileCheckpointer, HTTPCheckpointer, MemoryCheckpointer


# =============================================================================
# Conversation Manager Tests
# =============================================================================


class TestNullManager:
    """Tests for NullManager."""

    def test_returns_all_messages(self):
        """NullManager returns all messages unchanged."""
        manager = NullManager()
        messages = [
            Message.system("You are helpful"),
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]

        result = manager.apply(messages)

        assert len(result) == 3
        assert result == messages

    def test_empty_messages(self):
        """NullManager handles empty list."""
        manager = NullManager()

        result = manager.apply([])

        assert result == []

    def test_returns_copy(self):
        """NullManager returns a copy, not the original."""
        manager = NullManager()
        messages = [Message.user("Hello")]

        result = manager.apply(messages)

        assert result is not messages
        assert result == messages


class TestSlidingWindowManager:
    """Tests for SlidingWindowManager."""

    def test_keeps_last_n_messages(self):
        """SlidingWindowManager keeps only last N messages."""
        manager = SlidingWindowManager(window_size=3)
        messages = [Message.user(f"Message {i}") for i in range(10)]

        result = manager.apply(messages)

        assert len(result) == 3
        assert result[0].content == "Message 7"
        assert result[1].content == "Message 8"
        assert result[2].content == "Message 9"

    def test_preserves_system_message(self):
        """SlidingWindowManager preserves system message."""
        manager = SlidingWindowManager(window_size=2, preserve_system=True)
        messages = [
            Message.system("System prompt"),
            Message.user("Message 1"),
            Message.user("Message 2"),
            Message.user("Message 3"),
        ]

        result = manager.apply(messages)

        assert len(result) == 3  # system + 2 recent
        assert result[0].role == Role.SYSTEM
        assert result[1].content == "Message 2"
        assert result[2].content == "Message 3"

    def test_no_preserve_system(self):
        """SlidingWindowManager can exclude system message."""
        manager = SlidingWindowManager(window_size=2, preserve_system=False)
        messages = [
            Message.system("System prompt"),
            Message.user("Message 1"),
            Message.user("Message 2"),
            Message.user("Message 3"),
        ]

        result = manager.apply(messages)

        assert len(result) == 2
        assert result[0].content == "Message 2"
        assert result[1].content == "Message 3"

    def test_fewer_than_window_size(self):
        """SlidingWindowManager handles fewer messages than window."""
        manager = SlidingWindowManager(window_size=10)
        messages = [Message.user("Hello"), Message.user("World")]

        result = manager.apply(messages)

        assert len(result) == 2

    def test_invalid_window_size(self):
        """SlidingWindowManager rejects invalid window size."""
        with pytest.raises(ValueError, match="window_size must be at least 1"):
            SlidingWindowManager(window_size=0)

    def test_empty_messages(self):
        """SlidingWindowManager handles empty list."""
        manager = SlidingWindowManager(window_size=5)

        result = manager.apply([])

        assert result == []


class TestSummarizingManager:
    """Tests for SummarizingManager."""

    def test_no_summarization_under_threshold(self):
        """No summarization when under threshold."""
        manager = SummarizingManager(threshold=10, keep_recent=5)
        messages = [Message.user(f"Message {i}") for i in range(5)]

        result = manager.apply(messages)

        assert len(result) == 5
        assert all(m.role == Role.USER for m in result)

    def test_summarizes_when_over_threshold(self):
        """Summarizes older messages when over threshold."""
        manager = SummarizingManager(threshold=10, keep_recent=5)
        messages = [Message.user(f"Message {i}") for i in range(15)]

        result = manager.apply(messages)

        # Should have: summary message + 5 recent messages
        assert len(result) == 6
        assert result[0].role == Role.SYSTEM  # Summary
        assert "Summary" in (result[0].content or "")
        # Recent messages preserved
        assert result[1].content == "Message 10"
        assert result[-1].content == "Message 14"

    def test_preserves_system_message(self):
        """Preserves system message in summary mode."""
        manager = SummarizingManager(threshold=5, keep_recent=2)
        messages = [
            Message.system("Original system"),
            *[Message.user(f"Message {i}") for i in range(10)],
        ]

        result = manager.apply(messages)

        # System message + summary + 2 recent
        assert result[0].role == Role.SYSTEM
        assert result[0].content == "Original system"
        assert result[1].role == Role.SYSTEM  # Summary
        assert "Summary" in (result[1].content or "")

    def test_invalid_threshold(self):
        """Rejects invalid threshold."""
        with pytest.raises(ValueError, match="threshold must be at least 1"):
            SummarizingManager(threshold=0)

    def test_invalid_keep_recent(self):
        """Rejects invalid keep_recent."""
        with pytest.raises(ValueError, match="keep_recent must be at least 1"):
            SummarizingManager(threshold=10, keep_recent=0)

    def test_keep_recent_exceeds_threshold(self):
        """Rejects keep_recent >= threshold."""
        with pytest.raises(ValueError, match="keep_recent must be less than threshold"):
            SummarizingManager(threshold=5, keep_recent=5)

    def test_empty_messages(self):
        """Handles empty list."""
        manager = SummarizingManager(threshold=10, keep_recent=5)

        result = manager.apply([])

        assert result == []


# =============================================================================
# Memory Checkpointer Tests
# =============================================================================


class TestMemoryCheckpointer:
    """Tests for MemoryCheckpointer."""

    @pytest.fixture
    def checkpointer(self):
        """Create a MemoryCheckpointer instance."""
        return MemoryCheckpointer()

    @pytest.fixture
    def sample_state(self):
        """Create a sample agent state."""
        return AgentState(
            iteration=5,
            confidence=0.75,
        ).with_message(Message.user("Hello"))

    @pytest.mark.asyncio
    async def test_save_and_load(self, checkpointer, sample_state):
        """Save and load state."""
        checkpoint_id = await checkpointer.save(sample_state, "thread-1")

        restored = await checkpointer.load("thread-1", checkpoint_id)

        assert restored is not None
        assert restored.iteration == 5
        assert restored.confidence == 0.75
        assert len(restored.messages) == 1

    @pytest.mark.asyncio
    async def test_load_latest(self, checkpointer, sample_state):
        """Load latest checkpoint when no ID specified."""
        await checkpointer.save(sample_state, "thread-1")

        newer_state = sample_state.with_confidence(0.9)
        await checkpointer.save(newer_state, "thread-1")

        restored = await checkpointer.load("thread-1")

        assert restored is not None
        assert restored.confidence == 0.9

    @pytest.mark.asyncio
    async def test_load_nonexistent_thread(self, checkpointer):
        """Load returns None for nonexistent thread."""
        result = await checkpointer.load("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, checkpointer, sample_state):
        """List checkpoints for a thread."""
        ids = []
        for i in range(5):
            state = sample_state.with_iteration(i)
            checkpoint_id = await checkpointer.save(state, "thread-1")
            ids.append(checkpoint_id)
            await asyncio.sleep(0.01)  # Ensure different timestamps

        listed = await checkpointer.list_checkpoints("thread-1", limit=3)

        assert len(listed) == 3
        # Should be newest first
        assert listed[0] == ids[-1]

    @pytest.mark.asyncio
    async def test_delete_specific_checkpoint(self, checkpointer, sample_state):
        """Delete a specific checkpoint."""
        cp1 = await checkpointer.save(sample_state, "thread-1")
        cp2 = await checkpointer.save(sample_state.with_confidence(0.9), "thread-1")

        result = await checkpointer.delete("thread-1", cp1)

        assert result is True
        assert await checkpointer.load("thread-1", cp1) is None
        assert await checkpointer.load("thread-1", cp2) is not None

    @pytest.mark.asyncio
    async def test_delete_all_checkpoints(self, checkpointer, sample_state):
        """Delete all checkpoints for a thread."""
        await checkpointer.save(sample_state, "thread-1")
        await checkpointer.save(sample_state, "thread-1")

        result = await checkpointer.delete("thread-1")

        assert result is True
        assert await checkpointer.list_checkpoints("thread-1") == []

    def test_clear(self, checkpointer):
        """Clear all stored checkpoints."""
        asyncio.run(checkpointer.save(AgentState(), "thread-1"))
        asyncio.run(checkpointer.save(AgentState(), "thread-2"))

        checkpointer.clear()

        assert checkpointer.get_checkpoint_count() == 0

    def test_get_thread_ids(self, checkpointer):
        """Get list of thread IDs."""
        asyncio.run(checkpointer.save(AgentState(), "thread-1"))
        asyncio.run(checkpointer.save(AgentState(), "thread-2"))

        thread_ids = checkpointer.get_thread_ids()

        assert set(thread_ids) == {"thread-1", "thread-2"}


# =============================================================================
# File Checkpointer Tests
# =============================================================================


class TestFileCheckpointer:
    """Tests for FileCheckpointer."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def checkpointer(self, temp_dir):
        """Create a FileCheckpointer instance."""
        return FileCheckpointer(temp_dir / "checkpoints")

    @pytest.fixture
    def sample_state(self):
        """Create a sample agent state."""
        return AgentState(iteration=3, confidence=0.5)

    @pytest.mark.asyncio
    async def test_save_creates_file(self, checkpointer, sample_state, temp_dir):
        """Save creates a checkpoint file."""
        checkpoint_id = await checkpointer.save(sample_state, "thread-1")

        # Check file exists
        thread_dir = temp_dir / "checkpoints" / "thread-1"
        assert thread_dir.exists()
        files = list(thread_dir.glob("*.json"))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_save_and_load(self, checkpointer, sample_state):
        """Save and load state from file."""
        checkpoint_id = await checkpointer.save(sample_state, "thread-1")

        restored = await checkpointer.load("thread-1", checkpoint_id)

        assert restored is not None
        assert restored.iteration == 3
        assert restored.confidence == 0.5

    @pytest.mark.asyncio
    async def test_load_latest(self, checkpointer, sample_state):
        """Load latest checkpoint."""
        await checkpointer.save(sample_state, "thread-1")
        await asyncio.sleep(0.05)
        newer_state = sample_state.with_iteration(10)
        await checkpointer.save(newer_state, "thread-1")

        restored = await checkpointer.load("thread-1")

        assert restored is not None
        assert restored.iteration == 10

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, checkpointer, sample_state):
        """List checkpoints from files."""
        ids = []
        for i in range(3):
            cp_id = await checkpointer.save(sample_state.with_iteration(i), "thread-1")
            ids.append(cp_id)
            await asyncio.sleep(0.05)

        listed = await checkpointer.list_checkpoints("thread-1")

        assert len(listed) == 3
        assert listed[0] == ids[-1]  # Newest first

    @pytest.mark.asyncio
    async def test_delete_specific_checkpoint(self, checkpointer, sample_state):
        """Delete a specific checkpoint file."""
        cp1 = await checkpointer.save(sample_state, "thread-1")
        cp2 = await checkpointer.save(sample_state.with_iteration(2), "thread-1")

        result = await checkpointer.delete("thread-1", cp1)

        assert result is True
        listed = await checkpointer.list_checkpoints("thread-1")
        assert cp1 not in listed
        assert cp2 in listed

    @pytest.mark.asyncio
    async def test_delete_thread(self, checkpointer, sample_state):
        """Delete all checkpoints for a thread."""
        await checkpointer.save(sample_state, "thread-1")
        await checkpointer.save(sample_state, "thread-1")

        result = await checkpointer.delete("thread-1")

        assert result is True
        listed = await checkpointer.list_checkpoints("thread-1")
        assert listed == []

    @pytest.mark.asyncio
    async def test_get_disk_usage(self, checkpointer, sample_state):
        """Get disk usage for checkpoints."""
        await checkpointer.save(sample_state, "thread-1")

        usage = await checkpointer.get_disk_usage("thread-1")

        assert usage > 0

    def test_sanitizes_thread_id(self, temp_dir):
        """Sanitizes thread ID for filesystem safety."""
        checkpointer = FileCheckpointer(temp_dir / "checkpoints")

        # Thread ID with special characters
        asyncio.run(checkpointer.save(AgentState(), "thread/with:special<chars>"))

        # Should create directory with safe name
        assert (temp_dir / "checkpoints").exists()


# =============================================================================
# HTTP Checkpointer Tests
# =============================================================================


class TestHTTPCheckpointer:
    """Tests for HTTPCheckpointer."""

    @pytest.fixture
    def checkpointer(self):
        """Create an HTTPCheckpointer instance."""
        return HTTPCheckpointer(
            base_url="https://api.example.com/v1",
            headers={"X-Custom": "header"},
        )

    @pytest.fixture
    def sample_state(self):
        """Create a sample agent state."""
        return AgentState(iteration=3)

    @pytest.mark.asyncio
    async def test_save_makes_post_request(self, checkpointer, sample_state):
        """Save makes POST request to API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"checkpoint_id": "test-123"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(checkpointer, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await checkpointer.save(sample_state, "thread-1")

            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert "/threads/thread-1/checkpoints" in call_args[0][0]
            assert result == "test-123"

    @pytest.mark.asyncio
    async def test_load_makes_get_request(self, checkpointer):
        """Load makes GET request to API."""
        state_data = AgentState(iteration=5).to_checkpoint()
        mock_response = MagicMock()
        mock_response.json.return_value = {"state": state_data}
        mock_response.raise_for_status = MagicMock()

        with patch.object(checkpointer, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await checkpointer.load("thread-1", "cp-123")

            assert result is not None
            assert result.iteration == 5

    @pytest.mark.asyncio
    async def test_list_checkpoints_parses_response(self, checkpointer):
        """List checkpoints parses various response formats."""
        mock_response = MagicMock()
        mock_response.json.return_value = ["cp-1", "cp-2", "cp-3"]
        mock_response.raise_for_status = MagicMock()

        with patch.object(checkpointer, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await checkpointer.list_checkpoints("thread-1")

            assert result == ["cp-1", "cp-2", "cp-3"]

    @pytest.mark.asyncio
    async def test_handles_wrapped_response(self, checkpointer):
        """Handles wrapped response format."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "checkpoints": [
                {"checkpoint_id": "cp-1"},
                {"checkpoint_id": "cp-2"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(checkpointer, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await checkpointer.list_checkpoints("thread-1")

            assert result == ["cp-1", "cp-2"]

    @pytest.mark.asyncio
    async def test_context_manager(self, checkpointer):
        """Test async context manager."""
        with patch.object(checkpointer, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client

            async with checkpointer as cp:
                assert cp is checkpointer


# =============================================================================
# Delta Checkpointer Tests
# =============================================================================


class TestDeltaCheckpointer:
    """Tests for DeltaCheckpointer."""

    @pytest.fixture
    def storage(self):
        """Create in-memory delta storage."""
        return InMemoryDeltaStorage()

    @pytest.fixture
    def checkpointer(self, storage):
        """Create a DeltaCheckpointer instance."""
        return DeltaCheckpointer(
            storage=storage,
            max_chain_depth=3,
            compression_level=6,
        )

    @pytest.fixture
    def sample_state(self):
        """Create a sample agent state."""
        return AgentState(
            iteration=5,
            confidence=0.75,
        ).with_message(Message.user("Hello"))

    @pytest.mark.asyncio
    async def test_first_checkpoint_is_full(self, checkpointer, storage, sample_state):
        """First checkpoint creates full snapshot."""
        checkpoint_id = await checkpointer.save(sample_state, "thread-1")

        checkpoint = await storage.retrieve("thread-1", checkpoint_id)

        assert checkpoint is not None
        assert checkpoint.metadata.is_full is True
        assert checkpoint.metadata.chain_depth == 0

    @pytest.mark.asyncio
    async def test_subsequent_checkpoints_are_deltas(self, checkpointer, storage, sample_state):
        """Subsequent checkpoints create deltas."""
        cp1 = await checkpointer.save(sample_state, "thread-1")

        modified_state = sample_state.with_confidence(0.9)
        cp2 = await checkpointer.save(modified_state, "thread-1")

        checkpoint = await storage.retrieve("thread-1", cp2)

        assert checkpoint is not None
        assert checkpoint.is_delta is True
        assert checkpoint.metadata.parent_id == cp1
        assert checkpoint.metadata.chain_depth == 1

    @pytest.mark.asyncio
    async def test_full_checkpoint_at_chain_limit(self, checkpointer, storage, sample_state):
        """Creates full checkpoint when chain limit reached."""
        # Create checkpoints up to the limit (max_chain_depth=3)
        # Checkpoint 1: full (no parent)
        # Checkpoint 2: delta (chain_depth=1)
        # Checkpoint 3: delta (chain_depth=2)
        # Checkpoint 4: delta (chain_depth=3)
        # Checkpoint 5: full (chain_depth >= max_chain_depth, reset)
        for i in range(5):
            state = sample_state.with_iteration(i)
            await checkpointer.save(state, "thread-1")

        # Get all checkpoints
        metadata_list = await storage.list_checkpoints("thread-1", limit=10)

        # First and fifth should be full, rest are deltas
        full_count = sum(1 for m in metadata_list if m.is_full)
        assert full_count == 2  # First and 5th (after chain reset)

    @pytest.mark.asyncio
    async def test_load_reconstructs_from_deltas(self, checkpointer, sample_state):
        """Load reconstructs full state from delta chain."""
        # Create initial state
        await checkpointer.save(sample_state, "thread-1")

        # Create modified states
        state2 = sample_state.with_confidence(0.8)
        await checkpointer.save(state2, "thread-1")

        state3 = state2.with_iteration(10)
        cp3 = await checkpointer.save(state3, "thread-1")

        # Load and verify reconstruction
        restored = await checkpointer.load("thread-1", cp3)

        assert restored is not None
        assert restored.confidence == 0.8
        assert restored.iteration == 10

    @pytest.mark.asyncio
    async def test_load_latest(self, checkpointer, sample_state):
        """Load returns latest checkpoint when no ID specified."""
        await checkpointer.save(sample_state, "thread-1")

        newer_state = sample_state.with_confidence(0.99)
        await checkpointer.save(newer_state, "thread-1")

        restored = await checkpointer.load("thread-1")

        assert restored is not None
        assert restored.confidence == 0.99

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, checkpointer):
        """Load returns None for nonexistent checkpoint."""
        result = await checkpointer.load("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, checkpointer, sample_state):
        """List checkpoints returns IDs."""
        ids = []
        for i in range(3):
            cp_id = await checkpointer.save(sample_state.with_iteration(i), "thread-1")
            ids.append(cp_id)

        listed = await checkpointer.list_checkpoints("thread-1")

        assert len(listed) == 3
        assert set(listed) == set(ids)

    @pytest.mark.asyncio
    async def test_compression_reduces_size(self, checkpointer, storage, sample_state):
        """Compression reduces stored size."""
        # Create a state with more data
        state = sample_state
        for i in range(10):
            state = state.with_message(Message.user(f"Message {i} " * 100))

        checkpoint_id = await checkpointer.save(state, "thread-1")

        checkpoint = await storage.retrieve("thread-1", checkpoint_id)

        assert checkpoint is not None
        assert checkpoint.metadata.compressed_size_bytes < checkpoint.metadata.size_bytes
        assert checkpoint.compression_ratio > 1.0

    @pytest.mark.asyncio
    async def test_delta_is_smaller_than_full(self, checkpointer, storage, sample_state):
        """Delta checkpoint is smaller than full when changes are small."""
        # Create initial full checkpoint
        cp1 = await checkpointer.save(sample_state, "thread-1")

        # Create delta with small change
        modified = sample_state.with_confidence(0.9)
        cp2 = await checkpointer.save(modified, "thread-1")

        full = await storage.retrieve("thread-1", cp1)
        delta = await storage.retrieve("thread-1", cp2)

        assert full is not None
        assert delta is not None
        # Delta should be smaller (compressed size)
        assert delta.metadata.compressed_size_bytes < full.metadata.compressed_size_bytes

    @pytest.mark.asyncio
    async def test_get_storage_stats(self, checkpointer, sample_state):
        """Get storage statistics."""
        for i in range(5):
            await checkpointer.save(sample_state.with_iteration(i), "thread-1")

        stats = await checkpointer.get_storage_stats("thread-1")

        assert stats["total_checkpoints"] == 5
        assert stats["total_size"] > 0
        assert stats["compressed_size"] > 0
        assert stats["full_checkpoints"] >= 1
        assert stats["delta_checkpoints"] >= 1

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, checkpointer, sample_state):
        """Delete a specific checkpoint."""
        cp1 = await checkpointer.save(sample_state, "thread-1")
        cp2 = await checkpointer.save(sample_state.with_iteration(2), "thread-1")

        result = await checkpointer.delete("thread-1", cp1)

        assert result is True
        listed = await checkpointer.list_checkpoints("thread-1")
        assert cp1 not in listed
        assert cp2 in listed

    @pytest.mark.asyncio
    async def test_delete_thread(self, checkpointer, sample_state):
        """Delete all checkpoints for a thread."""
        await checkpointer.save(sample_state, "thread-1")
        await checkpointer.save(sample_state, "thread-1")

        result = await checkpointer.delete("thread-1")

        assert result is True
        listed = await checkpointer.list_checkpoints("thread-1")
        assert listed == []

    @pytest.mark.asyncio
    async def test_get_metadata(self, checkpointer, sample_state):
        """Get metadata for a checkpoint."""
        checkpoint_id = await checkpointer.save(sample_state, "thread-1")

        metadata = await checkpointer.get_metadata("thread-1", checkpoint_id)

        assert metadata is not None
        assert metadata.checkpoint_id == checkpoint_id
        assert metadata.thread_id == "thread-1"
        assert metadata.is_full is True


class TestDeltaComputation:
    """Tests for delta computation logic."""

    def test_compute_delta_added_keys(self):
        """Delta captures added keys."""
        checkpointer = DeltaCheckpointer()
        old = {"a": 1, "b": 2}
        new = {"a": 1, "b": 2, "c": 3}

        delta = checkpointer._compute_delta(old, new)

        assert delta["__added__"] == {"c": 3}
        assert delta["__removed__"] == []
        assert delta["__changed__"] == {}

    def test_compute_delta_removed_keys(self):
        """Delta captures removed keys."""
        checkpointer = DeltaCheckpointer()
        old = {"a": 1, "b": 2, "c": 3}
        new = {"a": 1, "b": 2}

        delta = checkpointer._compute_delta(old, new)

        assert delta["__added__"] == {}
        assert delta["__removed__"] == ["c"]
        assert delta["__changed__"] == {}

    def test_compute_delta_changed_keys(self):
        """Delta captures changed keys."""
        checkpointer = DeltaCheckpointer()
        old = {"a": 1, "b": 2}
        new = {"a": 1, "b": 99}

        delta = checkpointer._compute_delta(old, new)

        assert delta["__added__"] == {}
        assert delta["__removed__"] == []
        assert delta["__changed__"] == {"b": 99}

    def test_apply_delta(self):
        """Apply delta reconstructs new state."""
        checkpointer = DeltaCheckpointer()
        base = {"a": 1, "b": 2, "c": 3}
        delta = {
            "__added__": {"d": 4},
            "__removed__": ["c"],
            "__changed__": {"b": 99},
        }

        result = checkpointer._apply_delta(base, delta)

        assert result == {"a": 1, "b": 99, "d": 4}


class TestInMemoryDeltaStorage:
    """Tests for InMemoryDeltaStorage."""

    @pytest.fixture
    def storage(self):
        """Create storage instance."""
        return InMemoryDeltaStorage()

    @pytest.fixture
    def sample_checkpoint(self):
        """Create a sample checkpoint."""
        metadata = CheckpointMetadata(
            checkpoint_id="cp-1",
            thread_id="thread-1",
            is_full=True,
        )
        return DeltaCheckpoint(
            metadata=metadata,
            data=b"compressed-data",
            is_delta=False,
        )

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, storage, sample_checkpoint):
        """Store and retrieve checkpoint."""
        await storage.store("thread-1", "cp-1", sample_checkpoint)

        retrieved = await storage.retrieve("thread-1", "cp-1")

        assert retrieved is not None
        assert retrieved.metadata.checkpoint_id == "cp-1"

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent(self, storage):
        """Retrieve returns None for nonexistent checkpoint."""
        result = await storage.retrieve("thread-1", "nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, storage):
        """List checkpoints returns metadata sorted by time."""
        for i in range(3):
            metadata = CheckpointMetadata(
                checkpoint_id=f"cp-{i}",
                thread_id="thread-1",
            )
            checkpoint = DeltaCheckpoint(
                metadata=metadata,
                data=b"data",
                is_delta=False,
            )
            await storage.store("thread-1", f"cp-{i}", checkpoint)
            await asyncio.sleep(0.01)

        listed = await storage.list_checkpoints("thread-1")

        assert len(listed) == 3
        # Newest first
        assert listed[0].checkpoint_id == "cp-2"

    @pytest.mark.asyncio
    async def test_delete_specific(self, storage, sample_checkpoint):
        """Delete specific checkpoint."""
        await storage.store("thread-1", "cp-1", sample_checkpoint)

        result = await storage.delete("thread-1", "cp-1")

        assert result is True
        assert await storage.retrieve("thread-1", "cp-1") is None

    @pytest.mark.asyncio
    async def test_delete_thread(self, storage, sample_checkpoint):
        """Delete all checkpoints for thread."""
        await storage.store("thread-1", "cp-1", sample_checkpoint)
        await storage.store("thread-1", "cp-2", sample_checkpoint)

        result = await storage.delete("thread-1")

        assert result is True
        listed = await storage.list_checkpoints("thread-1")
        assert len(listed) == 0
