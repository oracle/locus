# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for file checkpointer."""

import json
import tempfile
from pathlib import Path

import pytest

from locus.core.state import AgentState
from locus.memory.backends.file import FileCheckpointer


class TestFileCheckpointerInit:
    """Tests for FileCheckpointer initialization."""

    def test_default_init(self):
        """Test creating checkpointer with defaults."""
        checkpointer = FileCheckpointer()
        assert checkpointer.base_dir == Path(".locus_checkpoints")
        assert checkpointer.pretty is True

    def test_custom_base_dir_string(self):
        """Test creating checkpointer with custom base dir as string."""
        checkpointer = FileCheckpointer("/tmp/checkpoints")
        assert checkpointer.base_dir == Path("/tmp/checkpoints")

    def test_custom_base_dir_path(self):
        """Test creating checkpointer with custom base dir as Path."""
        checkpointer = FileCheckpointer(Path("/tmp/checkpoints"))
        assert checkpointer.base_dir == Path("/tmp/checkpoints")

    def test_pretty_false(self):
        """Test creating checkpointer with pretty=False."""
        checkpointer = FileCheckpointer(pretty=False)
        assert checkpointer.pretty is False

    def test_repr(self):
        """Test string representation."""
        checkpointer = FileCheckpointer("/tmp/test")
        assert "FileCheckpointer" in repr(checkpointer)
        assert "/tmp/test" in repr(checkpointer)


class TestFileCheckpointerPaths:
    """Tests for path generation."""

    @pytest.fixture
    def checkpointer(self):
        """Create checkpointer with temp dir."""
        return FileCheckpointer("/tmp/test_checkpoints")

    def test_get_thread_dir(self, checkpointer):
        """Test thread directory path generation."""
        path = checkpointer._get_thread_dir("thread-123")
        assert path == Path("/tmp/test_checkpoints/thread-123")

    def test_get_thread_dir_sanitizes_unsafe_chars(self, checkpointer):
        """Test that unsafe characters are sanitized."""
        path = checkpointer._get_thread_dir("thread/with\\special:chars")
        assert "/" not in path.name
        assert "\\" not in path.name
        assert ":" not in path.name

    def test_get_checkpoint_path(self, checkpointer):
        """Test checkpoint file path generation."""
        path = checkpointer._get_checkpoint_path("thread1", "cp-123")
        assert path == Path("/tmp/test_checkpoints/thread1/cp-123.json")

    def test_get_storage_path(self, checkpointer):
        """Test getting storage path."""
        assert checkpointer.get_storage_path() == Path("/tmp/test_checkpoints")


class TestFileCheckpointerSave:
    """Tests for save operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temp directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def checkpointer(self, temp_dir):
        """Create checkpointer with temp dir."""
        return FileCheckpointer(temp_dir)

    @pytest.fixture
    def state(self):
        """Create test state."""
        return AgentState()

    @pytest.mark.asyncio
    async def test_save_creates_directory(self, checkpointer, state, temp_dir):
        """Test that save creates thread directory."""
        await checkpointer.save(state, "new-thread")
        thread_dir = temp_dir / "new-thread"
        assert thread_dir.exists()

    @pytest.mark.asyncio
    async def test_save_creates_json_file(self, checkpointer, state, temp_dir):
        """Test that save creates JSON file."""
        cp_id = await checkpointer.save(state, "thread1", checkpoint_id="cp-123")
        file_path = temp_dir / "thread1" / "cp-123.json"
        assert file_path.exists()
        assert cp_id == "cp-123"

    @pytest.mark.asyncio
    async def test_save_generates_checkpoint_id(self, checkpointer, state):
        """Test that save generates checkpoint ID if not provided."""
        cp_id = await checkpointer.save(state, "thread1")
        assert cp_id is not None
        assert len(cp_id) == 32  # UUID hex

    @pytest.mark.asyncio
    async def test_save_json_structure(self, checkpointer, state, temp_dir):
        """Test saved JSON structure."""
        await checkpointer.save(state, "thread1", checkpoint_id="cp1")
        file_path = temp_dir / "thread1" / "cp1.json"

        with open(file_path) as f:
            data = json.load(f)

        assert data["checkpoint_id"] == "cp1"
        assert data["thread_id"] == "thread1"
        assert "created_at" in data
        assert "state" in data


class TestFileCheckpointerLoad:
    """Tests for load operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temp directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def checkpointer(self, temp_dir):
        """Create checkpointer with temp dir."""
        return FileCheckpointer(temp_dir)

    @pytest.fixture
    def state(self):
        """Create test state."""
        return AgentState()

    @pytest.mark.asyncio
    async def test_load_nonexistent_thread(self, checkpointer):
        """Test loading from nonexistent thread."""
        result = await checkpointer.load("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_empty_thread(self, checkpointer, temp_dir):
        """Test loading from empty thread directory."""
        # Create empty thread directory
        (temp_dir / "empty-thread").mkdir()
        result = await checkpointer.load("empty-thread")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_specific_checkpoint(self, checkpointer, state):
        """Test loading a specific checkpoint."""
        await checkpointer.save(state, "thread1", checkpoint_id="cp1")
        await checkpointer.save(state, "thread1", checkpoint_id="cp2")

        loaded = await checkpointer.load("thread1", checkpoint_id="cp1")
        assert loaded is not None

    @pytest.mark.asyncio
    async def test_load_latest_checkpoint(self, checkpointer, state):
        """Test loading latest checkpoint."""
        await checkpointer.save(state, "thread1", checkpoint_id="cp1")
        await checkpointer.save(state, "thread1", checkpoint_id="cp2")

        loaded = await checkpointer.load("thread1")
        assert loaded is not None

    @pytest.mark.asyncio
    async def test_load_nonexistent_checkpoint(self, checkpointer, state):
        """Test loading nonexistent checkpoint ID."""
        await checkpointer.save(state, "thread1", checkpoint_id="cp1")
        result = await checkpointer.load("thread1", checkpoint_id="nonexistent")
        assert result is None


class TestFileCheckpointerListCheckpoints:
    """Tests for list_checkpoints operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temp directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def checkpointer(self, temp_dir):
        """Create checkpointer with temp dir."""
        return FileCheckpointer(temp_dir)

    @pytest.mark.asyncio
    async def test_list_empty_thread(self, checkpointer):
        """Test listing checkpoints for nonexistent thread."""
        result = await checkpointer.list_checkpoints("nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, checkpointer):
        """Test listing checkpoints."""
        state = AgentState()
        await checkpointer.save(state, "thread1", checkpoint_id="cp1")
        await checkpointer.save(state, "thread1", checkpoint_id="cp2")
        await checkpointer.save(state, "thread1", checkpoint_id="cp3")

        result = await checkpointer.list_checkpoints("thread1")
        assert len(result) == 3
        assert set(result) == {"cp1", "cp2", "cp3"}

    @pytest.mark.asyncio
    async def test_list_checkpoints_with_limit(self, checkpointer):
        """Test listing checkpoints with limit."""
        state = AgentState()
        for i in range(10):
            await checkpointer.save(state, "thread1", checkpoint_id=f"cp{i}")

        result = await checkpointer.list_checkpoints("thread1", limit=5)
        assert len(result) == 5


class TestFileCheckpointerDelete:
    """Tests for delete operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temp directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def checkpointer(self, temp_dir):
        """Create checkpointer with temp dir."""
        return FileCheckpointer(temp_dir)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_thread(self, checkpointer):
        """Test deleting nonexistent thread."""
        result = await checkpointer.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_all_checkpoints(self, checkpointer, temp_dir):
        """Test deleting all checkpoints for a thread."""
        state = AgentState()
        await checkpointer.save(state, "thread1", checkpoint_id="cp1")
        await checkpointer.save(state, "thread1", checkpoint_id="cp2")

        result = await checkpointer.delete("thread1")
        assert result is True
        assert not (temp_dir / "thread1").exists()

    @pytest.mark.asyncio
    async def test_delete_specific_checkpoint(self, checkpointer, temp_dir):
        """Test deleting a specific checkpoint."""
        state = AgentState()
        await checkpointer.save(state, "thread1", checkpoint_id="cp1")
        await checkpointer.save(state, "thread1", checkpoint_id="cp2")

        result = await checkpointer.delete("thread1", checkpoint_id="cp1")
        assert result is True
        assert not (temp_dir / "thread1" / "cp1.json").exists()
        assert (temp_dir / "thread1" / "cp2.json").exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_checkpoint(self, checkpointer):
        """Test deleting nonexistent checkpoint."""
        state = AgentState()
        await checkpointer.save(state, "thread1", checkpoint_id="cp1")

        result = await checkpointer.delete("thread1", checkpoint_id="nonexistent")
        assert result is False


class TestFileCheckpointerDiskUsage:
    """Tests for disk usage operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temp directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def checkpointer(self, temp_dir):
        """Create checkpointer with temp dir."""
        return FileCheckpointer(temp_dir)

    @pytest.mark.asyncio
    async def test_disk_usage_nonexistent_thread(self, checkpointer):
        """Test disk usage for nonexistent thread."""
        usage = await checkpointer.get_disk_usage("nonexistent")
        assert usage == 0

    @pytest.mark.asyncio
    async def test_disk_usage_for_thread(self, checkpointer):
        """Test disk usage for specific thread."""
        state = AgentState()
        await checkpointer.save(state, "thread1", checkpoint_id="cp1")

        usage = await checkpointer.get_disk_usage("thread1")
        assert usage > 0

    @pytest.mark.asyncio
    async def test_disk_usage_total_nonexistent_dir(self, temp_dir):
        """Test total disk usage when base dir doesn't exist."""
        checkpointer = FileCheckpointer(temp_dir / "nonexistent")
        usage = await checkpointer.get_disk_usage()
        assert usage == 0

    @pytest.mark.asyncio
    async def test_disk_usage_total(self, checkpointer):
        """Test total disk usage."""
        state = AgentState()
        await checkpointer.save(state, "thread1", checkpoint_id="cp1")
        await checkpointer.save(state, "thread2", checkpoint_id="cp1")

        usage = await checkpointer.get_disk_usage()
        assert usage > 0


class TestFileCheckpointerJsonWriting:
    """Tests for JSON writing functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temp directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_write_json_pretty(self, temp_dir):
        """Test writing pretty JSON."""
        checkpointer = FileCheckpointer(temp_dir, pretty=True)
        path = temp_dir / "test.json"
        data = {"key": "value", "nested": {"a": 1}}

        checkpointer._write_json(path, data)

        content = path.read_text()
        assert "  " in content  # Indentation

    def test_write_json_compact(self, temp_dir):
        """Test writing compact JSON."""
        checkpointer = FileCheckpointer(temp_dir, pretty=False)
        path = temp_dir / "test.json"
        data = {"key": "value"}

        checkpointer._write_json(path, data)

        content = path.read_text()
        assert "\n" not in content.strip()

    def test_read_json_nonexistent(self, temp_dir):
        """Test reading nonexistent JSON file."""
        checkpointer = FileCheckpointer(temp_dir)
        result = checkpointer._read_json(temp_dir / "nonexistent.json")
        assert result is None

    def test_read_json_existing(self, temp_dir):
        """Test reading existing JSON file."""
        checkpointer = FileCheckpointer(temp_dir)
        path = temp_dir / "test.json"
        path.write_text('{"key": "value"}')

        result = checkpointer._read_json(path)
        assert result == {"key": "value"}
