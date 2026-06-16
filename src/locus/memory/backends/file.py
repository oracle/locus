# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""File-based checkpoint backend for Locus.

This backend stores checkpoints as JSON files on the local filesystem,
providing:
- Persistent storage across process restarts
- Easy inspection and debugging
- Simple setup with no external dependencies

Directory structure:
    base_dir/
        thread_id_1/
            checkpoint_1.json
            checkpoint_2.json
        thread_id_2/
            checkpoint_1.json
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from locus.memory.checkpointer import BaseCheckpointer


if TYPE_CHECKING:
    from locus.core.state import AgentState


class FileCheckpointer(BaseCheckpointer):
    """
    File-based checkpointer for persistent local storage.

    Stores each checkpoint as a JSON file, organized by thread ID.
    Provides durable storage that survives process restarts.

    Args:
        base_dir: Base directory for checkpoint storage.
                  Defaults to ".locus_checkpoints" in current directory.
        pretty: Whether to format JSON for readability (default True)

    Example:
        ```python
        checkpointer = FileCheckpointer("./checkpoints")

        # Save state
        checkpoint_id = await checkpointer.save(state, "thread-1")

        # Load state
        restored = await checkpointer.load("thread-1")

        # Files are stored at: ./checkpoints/thread-1/{checkpoint_id}.json
        ```
    """

    def __init__(
        self,
        base_dir: str | Path = ".locus_checkpoints",
        pretty: bool = True,
    ):
        self.base_dir = Path(base_dir)
        self.pretty = pretty
        self._lock = asyncio.Lock()

    def _get_thread_dir(self, thread_id: str) -> Path:
        """Get directory path for a thread."""
        # Sanitize thread_id to be filesystem-safe
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in thread_id)
        return self.base_dir / safe_id

    def _get_checkpoint_path(self, thread_id: str, checkpoint_id: str) -> Path:
        """Get file path for a checkpoint."""
        safe_cp_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in checkpoint_id)
        return self._get_thread_dir(thread_id) / f"{safe_cp_id}.json"

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save agent state to a JSON file.

        Args:
            state: Current agent state
            thread_id: Thread identifier
            checkpoint_id: Optional specific checkpoint ID
            metadata: Optional metadata for querying/filtering checkpoints

        Returns:
            Checkpoint ID for the saved state
        """
        checkpoint_id = checkpoint_id or uuid4().hex

        async with self._lock:
            thread_dir = self._get_thread_dir(thread_id)
            thread_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = self._get_checkpoint_path(thread_id, checkpoint_id)

            # Prepare data with metadata
            data = {
                "checkpoint_id": checkpoint_id,
                "thread_id": thread_id,
                "created_at": datetime.now(UTC).isoformat(),
                "state": state.to_checkpoint(),
                "metadata": metadata or {},
            }

            # Write to file (run in executor to not block)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._write_json, checkpoint_path, data)

        return checkpoint_id

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        """Write JSON data to file (sync, for executor)."""
        with open(path, "w", encoding="utf-8") as f:
            if self.pretty:
                json.dump(data, f, indent=2, default=str)
            else:
                json.dump(data, f, default=str)

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        """Read JSON data from file (sync, for executor)."""
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            return data

    async def load(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> AgentState | None:
        """
        Load agent state from a JSON file.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint ID (latest if None)

        Returns:
            Restored AgentState or None if not found
        """
        from locus.core.state import AgentState

        thread_dir = self._get_thread_dir(thread_id)

        if not thread_dir.exists():
            return None

        if checkpoint_id is None:
            # Get latest checkpoint
            checkpoints = await self.list_checkpoints(thread_id, limit=1)
            if not checkpoints:
                return None
            checkpoint_id = checkpoints[0]

        checkpoint_path = self._get_checkpoint_path(thread_id, checkpoint_id)

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self._read_json, checkpoint_path)

        if data is None:
            return None

        return AgentState.from_checkpoint(data["state"])

    async def list_checkpoints(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[str]:
        """
        List available checkpoints for a thread.

        Reads checkpoint files and returns IDs sorted by creation time
        (newest first).

        Args:
            thread_id: Thread identifier
            limit: Maximum number to return

        Returns:
            List of checkpoint IDs, newest first
        """
        thread_dir = self._get_thread_dir(thread_id)

        if not thread_dir.exists():
            return []

        # Get all checkpoint files with their metadata
        checkpoints: list[tuple[str, datetime]] = []

        loop = asyncio.get_event_loop()

        for path in thread_dir.glob("*.json"):
            data = await loop.run_in_executor(None, self._read_json, path)
            if data and "checkpoint_id" in data:
                created_at = datetime.fromisoformat(
                    data.get("created_at", "1970-01-01T00:00:00+00:00")
                )
                checkpoints.append((data["checkpoint_id"], created_at))

        # Sort by creation time descending
        checkpoints.sort(key=lambda x: x[1], reverse=True)

        return [cp_id for cp_id, _ in checkpoints[:limit]]

    async def delete(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> bool:
        """
        Delete checkpoint file(s).

        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint to delete (all if None)

        Returns:
            True if deletion was successful
        """
        import shutil

        thread_dir = self._get_thread_dir(thread_id)

        if not thread_dir.exists():
            return False

        async with self._lock:
            if checkpoint_id is None:
                # Delete entire thread directory
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, lambda: shutil.rmtree(thread_dir, ignore_errors=True)
                )
                return True
            checkpoint_path = self._get_checkpoint_path(thread_id, checkpoint_id)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                return True
            return False

    def get_storage_path(self) -> Path:
        """Get the base storage directory path."""
        return self.base_dir

    async def get_disk_usage(self, thread_id: str | None = None) -> int:
        """
        Get total disk usage in bytes.

        Args:
            thread_id: Specific thread (all threads if None)

        Returns:
            Total size in bytes
        """
        if thread_id is not None:
            thread_dir = self._get_thread_dir(thread_id)
            if not thread_dir.exists():
                return 0
            return sum(f.stat().st_size for f in thread_dir.glob("*.json"))

        if not self.base_dir.exists():
            return 0

        total = 0
        for thread_dir in self.base_dir.iterdir():
            if thread_dir.is_dir():
                total += sum(f.stat().st_size for f in thread_dir.glob("*.json"))
        return total

    def __repr__(self) -> str:
        return f"FileCheckpointer(base_dir={self.base_dir!r})"
