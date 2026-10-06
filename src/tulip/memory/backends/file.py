# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""File-based checkpoint backend for Tulip.

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

Each checkpoint is written to a temporary file and renamed into place, so a
process killed mid-write leaves the previous checkpoint intact rather than a
truncated JSON file; a file that does not parse anyway (written by an older
version, or by hand) is skipped with a warning rather than making the whole
thread unloadable.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import re
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from tulip.core.protocols import CheckpointerCapabilities
from tulip.memory.checkpointer import BaseCheckpointer


if TYPE_CHECKING:
    from tulip.core.state import AgentState


logger = logging.getLogger(__name__)

#: Bytes read from the head of a checkpoint file to find its id and time.
_HEADER_BYTES = 1024
#: A JSON string value: the quotes and everything up to the closing one.
_JSON_STRING = r'("(?:[^"\\]|\\.)*")'
_CHECKPOINT_ID = re.compile(rb'^\{\s*"checkpoint_id":\s*' + _JSON_STRING.encode())
_CREATED_AT = re.compile(rb'"created_at":\s*' + _JSON_STRING.encode())
_EPOCH = "1970-01-01T00:00:00+00:00"


class FileCheckpointer(BaseCheckpointer):
    """
    File-based checkpointer for persistent local storage.

    Stores each checkpoint as a JSON file, organized by thread ID.
    Provides durable storage that survives process restarts.

    Args:
        base_dir: Base directory for checkpoint storage.
                  Defaults to ".tulip_checkpoints" in current directory.
        pretty: Whether to format JSON for readability (default True)
        max_checkpoints_per_thread: Keep only this many of a thread's newest
            checkpoints, deleting older ones after each save. ``None`` (the
            default) keeps them all. Each checkpoint holds the whole
            conversation, so a long thread's storage grows with the square
            of its length; a cap bounds it, at the cost of ``fork`` and
            ``get_state_history`` reaching back only that far.

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
        base_dir: str | Path = ".tulip_checkpoints",
        pretty: bool = True,
        max_checkpoints_per_thread: int | None = None,
    ):
        if max_checkpoints_per_thread is not None and max_checkpoints_per_thread < 1:
            raise ValueError("max_checkpoints_per_thread must be at least 1")
        self.base_dir = Path(base_dir)
        self.pretty = pretty
        self.max_checkpoints_per_thread = max_checkpoints_per_thread
        self._lock = asyncio.Lock()

    @property
    def capabilities(self) -> CheckpointerCapabilities:
        """Threads can be listed and old checkpoints vacuumed."""
        return CheckpointerCapabilities(
            vacuum=True,
            list_threads=True,
            list_with_metadata=True,
            persistent_checkpoint_ids=True,
        )

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

        if self.max_checkpoints_per_thread is not None:
            await self._trim(thread_id, self.max_checkpoints_per_thread)

        return checkpoint_id

    async def _trim(self, thread_id: str, keep: int) -> None:
        """Delete all but the ``keep`` newest checkpoints of a thread."""
        for old in (await self.list_checkpoints(thread_id, limit=1_000_000))[keep:]:
            await self.delete(thread_id, old)

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        """Write JSON data to file (sync, for executor).

        Written beside the target and renamed over it: the rename is atomic,
        so a reader — or a process restarted after a kill — sees the old
        checkpoint or the new one, never half of one.
        """
        tmp = path.with_name(f"{path.name}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            if self.pretty:
                json.dump(data, f, indent=2, default=str)
            else:
                json.dump(data, f, default=str)
        tmp.replace(path)

    def _read_header(self, path: Path) -> tuple[str, str] | None:
        """``(checkpoint_id, created_at)`` of a checkpoint file, without parsing its state.

        Listing a thread used to parse every checkpoint in full — each one the
        whole conversation — to sort them, which made resuming a long session
        slow. The two fields are written ahead of the state, so the head of
        the file has them; a file whose tail is not the closing brace was cut
        short and is left out. Anything unusual falls back to a full parse.
        """
        try:
            with open(path, "rb") as f:
                head = f.read(_HEADER_BYTES)
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 16))
                tail = f.read()
        except OSError:
            return None
        if not tail.rstrip().endswith(b"}"):
            logger.warning("skipping truncated checkpoint file %s", path)
            return None
        found_id = _CHECKPOINT_ID.search(head)
        found_at = _CREATED_AT.search(head)
        if found_id and found_at:
            return json.loads(found_id.group(1)), json.loads(found_at.group(1))
        data = self._read_json(path)
        if not data or "checkpoint_id" not in data:
            return None
        return str(data["checkpoint_id"]), str(data.get("created_at", _EPOCH))

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        """Read JSON data from file (sync, for executor).

        ``None`` for a file that is missing or does not parse. One unreadable
        checkpoint must not make the rest of its thread unreachable — the
        latest good one is what a resume needs.
        """
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
        except (OSError, ValueError):
            logger.warning("skipping unreadable checkpoint file %s", path, exc_info=True)
            return None
        return data if isinstance(data, dict) else None

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
        from tulip.core.state import AgentState

        thread_dir = self._get_thread_dir(thread_id)

        if not thread_dir.exists():
            return None

        loop = asyncio.get_event_loop()
        if checkpoint_id is None:
            # The newest checkpoint that parses. Listing reads only each
            # file's head and tail, so a damaged body shows up here, and the
            # one before it is the thread's latest good state.
            for candidate in await self.list_checkpoints(thread_id, limit=1_000_000):
                path = self._get_checkpoint_path(thread_id, candidate)
                data = await loop.run_in_executor(None, self._read_json, path)
                if data is not None and "state" in data:
                    return AgentState.from_checkpoint(data["state"])
            return None

        checkpoint_path = self._get_checkpoint_path(thread_id, checkpoint_id)
        data = await loop.run_in_executor(None, self._read_json, checkpoint_path)

        if data is None or "state" not in data:
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
            header = await loop.run_in_executor(None, self._read_header, path)
            if header is not None:
                checkpoint_id, created_at = header
                checkpoints.append((checkpoint_id, datetime.fromisoformat(created_at)))

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

    def _thread_dirs(self) -> list[tuple[float, Path]]:
        """Thread directories with the time of their newest checkpoint, newest first."""
        if not self.base_dir.exists():
            return []
        found: list[tuple[float, Path]] = []
        for thread_dir in self.base_dir.iterdir():
            if not thread_dir.is_dir():
                continue
            stamps = [f.stat().st_mtime for f in thread_dir.glob("*.json")]
            if stamps:
                found.append((max(stamps), thread_dir))
        found.sort(key=lambda item: item[0], reverse=True)
        return found

    def _recorded_thread_id(self, thread_dir: Path) -> str:
        """The thread id a directory's newest checkpoint records.

        Directory names are sanitised (``a#1`` is stored as ``a_1``), so the
        id to hand back to :meth:`load` is the one inside the file.
        """
        newest = max(thread_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, default=None)
        data = self._read_json(newest) if newest is not None else None
        recorded = (data or {}).get("thread_id")
        return recorded if isinstance(recorded, str) and recorded else thread_dir.name

    async def list_threads(self, limit: int = 100, pattern: str = "*") -> list[str]:
        """Thread ids, most recently saved first.

        Args:
            limit: Maximum threads to return.
            pattern: A shell-style pattern the thread id must match.
        """
        loop = asyncio.get_event_loop()
        threads: list[str] = []
        for _, thread_dir in await loop.run_in_executor(None, self._thread_dirs):
            thread_id = await loop.run_in_executor(None, self._recorded_thread_id, thread_dir)
            if fnmatch.fnmatchcase(thread_id, pattern):
                threads.append(thread_id)
                if len(threads) >= limit:
                    break
        return threads

    async def list_with_metadata(self, limit: int = 100) -> list[dict[str, Any]]:
        """Checkpoints across all threads, newest first, without their state.

        Each entry has ``thread_id``, ``checkpoint_id``, ``created_at`` (an
        ISO-8601 string) and the ``metadata`` it was saved with.
        """
        loop = asyncio.get_event_loop()
        entries: list[dict[str, Any]] = []
        for _, thread_dir in await loop.run_in_executor(None, self._thread_dirs):
            for path in thread_dir.glob("*.json"):
                data = await loop.run_in_executor(None, self._read_json, path)
                if not data or "checkpoint_id" not in data:
                    continue
                entries.append(
                    {
                        "thread_id": data.get("thread_id", thread_dir.name),
                        "checkpoint_id": data["checkpoint_id"],
                        "created_at": data.get("created_at", "1970-01-01T00:00:00+00:00"),
                        "metadata": data.get("metadata") or {},
                    }
                )
        entries.sort(key=lambda e: datetime.fromisoformat(e["created_at"]), reverse=True)
        return entries[:limit]

    async def vacuum(self, older_than_days: int = 30) -> int:
        """Delete checkpoints saved more than ``older_than_days`` days ago.

        A thread left with no checkpoints is removed with its directory.
        Returns how many checkpoints were deleted.
        """
        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).timestamp()

        def sweep() -> int:
            deleted = 0
            for _, thread_dir in self._thread_dirs():
                for path in thread_dir.glob("*.json"):
                    if path.stat().st_mtime < cutoff:
                        path.unlink(missing_ok=True)
                        deleted += 1
                if not any(thread_dir.glob("*.json")):
                    shutil.rmtree(thread_dir, ignore_errors=True)
            return deleted

        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, sweep)

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
