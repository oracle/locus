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

from locus.core.protocols import CheckpointerCapabilities
from locus.memory.checkpointer import BaseCheckpointer
from locus.memory.expiry import is_expired
from locus.memory.time_travel import (
    CheckpointInfo,
    checkpoint_iteration,
    checkpoint_size,
    info_from,
    link_parents,
)


if TYPE_CHECKING:
    from locus.core.state import AgentState
    from locus.memory.encryption import EncryptedSerializer
    from locus.memory.scope import CheckpointScope


logger = logging.getLogger(__name__)

#: Bytes read from the head of a checkpoint file to find its id and time.
_HEADER_BYTES = 1024
#: Bytes read from the head to find the step fields and metadata as well;
#: they are written ahead of the state, so this covers any metadata of a
#: sane size, and :meth:`FileCheckpointer.list` parses the file when not.
_INFO_BYTES = 8192
#: A JSON string value: the quotes and everything up to the closing one.
_JSON_STRING = r'("(?:[^"\\]|\\.)*")'
_CHECKPOINT_ID = re.compile(rb'^\{\s*"checkpoint_id":\s*' + _JSON_STRING.encode())
_CREATED_AT = re.compile(rb'"created_at":\s*' + _JSON_STRING.encode())
#: The expiry a file written with a ``ttl`` records, right after ``created_at``.
_EXPIRES_AT = re.compile(rb'"expires_at":\s*' + _JSON_STRING.encode())
#: Where the state begins: everything before it is the header.
_STATE_KEY = re.compile(rb'"state":\s*')
#: The header fields a file written since the step fields were added carries.
_INFO_FIELDS = ("checkpoint_id", "created_at", "iteration", "size", "metadata")
_EPOCH = "1970-01-01T00:00:00+00:00"


class FileCheckpointer(BaseCheckpointer):
    """
    File-based checkpointer for persistent local storage.

    Stores each checkpoint as a JSON file, organized by thread ID.
    Provides durable storage that survives process restarts.

    Args:
        base_dir: Base directory for checkpoint storage.
                  Defaults to ".locus_checkpoints" in current directory.
        pretty: Whether to format JSON for readability (default True)
        max_checkpoints_per_thread: Keep only this many of a thread's newest
            checkpoints, deleting older ones after each save. ``None`` (the
            default) keeps them all. Each checkpoint holds the whole
            conversation, so a long thread's storage grows with the square
            of its length; a cap bounds it, at the cost of ``fork`` and
            ``get_state_history`` reaching back only that far.
        ttl: How long a checkpoint stays loadable after its save (seconds or
            a ``timedelta``); ``None`` keeps it until deleted. The expiry is
            written in the file's header and :meth:`expire` deletes the files
            whose time has passed.
        encryption: Seals the state before it is written
            (:class:`~locus.memory.encryption.EncryptedSerializer`); the
            header stays readable.
        scope: The application and user the threads belong to
            (:class:`~locus.memory.scope.CheckpointScope`); a scoped thread's
            directory is named for its scope and its id.

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
        max_checkpoints_per_thread: int | None = None,
        *,
        ttl: float | timedelta | None = None,
        encryption: EncryptedSerializer | None = None,
        scope: CheckpointScope | None = None,
    ):
        if max_checkpoints_per_thread is not None and max_checkpoints_per_thread < 1:
            raise ValueError("max_checkpoints_per_thread must be at least 1")
        self.base_dir = Path(base_dir)
        self.pretty = pretty
        self.max_checkpoints_per_thread = max_checkpoints_per_thread
        self._configure(ttl=ttl, encryption=encryption, scope=scope)
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

    def _get_thread_dir(self, thread_key: str) -> Path:
        """Get directory path for a thread, from the id it is stored under (its scope included)."""
        # Sanitize thread_id to be filesystem-safe
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in thread_key)
        return self.base_dir / safe_id

    def _get_checkpoint_path(self, thread_key: str, checkpoint_id: str) -> Path:
        """Get file path for a checkpoint."""
        safe_cp_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in checkpoint_id)
        return self._get_thread_dir(thread_key) / f"{safe_cp_id}.json"

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
        key = self._scoped(thread_id)

        async with self._lock:
            thread_dir = self._get_thread_dir(key)
            thread_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = self._get_checkpoint_path(key, checkpoint_id)

            # The fields ``list`` reports come before the state, so a listing
            # reads the head of each file and never the conversation itself.
            # The expiry follows the write time, where the header read finds it.
            state_data = self._encode_state(state)
            now = datetime.now(UTC)
            expires_at = self._expires_at(now)
            data = {
                "checkpoint_id": checkpoint_id,
                "thread_id": key,
                "created_at": now.isoformat(),
                "expires_at": expires_at.isoformat() if expires_at else None,
                "iteration": checkpoint_iteration(state_data),
                "size": checkpoint_size(state_data),
                "metadata": metadata or {},
                "state": state_data,
            }

            # Write to file (run in executor to not block)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._write_json, checkpoint_path, data)

        if self.max_checkpoints_per_thread is not None:
            await self._trim_history(thread_id, self.max_checkpoints_per_thread)

        return checkpoint_id

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
        """``(checkpoint_id, created_at)`` of a live checkpoint file, without parsing its state.

        Listing a thread used to parse every checkpoint in full — each one the
        whole conversation — to sort them, which made resuming a long session
        slow. The fields are written ahead of the state, so the head of the
        file has them; a file whose tail is not the closing brace was cut
        short and is left out, and so is one whose ``expires_at`` has passed.
        Anything unusual falls back to a full parse.
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
            found_expiry = _EXPIRES_AT.search(head)
            if found_expiry and is_expired(json.loads(found_expiry.group(1))):
                return None
            return json.loads(found_id.group(1)), json.loads(found_at.group(1))
        data = self._read_json(path)
        if not data or "checkpoint_id" not in data or is_expired(data.get("expires_at")):
            return None
        return str(data["checkpoint_id"]), str(data.get("created_at", _EPOCH))

    @staticmethod
    def _header_fields(head: bytes) -> dict[str, Any] | None:
        """The fields written ahead of the state, parsed on their own.

        The header runs up to the ``"state"`` key, so it is parsed as a
        document of its own with the state left out; nothing is searched
        for inside the state, whose messages and tool arguments may well
        hold a ``size`` or a ``metadata`` of their own. ``None`` when the
        head does not hold the whole header (an older layout has no step
        fields before the state; a large metadata runs past the head) —
        the file is parsed in full then.
        """
        found = _STATE_KEY.search(head)
        if found is None:
            return None
        try:
            fields = json.loads(head[: found.start()] + b'"state":null}')
        except ValueError:
            return None
        if not isinstance(fields, dict) or not all(k in fields for k in _INFO_FIELDS):
            return None
        return fields if isinstance(fields["metadata"], dict) else None

    def _read_info(self, path: Path, thread_id: str) -> CheckpointInfo | None:
        """What :meth:`list` reports for one checkpoint file, from its head where possible.

        Files written since the step fields were added carry ``iteration``,
        ``size`` and ``metadata`` ahead of the state (:meth:`_header_fields`);
        a file without them (an older layout) or whose metadata runs past
        the head is parsed in full, and its size is the state's as compact
        JSON — the figure the header would have recorded.
        """
        try:
            with open(path, "rb") as f:
                head = f.read(_INFO_BYTES)
        except OSError:
            return None
        header = self._header_fields(head)
        if header is not None:
            if is_expired(header.get("expires_at")):
                return None
            return info_from(
                str(header["checkpoint_id"]),
                thread_id,
                created_at=header["created_at"],
                metadata=header["metadata"],
                iteration=header["iteration"],
                size=header["size"],
            )
        data = self._read_json(path)
        if not data or "checkpoint_id" not in data or "state" not in data:
            return None
        if is_expired(data.get("expires_at")):
            return None
        state = data["state"]
        return info_from(
            str(data["checkpoint_id"]),
            thread_id,
            created_at=data.get("created_at", _EPOCH),
            metadata=data.get("metadata") or {},
            iteration=state.get("iteration") if isinstance(state, dict) else None,
            size=checkpoint_size(state) if isinstance(state, dict) else None,
        )

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
        key = self._scoped(thread_id)
        thread_dir = self._get_thread_dir(key)

        if not thread_dir.exists():
            return None

        loop = asyncio.get_event_loop()
        if checkpoint_id is None:
            # The newest checkpoint that parses. Listing reads only each
            # file's head and tail, so a damaged body shows up here, and the
            # one before it is the thread's latest good state.
            for candidate in await self.list_checkpoints(thread_id, limit=1_000_000):
                path = self._get_checkpoint_path(key, candidate)
                data = await loop.run_in_executor(None, self._read_json, path)
                if data is not None and "state" in data:
                    return self._decode_state(data["state"])
            return None

        checkpoint_path = self._get_checkpoint_path(key, checkpoint_id)
        data = await loop.run_in_executor(None, self._read_json, checkpoint_path)

        if data is None or "state" not in data or is_expired(data.get("expires_at")):
            return None

        return self._decode_state(data["state"])

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
        thread_dir = self._get_thread_dir(self._scoped(thread_id))

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
        key = self._scoped(thread_id)
        thread_dir = self._get_thread_dir(key)

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
            checkpoint_path = self._get_checkpoint_path(key, checkpoint_id)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                return True
            return False

    def _file_expired(self, path: Path) -> bool:
        """Whether a checkpoint file's recorded expiry has passed (never for one without)."""
        try:
            with open(path, "rb") as f:
                head = f.read(_HEADER_BYTES)
        except OSError:
            return False
        found = _EXPIRES_AT.search(head)
        if found:
            return is_expired(json.loads(found.group(1)))
        data = self._read_json(path)
        return is_expired((data or {}).get("expires_at"))

    async def expire(self) -> int:
        """Delete the checkpoint files whose expiry has passed, and the threads left empty."""

        def sweep() -> int:
            deleted = 0
            for _, thread_dir in self._thread_dirs():
                for path in thread_dir.glob("*.json"):
                    if self._file_expired(path):
                        path.unlink(missing_ok=True)
                        deleted += 1
                if not any(thread_dir.glob("*.json")):
                    shutil.rmtree(thread_dir, ignore_errors=True)
            return deleted

        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, sweep)

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

    def _recorded_thread_id(self, thread_dir: Path) -> str | None:
        """The stored thread id a directory's newest live checkpoint records.

        Directory names are sanitised (``a#1`` is stored as ``a_1``), so the
        id to hand back to :meth:`load` is the one inside the file. ``None``
        for a thread whose every checkpoint has expired.
        """
        newest = max(thread_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, default=None)
        if newest is None or self._file_expired(newest):
            return None
        data = self._read_json(newest)
        recorded = (data or {}).get("thread_id")
        return recorded if isinstance(recorded, str) and recorded else thread_dir.name

    async def list_threads(
        self, limit: int = 100, pattern: str = "*", *, scope: CheckpointScope | None = None
    ) -> list[str]:
        """Thread ids of one scope, most recently saved first.

        Args:
            limit: Maximum threads to return.
            pattern: A shell-style pattern the thread id must match.
            scope: The scope to list; default the call's, else the unscoped threads.
        """
        loop = asyncio.get_event_loop()
        listing = self._listing_scope(scope)
        threads: list[str] = []
        for _, thread_dir in await loop.run_in_executor(None, self._thread_dirs):
            stored = await loop.run_in_executor(None, self._recorded_thread_id, thread_dir)
            thread_id = listing.unkey(stored) if stored is not None else None
            if thread_id is not None and fnmatch.fnmatchcase(thread_id, pattern):
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
                if not data or "checkpoint_id" not in data or is_expired(data.get("expires_at")):
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
            thread_dir = self._get_thread_dir(self._scoped(thread_id))
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

    # Last on purpose: a method named ``list`` shadows the builtin for the
    # annotations of anything defined after it.
    async def list(self, thread_id: str, limit: int = 10) -> list[CheckpointInfo]:
        """The thread's checkpoints, newest first, from the head of each file."""
        thread_dir = self._get_thread_dir(self._scoped(thread_id))
        if not thread_dir.exists():
            return []
        loop = asyncio.get_event_loop()
        infos: list[CheckpointInfo] = []
        for path in thread_dir.glob("*.json"):
            info = await loop.run_in_executor(None, self._read_info, path, thread_id)
            if info is not None:
                infos.append(info)
        infos.sort(key=lambda i: i.created_at or datetime.fromisoformat(_EPOCH), reverse=True)
        # The checkpoint just past the window is the parent of the oldest in it.
        older = infos[limit].checkpoint_id if len(infos) > limit else None
        return link_parents(infos[:limit], older)
