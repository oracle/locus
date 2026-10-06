# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""FileCheckpointer as a session store: listing, retention and crash safety.

A coding agent keeps every conversation on disk and picks one up by id, the
latest one, or after the process died mid-write. That needs threads listed
newest first under the ids they were saved with, a bound on what a long
thread keeps, old sessions swept away, and a torn file that costs one
checkpoint rather than the whole thread.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tulip.core.messages import Message
from tulip.core.state import AgentState
from tulip.memory.backends.file import FileCheckpointer


def _state(text: str) -> AgentState:
    return AgentState(messages=(Message.user(text),))


def _age(path: Path, days: float) -> None:
    then = time.time() - days * 86_400
    os.utime(path, (then, then))


async def test_capabilities_advertise_listing_and_vacuum(tmp_path: Path) -> None:
    caps = FileCheckpointer(tmp_path).capabilities
    assert caps.list_threads
    assert caps.vacuum
    assert caps.list_with_metadata
    assert FileCheckpointer(tmp_path).deletes_single_checkpoints


async def test_list_threads_newest_first_under_the_ids_they_were_saved_with(
    tmp_path: Path,
) -> None:
    store = FileCheckpointer(tmp_path)
    await store.save(_state("a"), "older")
    await store.save(_state("b"), "session#1")
    _age(next((tmp_path / "older").glob("*.json")), 1)

    # The directory is sanitised to ``session_1``; the id handed back is the
    # one load() was given, so it round-trips.
    assert await store.list_threads() == ["session#1", "older"]
    assert await store.load("session#1") is not None
    assert await store.list_threads(limit=1) == ["session#1"]
    assert await store.list_threads(pattern="old*") == ["older"]


async def test_list_threads_on_an_empty_store(tmp_path: Path) -> None:
    assert await FileCheckpointer(tmp_path / "missing").list_threads() == []
    (tmp_path / "empty-thread").mkdir()
    (tmp_path / "stray-file").write_text("x")
    assert await FileCheckpointer(tmp_path).list_threads() == []


async def test_list_with_metadata_is_newest_first_and_carries_no_state(tmp_path: Path) -> None:
    store = FileCheckpointer(tmp_path)
    first = await store.save(_state("a"), "t1", metadata={"turn": 1})
    second = await store.save(_state("b"), "t2", metadata={"turn": 2})

    entries = await store.list_with_metadata()

    assert [e["checkpoint_id"] for e in entries] == [second, first]
    assert entries[0]["thread_id"] == "t2"
    assert entries[0]["metadata"] == {"turn": 2}
    assert "state" not in entries[0]
    assert len(await store.list_with_metadata(limit=1)) == 1


async def test_max_checkpoints_per_thread_keeps_the_newest(tmp_path: Path) -> None:
    store = FileCheckpointer(tmp_path, max_checkpoints_per_thread=2)
    ids = [await store.save(_state(str(i)), "t") for i in range(4)]

    assert await store.list_checkpoints("t") == [ids[3], ids[2]]
    latest = await store.load("t")
    assert latest is not None
    assert latest.messages[0].content == "3"


def test_max_checkpoints_per_thread_must_keep_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        FileCheckpointer(tmp_path, max_checkpoints_per_thread=0)


async def test_vacuum_deletes_old_checkpoints_and_empty_threads(tmp_path: Path) -> None:
    store = FileCheckpointer(tmp_path)
    await store.save(_state("stale"), "stale")
    old = await store.save(_state("old turn"), "live")
    new = await store.save(_state("new turn"), "live")
    _age(tmp_path / "stale" / next(os.scandir(tmp_path / "stale")).name, 10)
    _age(tmp_path / "live" / f"{old}.json", 10)

    assert await store.vacuum(older_than_days=7) == 2

    assert not (tmp_path / "stale").exists()
    assert await store.list_checkpoints("live") == [new]
    assert await store.vacuum(older_than_days=7) == 0


async def test_a_write_never_leaves_a_partial_file(tmp_path: Path) -> None:
    store = FileCheckpointer(tmp_path)
    checkpoint_id = await store.save(_state("x"), "t")

    files = sorted(p.name for p in (tmp_path / "t").iterdir())
    assert files == [f"{checkpoint_id}.json"]


async def test_a_torn_checkpoint_is_skipped_not_fatal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = FileCheckpointer(tmp_path)
    good = await store.save(_state("good"), "t")
    torn = tmp_path / "t" / "torn.json"
    torn.write_text('{"checkpoint_id": "torn", "state": {"mess')

    assert await store.list_checkpoints("t") == [good]
    loaded = await store.load("t")
    assert loaded is not None
    assert loaded.messages[0].content == "good"
    assert await store.load("t", "torn") is None
    assert "unreadable checkpoint" in caplog.text


async def test_a_checkpoint_file_holding_a_non_object_is_ignored(tmp_path: Path) -> None:
    store = FileCheckpointer(tmp_path)
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "list.json").write_text("[1, 2]")
    assert await store.list_checkpoints("t") == []
    assert await store.list_threads() == ["t"]


async def test_latest_falls_back_past_a_damaged_newest_checkpoint(tmp_path: Path) -> None:
    store = FileCheckpointer(tmp_path)
    await store.save(_state("good"), "t")
    # Ends in a brace, so listing (head and tail only) keeps it; the body
    # does not parse, so load() moves on to the one before.
    (tmp_path / "t" / "bad.json").write_text(
        '{"checkpoint_id": "bad", "thread_id": "t", '
        '"created_at": "2999-01-01T00:00:00+00:00", "state": {oops}'
    )

    assert (await store.list_checkpoints("t"))[0] == "bad"
    loaded = await store.load("t")
    assert loaded is not None
    assert loaded.messages[0].content == "good"


async def test_listing_reads_a_file_written_in_another_key_order(tmp_path: Path) -> None:
    store = FileCheckpointer(tmp_path)
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "odd.json").write_text(
        '{"state": {"messages": []}, "created_at": "2026-01-01T00:00:00+00:00", '
        '"checkpoint_id": "odd"}'
    )
    (tmp_path / "t" / "no-id.json").write_text('{"state": {}}')

    assert await store.list_checkpoints("t") == ["odd"]
    assert await store.load("t") is not None


async def test_listing_skips_a_file_it_cannot_open(tmp_path: Path) -> None:
    store = FileCheckpointer(tmp_path)
    good = await store.save(_state("good"), "t")
    (tmp_path / "t" / "dir.json").mkdir()

    assert await store.list_checkpoints("t") == [good]


async def test_list_with_metadata_skips_files_that_are_not_checkpoints(tmp_path: Path) -> None:
    store = FileCheckpointer(tmp_path)
    good = await store.save(_state("good"), "t")
    (tmp_path / "t" / "junk.json").write_text('{"state": {}}')

    assert [e["checkpoint_id"] for e in await store.list_with_metadata()] == [good]
