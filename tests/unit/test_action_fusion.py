# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Action Fusion: the change, the hash guard, the per-file lock, one result."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from tulip.observability.mechanisms import MechanismLedger, bind_ledger
from tulip.tools.action_fusion import (
    RAN,
    SKIPPED,
    THEN_RUN,
    THEN_RUN_NOTE,
    CommandOutcome,
    FileLocks,
    Mutation,
    ThenRun,
    ThenRunError,
    changed_since_write,
    fusable,
    fuse,
    fused_command,
    refuse_disabled,
    sha256_file,
    sha256_text,
)
from tulip.tools.decorator import tool


@pytest.fixture
def ledger() -> object:
    book = MechanismLedger()
    bind_ledger(book)
    yield book
    bind_ledger(None)


def _write(path: Path, text: str) -> Mutation:
    path.write_text(text, encoding="utf-8")
    return Mutation(report=f"wrote {path.name}", written={path: text})


# ---------------------------------------------------------------- then_run --


def test_then_run_is_parsed_from_what_models_send() -> None:
    assert ThenRun.parse(None) is None
    assert ThenRun.parse({}) is None
    assert ThenRun.parse("") is None
    assert ThenRun.parse("  ") is None
    assert ThenRun.parse({"command": " pytest -q "}) == ThenRun("pytest -q")
    assert ThenRun.parse("pytest -q") == ThenRun("pytest -q")
    assert ThenRun.parse({"command": "make", "timeout": "30"}) == ThenRun("make", 30)
    assert ThenRun.parse({"command": "make", "timeout": 0}) == ThenRun("make", None)
    for bad in ({"timeout": 3}, {"command": ""}, 7, {"command": "x", "timeout": "soon"}):
        with pytest.raises(ThenRunError):
            ThenRun.parse(bad)


def test_then_run_arguments_are_what_a_shell_call_takes() -> None:
    assert ThenRun("pytest").arguments() == {"command": "pytest"}
    assert ThenRun("pytest", 9).arguments() == {"command": "pytest", "timeout": 9}


def test_fused_command_reads_a_calls_arguments() -> None:
    assert fused_command({"then_run": {"command": "pytest"}}) == "pytest"
    assert fused_command({"path": "a.py"}) is None
    assert fused_command({"then_run": 3}) is None


# ---------------------------------------------------------------- the guard --


def test_the_hash_guard_names_files_that_moved(tmp_path: Path) -> None:
    kept, moved, gone, back = (tmp_path / n for n in ("kept", "moved", "gone", "back"))
    kept.write_text("same", encoding="utf-8")
    moved.write_text("formatted", encoding="utf-8")
    back.write_text("recreated", encoding="utf-8")
    assert sha256_file(kept) == sha256_text("same")
    assert sha256_file(gone) is None
    problems = changed_since_write({kept: "same", moved: "raw", gone: "x", back: None})
    assert problems == [
        f"{moved} changed after it was written",
        f"{gone} can no longer be read",
        f"{back} was deleted by the change and exists again",
    ]
    assert changed_since_write({kept: "same", tmp_path / "absent": None}) == []


# ------------------------------------------------------------------ fusion --


def test_a_fused_change_returns_one_result_with_the_command(
    tmp_path: Path, ledger: MechanismLedger
) -> None:
    target = tmp_path / "a.py"
    ran: list[ThenRun] = []

    def run(spec: ThenRun) -> CommandOutcome:
        ran.append(spec)
        return CommandOutcome("exit 0\n1 passed", exit_code=0)

    out = fuse(lambda: _write(target, "x = 1\n"), ThenRun("pytest -q"), run, paths=(target,))
    assert out == f"wrote a.py\n\n{RAN} $ pytest -q\nexit 0\n1 passed"
    assert ran == [ThenRun("pytest -q")]
    summary = ledger.summary(tokens_per_step=1000)["action_fusion"]
    assert summary["steps_saved"] == 1
    assert summary["tokens_saved_est"] == 1000
    assert summary["outcomes"] == {"ran": 1}


def test_without_then_run_the_change_alone_is_returned(
    tmp_path: Path, ledger: MechanismLedger
) -> None:
    target = tmp_path / "a.py"
    out = fuse(lambda: _write(target, "x"), None, _never, paths=(target,))
    assert out == "wrote a.py"
    assert ledger.records == []


def _never(spec: ThenRun) -> CommandOutcome:
    raise AssertionError(f"ran {spec.command}")


def test_a_failed_change_skips_the_command(tmp_path: Path, ledger: MechanismLedger) -> None:
    target = tmp_path / "a.py"
    out = fuse(
        lambda: Mutation("a.py: old text not found"), ThenRun("pytest"), _never, paths=(target,)
    )
    assert out.startswith("a.py: old text not found\n\n" + SKIPPED)
    assert "`pytest` was not run" in out
    assert ledger.summary()["action_fusion"]["outcomes"] == {"edit_failed": 1}


def test_a_file_changed_after_the_write_skips_the_command(
    tmp_path: Path, ledger: MechanismLedger
) -> None:
    target = tmp_path / "a.py"

    def mutate() -> Mutation:
        done = _write(target, "x = 1\n")
        # A formatter or a watcher rewrites it between the write and the run.
        target.write_text("x = 1  # reformatted\n", encoding="utf-8")
        return done

    out = fuse(mutate, ThenRun("pytest"), _never, paths=(target,))
    assert f"{SKIPPED} {target} changed after it was written" in out
    assert "Read the file again" in out
    assert ledger.summary()["action_fusion"]["outcomes"] == {"file_changed": 1}


def test_a_refused_command_keeps_the_change(tmp_path: Path, ledger: MechanismLedger) -> None:
    target = tmp_path / "a.py"
    out = fuse(
        lambda: _write(target, "x"),
        ThenRun("rm -rf /"),
        lambda _: CommandOutcome("bash refused: recursive delete", refused=True),
        paths=(target,),
    )
    assert out == f"wrote a.py\n\n{SKIPPED} bash refused: recursive delete\nThe change stands."
    assert ledger.summary()["action_fusion"]["steps_saved"] == 0


def test_the_result_shows_the_command_that_actually_ran(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    out = fuse(
        lambda: _write(target, "x"),
        ThenRun("pytest"),
        lambda _: CommandOutcome("exit 0", exit_code=0, command="pytest -q"),
        paths=(target,),
    )
    assert out.endswith(f"{RAN} $ pytest -q\nexit 0")


def test_an_unfinished_command_is_recorded_as_such(tmp_path: Path, ledger: MechanismLedger) -> None:
    target = tmp_path / "a.py"
    fuse(
        lambda: _write(target, "x"),
        ThenRun("sleep 999"),
        lambda _: CommandOutcome("timed out after 1s — still running as sh1"),
        paths=(target,),
    )
    assert ledger.summary()["action_fusion"]["outcomes"] == {"unfinished": 1}


def test_switched_off_the_command_is_not_run(ledger: MechanismLedger) -> None:
    out = refuse_disabled("edited a.py", ThenRun("pytest"))
    assert out.startswith(f"edited a.py\n\n{SKIPPED} then_run is switched off")
    assert ledger.summary()["action_fusion"]["outcomes"] == {"disabled": 1}


# ------------------------------------------------------------------- locks --


def test_two_fused_calls_on_one_file_never_interleave(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    events: list[str] = []
    entered = threading.Event()

    def call(name: str, spelled: Path) -> None:
        def mutate() -> Mutation:
            events.append(f"{name}:write")
            entered.set()
            return _write(target, name)

        def run(_: ThenRun) -> CommandOutcome:
            time.sleep(0.05)
            events.append(f"{name}:run")
            return CommandOutcome("exit 0", exit_code=0)

        fuse(mutate, ThenRun("check"), run, paths=(spelled,))

    (tmp_path / "sub").mkdir()
    first = threading.Thread(target=call, args=("one", target))
    first.start()
    entered.wait(2)
    # The same file by another spelling: still one lock.
    second = threading.Thread(target=call, args=("two", tmp_path / "sub" / ".." / "a.py"))
    second.start()
    first.join(2)
    second.join(2)
    assert events == ["one:write", "one:run", "two:write", "two:run"]


def test_locks_are_released_and_forgotten(tmp_path: Path) -> None:
    locks = FileLocks()
    with locks.hold(tmp_path / "a", tmp_path / "sub" / ".." / "a", tmp_path / "b"):
        assert locks.held() == 2
    assert locks.held() == 0
    assert FileLocks.key(tmp_path / "sub" / ".." / "a") == str((tmp_path / "a").resolve())


def test_a_raising_change_releases_its_lock(tmp_path: Path) -> None:
    locks = FileLocks()

    def boom() -> Mutation:
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        fuse(boom, ThenRun("x"), _never, paths=(tmp_path / "a",), locks=locks)
    assert locks.held() == 0


async def test_the_async_form_waits_for_a_thread_holding_the_lock(tmp_path: Path) -> None:
    locks = FileLocks()
    order: list[str] = []
    held = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with locks.hold(tmp_path / "a"):
            held.set()
            release.wait(2)
            order.append("thread")

    thread = threading.Thread(target=holder)
    thread.start()
    held.wait(2)

    async def waiter() -> None:
        async with locks.ahold(tmp_path / "a"):
            order.append("task")

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    assert order == []
    release.set()
    await task
    thread.join(2)
    assert order == ["thread", "task"]
    async with locks.ahold(tmp_path / "b"):
        assert locks.held() == 1
    assert locks.held() == 0


# ----------------------------------------------------------------- schemas --


@tool
def edit(path: str, old: str, new: str, then_run: dict[str, object] | None = None) -> str:
    """Replace text in a file.

    Args:
        path: File to edit.
        old: Text to replace.
        new: Replacement.
        then_run: Filled in by fusable().
    """
    return path


def test_fusable_offers_then_run_or_hides_it() -> None:
    on = fusable(edit)
    assert on.parameters["properties"][THEN_RUN]["required"] == ["command"]
    assert on.description.endswith(THEN_RUN_NOTE)
    assert THEN_RUN not in on.parameters["required"]
    # Idempotent: fitting twice does not repeat the note.
    assert fusable(on).description.count(THEN_RUN_NOTE) == 1
    off = fusable(edit, enabled=False)
    assert THEN_RUN not in off.parameters["properties"]
    assert THEN_RUN_NOTE not in off.description
    # The original tool is untouched.
    assert edit.parameters["properties"][THEN_RUN].get("required") is None
