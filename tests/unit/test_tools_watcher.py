# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``tulip.tools.watcher`` (hot-reload + initial load).

The watcher uses ``importlib.util.spec_from_file_location`` to load
tools from arbitrary Python files. The tests below build small
fixture files in ``tmp_path`` and exercise:

- one-shot loading (default ``dev_reload=False``)
- the dev-only hot-reload path (``dev_reload=True``)
- new / modified / deleted file detection
- callback invocation
- failure modes (broken file, missing directory, on_reload callback raise)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tulip.tools.registry import ToolRegistry
from tulip.tools.watcher import (
    ToolWatcher,
    load_tools_from_directory,
    load_tools_from_file,
)


def _wait_until(predicate, what: str, timeout: float = 10.0, interval: float = 0.05) -> None:
    """Block until ``predicate()`` holds, or fail saying what never happened.

    The watcher runs on its own thread, so every assertion about it races a
    poll loop. The previous form gave the watcher 2s and then asserted
    regardless of whether the wait had succeeded — so on a loaded runner it
    produced a confusing value comparison instead of "the watcher never
    reloaded". A longer deadline costs nothing when the reload is prompt: the
    loop returns the moment it lands.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout:g}s waiting for {what}")


# A reusable tool source shared by most tests.
_TOOL_SOURCE = """
from tulip.tools.decorator import tool

@tool(name="echo", description="Echo back the input.")
def echo(text: str) -> str:
    return text
"""


def _write_tool_file(directory: Path, filename: str, source: str = _TOOL_SOURCE) -> Path:
    path = directory / filename
    path.write_text(source)
    return path


# ---------------------------------------------------------------------------
# load_tools_from_file
# ---------------------------------------------------------------------------


class TestLoadToolsFromFile:
    def test_loads_tool_from_valid_file(self, tmp_path: Path) -> None:
        path = _write_tool_file(tmp_path, "echo_tool.py")
        tools = load_tools_from_file(path)
        assert len(tools) == 1
        assert tools[0].name == "echo"

    def test_skips_private_attributes(self, tmp_path: Path) -> None:
        # Names starting with ``_`` are skipped — including dunders that
        # stdlib injects into module namespaces.
        path = _write_tool_file(tmp_path, "private.py")
        tools = load_tools_from_file(path)
        assert all(t.name == "echo" for t in tools)

    def test_returns_empty_for_unloadable_file(self, tmp_path: Path) -> None:
        # Syntax error → ``exec_module`` raises and the loader returns [].
        path = tmp_path / "broken.py"
        path.write_text("this is not !! valid python {{{")
        tools = load_tools_from_file(path)
        assert tools == []

    def test_returns_empty_for_file_without_tools(self, tmp_path: Path) -> None:
        path = tmp_path / "no_tools.py"
        path.write_text("VALUE = 42\ndef plain_fn(): return 1\n")
        tools = load_tools_from_file(path)
        assert tools == []


# ---------------------------------------------------------------------------
# load_tools_from_directory
# ---------------------------------------------------------------------------


class TestLoadToolsFromDirectory:
    def test_loads_from_all_py_files(self, tmp_path: Path) -> None:
        _write_tool_file(tmp_path, "a.py")
        _write_tool_file(
            tmp_path,
            "b.py",
            source=_TOOL_SOURCE.replace('name="echo"', 'name="echo_b"'),
        )
        tools = load_tools_from_directory(tmp_path)
        names = sorted(t.name for t in tools)
        assert names == ["echo", "echo_b"]

    def test_skips_underscore_prefixed_files(self, tmp_path: Path) -> None:
        _write_tool_file(tmp_path, "_private.py")
        _write_tool_file(tmp_path, "public.py")
        tools = load_tools_from_directory(tmp_path)
        # Only ``public.py`` contributes.
        assert len(tools) == 1

    def test_returns_empty_for_non_directory(self, tmp_path: Path) -> None:
        # Pass a file path — the directory check fails, [] is returned.
        plain_file = tmp_path / "single_file.py"
        plain_file.write_text("# nothing")
        tools = load_tools_from_directory(plain_file)
        assert tools == []


# ---------------------------------------------------------------------------
# ToolWatcher — initial load (default, ``dev_reload=False``)
# ---------------------------------------------------------------------------


class TestToolWatcherInitialLoad:
    def test_initial_load_registers_tools(self, tmp_path: Path) -> None:
        _write_tool_file(tmp_path, "echo_tool.py")
        registry = ToolRegistry()
        watcher = ToolWatcher(tmp_path, registry=registry)
        watcher.start()
        try:
            assert "echo" in registry.tools
        finally:
            watcher.stop()

    def test_initial_load_without_registry(self, tmp_path: Path) -> None:
        # Without a registry, the watcher just records mtimes — no crash.
        _write_tool_file(tmp_path, "echo_tool.py")
        watcher = ToolWatcher(tmp_path)
        watcher.start()
        try:
            assert len(watcher._file_mtimes) == 1
        finally:
            watcher.stop()

    def test_initial_load_handles_already_registered_tool(self, tmp_path: Path) -> None:
        # If a tool with the same name is already in the registry, the
        # initial scan logs and continues — no exception.
        _write_tool_file(tmp_path, "echo_tool.py")
        registry = ToolRegistry()
        first = ToolWatcher(tmp_path, registry=registry)
        first.start()
        first.stop()
        # Second watcher instance over same directory & registry hits
        # the ``ValueError: already registered`` branch.
        second = ToolWatcher(tmp_path, registry=registry)
        second.start()
        second.stop()
        assert "echo" in registry.tools

    def test_no_op_on_double_start(self, tmp_path: Path) -> None:
        _write_tool_file(tmp_path, "echo_tool.py")
        registry = ToolRegistry()
        watcher = ToolWatcher(tmp_path, registry=registry)
        watcher.start()
        watcher.start()  # Second call must early-return.
        try:
            assert "echo" in registry.tools
        finally:
            watcher.stop()

    def test_handles_missing_directory(self, tmp_path: Path) -> None:
        watcher = ToolWatcher(tmp_path / "nonexistent")
        watcher.start()  # Must not crash.
        watcher.stop()
        assert watcher._file_mtimes == {}


# ---------------------------------------------------------------------------
# ToolWatcher — dev_reload (hot-reload of new + modified files)
# ---------------------------------------------------------------------------


class TestToolWatcherDevReload:
    def test_dev_reload_picks_up_new_file(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        watcher = ToolWatcher(tmp_path, registry=registry, poll_interval=0.05, dev_reload=True)
        watcher.start()
        try:
            # Drop a new file in after start — the poll loop must pick it up.
            _write_tool_file(tmp_path, "echo_tool.py")
            _wait_until(lambda: "echo" in registry.tools, "the new file to be picked up")
        finally:
            watcher.stop()

    def test_dev_reload_picks_up_modified_file(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        path = _write_tool_file(tmp_path, "echo_tool.py")
        watcher = ToolWatcher(tmp_path, registry=registry, poll_interval=0.05, dev_reload=True)
        watcher.start()
        try:
            assert "echo" in registry.tools
            # Modify with a different tool name — bump mtime explicitly
            # to defeat coarse FS resolution.
            new_source = _TOOL_SOURCE.replace('name="echo"', 'name="echo_v2"')
            path.write_text(new_source)
            future_mtime = time.time() + 1
            os.utime(path, (future_mtime, future_mtime))
            _wait_until(lambda: "echo_v2" in registry.tools, "the renamed tool to appear")
        finally:
            watcher.stop()

    def test_dev_reload_detects_deleted_file(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        path = _write_tool_file(tmp_path, "echo_tool.py")
        watcher = ToolWatcher(tmp_path, registry=registry, poll_interval=0.05, dev_reload=True)
        watcher.start()
        try:
            assert str(path) in watcher._file_mtimes
            path.unlink()
            _wait_until(
                lambda: str(path) not in watcher._file_mtimes,
                "the deleted file to leave the mtime map",
            )
        finally:
            watcher.stop()

    def test_dev_reload_invokes_on_reload_callbacks(self, tmp_path: Path) -> None:
        seen: list[tuple[Path, list]] = []

        watcher = ToolWatcher(tmp_path, poll_interval=0.05, dev_reload=True)
        watcher.on_reload(lambda p, ts: seen.append((p, ts)))
        watcher.start()
        try:
            _write_tool_file(tmp_path, "echo_tool.py")
            _wait_until(lambda: bool(seen), "the on-change callback to fire")
            assert seen[0][0].name == "echo_tool.py"
        finally:
            watcher.stop()

    def test_dev_reload_swallows_callback_exceptions(self, tmp_path: Path) -> None:
        # A callback that raises must not stop the watcher.
        def boom(_path, _tools):  # type: ignore[no-untyped-def]
            raise RuntimeError("user callback bug")

        watcher = ToolWatcher(tmp_path, poll_interval=0.05, dev_reload=True)
        watcher.on_reload(boom)
        watcher.start()
        try:
            _write_tool_file(tmp_path, "echo_tool.py")
            # A fixed sleep here raced the poll thread under ``pytest -n
            # auto``; wait for the observable outcome instead.
            _wait_until(
                lambda: any(k.endswith("echo_tool.py") for k in watcher._file_mtimes),
                "the watcher to record the new file despite the raising callback",
            )
            # Still alive after the callback raised: a later file is picked up.
            _write_tool_file(tmp_path, "second.py")
            _wait_until(
                lambda: any(k.endswith("second.py") for k in watcher._file_mtimes),
                "the watcher to keep polling after the callback raised",
            )
        finally:
            watcher.stop()

    def test_dev_reload_overwrites_existing_tool(self, tmp_path: Path) -> None:
        # When a reload finds a tool already in the registry under the
        # same name, it overwrites in place rather than calling
        # ``register`` (which would raise).
        registry = ToolRegistry()
        path = _write_tool_file(tmp_path, "t.py")
        watcher = ToolWatcher(tmp_path, registry=registry, poll_interval=0.05, dev_reload=True)
        watcher.start()
        try:
            initial_tool = registry.tools["echo"]
            new_source = _TOOL_SOURCE.replace("Echo back the input.", "Updated description.")
            path.write_text(new_source)
            future_mtime = time.time() + 1
            os.utime(path, (future_mtime, future_mtime))
            _wait_until(
                lambda: registry.tools["echo"] is not initial_tool,
                "the watcher to reload the changed file",
            )
            assert registry.tools["echo"].description == "Updated description."
        finally:
            watcher.stop()


# ---------------------------------------------------------------------------
# Change detection that mtime alone gets wrong
# ---------------------------------------------------------------------------


class TestToolWatcherChangeSignature:
    """Driven through ``_scan_directory`` / ``_check_for_changes`` directly, so
    each case is one deterministic poll instead of a race with the thread."""

    def test_content_written_within_the_same_mtime_tick_is_reloaded(self, tmp_path: Path) -> None:
        # ``write_text`` creates/truncates the file and then writes it. A poll
        # between the two steps records the empty file; on a coarse-clock
        # filesystem the write that follows keeps the same mtime. Pinning
        # the mtime reproduces that torn read deterministically.
        registry = ToolRegistry()
        path = tmp_path / "echo_tool.py"
        path.write_text("")
        watcher = ToolWatcher(tmp_path, registry=registry, dev_reload=True)
        watcher._scan_directory()
        assert registry.tools == {}

        before = path.stat()
        path.write_text(_TOOL_SOURCE)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        watcher._check_for_changes()

        assert "echo" in registry.tools

    def test_file_restored_to_an_older_mtime_is_reloaded(self, tmp_path: Path) -> None:
        # ``git checkout`` / ``cp -p`` / an editor restoring a backup can move
        # the mtime backwards; that is still a change to pick up.
        registry = ToolRegistry()
        path = _write_tool_file(tmp_path, "t.py")
        watcher = ToolWatcher(tmp_path, registry=registry, dev_reload=True)
        watcher._scan_directory()
        original = registry.tools["echo"]

        # Same length, different description — only the mtime can tell.
        path.write_text(_TOOL_SOURCE.replace("Echo back the input.", "Echo back the inpuT."))
        past = time.time() - 3600
        os.utime(path, (past, past))
        watcher._check_for_changes()

        assert registry.tools["echo"] is not original
        assert registry.tools["echo"].description == "Echo back the inpuT."

    def test_unchanged_file_is_not_reloaded(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        _write_tool_file(tmp_path, "t.py")
        watcher = ToolWatcher(tmp_path, registry=registry, dev_reload=True)
        watcher._scan_directory()
        original = registry.tools["echo"]

        watcher._check_for_changes()

        assert registry.tools["echo"] is original

    def test_file_vanishing_between_glob_and_stat_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tulip.tools.watcher as watcher_module

        path = _write_tool_file(tmp_path, "t.py")
        watcher = ToolWatcher(tmp_path, dev_reload=True)
        watcher._scan_directory()

        def gone(_p: Path) -> tuple[int, int]:
            raise FileNotFoundError(str(_p))

        monkeypatch.setattr(watcher_module, "_signature", gone)
        watcher._check_for_changes()  # Must not raise.

        assert str(path) not in watcher._file_mtimes


# ---------------------------------------------------------------------------
# ToolWatcher.stop on a never-started watcher
# ---------------------------------------------------------------------------


class TestToolWatcherStop:
    def test_stop_without_start_is_no_op(self, tmp_path: Path) -> None:
        watcher = ToolWatcher(tmp_path)
        watcher.stop()  # Must not crash.
