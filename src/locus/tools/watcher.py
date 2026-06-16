# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tool hot-reload — watch directories for tool changes during development.

Monitors a directory of Python tool files and automatically reloads them
when modified. Useful for rapid iteration during agent development.

Example:
    from locus.tools.watcher import ToolWatcher

    watcher = ToolWatcher("./tools", registry=agent.tools)
    watcher.start()

    # Edit tools/search.py → automatically reloaded
    # Agent's next call uses the updated tool

    watcher.stop()
"""

from __future__ import annotations

import importlib.util
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from locus.tools.decorator import Tool


if TYPE_CHECKING:
    from locus.tools.registry import ToolRegistry


logger = logging.getLogger(__name__)


def load_tools_from_file(file_path: Path) -> list[Tool]:
    """Load Tool instances from a Python file.

    Scans the file's module namespace for Tool instances.

    Args:
        file_path: Path to a .py file containing @tool decorated functions.

    Returns:
        List of Tool instances found in the file.
    """
    # Unique module name to avoid import caching across reloads
    import time as _time

    unique_name = f"locus_tools.{file_path.stem}_{int(_time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    if spec is None or spec.loader is None:
        return []

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        logger.exception("Failed to load tools from %s", file_path)
        return []

    tools: list[Tool] = []
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name, None)
        if attr is None:
            continue
        # Use duck typing: check for Tool-like attributes
        # isinstance can fail across dynamic imports due to different class objects
        if (
            hasattr(attr, "name")
            and hasattr(attr, "fn")
            and hasattr(attr, "parameters")
            and callable(getattr(attr, "execute", None))
        ):
            if isinstance(attr, Tool):
                tools.append(attr)
            else:
                tools.append(
                    Tool(
                        name=attr.name,
                        description=getattr(attr, "description", ""),
                        parameters=attr.parameters,
                        fn=attr.fn,
                    )
                )

    return tools


def load_tools_from_directory(directory: Path | str) -> list[Tool]:
    """Load all tools from a directory of Python files.

    Args:
        directory: Path to directory containing .py tool files.

    Returns:
        List of all Tool instances found.
    """
    directory = Path(directory)
    tools: list[Tool] = []

    if not directory.is_dir():
        return tools

    for py_file in sorted(directory.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        tools.extend(load_tools_from_file(py_file))

    return tools


class ToolWatcher:
    """Watch a directory for tool file changes and auto-reload.

    Uses polling (no external dependencies like watchdog needed).
    Checks for modifications every `poll_interval` seconds.

    Security note:
        Each reload calls ``importlib.util.spec_from_file_location(...)``
        followed by ``spec.loader.exec_module(...)``. Any actor able to
        drop a ``*.py`` file into the watched directory therefore gets
        arbitrary in-process code execution as the agent (CWE-94 / CWE-73).
        Because this is a developer-convenience feature, hot-reload of
        *new* and *modified* files is OFF by default and must be enabled
        explicitly via ``dev_reload=True``. Without the opt-in, the
        watcher still performs the initial one-shot load of files that
        existed at startup (matching the old behaviour of
        ``load_tools_from_directory``), but ignores every subsequent file
        mutation with a warning so surprise drops do not execute.

    Example:
        >>> # Production / default: no runtime reloads.
        >>> watcher = ToolWatcher("./tools", registry=agent.tools)
        >>> watcher.start()

        >>> # Development: explicit opt-in to hot-reload.
        >>> watcher = ToolWatcher(
        ...     "./tools",
        ...     registry=agent.tools,
        ...     dev_reload=True,
        ... )
        >>> watcher.start()
        >>> # ... edit files, they auto-reload ...
        >>> watcher.stop()
    """

    def __init__(
        self,
        directory: Path | str,
        registry: ToolRegistry | None = None,
        poll_interval: float = 1.0,
        dev_reload: bool = False,
    ) -> None:
        """Initialize the watcher.

        Args:
            directory: Directory to watch for .py tool files.
            registry: ToolRegistry to update when tools change.
            poll_interval: Seconds between file modification checks.
            dev_reload: If True, hot-reload new or modified files at
                runtime. Leave False in production: runtime reload
                imports arbitrary Python from the filesystem, so any
                write into the watched directory becomes RCE as the
                agent.
        """
        self._directory = Path(directory)
        self._registry = registry
        self._poll_interval = poll_interval
        self._dev_reload = dev_reload
        self._running = False
        self._thread: threading.Thread | None = None
        self._file_mtimes: dict[str, float] = {}
        self._on_reload: list[Any] = []

    def on_reload(self, callback: Any) -> None:
        """Register a callback for when tools are reloaded.

        Args:
            callback: Function called with (file_path, tools) on reload.
        """
        self._on_reload.append(callback)

    def start(self) -> None:
        """Start watching for file changes in a background thread.

        Without ``dev_reload=True`` the poll loop is not started; the
        watcher becomes a one-shot loader of files present at startup.
        """
        if self._running:
            return

        # Initial scan — always safe: these files were on disk at
        # startup, so the operator already trusts them.
        self._scan_directory()

        if not self._dev_reload:
            logger.info(
                "ToolWatcher: dev_reload=False; loaded %d files at startup "
                "and will NOT hot-reload new/modified files. Pass "
                "dev_reload=True to enable runtime reload (development "
                "only — runtime import of any *.py written to %s is "
                "remote code execution as the agent).",
                len(self._file_mtimes),
                self._directory,
            )
            return

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("ToolWatcher started (dev_reload=True): %s", self._directory)

    def stop(self) -> None:
        """Stop the watcher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("ToolWatcher stopped")

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            time.sleep(self._poll_interval)
            self._check_for_changes()

    def _scan_directory(self) -> None:
        """Initial directory scan — record file modification times."""
        if not self._directory.is_dir():
            return

        for py_file in self._directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            self._file_mtimes[str(py_file)] = py_file.stat().st_mtime

            # Load and register initial tools
            if self._registry is not None:
                tools = load_tools_from_file(py_file)
                for t in tools:
                    try:
                        self._registry.register(t)
                    except ValueError:
                        pass  # Already registered

    def _check_for_changes(self) -> None:
        """Check for new/modified/deleted files."""
        if not self._directory.is_dir():
            return

        current_files: set[str] = set()

        for py_file in self._directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            file_key = str(py_file)
            current_files.add(file_key)
            mtime = py_file.stat().st_mtime

            if file_key not in self._file_mtimes:
                # New file
                self._file_mtimes[file_key] = mtime
                self._reload_file(py_file)
            elif mtime > self._file_mtimes[file_key]:
                # Modified file
                self._file_mtimes[file_key] = mtime
                self._reload_file(py_file)

        # Check for deleted files
        for file_key in list(self._file_mtimes.keys()):
            if file_key not in current_files:
                del self._file_mtimes[file_key]
                logger.info("Tool file removed: %s", file_key)

    def _reload_file(self, file_path: Path) -> None:
        """Reload tools from a modified file."""
        logger.info("Reloading tools from: %s", file_path)

        tools = load_tools_from_file(file_path)

        if self._registry is not None:
            for t in tools:
                # Re-register (overwrite existing)
                if t.name in self._registry.tools:
                    self._registry.tools[t.name] = t
                else:
                    self._registry.register(t)

        for callback in self._on_reload:
            try:
                callback(file_path, tools)
            except Exception:
                logger.exception("Error in reload callback")

        logger.info("Reloaded %d tools from %s", len(tools), file_path.name)
