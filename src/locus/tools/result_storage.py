# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""External tool-result storage with reference-key substitution.

The agent's default behaviour for oversized tool output is head
truncation (``agent.py:643-659``). That's lossy — once the agent
decides the truncated head isn't what it needed, there's no way
back to the full result.

:class:`ToolResultStore` flips the trade: persist the full content
to a user-supplied backend (a :class:`~locus.memory.BaseCheckpointer`,
an S3 bucket, a local file, …) and substitute an inline reference
that preserves a prefix plus a recoverable key. The agent never
blows its context budget and the user keeps a way to fetch the real
payload later.

Example — wiring a store to the existing checkpointer::

    from locus.tools.result_storage import ToolResultStore


    def _save(key: str, content: str) -> None:
        agent.checkpointer.save_blob(key, content)


    def _load(key: str) -> str | None:
        return agent.checkpointer.load_blob(key)


    store = ToolResultStore(
        save=_save,
        load=_load,
        threshold_chars=32_000,
    )
    maybe_stored = store.maybe_offload(
        result=tool_result,
        run_id="run-42",
        iteration=7,
    )
    # `maybe_stored` is either the original `tool_result` (under threshold)
    # or a new ToolResult whose content is a summary + reference key.

The module is storage-agnostic — it does not depend on the Locus
checkpointer subsystem. Users who prefer a different backend wire
``save`` / ``load`` callables of their choice.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from locus.core.messages import ToolResult


__all__ = [
    "ToolResultStore",
    "extract_reference_key",
]


#: Marker used in substituted tool-result content. Consumers can
#: detect stored payloads by scanning for this prefix and then use
#: :func:`extract_reference_key` to recover the storage key.
REFERENCE_MARKER = "[TOOL RESULT STORED externally]"


SaveFn = Callable[[str, str], None]
LoadFn = Callable[[str], str | None]


class ToolResultStore:
    """Offload oversized tool results to an external store.

    Args:
        save: Callable that persists a ``(key, content)`` pair.
        load: Callable that looks up content by key; returns
            ``None`` if the key is unknown.
        threshold_chars: Only offload when the tool result's content
            exceeds this many characters. Default 32000, matching
            ``AgentConfig.max_tool_result_length``'s default.
        preview_chars: Number of leading characters to preserve
            inline so the agent still sees the shape of the output.
            Must be ``<= threshold_chars``.
    """

    def __init__(
        self,
        *,
        save: SaveFn,
        load: LoadFn,
        threshold_chars: int = 32_000,
        preview_chars: int = 8_000,
    ) -> None:
        if threshold_chars < 1:
            raise ValueError("threshold_chars must be positive")
        if preview_chars < 0:
            raise ValueError("preview_chars must be non-negative")
        if preview_chars > threshold_chars:
            raise ValueError("preview_chars must not exceed threshold_chars")
        self._save = save
        self._load = load
        self.threshold_chars = threshold_chars
        self.preview_chars = preview_chars

    # ------------------------------------------------------------------

    def maybe_offload(
        self,
        result: ToolResult,
        *,
        run_id: str,
        iteration: int,
    ) -> ToolResult:
        """Return the original result or a reference-bearing replacement.

        When ``len(result.content) <= threshold_chars`` the input is
        returned unchanged. Otherwise the full content is persisted
        via ``save``, and a new :class:`ToolResult` is returned with
        content replaced by ``{marker} — {len} chars, key={key}`` and
        a preview of the first ``preview_chars``.
        """
        from locus.core.messages import ToolResult

        content = result.content or ""
        if len(content) <= self.threshold_chars:
            return result

        key = self._build_key(run_id=run_id, iteration=iteration, tool=result.name)
        self._save(key, content)

        preview = content[: self.preview_chars]
        replacement = (
            f"{REFERENCE_MARKER} — {len(content)} chars, key={key}\n"
            f"First {self.preview_chars} chars follow:\n{preview}"
        )
        return ToolResult(
            tool_call_id=result.tool_call_id,
            name=result.name,
            content=replacement,
            error=result.error,
            duration_ms=result.duration_ms,
        )

    def load(self, key: str) -> str | None:
        """Recover the full content for a previously-offloaded result."""
        return self._load(key)

    # ------------------------------------------------------------------

    @staticmethod
    def _build_key(*, run_id: str, iteration: int, tool: str) -> str:
        # Keys are human-readable for grep-ability, but the caller
        # shouldn't rely on the exact format — treat as opaque.
        safe_run = run_id.replace(":", "_").replace("/", "_")
        safe_tool = tool.replace(":", "_").replace("/", "_") or "tool"
        return f"locus:result:{safe_run}:{iteration}:{safe_tool}"


def extract_reference_key(content: str) -> str | None:
    """Return the storage key embedded in ``content`` or ``None``.

    Scans for the ``key=<value>`` fragment that
    :meth:`ToolResultStore.maybe_offload` inserts. Safe to call on
    any string — non-matching content returns ``None``.
    """
    if not content or REFERENCE_MARKER not in content:
        return None
    marker_idx = content.find("key=")
    if marker_idx < 0:
        return None
    tail = content[marker_idx + len("key=") :]
    # Key runs until whitespace or newline.
    end = len(tail)
    for i, ch in enumerate(tail):
        if ch in (" ", "\n", "\t"):
            end = i
            break
    return tail[:end].strip() or None
