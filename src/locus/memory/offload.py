# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Offload large tool results to storage and keep a handle in the conversation.

A coding or research run reads files, runs test suites and fetches pages, and
each of those outputs lands in the conversation whole. Most of it is read
once. :class:`ToolResultOffload` moves any tool result over a size threshold
into a :class:`~locus.tools.result_storage.ToolResultStore` the moment it is
produced; the message the model sees keeps a *handle* and a short preview,
and the ``fetch_tool_result`` tool (:meth:`ToolResultOffload.as_tool`) reads
the stored text back a page at a time when the model needs it again.

The handle is written into the message text, so it survives everything a
message survives: a checkpoint round trip, a copy of the thread, and every
compaction tier (a cleared stub names the handle rather than dropping it).
:func:`offload_handle` reads it back from a message or its content.

Example::

    from locus.agent import Agent
    from locus.memory import ToolResultOffload

    agent = Agent(
        model="oci:openai.gpt-4.1",
        tools=[read_file, run_tests],
        tool_result_offload=ToolResultOffload(threshold_chars=16_000),
    )

With no ``store`` the results live in process memory for the life of the
offload object; pass a :class:`~locus.tools.result_storage.ToolResultStore`
over a checkpointer, a bucket or a database for anything that must outlive
the process.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from locus.tools.result_storage import ToolResultStore


if TYPE_CHECKING:
    from locus.core.messages import Message, ToolResult
    from locus.tools.decorator import Tool


__all__ = [
    "FETCH_TOOL_NAME",
    "OFFLOAD_MARKER",
    "ToolResultOffload",
    "offload_handle",
    "page_of",
]

#: Opens the content of every offloaded tool result.
OFFLOAD_MARKER = "[tool result offloaded:"

#: Most characters one ``fetch_tool_result`` call returns, header included.
DEFAULT_PAGE_CHARS = 16_000

#: The tool that reads offloaded results back. Its own output is never
#: offloaded, whatever its size: a page the model asked for must arrive.
FETCH_TOOL_NAME = "fetch_tool_result"

_PAGE_HEADER = f"[{FETCH_TOOL_NAME} "

_HANDLE = re.compile(r"handle=(\S+?)[\]\s,;]")


class ToolResultOffload:
    """Move tool results over ``threshold_chars`` to storage, by handle.

    Args:
        store: Where the full text goes. ``None`` keeps it in process memory
            for the life of this object.
        threshold_chars: A result longer than this is offloaded.
        preview_chars: How much of the start of the result stays inline,
            so the model sees the shape of the output and can decide
            whether to fetch the rest. Capped at ``threshold_chars``.
        page_chars: Most characters one ``fetch_tool_result`` call returns.
    """

    def __init__(
        self,
        store: ToolResultStore | None = None,
        *,
        threshold_chars: int = 16_000,
        preview_chars: int = 2_000,
        page_chars: int = DEFAULT_PAGE_CHARS,
    ) -> None:
        if threshold_chars < 1:
            raise ValueError("threshold_chars must be positive")
        if preview_chars < 0:
            raise ValueError("preview_chars must be non-negative")
        if page_chars < 100:
            raise ValueError("page_chars must be at least 100")
        if store is None:
            memory: dict[str, str] = {}
            # Only ``put`` and ``load`` are used, so the store's own cap and
            # preview do not matter; a tiny threshold must not trip them.
            store = ToolResultStore(
                save=memory.__setitem__,
                load=memory.get,
                threshold_chars=threshold_chars,
                preview_chars=0,
            )
        self.store = store
        self.threshold_chars = threshold_chars
        # A preview longer than the threshold would be the whole result.
        self.preview_chars = min(preview_chars, threshold_chars)
        self.page_chars = page_chars

    # ------------------------------------------------------------------
    # Offloading
    # ------------------------------------------------------------------

    def should_offload(self, content: str | None) -> bool:
        """Whether a result with ``content`` goes to storage."""
        return bool(content) and len(content or "") > self.threshold_chars

    def offload(self, result: ToolResult, *, run_id: str, iteration: int) -> ToolResult:
        """``result`` with its content stored and replaced by a handle and a preview.

        A result under the threshold, one already offloaded, and a page
        ``fetch_tool_result`` itself returned are returned unchanged.
        """
        content = result.content or ""
        if (
            not self.should_offload(content)
            or content.startswith((OFFLOAD_MARKER, _PAGE_HEADER))
            or result.name == FETCH_TOOL_NAME
        ):
            return result
        handle = self.store.put(
            content,
            run_id=run_id,
            iteration=iteration,
            tool=result.name,
            tool_call_id=result.tool_call_id,
        )
        return result.model_copy(update={"content": self.render(handle, content)})

    def render(self, handle: str, content: str) -> str:
        """The message text standing in for ``content``: handle, how to fetch, preview."""
        preview = content[: self.preview_chars]
        lines = [
            f"{OFFLOAD_MARKER} {len(content):,} characters; handle={handle}] "
            f'The full output is stored. Call fetch_tool_result(handle="{handle}", '
            f"offset=0) to read it in pages of up to {self.page_chars:,} characters, "
            "continuing from the next_offset each page reports.",
        ]
        if preview:
            lines.append(f"The first {len(preview):,} characters follow:\n{preview}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Fetching back
    # ------------------------------------------------------------------

    def load(self, handle: str) -> str | None:
        """The whole stored text under ``handle``, or ``None`` when unknown."""
        return self.store.load(handle)

    def fetch(self, handle: str, *, offset: int = 0, limit: int | None = None) -> str:
        """One page of the stored text, with a header saying where the next starts.

        Raises:
            KeyError: ``handle`` names nothing in the store.
        """
        content = self.load(handle)
        if content is None:
            raise KeyError(handle)
        return page_of(content, handle, offset=offset, limit=limit or self.page_chars)

    def as_tool(self) -> Tool:
        """``fetch_tool_result``: read an offloaded result back, a page at a time."""
        from locus.tools.decorator import tool as tool_decorator  # noqa: PLC0415

        page_chars = self.page_chars
        offload = self

        @tool_decorator(
            name=FETCH_TOOL_NAME,
            description=(
                "Read back the full text of an earlier tool result that was offloaded to "
                "save context. Pass the handle from its placeholder and a character "
                f"offset (0 to start, then the returned next_offset). Returns up to "
                f"{page_chars:,} characters per call."
            ),
            idempotent=True,
        )
        def fetch_tool_result(handle: str, offset: int = 0, limit: int | None = None) -> str:
            """Read a page of an offloaded tool result.

            Args:
                handle: The handle from the result's placeholder.
                offset: Character offset to start at; use the previous page's next_offset.
                limit: Characters to return, at most the page size.
            """
            try:
                return offload.fetch(handle, offset=offset, limit=limit)
            except KeyError:
                return (
                    f"No stored result under handle {handle!r}. It may belong to another "
                    "run or have expired; call the original tool again if you need it."
                )

        return fetch_tool_result

    def __repr__(self) -> str:
        return (
            f"ToolResultOffload(threshold_chars={self.threshold_chars}, "
            f"preview_chars={self.preview_chars}, page_chars={self.page_chars})"
        )


def offload_handle(message_or_content: Message | str | None) -> str | None:
    """The storage handle an offloaded tool result names, or ``None``.

    Reads a :class:`~locus.core.messages.Message`, a
    :class:`~locus.core.messages.ToolResult` or a plain string: anything whose
    text starts with :data:`OFFLOAD_MARKER`.
    """
    content: Any = message_or_content
    if content is not None and not isinstance(content, str):
        content = getattr(content, "content", None)
    if not isinstance(content, str) or not content.startswith(OFFLOAD_MARKER):
        return None
    match = _HANDLE.search(content)
    return match.group(1) if match else None


def page_of(content: str, handle: str, *, offset: int = 0, limit: int) -> str:
    """``content[offset:offset + limit]`` under a header that says what is left.

    The header carries the handle, the offsets and the total, so the model
    can page without counting characters itself; ``next_offset=end`` means
    the page reached the end.
    """
    total = len(content)
    start = max(0, min(offset, total))
    limit = max(1, limit)
    end = min(total, start + limit)
    where = "end" if end >= total else str(end)
    header = (
        f"[fetch_tool_result handle={handle} offset={start} next_offset={where} "
        f"of {total:,} characters]"
    )
    return f"{header}\n{content[start:end]}"
