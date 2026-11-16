# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``fetch_tool_result``: read back a command output the harness stored whole.

``bash`` hands the model a capped head and tail of a long output and, with a
:class:`~locus.tools.result_storage.ToolResultStore` in the
:class:`~locus.harness.tools.common.HarnessConfig`, keeps the whole of it
under a key. This tool pages that text back, so a test run's middle or a
build log's one interesting line is a call away instead of a re-run.
:func:`~locus.harness.toolset.build_harness` includes it whenever the
config names a store.
"""

from __future__ import annotations

from locus.harness.tools.common import HarnessContext
from locus.memory.offload import DEFAULT_PAGE_CHARS, page_of
from locus.tools.decorator import Tool, tool


__all__ = ["make_fetch_tool_result"]


def make_fetch_tool_result(h: HarnessContext) -> Tool:
    """Build ``fetch_tool_result`` over the harness's result store."""

    @tool(
        name="fetch_tool_result",
        description=(
            "Read back the full text of an earlier tool output that was too long to show "
            "and was stored under a key (or handle). Pass the key from the output's note "
            "and a character offset (0 to start, then the returned next_offset). Returns "
            f"up to {DEFAULT_PAGE_CHARS:,} characters per call."
        ),
    )
    def fetch_tool_result(key: str, offset: int = 0, limit: int = DEFAULT_PAGE_CHARS) -> str:
        """Read a page of a stored tool output.

        Args:
            key: The key (or handle) the output's note names.
            offset: Character offset to start at; use the previous page's next_offset.
            limit: Characters to return, at most the page size.
        """
        store = h.config.result_store
        if store is None:
            return "No result store is configured for this harness; nothing was stored."
        content = store.load(key)
        if content is None:
            return (
                f"No stored output under key {key!r}. It may belong to another session or "
                "have expired; run the command again if you need it."
            )
        return page_of(content, key, offset=offset, limit=min(limit, DEFAULT_PAGE_CHARS))

    return fetch_tool_result
