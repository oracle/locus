# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Rich tool output — text for the model, data for the application.

A tool result is a string everywhere in Tulip (see :mod:`tulip.core.media`),
because the model reads text and checkpoints, audit trails and durable
engines all move strings. Some tools have more to say than the model should
read: an MCP server returns ``structuredContent`` a UI renders as a widget,
flags a failure with ``isError``, or returns an image.

:class:`ToolOutput` carries that without breaking the string contract. It *is*
a ``str`` — the text the model sees — so every caller that treats a tool
result as a string keeps working, and it additionally carries:

``structured_content``
    Machine-readable data, surfaced on
    :attr:`~tulip.core.events.ToolCompleteEvent.structured_content`. Never
    sent to the model.
``content_blocks``
    Non-text content (images, audio, resources) in the protocol's own shape,
    surfaced on :attr:`~tulip.core.events.ToolCompleteEvent.content_blocks`.
``is_error``
    The tool *returned* a failure rather than raising one. The executor
    records it as the call's error, so ``ToolCompleteEvent.error`` is set and
    the model is told the call failed.

A local tool can return one directly::

    @tool
    def search_hotels(city: str) -> ToolOutput:
        hits = catalogue.search(city)
        return ToolOutput(
            f"Found {len(hits)} hotels in {city}.",
            structured_content={"hotels": [h.model_dump() for h in hits]},
        )
"""

from __future__ import annotations

from typing import Any


__all__ = ["ToolOutput"]


def _rebuild(
    text: str,
    structured_content: dict[str, Any] | None,
    content_blocks: list[dict[str, Any]] | None,
    is_error: bool,
) -> ToolOutput:
    """Unpickle helper — ``str`` subclasses do not round-trip their extras."""
    return ToolOutput(
        text,
        structured_content=structured_content,
        content_blocks=content_blocks,
        is_error=is_error,
    )


class ToolOutput(str):  # noqa: SLOT000 — str subtypes cannot have non-empty __slots__; the extras need a __dict__
    """A tool result string that also carries structured data.

    Args:
        text: What the model reads. For an error, the error message.
        structured_content: Machine-readable result for the application.
        content_blocks: Non-text content blocks, JSON-mode dicts.
        is_error: The tool reports failure; recorded as the call's error.
    """

    structured_content: dict[str, Any] | None
    content_blocks: list[dict[str, Any]] | None
    is_error: bool

    def __new__(
        cls,
        text: str = "",
        *,
        structured_content: dict[str, Any] | None = None,
        content_blocks: list[dict[str, Any]] | None = None,
        is_error: bool = False,
    ) -> ToolOutput:
        obj = super().__new__(cls, text)
        obj.structured_content = structured_content
        obj.content_blocks = content_blocks or None
        obj.is_error = is_error
        return obj

    @property
    def text(self) -> str:
        """The plain-``str`` text, without the extras."""
        return str.__str__(self)

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _rebuild,
            (self.text, self.structured_content, self.content_blocks, self.is_error),
        )

    def __repr__(self) -> str:
        extras = []
        if self.structured_content is not None:
            extras.append("structured")
        if self.content_blocks:
            extras.append(f"{len(self.content_blocks)} block(s)")
        if self.is_error:
            extras.append("error")
        suffix = f" [{', '.join(extras)}]" if extras else ""
        return f"ToolOutput({str.__repr__(self)}{suffix})"
