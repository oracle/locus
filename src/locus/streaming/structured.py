# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Streaming partial Pydantic objects from an agent run.

The standard pattern in :mod:`locus` is to set ``output_schema=`` on an
:class:`~locus.agent.Agent` and read the parsed model off
``AgentResult.parsed`` after :meth:`Agent.run_sync`. For streaming UIs you
also want *incremental* snapshots — show the user the model in flight as
fields populate.

:class:`StructuredStream` wraps any ``AsyncIterator[LocusEvent]`` and yields
each best-effort parse of the accumulated assistant content. The first
``ModelChunkEvent`` whose buffer parses against the schema produces a
partial; later chunks may overwrite that partial as more fields stream in.
The final fully-validated instance is exposed via the ``final`` attribute
once the stream completes.

Usage::

    from locus import Agent
    from locus.streaming.structured import StructuredStream

    agent = Agent(model="oci:openai.gpt-5-mini", output_schema=VendorList)

    stream = StructuredStream(agent.run("Top 3 vendors."), schema=VendorList)
    async for partial in stream:
        ui.render(partial)
    final = stream.final
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from locus.core.events import LocusEvent, ModelChunkEvent, TerminateEvent
from locus.core.structured import parse_partial, parse_structured


T = TypeVar("T", bound=BaseModel)


class StructuredStream(Generic[T]):
    """Async iterator yielding partial Pydantic instances from an agent run."""

    def __init__(
        self,
        events: AsyncIterator[LocusEvent],
        schema: type[T],
        *,
        emit_unchanged: bool = False,
    ) -> None:
        """Build a structured stream wrapper.

        Args:
            events: The event iterator from ``Agent.run(...)``.
            schema: The Pydantic schema to coerce the streamed JSON into.
            emit_unchanged: When ``False`` (default) we only yield a partial
                if the parsed model differs from the previous emit; when
                ``True`` every new chunk that successfully parses is yielded.
                The default reduces churn for UIs that rerender on each yield.
        """
        self._events = events
        self._schema = schema
        self._emit_unchanged = emit_unchanged
        self._buffer = ""
        self._final: T | None = None
        self._final_message: str | None = None
        self._terminate_reason: str | None = None
        self._last_emit: T | None = None

    @property
    def final(self) -> T | None:
        """The final fully-validated instance (or ``None`` if parsing failed)."""
        return self._final

    @property
    def terminate_reason(self) -> str | None:
        """The ``TerminateEvent.reason`` that ended the stream, if any."""
        return self._terminate_reason

    async def __aiter__(self) -> AsyncIterator[T]:
        async for event in self._events:
            partial = self._handle(event)
            if partial is None:
                continue
            if not self._emit_unchanged and self._eq(partial, self._last_emit):
                continue
            self._last_emit = partial
            yield partial

        # Run a final validation on the complete buffer so callers get the
        # canonical instance via ``stream.final`` even when the last partial
        # snapshot came up missing optional fields.
        if self._final_message:
            full = parse_structured(self._final_message, self._schema, strict=False)
            if full.success:
                self._final = full.parsed  # type: ignore[assignment]
            elif self._final is None and self._last_emit is not None:
                self._final = self._last_emit

    def _handle(self, event: LocusEvent) -> T | None:
        if isinstance(event, ModelChunkEvent):
            if event.content:
                self._buffer += event.content
            return parse_partial(self._buffer, self._schema)
        if isinstance(event, TerminateEvent):
            self._terminate_reason = event.reason
            self._final_message = event.final_message or self._buffer
        return None

    @staticmethod
    def _eq(a: BaseModel, b: BaseModel | None) -> bool:
        if b is None:
            return False
        try:
            return a.model_dump() == b.model_dump()
        except Exception:  # pragma: no cover  # noqa: BLE001
            return False


def stream_structured(
    events: AsyncIterator[LocusEvent],
    schema: type[T],
    *,
    emit_unchanged: bool = False,
) -> StructuredStream[T]:
    """Convenience factory mirroring :class:`StructuredStream`.

    Equivalent to ``StructuredStream(events, schema)`` but reads as a verb in
    user code: ``async for snap in stream_structured(agent.run(...), Foo): ...``.
    """
    return StructuredStream(events, schema, emit_unchanged=emit_unchanged)


__all__: list[str] = ["StructuredStream", "stream_structured"]


# Keep type-checkers happy for ``Any`` re-export.
_ = Any
