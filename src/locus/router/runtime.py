# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Top-level :class:`Router` — the only public surface of the module.

The Router takes natural language, asks an extractor agent to fill a
:class:`GoalFrame`, hands it to the :class:`CognitiveCompiler`, and
executes the compiled :class:`Runnable`. Everything else in
``locus.router`` is internal machinery.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from locus.observability.context import reset_run_id, set_run_id
from locus.observability.event_bus import get_event_bus
from locus.observability.router_events import (
    emit_frame_extracted,
    emit_frame_failed,
    emit_runnable_executed,
    emit_runnable_executing,
    emit_runnable_failed,
)
from locus.router.compiler import CognitiveCompiler
from locus.router.goal_frame import GoalFrame
from locus.router.runnable import RunnableResult


if TYPE_CHECKING:
    from locus.agent.agent import Agent


class FrameExtractionError(RuntimeError):
    """Raised when the extractor agent failed to produce a valid GoalFrame."""


class Router:
    """One-shot dispatcher: text in, :class:`RunnableResult` out.

    Parameters
    ----------
    extractor:
        :class:`~locus.Agent` configured with
        ``output_schema=GoalFrame``. Its parsed result drives selection.
    compiler:
        :class:`CognitiveCompiler` set up with protocols, capabilities,
        policy, and model.
    on_frame:
        Optional callback fired right after a frame is extracted. Useful
        for telemetry / workbench display.
    """

    def __init__(
        self,
        *,
        extractor: Agent,
        compiler: CognitiveCompiler,
        on_frame: Callable[[GoalFrame], None] | None = None,
    ) -> None:
        self.extractor = extractor
        self.compiler = compiler
        self._on_frame = on_frame

    async def extract(self, user_input: str, run_id: str | None = None) -> GoalFrame:
        """Run the extractor and return the parsed :class:`GoalFrame`.

        Emits ``router.frame.extracted`` on success or
        ``router.frame.failed`` on schema rejection, scoped to
        ``run_id`` when supplied. Raises :class:`FrameExtractionError`
        on failure — the compiler can't recover from that.
        """
        result: Any = await asyncio.to_thread(self.extractor.invoke, user_input)
        parsed = result.parsed
        if not isinstance(parsed, GoalFrame):
            err_msg = (
                "Extractor did not produce a GoalFrame. "
                f"parse_error={result.parse_error!r}, message={result.message!r}"
            )
            if run_id:
                await emit_frame_failed(run_id, err_msg)
            raise FrameExtractionError(err_msg)
        if run_id:
            await emit_frame_extracted(run_id, parsed)
        if self._on_frame is not None:
            self._on_frame(parsed)
        return parsed

    async def dispatch(self, user_input: str, run_id: str | None = None) -> RunnableResult:
        """Extract a frame, compile a runnable, execute it.

        ``run_id`` scopes every emitted :class:`StreamEvent` to one
        cognitive dispatch. Defaults to a fresh ``uuid4`` if omitted —
        callers that want their own correlation id (a request id, a
        Slack thread, etc.) should pass one in. The id is also
        attached to the :class:`RunnableResult` raw payload so
        downstream callers can correlate without re-parsing the bus.
        """
        rid = run_id or str(uuid4())
        # Pin the run_id on the current asyncio context so any
        # nested instrumentation (orchestrators, specialists,
        # pipelines, skills) auto-tags its events without us
        # threading the id through every signature.
        token = set_run_id(rid)
        try:
            frame = await self.extract(user_input, run_id=rid)
            runnable = await self.compiler.compile(frame, run_id=rid)
            protocol_id = getattr(runnable, "protocol_id", "?")
            inner = getattr(runnable, "inner", None)
            if protocol_id == "?" and inner is not None:
                protocol_id = getattr(inner, "protocol_id", "?")
            await emit_runnable_executing(rid, protocol_id)
            try:
                result = await runnable.execute(user_input)
            except Exception as exc:  # noqa: BLE001 — surface every failure on bus
                await emit_runnable_failed(rid, protocol_id, f"{type(exc).__name__}: {exc}")
                raise
            await emit_runnable_executed(rid, protocol_id, len(result.text or ""))
            return result
        finally:
            # Drain history and signal end-of-stream to subscribers so
            # the SSE consumer sees a clean termination, not a hung
            # connection. Then unbind the contextvar.
            await get_event_bus().close_stream(rid)
            reset_run_id(token)
