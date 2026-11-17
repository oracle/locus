# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Top-level :class:`Router` — the only public surface of the module.

The Router takes natural language, asks a :class:`RequestClassifier` (by
default an extractor agent) to fill a :class:`GoalFrame`, hands it to the
:class:`CognitiveCompiler`, and executes the compiled :class:`Runnable`.
Everything else in ``locus.router`` is internal machinery.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
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
from locus.router.classifier import (
    AgentFrameExtractor,
    FrameExtractionError,
    RequestClassifier,
)
from locus.router.compiler import CognitiveCompiler
from locus.router.explain import RoutingExplanation
from locus.router.goal_frame import GoalFrame
from locus.router.runnable import Runnable, RunnableResult


if TYPE_CHECKING:
    from locus.agent.agent import Agent


class Router:
    """One-shot dispatcher: text in, :class:`RunnableResult` out.

    Parameters
    ----------
    extractor:
        :class:`~locus.Agent` configured with
        ``output_schema=GoalFrame``. Its parsed result drives selection.
        Wrapped in an :class:`AgentFrameExtractor`; pass ``classifier``
        instead to supply the frame some other way.
    classifier:
        Any :class:`RequestClassifier` — ``async classify(text) ->
        GoalFrame``. Exactly one of ``extractor`` and ``classifier`` is
        given; the compiler sees the same frame either way.
    compiler:
        :class:`CognitiveCompiler` set up with protocols, capabilities,
        policy, and model.
    on_frame:
        Optional callback fired right after a frame is extracted. Useful
        for telemetry / UI display.
    """

    def __init__(
        self,
        *,
        compiler: CognitiveCompiler,
        extractor: Agent | None = None,
        classifier: RequestClassifier | None = None,
        on_frame: Callable[[GoalFrame], None] | None = None,
    ) -> None:
        if classifier is not None:
            if extractor is not None:
                err_msg = "Router takes either extractor= or classifier=, not both."
                raise ValueError(err_msg)
            self.classifier: RequestClassifier = classifier
        elif extractor is not None:
            self.classifier = AgentFrameExtractor(extractor)
        else:
            err_msg = "Router needs an extractor= agent or a classifier=."
            raise ValueError(err_msg)
        self.extractor = extractor
        self.compiler = compiler
        self._on_frame = on_frame

    async def extract(self, user_input: str, run_id: str | None = None) -> GoalFrame:
        """Run the classifier and return the :class:`GoalFrame`.

        Emits ``router.frame.extracted`` on success or
        ``router.frame.failed`` when the classifier raises
        :class:`FrameExtractionError`, scoped to ``run_id`` when supplied.
        The error is re-raised — the compiler can't recover from that.
        """
        try:
            frame = await self.classifier.classify(user_input)
        except FrameExtractionError as exc:
            if run_id:
                await emit_frame_failed(run_id, str(exc))
            raise
        if run_id:
            await emit_frame_extracted(run_id, frame)
        if self._on_frame is not None:
            self._on_frame(frame)
        return frame

    async def explain(
        self,
        user_input: str,
        frame: GoalFrame | None = None,
    ) -> RoutingExplanation:
        """Say what the router would choose, and why. Nothing is executed.

        One model call, for the frame — pass ``frame`` to spend none at all.
        Selection, ranking and the policy check are pure functions of the
        frame and the registry, so everything after extraction is free and
        deterministic.

        Returns the evidence the compiler computes and discards: every
        protocol that was ruled out and which of the three gates stopped it,
        the survivors in rank order with the terms that ordered them, and the
        policy verdict.

            >>> e = await router.explain("Delete every record older than 7 years")
            >>> e.selected_protocol
            'approval_gated_execution'
            >>> e.requires_approval
            True

        This is the read-only sibling of :meth:`dispatch`; use
        :meth:`compile` when you want the topology that would be built.
        """
        goal_frame = frame if frame is not None else await self.extract(user_input)
        return self.compiler.explain(goal_frame, goal=user_input)

    async def compile(
        self,
        user_input: str,
        frame: GoalFrame | None = None,
    ) -> Runnable:
        """Build the topology without running it.

        The compiler has always separated building from executing; this is
        the front door for it, so a caller does not have to reach through
        ``router.compiler`` and construct a frame by hand.
        """
        goal_frame = frame if frame is not None else await self.extract(user_input)
        return await self.compiler.compile(goal_frame)

    async def dispatch(
        self,
        user_input: str,
        run_id: str | None = None,
        frame: GoalFrame | None = None,
    ) -> RunnableResult:
        """Extract a frame, compile a runnable, execute it.

        ``run_id`` scopes every emitted :class:`StreamEvent` to one
        cognitive dispatch. Defaults to a fresh ``uuid4`` if omitted —
        callers that want their own correlation id (a request id, a
        Slack thread, etc.) should pass one in. The id is also
        attached to the :class:`RunnableResult` raw payload so
        downstream callers can correlate without re-parsing the bus.

        ``frame`` pins a pre-extracted :class:`GoalFrame` and skips the
        extractor entirely. This is the resume seam: a governed dispatch
        that parked on a hold must replay under the SAME frame a human's
        approval was granted against — a live extractor re-reading the
        goal can frame it differently, select a different protocol, and
        re-hold what was already approved, forever. The pinned frame
        still flows through ``on_frame`` and the event bus, so consumers
        see an identical stream either way.
        """
        rid = run_id or str(uuid4())
        # Pin the run_id on the current asyncio context so any
        # nested instrumentation (orchestrators, specialists,
        # pipelines, skills) auto-tags its events without us
        # threading the id through every signature.
        token = set_run_id(rid)
        try:
            if frame is None:
                frame = await self.extract(user_input, run_id=rid)
            else:
                await emit_frame_extracted(rid, frame)
                if self._on_frame is not None:
                    self._on_frame(frame)
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
