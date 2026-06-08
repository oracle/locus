# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Uniform :class:`Runnable` contract over heterogeneous primitives.

Tulip primitives expose three different methods (`invoke`, `run`,
`execute`) returning four different result types (`AgentResult`,
`PipelineResult`, `OrchestratorResult`, plain `str`). The router
needs a single shape so :class:`~tulip.router.compiler.CognitiveCompiler`
can return any of them uniformly. This module provides the shared
contract plus thin per-primitive adapters.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from tulip.agent.agent import Agent
    from tulip.agent.composition import LoopAgent, ParallelPipeline, SequentialPipeline
    from tulip.multiagent.orchestrator import Orchestrator
    from tulip.router.goal_frame import GoalFrame


class RunnableResult(BaseModel):
    """Normalized result envelope for any compiled router execution."""

    text: str = Field(
        ...,
        description="Surface text — what the caller would print to a user.",
    )
    parsed: BaseModel | None = Field(
        default=None,
        description="Mirrors AgentResult.parsed when output_schema was set.",
    )
    raw: Any = Field(
        ...,
        description="The underlying primitive's full result object.",
    )
    protocol_id: str = Field(
        ...,
        description="Which Protocol.id produced this. Useful for audit + telemetry.",
    )
    frame: Any = Field(
        ...,
        description="The GoalFrame that drove protocol selection.",
    )

    model_config = {"arbitrary_types_allowed": True, "frozen": True}


@runtime_checkable
class Runnable(Protocol):
    """Anything the router compiler can return.

    The contract is intentionally minimal: a single async ``execute``
    that yields a normalized :class:`RunnableResult`. Builders wrap
    each emitted primitive in the matching adapter below — call sites
    never branch on the concrete primitive type.
    """

    async def execute(self, task: str) -> RunnableResult: ...


class AgentRunnable(BaseModel):
    """Adapter from :class:`tulip.Agent` to :class:`Runnable`.

    ``Agent.invoke`` is sync but already handles being called from a
    running event loop (it spawns a worker thread internally). We wrap
    it in :func:`asyncio.to_thread` so the surrounding coroutine yields
    properly while the agent runs.
    """

    agent: Any
    protocol_id: str
    frame: Any

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, task: str) -> RunnableResult:
        result = await asyncio.to_thread(self.agent.invoke, task)
        return RunnableResult(
            text=result.message or "",
            parsed=result.parsed,
            raw=result,
            protocol_id=self.protocol_id,
            frame=self.frame,
        )


class PipelineRunnable(BaseModel):
    """Adapter from any composition primitive to :class:`Runnable`.

    Works for :class:`SequentialPipeline`, :class:`ParallelPipeline`,
    and :class:`LoopAgent` — they all share the
    ``async run(task) -> PipelineResult`` shape.
    """

    pipeline: Any
    protocol_id: str
    frame: Any

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, task: str) -> RunnableResult:
        result = await self.pipeline.run(task)
        return RunnableResult(
            text=result.final_output or "",
            parsed=None,
            raw=result,
            protocol_id=self.protocol_id,
            frame=self.frame,
        )


class OrchestratorRunnable(BaseModel):
    """Adapter from :class:`tulip.Orchestrator` to :class:`Runnable`."""

    orchestrator: Any
    protocol_id: str
    frame: Any

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, task: str) -> RunnableResult:
        result = await self.orchestrator.execute(task)
        return RunnableResult(
            text=result.summary or "",
            parsed=None,
            raw=result,
            protocol_id=self.protocol_id,
            frame=self.frame,
        )


class DebateRunnable(BaseModel):
    """Adapter for the *debate* protocol.

    Runs N debater agents (each a :class:`Agent`) in parallel against
    the same task, then feeds the joined transcript to a single judge
    :class:`Agent`
    that picks the strongest argument. The judge's verdict is the
    surface text. There is no native tulip "debate" primitive — this
    builder composes :class:`ParallelPipeline` + an extra
    :class:`Agent` step, which is small enough to live in this adapter.
    """

    debaters: Any  # ParallelPipeline of Agents
    judge: Any  # single Agent
    protocol_id: str
    frame: Any

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, task: str) -> RunnableResult:
        import asyncio as _asyncio

        debate_result = await self.debaters.run(task)
        judge_prompt = (
            f"Original question: {task}\n\n"
            f"Debater transcripts (separated by --- ):\n\n{debate_result.final_output}\n\n"
            "Pick the strongest argument and explain in two sentences why it wins. "
            "End with one line: 'WINNER: <debater label or 'inconclusive'>'."
        )
        verdict = await _asyncio.to_thread(self.judge.invoke, judge_prompt)
        return RunnableResult(
            text=verdict.message or "",
            parsed=verdict.parsed,
            raw={"debate": debate_result, "verdict": verdict},
            protocol_id=self.protocol_id,
            frame=self.frame,
        )


class A2ARunnable(BaseModel):
    """Adapter for the *a2a_delegate* protocol.

    Wraps an :class:`A2AClient` so a remote, separately-deployed agent
    can serve a request. The remote agent is responsible for any tool
    calls / orchestration on its end; the router simply forwards the
    user prompt and packages the response.
    """

    client: Any  # A2AClient
    protocol_id: str
    frame: Any

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, task: str) -> RunnableResult:
        from tulip.a2a import Message, TextPart

        message = Message(
            role="user",
            parts=[TextPart(text=task)],
            messageId=uuid.uuid4().hex,
        )
        try:
            remote_task = await self.client.send_message(message)
            text = self._extract_task_text(remote_task)
            raw: Any = remote_task
        except RuntimeError as exc:
            if "A2A error -32601" not in str(exc):
                raise
            text = await self.client.invoke(task)
            raw = text
        return RunnableResult(
            text=text or "",
            parsed=None,
            raw=raw,
            protocol_id=self.protocol_id,
            frame=self.frame,
        )

    @staticmethod
    def _extract_task_text(task: Any) -> str:
        if getattr(task, "artifacts", None):
            for artifact in reversed(task.artifacts):
                for part in reversed(getattr(artifact, "parts", [])):
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text:
                        return text
        status = getattr(task, "status", None)
        message = getattr(status, "message", None)
        if message is not None:
            for part in reversed(getattr(message, "parts", [])):
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    return text
        return ""


def wrap_agent(agent: Agent, protocol_id: str, frame: GoalFrame) -> Runnable:
    """Helper used by protocol builders for the single-agent shape."""
    return AgentRunnable(agent=agent, protocol_id=protocol_id, frame=frame)


def wrap_pipeline(
    pipeline: SequentialPipeline | ParallelPipeline | LoopAgent,
    protocol_id: str,
    frame: GoalFrame,
) -> Runnable:
    """Helper for any of the composition primitives."""
    return PipelineRunnable(pipeline=pipeline, protocol_id=protocol_id, frame=frame)


def wrap_orchestrator(orchestrator: Orchestrator, protocol_id: str, frame: GoalFrame) -> Runnable:
    """Helper for the orchestrator + specialists shape."""
    return OrchestratorRunnable(orchestrator=orchestrator, protocol_id=protocol_id, frame=frame)


def wrap_debate(
    debaters: ParallelPipeline, judge: Agent, protocol_id: str, frame: GoalFrame
) -> Runnable:
    """Helper for the parallel-debaters-then-judge shape."""
    return DebateRunnable(debaters=debaters, judge=judge, protocol_id=protocol_id, frame=frame)


def wrap_a2a(client: Any, protocol_id: str, frame: GoalFrame) -> Runnable:
    """Helper for cross-process delegation via :class:`A2AClient`."""
    return A2ARunnable(client=client, protocol_id=protocol_id, frame=frame)
