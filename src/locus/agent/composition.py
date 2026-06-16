# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Agent composition primitives.

Declarative patterns for composing agents:
- SequentialPipeline: Run agents in order, passing output to next
- ParallelPipeline: Run agents concurrently, merge results
- LoopAgent: Run an agent repeatedly until a condition is met
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class PipelineResult(BaseModel):
    """Result from a pipeline execution."""

    success: bool = True
    outputs: list[str] = Field(default_factory=list)
    final_output: str = ""
    duration_ms: float = 0.0
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class SequentialPipeline(BaseModel):
    """Run agents in order, passing each output as the next agent's prompt.

    Each agent receives either the original task (first agent) or the
    previous agent's output (subsequent agents). A prompt_template can
    customize how the previous output is passed.

    Example:
        >>> pipeline = SequentialPipeline(
        ...     agents=[researcher, writer, editor],
        ...     prompt_template="Based on the following:\\n{previous_output}\\n\\nOriginal task: {task}",
        ... )
        >>> result = await pipeline.run("Write about quantum computing")
    """

    agents: list[Any] = Field(default_factory=list)
    prompt_template: str = Field(
        default="{previous_output}\n\nContinue with the next step of: {task}",
        description="Template for subsequent agents. Available vars: {previous_output}, {task}",
    )

    model_config = {"arbitrary_types_allowed": True}

    async def run(self, task: str) -> PipelineResult:
        """Execute agents sequentially, chaining outputs."""
        # Local import — observability is optional; emit is a no-op
        # outside an active run_context.
        from locus.observability.emit import (  # noqa: PLC0415
            EV_PIPELINE_STAGE_COMPLETED,
            EV_PIPELINE_STAGE_STARTED,
            emit,
        )

        start_time = time.perf_counter()
        outputs: list[str] = []
        current_input = task

        try:
            for i, agent in enumerate(self.agents):
                if i > 0 and outputs:
                    # Format prompt with previous output
                    current_input = self.prompt_template.format(
                        previous_output=outputs[-1],
                        task=task,
                    )

                stage_started = time.perf_counter()
                await emit(
                    EV_PIPELINE_STAGE_STARTED,
                    pipeline_kind="sequential",
                    stage=i,
                    stage_count=len(self.agents),
                )
                result = agent.run_sync(current_input)
                output = result.message or ""
                outputs.append(output)
                await emit(
                    EV_PIPELINE_STAGE_COMPLETED,
                    pipeline_kind="sequential",
                    stage=i,
                    output_length=len(output),
                    duration_ms=(time.perf_counter() - stage_started) * 1000,
                    success=True,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000
            return PipelineResult(
                success=True,
                outputs=outputs,
                final_output=outputs[-1] if outputs else "",
                duration_ms=duration_ms,
            )

        except Exception as e:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start_time) * 1000
            return PipelineResult(
                success=False,
                outputs=outputs,
                final_output=outputs[-1] if outputs else "",
                duration_ms=duration_ms,
                error=str(e),
            )


class ParallelPipeline(BaseModel):
    """Run agents concurrently and merge their results.

    All agents receive the same task (or custom prompts via task_map).
    Results are collected and merged using the merge_strategy.

    Example:
        >>> pipeline = ParallelPipeline(
        ...     agents=[fact_checker, analyst, summarizer],
        ...     merge_strategy="concatenate",
        ... )
        >>> result = await pipeline.run("Analyze climate change impacts")
    """

    agents: list[Any] = Field(default_factory=list)
    merge_strategy: str = Field(
        default="concatenate",
        description="How to merge results: 'concatenate' or 'last'",
    )
    separator: str = Field(
        default="\n\n---\n\n",
        description="Separator for concatenated results",
    )

    model_config = {"arbitrary_types_allowed": True}

    async def run(self, task: str, task_map: dict[int, str] | None = None) -> PipelineResult:
        """Execute agents in parallel and merge results.

        Args:
            task: Default task for all agents
            task_map: Optional mapping of agent index to custom task
        """
        from locus.observability.emit import (  # noqa: PLC0415
            EV_PIPELINE_FANOUT_COMPLETED,
            EV_PIPELINE_FANOUT_STARTED,
            emit,
        )

        start_time = time.perf_counter()
        await emit(
            EV_PIPELINE_FANOUT_STARTED,
            agent_count=len(self.agents),
            merge_strategy=self.merge_strategy,
        )

        async def run_agent(index: int, agent: Any) -> str:
            prompt = task_map.get(index, task) if task_map else task
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, agent.run_sync, prompt)
            return result.message or ""

        # ``return_exceptions=True`` so one stuck/failed agent doesn't
        # collapse the whole result into an empty ``outputs=[]`` (which
        # forced every caller into defensive ``if result.success`` and
        # ate which-agent-failed context). Now each slot in ``outputs``
        # is either the agent's reply text or the stringified exception,
        # and ``error`` summarises the per-agent failures.
        tasks = [run_agent(i, agent) for i, agent in enumerate(self.agents)]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        outputs: list[str] = []
        agent_errors: list[str] = []
        for i, item in enumerate(gathered):
            if isinstance(item, BaseException):
                outputs.append("")
                agent_errors.append(f"agent[{i}] {type(item).__name__}: {item}")
            else:
                outputs.append(item)

        if self.merge_strategy == "last":
            final = outputs[-1] if outputs else ""
        else:
            final = self.separator.join(o for o in outputs if o)

        duration_ms = (time.perf_counter() - start_time) * 1000
        await emit(
            EV_PIPELINE_FANOUT_COMPLETED,
            agents_succeeded=len(self.agents) - len(agent_errors),
            agents_failed=len(agent_errors),
            duration_ms=duration_ms,
        )
        return PipelineResult(
            success=not agent_errors,
            outputs=outputs,
            final_output=final,
            duration_ms=duration_ms,
            error="; ".join(agent_errors) if agent_errors else None,
        )


class LoopAgent(BaseModel):
    """Run an agent repeatedly until a condition is met.

    The agent is called in a loop. After each iteration, the condition
    function is called with the latest output. If it returns True, the
    loop stops. A max_loops limit prevents infinite execution.

    Example:
        >>> loop = LoopAgent(
        ...     agent=editor,
        ...     condition=lambda output: "APPROVED" in output,
        ...     max_loops=5,
        ...     loop_prompt="Review and improve:\\n{previous_output}\\n\\nSay APPROVED if quality is good.",
        ... )
        >>> result = await loop.run("Write a haiku about Python")
    """

    agent: Any = None
    condition: Callable[[str], bool] = Field(
        default=lambda _output: False,  # Never stop by default — relies on max_loops
        description="Function that returns True when the loop should stop",
    )
    max_loops: int = Field(default=5, ge=1, le=50)
    loop_prompt: str = Field(
        default="Review and improve the following:\n{previous_output}\n\nOriginal task: {task}",
        description="Prompt template for loop iterations. Vars: {previous_output}, {task}",
    )

    model_config = {"arbitrary_types_allowed": True}

    async def run(self, task: str) -> PipelineResult:
        """Execute agent in a loop until condition is met or max_loops reached."""
        from locus.observability.emit import (  # noqa: PLC0415
            EV_LOOP_ITERATION_COMPLETED,
            EV_LOOP_ITERATION_STARTED,
            EV_LOOP_TERMINATED,
            emit,
        )

        start_time = time.perf_counter()
        outputs: list[str] = []
        current_input = task
        terminated_by = "max_loops"
        try:
            for i in range(self.max_loops):
                await emit(
                    EV_LOOP_ITERATION_STARTED,
                    iteration=i,
                    max_loops=self.max_loops,
                )
                result = self.agent.run_sync(current_input)
                output = result.message or ""
                outputs.append(output)
                stopped = self.condition(output)
                await emit(
                    EV_LOOP_ITERATION_COMPLETED,
                    iteration=i,
                    output_length=len(output),
                    condition_met=stopped,
                )

                # Check termination condition
                if stopped:
                    terminated_by = "condition"
                    break

                # Prepare next iteration prompt
                if i < self.max_loops - 1:
                    current_input = self.loop_prompt.format(
                        previous_output=output,
                        task=task,
                    )

            duration_ms = (time.perf_counter() - start_time) * 1000
            await emit(
                EV_LOOP_TERMINATED,
                terminated_by=terminated_by,
                iterations=len(outputs),
                duration_ms=duration_ms,
            )
            return PipelineResult(
                success=True,
                outputs=outputs,
                final_output=outputs[-1] if outputs else "",
                duration_ms=duration_ms,
            )

        except Exception as e:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start_time) * 1000
            return PipelineResult(
                success=False,
                outputs=outputs,
                final_output=outputs[-1] if outputs else "",
                duration_ms=duration_ms,
                error=str(e),
            )


def sequential(*agents: Any, prompt_template: str | None = None) -> SequentialPipeline:
    """Create a sequential pipeline from agents.

    Args:
        *agents: Agents to run in order
        prompt_template: Optional template for passing output between agents
    """
    kwargs: dict[str, Any] = {"agents": list(agents)}
    if prompt_template:
        kwargs["prompt_template"] = prompt_template
    return SequentialPipeline(**kwargs)


def parallel(*agents: Any, merge_strategy: str = "concatenate") -> ParallelPipeline:
    """Create a parallel pipeline from agents.

    Args:
        *agents: Agents to run concurrently
        merge_strategy: How to merge results ('concatenate' or 'last')
    """
    return ParallelPipeline(agents=list(agents), merge_strategy=merge_strategy)


def loop(
    agent: Any,
    condition: Callable[[str], bool],
    max_loops: int = 5,
    loop_prompt: str | None = None,
) -> LoopAgent:
    """Create a loop agent.

    Args:
        agent: Agent to run repeatedly
        condition: Function returning True when loop should stop
        max_loops: Maximum iterations
        loop_prompt: Template for loop iteration prompts
    """
    kwargs: dict[str, Any] = {
        "agent": agent,
        "condition": condition,
        "max_loops": max_loops,
    }
    if loop_prompt:
        kwargs["loop_prompt"] = loop_prompt
    return LoopAgent(**kwargs)
