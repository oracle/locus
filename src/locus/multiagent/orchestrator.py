# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Orchestrator pattern for multi-agent coordination."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from locus.core.messages import Message
from locus.multiagent.specialist import Specialist, SpecialistResult


class RoutingDecision(BaseModel):
    """A decision made by the orchestrator."""

    decision_type: str  # "invoke", "correlate", "summarize", "finalize"
    specialists: list[str] = Field(default_factory=list)
    reasoning: str = ""
    context: dict[str, Any] = Field(default_factory=dict)


class OrchestratorResult(BaseModel):
    """Result from the orchestrator execution."""

    orchestrator_id: str
    success: bool
    summary: str | None = None
    specialist_results: dict[str, SpecialistResult] = Field(default_factory=dict)
    decisions: list[RoutingDecision] = Field(default_factory=list)
    duration_ms: float = 0.0
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class Orchestrator(BaseModel):
    """
    Orchestrator for coordinating specialist agents.

    Features:
    - Selects which specialists to invoke based on the task
    - Routes tasks to appropriate specialists
    - Correlates findings from multiple specialists
    - Summarizes results into a coherent response
    """

    id: str = Field(default_factory=lambda: f"orchestrator_{uuid4().hex[:8]}")
    name: str = "Orchestrator"
    description: str = ""

    # Available specialists
    specialists: dict[str, Specialist] = Field(default_factory=dict)

    # Orchestrator configuration
    system_prompt: str = """You are an orchestrator coordinating specialist agents.
Your role is to:
1. Analyze the task and determine which specialists should handle it
2. Route sub-tasks to appropriate specialists
3. Correlate findings from multiple specialists
4. Synthesize a final response

Available specialists will be listed. Select the most appropriate ones for each task."""

    # Execution settings
    max_parallel_specialists: int = 5
    correlation_threshold: float = 0.7

    # The model to use
    model: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def register_specialist(self, specialist: Specialist) -> None:
        """Register a specialist with the orchestrator."""
        self.specialists[specialist.id] = specialist

    def register_specialists(self, specialists: list[Specialist]) -> None:
        """Register multiple specialists."""
        for specialist in specialists:
            self.register_specialist(specialist)

    def _build_routing_prompt(self, task: str) -> str:
        """Build a prompt for the routing decision."""
        specialist_descriptions = []
        for spec_id, spec in self.specialists.items():
            specialist_descriptions.append(f"- **{spec.name}** (id: {spec_id}): {spec.description}")

        return f"""## Task
{task}

## Available Specialists
{chr(10).join(specialist_descriptions)}

## Instructions
Based on the task, determine which specialists should be invoked.
Respond with a JSON object containing:
- "specialists": list of specialist IDs to invoke
- "reasoning": explanation of your selection
- "subtasks": dict mapping specialist ID to their specific subtask

Example response:
```json
{{
    "specialists": ["specialist_abc123", "specialist_def456"],
    "reasoning": "This task requires log analysis and metrics correlation",
    "subtasks": {{
        "specialist_abc123": "Analyze the error logs for the time period",
        "specialist_def456": "Check CPU and memory metrics during the incident"
    }}
}}
```"""

    def _build_correlation_prompt(
        self,
        task: str,
        results: dict[str, SpecialistResult],
    ) -> str:
        """Build a prompt for correlating specialist findings."""
        findings = []
        for spec_id, result in results.items():
            spec = self.specialists.get(spec_id)
            name = spec.name if spec else spec_id
            findings.append(f"### {name}\n{result.output or 'No output'}")

        return f"""## Original Task
{task}

## Specialist Findings
{chr(10).join(findings)}

## Instructions
Correlate the findings from all specialists. Look for:
1. Common themes or patterns
2. Contradictions that need resolution
3. Gaps in the analysis
4. Causal relationships between findings

Provide a structured correlation analysis."""

    def _build_summary_prompt(
        self,
        task: str,
        correlation: str,
        results: dict[str, SpecialistResult],
    ) -> str:
        """Build a prompt for final summarization."""
        return f"""## Original Task
{task}

## Correlation Analysis
{correlation}

## Instructions
Synthesize the analysis into a clear, actionable summary.
Include:
1. Key findings
2. Root cause (if identified)
3. Recommended actions
4. Confidence level in the analysis"""

    async def _make_routing_decision(self, task: str) -> RoutingDecision:
        """Use the model to decide which specialists to invoke."""
        if self.model is None:
            # Default: invoke all specialists
            return RoutingDecision(
                decision_type="invoke",
                specialists=list(self.specialists.keys()),
                reasoning="No model available, invoking all specialists",
            )

        prompt = self._build_routing_prompt(task)
        messages = [
            Message.system(self.system_prompt),
            Message.user(prompt),
        ]

        response = await self.model.complete(messages=messages)
        content = response.message.content or ""

        # Parse the response (simple extraction)
        import json
        import re

        # Try to extract JSON from the response
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return RoutingDecision(
                    decision_type="invoke",
                    specialists=data.get("specialists", []),
                    reasoning=data.get("reasoning", ""),
                    context={"subtasks": data.get("subtasks", {})},
                )
            except json.JSONDecodeError:
                pass

        # Fallback: try to find specialist IDs mentioned in the response
        mentioned_specialists = [spec_id for spec_id in self.specialists if spec_id in content]

        return RoutingDecision(
            decision_type="invoke",
            specialists=mentioned_specialists or list(self.specialists.keys()),
            reasoning=content,
        )

    async def _invoke_specialists(
        self,
        task: str,
        decision: RoutingDecision,
    ) -> dict[str, SpecialistResult]:
        """Invoke selected specialists in parallel, bounded by
        ``max_parallel_specialists``.

        Each specialist gets a retry if its first response was empty,
        and a per-specialist exception is captured into the
        ``SpecialistResult.error`` field instead of bringing down the
        whole batch — same shape ``ParallelPipeline`` uses.

        Concurrency is capped by an :class:`asyncio.Semaphore` so the
        orchestrator can fan out 10+ specialists without flooding the
        provider's rate limits — set ``max_parallel_specialists=1`` to
        get the old serial behaviour.
        """
        subtasks = decision.context.get("subtasks", {})
        semaphore = asyncio.Semaphore(max(1, self.max_parallel_specialists))

        async def _run_one(spec_id: str) -> tuple[str, SpecialistResult]:
            specialist = self.specialists.get(spec_id)
            if specialist is None:
                return spec_id, SpecialistResult(
                    specialist_id=spec_id,
                    specialist_type="unknown",
                    error=f"Specialist not found: {spec_id}",
                )
            spec_task = subtasks.get(spec_id, task)
            context = {"original_task": task} if spec_task != task else None
            async with semaphore:
                # Retry once if the first attempt comes back empty.
                result = await specialist.execute(task=spec_task, context=context)
                if not result.output:
                    result = await specialist.execute(task=spec_task, context=context)
            return spec_id, result

        gathered = await asyncio.gather(
            *(_run_one(sid) for sid in decision.specialists),
            return_exceptions=True,
        )

        results: dict[str, SpecialistResult] = {}
        for sid, item in zip(decision.specialists, gathered, strict=False):
            if isinstance(item, BaseException):
                spec = self.specialists.get(sid)
                results[sid] = SpecialistResult(
                    specialist_id=sid,
                    specialist_type=spec.specialist_type if spec else "unknown",
                    error=f"{type(item).__name__}: {item}",
                )
            else:
                _spec_id, spec_result = item
                results[_spec_id] = spec_result
        return results

    async def _correlate_findings(
        self,
        task: str,
        results: dict[str, SpecialistResult],
    ) -> str:
        """Correlate findings from multiple specialists."""
        if self.model is None:
            # Simple concatenation without model
            findings = []
            for spec_id, result in results.items():
                spec = self.specialists.get(spec_id)
                name = spec.name if spec else spec_id
                findings.append(f"## {name}\n{result.output or 'No output'}")
            return "\n\n".join(findings)

        prompt = self._build_correlation_prompt(task, results)
        messages = [
            Message.system(self.system_prompt),
            Message.user(prompt),
        ]

        response = await self.model.complete(messages=messages)
        return response.message.content or ""

    async def _summarize(
        self,
        task: str,
        correlation: str,
        results: dict[str, SpecialistResult],
    ) -> str:
        """Generate final summary."""
        if self.model is None:
            return correlation

        prompt = self._build_summary_prompt(task, correlation, results)
        messages = [
            Message.system(self.system_prompt),
            Message.user(prompt),
        ]

        response = await self.model.complete(messages=messages)
        return response.message.content or ""

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> OrchestratorResult:
        """
        Execute the orchestration workflow.

        Args:
            task: The task to process
            context: Optional additional context

        Returns:
            OrchestratorResult with summary and all findings
        """
        # Local import — keeps observability optional. If the user
        # never enters a run_context, ``emit`` is a no-op and the
        # bus singleton is never instantiated.
        from locus.observability.emit import (  # noqa: PLC0415
            EV_ORCHESTRATOR_DECISION,
            EV_ORCHESTRATOR_ROUTING,
            EV_ORCHESTRATOR_SPECIALISTS_INVOKED,
            EV_ORCHESTRATOR_SUMMARY,
            emit,
        )

        start_time = time.perf_counter()
        decisions: list[RoutingDecision] = []

        try:
            await emit(
                EV_ORCHESTRATOR_ROUTING,
                orchestrator_id=self.id,
                task_preview=task[:160],
                specialist_count=len(self.specialists),
            )

            # Step 1: Make routing decision
            routing_decision = await self._make_routing_decision(task)
            decisions.append(routing_decision)

            await emit(
                EV_ORCHESTRATOR_DECISION,
                orchestrator_id=self.id,
                decision="invoke_specialist",
                specialists_selected=routing_decision.specialists,
                reasoning=routing_decision.reasoning,
            )

            # Step 2: Invoke specialists
            specialist_results = await self._invoke_specialists(task, routing_decision)

            await emit(
                EV_ORCHESTRATOR_SPECIALISTS_INVOKED,
                orchestrator_id=self.id,
                specialists_invoked=list(specialist_results.keys()),
                specialists_succeeded=[sid for sid, r in specialist_results.items() if r.success],
                specialists_failed=[sid for sid, r in specialist_results.items() if not r.success],
            )

            # Step 3: Correlate findings
            correlation_decision = RoutingDecision(
                decision_type="correlate",
                reasoning="Correlating findings from specialists",
            )
            decisions.append(correlation_decision)

            await emit(
                EV_ORCHESTRATOR_DECISION,
                orchestrator_id=self.id,
                decision="correlate",
                reasoning="Correlating specialist findings",
            )

            correlation = await self._correlate_findings(task, specialist_results)

            # Step 4: Summarize
            summary_decision = RoutingDecision(
                decision_type="summarize",
                reasoning="Generating final summary",
            )
            decisions.append(summary_decision)

            await emit(
                EV_ORCHESTRATOR_DECISION,
                orchestrator_id=self.id,
                decision="summarize",
                reasoning="Generating final summary",
            )

            summary = await self._summarize(task, correlation, specialist_results)
            await emit(
                EV_ORCHESTRATOR_SUMMARY,
                orchestrator_id=self.id,
                summary_length=len(summary or ""),
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            return OrchestratorResult(
                orchestrator_id=self.id,
                success=True,
                summary=summary,
                specialist_results=specialist_results,
                decisions=decisions,
                duration_ms=duration_ms,
            )

        except Exception as e:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start_time) * 1000
            return OrchestratorResult(
                orchestrator_id=self.id,
                success=False,
                decisions=decisions,
                duration_ms=duration_ms,
                error=str(e),
            )

    def with_model(self, model: Any) -> Orchestrator:
        """Return a copy of this orchestrator with the given model."""
        # Also update specialists with the model
        updated_specialists = {
            spec_id: spec.with_model(model) for spec_id, spec in self.specialists.items()
        }
        return self.model_copy(
            update={
                "model": model,
                "specialists": updated_specialists,
            }
        )


def create_orchestrator(
    name: str = "Orchestrator",
    specialists: list[Specialist] | None = None,
    model: Any = None,
) -> Orchestrator:
    """
    Create an orchestrator with the given specialists.

    Args:
        name: Orchestrator name
        specialists: List of specialists to register
        model: Model for decision making

    Returns:
        Configured Orchestrator instance
    """
    orchestrator = Orchestrator(name=name, model=model)

    if specialists:
        orchestrator.register_specialists(specialists)

    return orchestrator
