# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Agent-to-agent handoff mechanism."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from locus.core.events import LocusEvent
from locus.core.messages import Message
from locus.core.state import AgentState
from locus.tools.decorator import Tool


class HandoffReason(StrEnum):
    """Reason for a handoff between agents."""

    SPECIALIZATION = "specialization"  # Target has better capabilities
    ESCALATION = "escalation"  # Issue needs higher authority
    DELEGATION = "delegation"  # Sub-task delegation
    COMPLETION = "completion"  # Task completed, returning to parent
    FAILURE = "failure"  # Agent failed, trying another


class HandoffEvent(LocusEvent):
    """Event emitted when a handoff occurs."""

    event_type: str = "handoff"
    source_agent_id: str
    target_agent_id: str
    reason: HandoffReason
    context_summary: str | None = None


class HandoffContext(BaseModel):
    """
    Context transferred during a handoff.

    Contains all information needed for the target agent to continue.
    """

    # Handoff metadata
    handoff_id: str = Field(default_factory=lambda: f"handoff_{uuid4().hex[:8]}")
    source_agent_id: str
    target_agent_id: str
    reason: HandoffReason

    # Original task
    original_task: str

    # Conversation history (key messages)
    conversation_summary: str | None = None
    key_messages: list[Message] = Field(default_factory=list)

    # State snapshot
    state_snapshot: dict[str, Any] = Field(default_factory=dict)

    # Findings and progress
    findings: dict[str, Any] = Field(default_factory=dict)
    progress_summary: str | None = None
    confidence: float = 0.0

    # Specific instructions for target
    instructions: str | None = None

    # Chain of custody
    handoff_chain: list[str] = Field(default_factory=list)

    # Timing
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"arbitrary_types_allowed": True}

    def to_prompt(self) -> str:
        """Convert handoff context to a prompt for the target agent."""
        lines = [
            "## Handoff Context",
            "",
            f"**Reason:** {self.reason.value}",
            f"**From:** {self.source_agent_id}",
            f"**Confidence so far:** {self.confidence:.2f}",
            "",
            "### Original Task",
            self.original_task,
            "",
        ]

        if self.progress_summary:
            lines.extend(
                [
                    "### Progress So Far",
                    self.progress_summary,
                    "",
                ]
            )

        if self.findings:
            lines.append("### Findings")
            for key, value in self.findings.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        if self.conversation_summary:
            lines.extend(
                [
                    "### Conversation Summary",
                    self.conversation_summary,
                    "",
                ]
            )

        if self.instructions:
            lines.extend(
                [
                    "### Instructions",
                    self.instructions,
                    "",
                ]
            )

        if self.handoff_chain:
            lines.extend(
                [
                    "### Handoff Chain",
                    " -> ".join(self.handoff_chain + [self.target_agent_id]),
                    "",
                ]
            )

        return "\n".join(lines)


class HandoffResult(BaseModel):
    """Result from a handoff operation."""

    handoff_id: str
    success: bool
    source_agent_id: str
    target_agent_id: str
    output: str | None = None
    final_confidence: float = 0.0
    duration_ms: float = 0.0
    error: str | None = None
    returned_context: HandoffContext | None = None

    model_config = {"arbitrary_types_allowed": True}


class HandoffAgent(BaseModel):
    """
    An agent that can participate in handoffs.

    Supports receiving context from other agents and
    transferring context when handing off.
    """

    id: str = Field(default_factory=lambda: f"agent_{uuid4().hex[:8]}")
    name: str
    description: str = ""
    system_prompt: str = ""
    tools: list[Tool] = Field(default_factory=list)
    model: Any = None

    # Handoff configuration
    can_escalate_to: list[str] = Field(default_factory=list)  # Agent IDs
    can_delegate_to: list[str] = Field(default_factory=list)  # Agent IDs

    model_config = {"arbitrary_types_allowed": True}

    async def receive_handoff(
        self,
        context: HandoffContext,
    ) -> HandoffResult:
        """
        Receive a handoff from another agent.

        Args:
            context: The handoff context

        Returns:
            HandoffResult with output
        """
        if self.model is None:
            return HandoffResult(
                handoff_id=context.handoff_id,
                success=False,
                source_agent_id=context.source_agent_id,
                target_agent_id=self.id,
                error="No model configured for agent",
            )

        start_time = time.perf_counter()

        # Build prompt from context
        handoff_prompt = context.to_prompt()

        # Create messages
        messages = [
            Message.system(self.system_prompt),
            Message.user(handoff_prompt),
        ]

        # Add key messages from context
        for msg in context.key_messages:
            messages.append(msg)

        # Final instruction
        messages.append(
            Message.user("Continue working on the task. Report your findings and conclusions.")
        )

        try:
            # Get tool schemas
            tool_schemas = None
            if self.tools:
                tool_schemas = [tool.to_openai_schema() for tool in self.tools]

            response = await self.model.complete(
                messages=messages,
                tools=tool_schemas,
            )
            content = response.message.content or ""

            # If the provider returned an empty body (some endpoints do this
            # when the prompt is mostly structured headers), retry once with
            # a more direct continuation instruction so callers don't silently
            # receive an empty handoff result.
            if not content.strip():
                retry_messages = [
                    *messages,
                    Message.user(
                        "Please respond now with your findings and conclusions "
                        "as a short paragraph. Do not return an empty response."
                    ),
                ]
                retry_response = await self.model.complete(
                    messages=retry_messages,
                    tools=tool_schemas,
                )
                content = retry_response.message.content or ""

            # Estimate new confidence
            confidence = self._estimate_confidence(content, context.confidence)

            duration_ms = (time.perf_counter() - start_time) * 1000

            return HandoffResult(
                handoff_id=context.handoff_id,
                success=bool(content.strip()),
                source_agent_id=context.source_agent_id,
                target_agent_id=self.id,
                output=content,
                final_confidence=confidence,
                duration_ms=duration_ms,
                error=None if content.strip() else "model returned empty content twice",
            )

        except Exception as e:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start_time) * 1000
            return HandoffResult(
                handoff_id=context.handoff_id,
                success=False,
                source_agent_id=context.source_agent_id,
                target_agent_id=self.id,
                error=str(e),
                duration_ms=duration_ms,
            )

    def _estimate_confidence(self, response: str, base_confidence: float) -> float:
        """Estimate confidence based on response and prior confidence."""
        response_lower = response.lower()

        # Confidence modifiers
        if any(word in response_lower for word in ["solved", "resolved", "confirmed"]):
            return min(1.0, base_confidence + 0.2)

        if any(word in response_lower for word in ["unclear", "uncertain", "need more"]):
            return max(0.0, base_confidence - 0.1)

        return min(1.0, base_confidence + 0.1)

    def with_model(self, model: Any) -> HandoffAgent:
        """Return a copy with the given model."""
        return self.model_copy(update={"model": model})


class Handoff(BaseModel):
    """
    Manages handoffs between agents.

    Features:
    - Context transfer between agents
    - State preservation
    - Handoff event emission
    - Chain of custody tracking
    """

    id: str = Field(default_factory=lambda: f"handoff_mgr_{uuid4().hex[:8]}")

    # Registered agents
    agents: dict[str, HandoffAgent] = Field(default_factory=dict)

    # Handoff history
    history: list[HandoffContext] = Field(default_factory=list)

    # Configuration
    max_handoff_chain: int = 5  # Maximum number of handoffs
    preserve_full_history: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def register_agent(self, agent: HandoffAgent) -> None:
        """Register an agent for handoffs."""
        self.agents[agent.id] = agent

    def register_agents(self, agents: list[HandoffAgent]) -> None:
        """Register multiple agents."""
        for agent in agents:
            self.register_agent(agent)

    def _extract_key_messages(
        self,
        state: AgentState,
        max_messages: int = 5,
    ) -> list[Message]:
        """Extract key messages from agent state for handoff."""
        messages = list(state.messages)

        if len(messages) <= max_messages:
            return messages

        # Keep system message + last N messages
        key_messages = []

        # Always keep system message
        for msg in messages:
            if msg.role.value == "system":
                key_messages.append(msg)
                break

        # Add last N messages
        key_messages.extend(messages[-max_messages:])

        return key_messages

    def _summarize_conversation(self, messages: list[Message]) -> str:
        """Create a summary of the conversation."""
        lines = []
        for msg in messages:
            role = msg.role.value.upper()
            content = msg.content or ""
            if len(content) > 200:
                content = content[:200] + "..."
            if content:
                lines.append(f"[{role}]: {content}")

        return "\n".join(lines)

    async def create_handoff(
        self,
        source_agent: HandoffAgent,
        target_agent_id: str,
        task: str,
        reason: HandoffReason,
        state: AgentState | None = None,
        findings: dict[str, Any] | None = None,
        instructions: str | None = None,
    ) -> HandoffContext:
        """
        Create a handoff context.

        Args:
            source_agent: The agent initiating the handoff
            target_agent_id: ID of the target agent
            task: The original task
            reason: Reason for the handoff
            state: Current agent state
            findings: Findings to transfer
            instructions: Specific instructions for target

        Returns:
            HandoffContext for the target agent
        """
        # Extract information from state
        key_messages: list[Message] = []
        state_snapshot: dict[str, Any] = {}
        conversation_summary: str | None = None
        confidence = 0.0

        if state:
            key_messages = self._extract_key_messages(state)
            conversation_summary = self._summarize_conversation(list(state.messages))
            state_snapshot = {
                "iteration": state.iteration,
                "tool_history": list(state.tool_history[-5:]),
                "errors": list(state.errors[-3:]),
            }
            confidence = state.confidence

        # Build handoff chain
        handoff_chain = [source_agent.id]
        if self.history:
            last_context = self.history[-1]
            if last_context.target_agent_id == source_agent.id:
                handoff_chain = last_context.handoff_chain + [source_agent.id]

        context = HandoffContext(
            source_agent_id=source_agent.id,
            target_agent_id=target_agent_id,
            reason=reason,
            original_task=task,
            conversation_summary=conversation_summary,
            key_messages=key_messages if self.preserve_full_history else [],
            state_snapshot=state_snapshot,
            findings=findings or {},
            confidence=confidence,
            instructions=instructions,
            handoff_chain=handoff_chain,
        )

        self.history.append(context)

        return context

    async def execute_handoff(
        self,
        source_agent: HandoffAgent,
        target_agent_id: str,
        task: str,
        reason: HandoffReason,
        state: AgentState | None = None,
        findings: dict[str, Any] | None = None,
        instructions: str | None = None,
    ) -> HandoffResult:
        """
        Execute a complete handoff.

        Args:
            source_agent: The agent initiating the handoff
            target_agent_id: ID of the target agent
            task: The original task
            reason: Reason for the handoff
            state: Current agent state
            findings: Findings to transfer
            instructions: Specific instructions for target

        Returns:
            HandoffResult from the target agent
        """
        # Validate target exists
        target_agent = self.agents.get(target_agent_id)
        if target_agent is None:
            return HandoffResult(
                handoff_id="",
                success=False,
                source_agent_id=source_agent.id,
                target_agent_id=target_agent_id,
                error=f"Target agent not found: {target_agent_id}",
            )

        # Check handoff chain limit
        chain_length = len(self.history)
        if chain_length >= self.max_handoff_chain:
            return HandoffResult(
                handoff_id="",
                success=False,
                source_agent_id=source_agent.id,
                target_agent_id=target_agent_id,
                error=f"Maximum handoff chain length ({self.max_handoff_chain}) exceeded",
            )

        # Create handoff context
        context = await self.create_handoff(
            source_agent=source_agent,
            target_agent_id=target_agent_id,
            task=task,
            reason=reason,
            state=state,
            findings=findings,
            instructions=instructions,
        )

        # Local import — observability is optional. The emit calls
        # below are no-ops when there is no run_context active.
        from locus.observability.emit import (  # noqa: PLC0415
            EV_HANDOFF_COMPLETED,
            EV_HANDOFF_INITIATED,
            emit,
        )

        await emit(
            EV_HANDOFF_INITIATED,
            source_agent_id=source_agent.id,
            target_agent_id=target_agent_id,
            reason=getattr(reason, "value", str(reason)),
            context_summary=context.progress_summary,
        )

        # Construct the typed event for back-compat consumers — emit
        # above is the live publication path.
        _handoff_event = HandoffEvent(  # noqa: F841
            source_agent_id=source_agent.id,
            target_agent_id=target_agent_id,
            reason=reason,
            context_summary=context.progress_summary,
        )

        # Execute handoff
        result = await target_agent.receive_handoff(context)
        result.returned_context = context

        await emit(
            EV_HANDOFF_COMPLETED,
            source_agent_id=source_agent.id,
            target_agent_id=target_agent_id,
            success=result.success,
            output_length=len(result.output or ""),
        )

        return result

    async def chain_handoff(
        self,
        agent_chain: list[str],
        task: str,
        initial_state: AgentState | None = None,
    ) -> list[HandoffResult]:
        """
        Execute a chain of handoffs through multiple agents.

        Args:
            agent_chain: List of agent IDs to process through
            task: The task to process
            initial_state: Initial state

        Returns:
            List of results from each handoff
        """
        results: list[HandoffResult] = []
        current_state = initial_state
        current_findings: dict[str, Any] = {}

        for i in range(len(agent_chain) - 1):
            source_id = agent_chain[i]
            target_id = agent_chain[i + 1]

            source_agent = self.agents.get(source_id)
            if source_agent is None:
                results.append(
                    HandoffResult(
                        handoff_id="",
                        success=False,
                        source_agent_id=source_id,
                        target_agent_id=target_id,
                        error=f"Source agent not found: {source_id}",
                    )
                )
                break

            result = await self.execute_handoff(
                source_agent=source_agent,
                target_agent_id=target_id,
                task=task,
                reason=HandoffReason.DELEGATION,
                state=current_state,
                findings=current_findings,
            )

            results.append(result)

            if not result.success:
                break

            # Update findings for next handoff
            if result.output:
                current_findings[f"from_{source_id}"] = result.output

        return results


def create_handoff_manager(
    agents: list[HandoffAgent] | None = None,
    max_chain: int = 5,
) -> Handoff:
    """
    Create a handoff manager.

    Args:
        agents: Agents to register
        max_chain: Maximum handoff chain length

    Returns:
        Configured Handoff manager
    """
    manager = Handoff(max_handoff_chain=max_chain)

    if agents:
        manager.register_agents(agents)

    return manager


def create_handoff_agent(
    name: str,
    description: str = "",
    system_prompt: str = "",
    tools: list[Tool] | None = None,
    model: Any = None,
) -> HandoffAgent:
    """
    Create a handoff-capable agent.

    Args:
        name: Agent name
        description: Agent description
        system_prompt: System prompt
        tools: Available tools
        model: Model for the agent

    Returns:
        Configured HandoffAgent
    """
    return HandoffAgent(
        name=name,
        description=description,
        system_prompt=system_prompt,
        tools=tools or [],
        model=model,
    )
