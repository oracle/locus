# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Steering system — LLM-powered real-time agent guidance.

Evaluates tool calls and model responses using a separate LLM to decide:
- Proceed: allow the action
- Guide: cancel and inject corrective feedback
- Interrupt: pause for human approval

This goes beyond regex guardrails — the steering LLM understands context
and can make nuanced decisions about whether an action is appropriate.

Example:
    from locus.hooks.builtin.steering import SteeringHook

    steering = SteeringHook(
        model=steering_model,  # Can be a smaller/cheaper model
        policy="Only allow database reads, never writes or deletes.",
    )

    agent = Agent(config=AgentConfig(
        model=main_model,
        hooks=[steering],
    ))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from locus.hooks.provider import HookPriority, HookProvider


logger = logging.getLogger(__name__)


class SteeringAction(StrEnum):
    """Action the steering system can take."""

    PROCEED = "proceed"  # Allow the action
    GUIDE = "guide"  # Cancel and provide feedback
    INTERRUPT = "interrupt"  # Pause for human approval


@dataclass
class SteeringDecision:
    """Decision from the steering evaluator."""

    action: SteeringAction
    reason: str = ""
    guidance: str = ""  # Feedback message for GUIDE action
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class SteeringContext:
    """Activity ledger tracking agent behavior for steering decisions."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    policy: str = ""

    def record_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Record a tool call for context."""
        self.tool_calls.append(
            {
                "tool": tool_name,
                "arguments": arguments,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def to_prompt(self) -> str:
        """Format context as a prompt for the steering LLM."""
        parts = []
        if self.policy:
            parts.append(f"## Policy\n{self.policy}")

        if self.tool_calls:
            recent = self.tool_calls[-5:]  # Last 5 calls
            calls = "\n".join(f"- {c['tool']}({c['arguments']})" for c in recent)
            parts.append(f"## Recent Tool Calls\n{calls}")

        parts.append(f"## Stats\nModel calls: {self.model_calls}")
        return "\n\n".join(parts)


class SteeringHook(HookProvider):
    """LLM-powered steering for real-time agent guidance.

    Evaluates each tool call before execution using a separate LLM.
    The steering model decides whether to proceed, guide (cancel with
    feedback), or interrupt (pause for human).

    Args:
        model: LLM for steering decisions (can be smaller/cheaper than main model).
        policy: Natural language policy the agent must follow.
        evaluate_tools: If True, evaluate tool calls before execution.
        evaluate_responses: If True, evaluate model responses after generation.
        interrupt_tools: Tools that always require human approval.
    """

    def __init__(
        self,
        model: Any,
        policy: str = "",
        evaluate_tools: bool = True,
        evaluate_responses: bool = False,
        interrupt_tools: set[str] | None = None,
        priority: int = HookPriority.SECURITY_DEFAULT,
    ) -> None:
        self._model = model
        self._policy = policy
        self._evaluate_tools = evaluate_tools
        self._evaluate_responses = evaluate_responses
        self._interrupt_tools = interrupt_tools or set()
        self._priority = priority
        self._context = SteeringContext(policy=policy)
        self.decisions: list[SteeringDecision] = []

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def name(self) -> str:
        return "SteeringHook"

    async def _evaluate_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> SteeringDecision:
        """Use the steering LLM to evaluate a tool call."""
        from locus.core.messages import Message

        prompt = (
            f"You are a safety evaluator. Evaluate whether this tool call "
            f"should be allowed based on the policy.\n\n"
            f"{self._context.to_prompt()}\n\n"
            f"## Proposed Tool Call\n"
            f"Tool: {tool_name}\n"
            f"Arguments: {arguments}\n\n"
            f"Respond with exactly one of:\n"
            f"- PROCEED: if the call is safe and within policy\n"
            f"- GUIDE: <reason> if the call should be blocked with feedback\n"
            f"- INTERRUPT: <reason> if human approval is needed"
        )

        try:
            response = await self._model.complete(
                messages=[
                    Message.system("You are a concise safety evaluator."),
                    Message.user(prompt),
                ],
                max_tokens=100,
            )

            text = (response.message.content or "PROCEED").strip().upper()

            if text.startswith("GUIDE"):
                reason = text[5:].strip(": ")
                return SteeringDecision(
                    action=SteeringAction.GUIDE,
                    reason=reason,
                    guidance=f"Steering blocked: {reason}",
                )
            elif text.startswith("INTERRUPT"):
                reason = text[9:].strip(": ")
                return SteeringDecision(
                    action=SteeringAction.INTERRUPT,
                    reason=reason,
                )
            else:
                return SteeringDecision(action=SteeringAction.PROCEED)

        except Exception:
            logger.exception("Steering evaluation failed, defaulting to PROCEED")
            return SteeringDecision(action=SteeringAction.PROCEED)

    async def on_before_tool_call(self, event: Any) -> None:
        """Evaluate tool call before execution."""
        if not self._evaluate_tools:
            return

        # Always interrupt for specified tools
        if event.tool_name in self._interrupt_tools:
            decision = SteeringDecision(
                action=SteeringAction.INTERRUPT,
                reason=f"Tool '{event.tool_name}' requires human approval",
            )
            self.decisions.append(decision)
            event.cancel = f"REQUIRES APPROVAL: {event.tool_name} needs human approval"
            return

        # LLM evaluation
        decision = await self._evaluate_tool_call(event.tool_name, event.arguments)
        self.decisions.append(decision)

        from locus.observability.emit import EV_HOOK_STEERING_APPLIED, emit  # noqa: PLC0415

        if decision.action == SteeringAction.GUIDE:
            event.cancel = decision.guidance
            logger.info(
                "Steering GUIDE: %s(%s) — %s", event.tool_name, event.arguments, decision.reason
            )
            await emit(
                EV_HOOK_STEERING_APPLIED,
                action="guide",
                tool_name=event.tool_name,
                reason=decision.reason,
            )

        elif decision.action == SteeringAction.INTERRUPT:
            event.cancel = f"REQUIRES APPROVAL: {decision.reason}"
            logger.info("Steering INTERRUPT: %s — %s", event.tool_name, decision.reason)
            await emit(
                EV_HOOK_STEERING_APPLIED,
                action="interrupt",
                tool_name=event.tool_name,
                reason=decision.reason,
            )

        # Record in context
        self._context.record_tool_call(event.tool_name, event.arguments)

    async def on_before_model_call(self, event: Any) -> None:
        """Track model calls in context."""
        self._context.model_calls += 1

    async def on_after_model_call(self, event: Any) -> None:
        """Optionally evaluate model responses."""
        if not self._evaluate_responses:
            return

        content = event.response.message.content or ""
        if not content:
            return

        # Simple policy check — could be expanded to use steering LLM
        if self._policy and any(
            word in content.lower() for word in ["password", "secret", "credential"]
        ):
            logger.warning("Steering: response may contain sensitive info")
