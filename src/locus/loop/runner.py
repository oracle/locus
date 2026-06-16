# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Loop executor and utilities - 100% Pydantic."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr

from locus.core.events import LoopEvent, TerminateEvent
from locus.core.state import AgentState
from locus.loop.react import ReActLoop, ReActLoopConfig


if TYPE_CHECKING:
    from locus.core.protocols import ModelProtocol
    from locus.tools.registry import ToolRegistry


class LoopRunner(BaseModel):
    """
    High-level executor for ReAct loops.

    Provides additional features:
    - Event callbacks/hooks
    - Error handling and retries
    - Timeout management
    - Progress tracking
    """

    loop: ReActLoop

    # Event callbacks
    on_event: Callable[[LoopEvent], None] | None = None
    on_error: Callable[[Exception, AgentState], None] | None = None
    on_complete: Callable[[AgentState, list[LoopEvent]], None] | None = None

    # Execution options
    timeout: float | None = Field(default=None, description="Timeout in seconds")
    retry_on_error: bool = False
    max_retries: int = Field(default=3, ge=0)

    # Private state for tracking
    _events: list[LoopEvent] = PrivateAttr(default_factory=list)
    _final_state: AgentState | None = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    async def run(
        self,
        prompt: str,
        initial_state: AgentState | None = None,
        **state_kwargs: Any,
    ) -> AsyncIterator[LoopEvent]:
        """
        Run the loop with callbacks and error handling.

        Args:
            prompt: User prompt
            initial_state: Optional initial state
            **state_kwargs: Additional state configuration

        Yields:
            Loop events
        """
        self._events = []
        self._final_state = None
        retries = 0

        while True:
            try:
                async for event in self._run_with_timeout(prompt, initial_state, **state_kwargs):
                    self._events.append(event)

                    # Call event callback
                    if self.on_event:
                        self.on_event(event)

                    yield event

                    # Track final state from terminate event
                    if isinstance(event, TerminateEvent):
                        break

                # Success - exit retry loop
                break

            except Exception as e:
                retries += 1

                if self.on_error:
                    # Create state for error callback
                    error_state = initial_state or AgentState()
                    self.on_error(e, error_state)

                if not self.retry_on_error or retries > self.max_retries:
                    raise

                # Wait before retry (exponential backoff)
                await asyncio.sleep(min(2**retries, 30))

        # Call completion callback
        if self.on_complete and self._final_state:
            self.on_complete(self._final_state, self._events)

    async def _run_with_timeout(
        self,
        prompt: str,
        initial_state: AgentState | None,
        **state_kwargs: Any,
    ) -> AsyncIterator[LoopEvent]:
        """Run the loop with optional timeout."""
        if self.timeout is None:
            async for event in self.loop.run(prompt, initial_state, **state_kwargs):
                yield event
        else:
            try:
                async with asyncio.timeout(self.timeout):
                    async for event in self.loop.run(prompt, initial_state, **state_kwargs):
                        yield event
            except TimeoutError:
                yield TerminateEvent(
                    reason="timeout",
                    iterations_used=0,
                    final_confidence=0.0,
                    total_tool_calls=0,
                )

    async def run_to_completion(
        self,
        prompt: str,
        initial_state: AgentState | None = None,
        **state_kwargs: Any,
    ) -> tuple[AgentState, list[LoopEvent]]:
        """
        Run to completion and return final state with events.

        Args:
            prompt: User prompt
            initial_state: Optional initial state
            **state_kwargs: Additional state configuration

        Returns:
            Tuple of (final_state, events)
        """
        events: list[LoopEvent] = []

        async for event in self.run(prompt, initial_state, **state_kwargs):
            events.append(event)

        # Get final state from the loop
        state, _ = await self.loop.run_to_completion(prompt, initial_state, **state_kwargs)

        return state, events

    @property
    def events(self) -> list[LoopEvent]:
        """Get events from the last run."""
        return list(self._events)

    @property
    def final_state(self) -> AgentState | None:
        """Get final state from the last run."""
        return self._final_state


class BatchRunner(BaseModel):
    """
    Run multiple prompts through the loop.

    Supports parallel execution with concurrency limits.
    """

    loop: ReActLoop
    max_concurrency: int = Field(default=5, ge=1)

    model_config = {"arbitrary_types_allowed": True}

    async def run_batch(
        self,
        prompts: list[str],
        on_result: Callable[[str, AgentState, list[LoopEvent]], None] | None = None,
    ) -> list[tuple[str, AgentState, list[LoopEvent]]]:
        """
        Run multiple prompts in parallel.

        Args:
            prompts: List of prompts to process
            on_result: Optional callback for each result

        Returns:
            List of (prompt, final_state, events) tuples
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        results: list[tuple[str, AgentState, list[LoopEvent]]] = []

        async def run_one(prompt: str) -> tuple[str, AgentState, list[LoopEvent]]:
            async with semaphore:
                state, events = await self.loop.run_to_completion(prompt)
                if on_result:
                    on_result(prompt, state, events)
                return (prompt, state, events)

        tasks = [run_one(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)

        return list(results)


class StreamingCollector(BaseModel):
    """
    Collect events from a streaming loop run.

    Useful for capturing events while also processing them.
    """

    events: list[LoopEvent] = Field(default_factory=list)
    think_events: list[Any] = Field(default_factory=list)
    tool_events: list[Any] = Field(default_factory=list)
    reflect_events: list[Any] = Field(default_factory=list)
    terminate_event: TerminateEvent | None = None

    model_config = {"arbitrary_types_allowed": True}

    def collect(self, event: LoopEvent) -> None:
        """Add an event to the collection."""
        self.events.append(event)

        if event.event_type == "think":
            self.think_events.append(event)
        elif event.event_type in ("tool_start", "tool_complete"):
            self.tool_events.append(event)
        elif event.event_type == "reflect":
            self.reflect_events.append(event)
        elif event.event_type == "terminate":
            self.terminate_event = event

    @property
    def is_complete(self) -> bool:
        """Check if the loop has completed."""
        return self.terminate_event is not None

    @property
    def iterations(self) -> int:
        """Get number of iterations completed."""
        return self.terminate_event.iterations_used if self.terminate_event else 0

    @property
    def final_confidence(self) -> float:
        """Get final confidence score."""
        return self.terminate_event.final_confidence if self.terminate_event else 0.0

    def reset(self) -> None:
        """Reset the collector for reuse."""
        self.events = []
        self.think_events = []
        self.tool_events = []
        self.reflect_events = []
        self.terminate_event = None


def create_runner(
    model: ModelProtocol,
    registry: ToolRegistry,
    *,
    max_iterations: int = 20,
    confidence_threshold: float = 0.85,
    enable_reflection: bool = True,
    system_prompt: str | None = None,
    timeout: float | None = None,
    on_event: Callable[[LoopEvent], None] | None = None,
) -> LoopRunner:
    """
    Factory function to create a configured LoopRunner.

    Args:
        model: LLM model for reasoning
        registry: Tool registry
        max_iterations: Maximum iterations
        confidence_threshold: Confidence for completion
        enable_reflection: Enable reflection step
        system_prompt: System prompt
        timeout: Optional timeout in seconds
        on_event: Optional event callback

    Returns:
        Configured LoopRunner
    """
    config = ReActLoopConfig(
        max_iterations=max_iterations,
        confidence_threshold=confidence_threshold,
        enable_reflection=enable_reflection,
        system_prompt=system_prompt,
    )

    loop = ReActLoop(model=model, registry=registry, config=config)

    return LoopRunner(
        loop=loop,
        timeout=timeout,
        on_event=on_event,
    )
