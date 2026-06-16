# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Console streaming handler with rich text output."""

from __future__ import annotations

import sys
from typing import IO

from locus.core.events import (
    CausalEdgeEvent,
    CausalNodeEvent,
    GroundingEvent,
    LocusEvent,
    ModelChunkEvent,
    ModelCompleteEvent,
    OrchestratorDecisionEvent,
    ReflectEvent,
    SpecialistCompleteEvent,
    SpecialistStartEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from locus.streaming.handler import BaseStreamHandler


class ConsoleHandler(BaseStreamHandler):
    """Stream handler that outputs to console with rich formatting.

    Provides visual feedback during agent execution including:
    - Progress indicators
    - Tool call visualization
    - Reasoning display
    - Color-coded output (when terminal supports it)

    Example:
        >>> handler = ConsoleHandler(show_reasoning=True)
        >>> await handler.on_event(think_event)
    """

    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
    }

    # Event type symbols
    SYMBOLS = {
        "think": "💭",
        "tool_start": "🔧",
        "tool_complete": "✓",
        "tool_error": "✗",
        "reflect": "🔍",
        "grounding": "📍",
        "terminate": "🏁",
        "specialist_start": "👤",
        "specialist_complete": "👤",
        "orchestrator": "🎯",
        "causal_node": "📊",
        "causal_edge": "→",
        "model_chunk": "·",
        "error": "❌",
        "warning": "⚠️",
        "info": "i",
    }

    def __init__(
        self,
        output: IO[str] | None = None,
        show_reasoning: bool = True,
        show_tool_args: bool = False,
        show_tool_results: bool = True,
        show_timestamps: bool = False,
        show_progress: bool = True,
        use_color: bool = True,
        use_emoji: bool = True,
        max_result_length: int = 500,
        indent: str = "  ",
    ):
        """Initialize the console handler.

        Args:
            output: Output stream (defaults to sys.stdout)
            show_reasoning: Whether to show agent reasoning
            show_tool_args: Whether to show tool arguments
            show_tool_results: Whether to show tool results
            show_timestamps: Whether to show timestamps
            show_progress: Whether to show progress indicators
            use_color: Whether to use ANSI colors
            use_emoji: Whether to use emoji symbols
            max_result_length: Maximum length for tool results
            indent: Indentation string
        """
        self.output = output or sys.stdout
        self.show_reasoning = show_reasoning
        self.show_tool_args = show_tool_args
        self.show_tool_results = show_tool_results
        self.show_timestamps = show_timestamps
        self.show_progress = show_progress
        self.use_color = use_color and self._supports_color()
        self.use_emoji = use_emoji
        self.max_result_length = max_result_length
        self.indent = indent

        self._iteration = 0
        self._tool_count = 0
        self._active_tools: dict[str, str] = {}

    def _supports_color(self) -> bool:
        """Check if terminal supports color."""
        if hasattr(self.output, "isatty"):
            return self.output.isatty()
        return False

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if enabled."""
        if not self.use_color:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def _symbol(self, name: str) -> str:
        """Get symbol for event type."""
        if not self.use_emoji:
            return ""
        return self.SYMBOLS.get(name, "")

    def _write(self, text: str, newline: bool = True) -> None:
        """Write text to output."""
        self.output.write(text)
        if newline:
            self.output.write("\n")
        self.output.flush()

    def _format_timestamp(self, event: LocusEvent) -> str:
        """Format event timestamp."""
        if not self.show_timestamps:
            return ""
        return f"[{event.timestamp.strftime('%H:%M:%S.%f')[:-3]}] "

    def _truncate(self, text: str, max_length: int | None = None) -> str:
        """Truncate text to max length."""
        max_len = max_length or self.max_result_length
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    async def on_event(self, event: LocusEvent) -> None:
        """Handle a streaming event.

        Args:
            event: The event to process
        """
        handler = getattr(self, f"_handle_{event.event_type}", None)
        if handler:
            handler(event)
        else:
            self._handle_unknown(event)

    async def on_complete(self) -> None:
        """Called when streaming is complete."""
        self._write("")  # Empty line
        self._write(
            f"{self._symbol('terminate')} "
            f"{self._color('Execution complete', 'green')} "
            f"({self._tool_count} tool calls)"
        )

    async def on_error(self, error: Exception) -> None:
        """Handle a streaming error.

        Args:
            error: The error that occurred
        """
        self._write(f"{self._symbol('error')} {self._color('Error:', 'red')} {error!s}")

    def _handle_think(self, event: ThinkEvent) -> None:
        """Handle think event."""
        self._iteration = event.iteration

        if self.show_progress:
            self._write("")
            self._write(
                f"{self._symbol('think')} {self._color(f'Iteration {event.iteration}', 'cyan')}"
            )

        if self.show_reasoning and event.reasoning:
            for line in event.reasoning.split("\n"):
                self._write(f"{self.indent}{self._color(line, 'dim')}")

        if event.tool_calls:
            count = len(event.tool_calls)
            self._write(f"{self.indent}Planning {count} tool call{'s' if count > 1 else ''}")

    def _handle_tool_start(self, event: ToolStartEvent) -> None:
        """Handle tool start event."""
        self._active_tools[event.tool_call_id] = event.tool_name
        self._tool_count += 1

        prefix = f"{self._format_timestamp(event)}{self._symbol('tool_start')}"
        self._write(
            f"{prefix} {self._color(event.tool_name, 'yellow')}",
            newline=not self.show_tool_args,
        )

        if self.show_tool_args and event.arguments:
            args_str = ", ".join(f"{k}={v!r}" for k, v in event.arguments.items())
            self._write(f"({self._truncate(args_str, 200)})")

    def _handle_tool_complete(self, event: ToolCompleteEvent) -> None:
        """Handle tool complete event."""
        self._active_tools.pop(event.tool_call_id, None)

        if event.error:
            prefix = f"{self._format_timestamp(event)}{self._symbol('tool_error')}"
            self._write(
                f"{prefix} {self._color(event.tool_name, 'red')}: {self._color(event.error, 'red')}"
            )
        else:
            prefix = f"{self._format_timestamp(event)}{self._symbol('tool_complete')}"
            duration = f" ({event.duration_ms:.0f}ms)" if event.duration_ms else ""
            self._write(f"{prefix} {self._color(event.tool_name, 'green')}{duration}")

            if self.show_tool_results and event.result:
                result = self._truncate(event.result)
                for line in result.split("\n")[:5]:  # Max 5 lines
                    self._write(f"{self.indent}{self._color(line, 'dim')}")

    def _handle_reflect(self, event: ReflectEvent) -> None:
        """Handle reflect event."""
        color = "green" if event.assessment == "on_track" else "yellow"
        if event.assessment in ("stuck", "loop_detected"):
            color = "red"

        self._write(
            f"{self._symbol('reflect')} "
            f"{self._color(f'Reflection: {event.assessment}', color)} "
            f"(confidence: {event.new_confidence:.2f})"
        )

        if event.guidance:
            self._write(f"{self.indent}{event.guidance}")

    def _handle_grounding(self, event: GroundingEvent) -> None:
        """Handle grounding event."""
        color = "green" if event.score >= 0.8 else ("yellow" if event.score >= 0.5 else "red")

        self._write(
            f"{self._symbol('grounding')} "
            f"{self._color(f'Grounding score: {event.score:.2f}', color)} "
            f"({event.claims_evaluated} claims)"
        )

        if event.ungrounded_claims:
            for claim in event.ungrounded_claims[:3]:
                self._write(f"{self.indent}{self._color(f'Ungrounded: {claim}', 'yellow')}")

    def _handle_terminate(self, event: TerminateEvent) -> None:
        """Handle terminate event."""
        color = "green" if event.reason == "complete" else "yellow"
        if event.reason == "error":
            color = "red"

        self._write("")
        self._write(
            f"{self._symbol('terminate')} {self._color(f'Terminated: {event.reason}', color)}"
        )
        self._write(
            f"{self.indent}Iterations: {event.iterations_used}, "
            f"Tool calls: {event.total_tool_calls}, "
            f"Confidence: {event.final_confidence:.2f}"
        )

    def _handle_specialist_start(self, event: SpecialistStartEvent) -> None:
        """Handle specialist start event."""
        self._write(
            f"{self._symbol('specialist_start')} "
            f"Starting specialist: {self._color(event.specialist_type, 'magenta')}"
        )
        self._write(f"{self.indent}Task: {self._truncate(event.task, 100)}")

    def _handle_specialist_complete(self, event: SpecialistCompleteEvent) -> None:
        """Handle specialist complete event."""
        self._write(
            f"{self._symbol('specialist_complete')} "
            f"Specialist {self._color(event.specialist_type, 'magenta')} complete "
            f"(confidence: {event.confidence:.2f}, {event.duration_ms:.0f}ms)"
        )

    def _handle_orchestrator_decision(self, event: OrchestratorDecisionEvent) -> None:
        """Handle orchestrator decision event."""
        self._write(
            f"{self._symbol('orchestrator')} Orchestrator: {self._color(event.decision, 'cyan')}"
        )
        if event.specialists_selected:
            self._write(f"{self.indent}Specialists: {', '.join(event.specialists_selected)}")

    def _handle_model_chunk(self, event: ModelChunkEvent) -> None:
        """Handle model chunk event (streaming)."""
        if event.content:
            self._write(event.content, newline=False)
        if event.done:
            self._write("")  # Newline at end

    def _handle_model_complete(self, event: ModelCompleteEvent) -> None:
        """Handle model complete event."""
        # Usually don't need to display this

    def _handle_causal_node(self, event: CausalNodeEvent) -> None:
        """Handle causal node event."""
        color = "red" if event.node_type == "root_cause" else "yellow"
        self._write(
            f"{self._symbol('causal_node')} {self._color(event.label, color)} ({event.node_type})"
        )

    def _handle_causal_edge(self, event: CausalEdgeEvent) -> None:
        """Handle causal edge event."""
        self._write(
            f"{self.indent}{self._symbol('causal_edge')} "
            f"{event.source_id} {event.relationship} {event.target_id} "
            f"(confidence: {event.confidence:.2f})"
        )

    def _handle_unknown(self, event: LocusEvent) -> None:
        """Handle unknown event type."""
        self._write(f"{self._symbol('info')} Event: {event.event_type}")


class MinimalConsoleHandler(BaseStreamHandler):
    """Minimal console handler showing only essential output.

    Shows tool calls and final result, hiding reasoning and details.
    """

    def __init__(self, output: IO[str] | None = None):
        """Initialize minimal handler.

        Args:
            output: Output stream (defaults to sys.stdout)
        """
        self.output = output or sys.stdout
        self._result: str | None = None

    async def on_event(self, event: LocusEvent) -> None:
        """Handle events minimally."""
        if isinstance(event, ToolStartEvent):
            self.output.write(f"• {event.tool_name}\n")
            self.output.flush()
        elif isinstance(event, ToolCompleteEvent) and event.error:
            self.output.write(f"  Error: {event.error}\n")
            self.output.flush()
        elif isinstance(event, TerminateEvent):
            self.output.write(f"\nCompleted in {event.iterations_used} iterations\n")
            self.output.flush()

    async def on_complete(self) -> None:
        """Handle completion."""

    async def on_error(self, error: Exception) -> None:
        """Handle error."""
        self.output.write(f"Error: {error}\n")
        self.output.flush()
