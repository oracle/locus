# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Protocol definitions for Locus - dependency injection contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from locus.core.events import ModelChunkEvent
    from locus.core.messages import Message, ToolCall
    from locus.core.state import AgentState


# =============================================================================
# Model Protocol
# =============================================================================


class ModelResponse:
    """Response from a model completion."""

    def __init__(
        self,
        message: Message,
        usage: dict[str, int] | None = None,
        stop_reason: str | None = None,
    ):
        self.message = message
        self.usage = usage or {}
        self.stop_reason = stop_reason


@runtime_checkable
class ModelProtocol(Protocol):
    """
    Protocol for LLM providers.

    Implementations must support both completion and streaming.
    """

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Complete a chat request.

        Args:
            messages: Conversation history
            tools: Tool schemas in OpenAI format
            **kwargs: Provider-specific options

        Returns:
            Model response with message and metadata
        """
        ...

    # Declared as ``def`` (not ``async def``) so the Protocol matches
    # concrete async-generator implementations: an ``async def`` body
    # containing ``yield`` returns ``AsyncIterator[X]`` directly, not
    # ``Coroutine[..., AsyncIterator[X]]``.
    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        """
        Stream a chat response.

        Args:
            messages: Conversation history
            tools: Tool schemas in OpenAI format
            **kwargs: Provider-specific options

        Yields:
            Streaming chunks with content and/or tool calls
        """
        ...


# =============================================================================
# Tool Protocol
# =============================================================================


@runtime_checkable
class ToolProtocol(Protocol):
    """
    Protocol for tools that can be called by agents.

    Tools must have a name, description, parameters schema, and be callable.
    """

    @property
    def name(self) -> str:
        """Unique name of the tool."""
        ...

    @property
    def description(self) -> str:
        """Description of what the tool does."""
        ...

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        ...

    async def execute(self, **kwargs: Any) -> Any:
        """
        Execute the tool with given arguments.

        Args:
            **kwargs: Tool arguments matching the parameters schema

        Returns:
            Tool result (will be converted to string for LLM)
        """
        ...


# =============================================================================
# Checkpointer Protocol
# =============================================================================


@dataclass(frozen=True)
class CheckpointerCapabilities:
    """
    Capabilities supported by a checkpointer.

    Use this to discover what features a checkpointer supports before
    calling optional methods.

    Example:
        >>> if checkpointer.capabilities.search:
        ...     results = await checkpointer.search("error handling")
    """

    # Extended capabilities (vary by backend)
    search: bool = False  # Full-text search across checkpoints
    metadata_query: bool = False  # Query checkpoints by metadata fields
    vacuum: bool = False  # Cleanup old checkpoints
    branching: bool = False  # Copy/fork threads
    ttl: bool = False  # Time-to-live / auto-expiration
    list_threads: bool = False  # List all thread IDs
    list_with_metadata: bool = False  # List checkpoints with metadata
    persistent_checkpoint_ids: bool = False  # Checkpoint IDs persist across restarts


@runtime_checkable
class CheckpointerProtocol(Protocol):
    """
    Protocol for state persistence.

    Implementations handle saving and loading agent state.
    Extended methods are optional based on capabilities.
    """

    @property
    def capabilities(self) -> CheckpointerCapabilities:
        """Return the capabilities of this checkpointer."""
        ...

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save agent state.

        Args:
            state: Current agent state
            thread_id: Unique identifier for the conversation thread
            checkpoint_id: Optional specific checkpoint ID
            metadata: Optional metadata for querying/filtering

        Returns:
            Checkpoint ID that can be used to restore
        """
        ...

    async def load(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> AgentState | None:
        """
        Load agent state.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Optional specific checkpoint (latest if None)

        Returns:
            Restored state or None if not found
        """
        ...

    async def list_checkpoints(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[str]:
        """
        List available checkpoints for a thread.

        Args:
            thread_id: Thread identifier
            limit: Maximum number to return

        Returns:
            List of checkpoint IDs, newest first
        """
        ...


# =============================================================================
# Hook Protocol
# =============================================================================


@runtime_checkable
class HookProtocol(Protocol):
    """
    Protocol for lifecycle hooks.

    Hooks can observe and modify agent behavior at specific points.
    """

    @property
    def priority(self) -> int:
        """
        Hook priority (lower = earlier).

        Standard priorities:
        - 0-99: Security hooks
        - 100-199: Observability hooks
        - 200-299: Business logic hooks
        - 300+: Default hooks
        """
        ...

    async def on_before_invocation(
        self,
        prompt: str,
        state: AgentState,
    ) -> AgentState:
        """Called before agent starts processing."""
        ...

    async def on_after_invocation(
        self,
        state: AgentState,
        success: bool,
    ) -> None:
        """Called after agent completes processing."""
        ...

    async def on_before_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Called before tool execution.

        Returns potentially modified arguments.
        """
        ...

    async def on_after_tool_call(
        self,
        tool_name: str,
        result: Any,
        error: str | None,
    ) -> None:
        """Called after tool execution."""
        ...


# =============================================================================
# Executor Protocol
# =============================================================================


@runtime_checkable
class ExecutorProtocol(Protocol):
    """
    Protocol for tool execution strategies.

    Implementations can execute tools sequentially, concurrently, etc.
    """

    async def execute(
        self,
        tool_calls: list[ToolCall],
        tool_registry: dict[str, ToolProtocol],
    ) -> list[tuple[str, Any | None, str | None]]:
        """
        Execute a batch of tool calls.

        Args:
            tool_calls: Tool calls to execute
            tool_registry: Available tools by name

        Returns:
            List of (tool_call_id, result, error) tuples
        """
        ...


# =============================================================================
# Conversation Manager Protocol
# =============================================================================


@runtime_checkable
class ConversationManagerProtocol(Protocol):
    """
    Protocol for managing conversation history.

    Implementations handle message trimming, summarization, etc.
    """

    def apply(self, messages: list[Message]) -> list[Message]:
        """
        Apply conversation management to messages.

        Args:
            messages: Full message history

        Returns:
            Managed message list (potentially trimmed/summarized)
        """
        ...
