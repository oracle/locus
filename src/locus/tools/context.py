# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Tool execution context - 100% Pydantic."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolContext(BaseModel):
    """
    Context passed to tools during execution.

    Provides access to agent state, metadata, and utilities.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identifiers
    tool_call_id: str = Field(..., description="Unique ID of this tool call")
    tool_name: str = Field(..., description="Name of the tool being called")

    # Agent context
    agent_id: str | None = Field(default=None, description="ID of the calling agent")
    run_id: str = Field(..., description="ID of the current agent run")
    iteration: int = Field(..., description="Current iteration number")

    # State access (read-only view)
    state: Any = Field(default=None, description="Current agent state")

    # User-provided metadata
    invocation_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata passed at invocation time",
    )

    # Tool-specific config
    tool_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific configuration",
    )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value."""
        return self.invocation_metadata.get(key, default)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a tool config value."""
        return self.tool_config.get(key, default)

    @property
    def messages(self) -> list[Any]:
        """Get conversation messages (if state available)."""
        if self.state is None:
            return []
        return list(self.state.messages)

    @property
    def confidence(self) -> float:
        """Get current confidence score (if state available)."""
        if self.state is None:
            return 0.0
        # ``self.state`` is typed as ``Any`` upstream — narrow the return
        # to ``float`` to satisfy strict mypy.
        return float(self.state.confidence)
