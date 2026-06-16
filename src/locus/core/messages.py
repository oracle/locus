# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Message types for Locus - 100% Pydantic."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class Role(StrEnum):
    """Message role in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A tool call requested by the model."""

    id: str = Field(default_factory=lambda: f"call_{uuid4().hex[:12]}")
    name: str = Field(..., description="Name of the tool to call")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_args_alias(cls, data: Any) -> Any:
        """Accept ``args`` as an alias for ``arguments`` (parallel-batch compat)."""
        if isinstance(data, dict) and "args" in data and "arguments" not in data:
            data = dict(data)
            data["arguments"] = data.pop("args")
        return data

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


class ToolResult(BaseModel):
    """Result from a tool execution."""

    tool_call_id: str = Field(..., description="ID of the tool call this responds to")
    name: str = Field(..., description="Name of the tool")
    content: str = Field(..., description="String result from the tool")
    error: str | None = Field(default=None, description="Error message if tool failed")
    duration_ms: float | None = Field(default=None, description="Execution time in milliseconds")

    @property
    def success(self) -> bool:
        """Whether the tool execution succeeded."""
        return self.error is None


class Message(BaseModel):
    """A message in the conversation."""

    role: Role = Field(..., description="Role of the message sender")
    content: str | None = Field(default=None, description="Text content of the message")
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool calls requested by assistant",
    )
    tool_call_id: str | None = Field(
        default=None,
        description="For tool messages, the ID of the call being responded to",
    )
    name: str | None = Field(
        default=None,
        description="For tool messages, the name of the tool",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Opaque out-of-band annotations — e.g. prompt-cache "
            "breakpoints (``cache_control``), provider-specific hints, "
            "or user-authored tags. Adapters ignore unrecognised keys "
            "and must not transmit the dict in API payloads without "
            "explicit handling for each known key."
        ),
    )

    model_config = {"frozen": True}

    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """Create a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls,
        content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> Message:
        """Create an assistant message."""
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls or [],
        )

    @classmethod
    def tool(cls, result: ToolResult) -> Message:
        """Create a tool result message."""
        return cls(
            role=Role.TOOL,
            content=result.content,
            tool_call_id=result.tool_call_id,
            name=result.name,
        )

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        msg: dict[str, Any] = {"role": self.role.value}

        if self.content is not None:
            msg["content"] = self.content

        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai_format() for tc in self.tool_calls]

        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id

        if self.name:
            msg["name"] = self.name

        return msg
