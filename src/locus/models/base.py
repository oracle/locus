# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Base model types - 100% Pydantic."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from locus.core.messages import Message


if TYPE_CHECKING:
    from locus.core.events import ModelChunkEvent


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol defining the model interface."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Complete a chat request."""
        ...

    # NOTE: declared as ``def`` (not ``async def``) so the Protocol
    # matches concrete async-generator implementations. An ``async def``
    # function whose body contains ``yield`` returns ``AsyncIterator[X]``
    # directly, not a coroutine — mypy types the Protocol that way only
    # when the declaration is plain ``def`` returning ``AsyncIterator``.
    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        """Stream a chat response."""
        ...


@runtime_checkable
class RequestBuilder(Protocol):
    """Protocol for building provider-specific requests."""

    def build(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        **kwargs: Any,
    ) -> Any:
        """Build a provider-specific request."""
        ...


@runtime_checkable
class ResponseParser(Protocol):
    """Protocol for parsing provider-specific responses."""

    def parse(self, response: Any) -> ModelResponse:
        """Parse a provider-specific response."""
        ...


class ModelResponse(BaseModel):
    """Response from a model completion."""

    message: Message
    usage: dict[str, int] = Field(default_factory=dict)
    stop_reason: str | None = None
    # Opaque per-provider continuation state. Stateless transports
    # (the default — chat/completions-style) leave this as ``None``.
    # Server-stateful transports (e.g. ``OCIResponsesModel``) return
    # a continuation token here (e.g. ``{"previous_response_id":
    # "resp_abc"}``) so the agent can thread it into the next
    # ``complete()`` call without resending the full history.
    provider_state: dict[str, Any] | None = None

    @property
    def content(self) -> str | None:
        """Get response content."""
        return self.message.content

    @property
    def tool_calls(self) -> list[Any]:
        """Get tool calls."""
        return self.message.tool_calls

    @property
    def prompt_tokens(self) -> int:
        """Get prompt token count."""
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        """Get completion token count."""
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.prompt_tokens + self.completion_tokens


class ModelConfig(BaseModel):
    """Base configuration for models."""

    model: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}
