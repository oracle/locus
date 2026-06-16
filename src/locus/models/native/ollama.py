# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Ollama model provider for local LLMs."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from locus.core.events import ModelChunkEvent
from locus.core.messages import Message, ToolCall
from locus.models.base import ModelConfig, ModelResponse


class OllamaConfig(ModelConfig):
    """Configuration for Ollama models."""

    model: str = "llama3.3"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")


class OllamaModel(BaseModel):
    """Ollama model provider for local LLMs.

    Supports any model available in Ollama (Llama, Mistral, Gemma, etc.)
    with tool calling support.

    Example:
        >>> model = OllamaModel(model="llama3.3")
        >>> response = await model.complete([Message.user("Hello!")])
    """

    config: OllamaConfig
    _client: Any = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def supports_structured_output(self) -> bool:
        """Ollama doesn't yet ship OpenAI-style ``response_format``.

        The agent loop falls back to the prompted-JSON path with
        post-hoc parsing for Ollama models.
        """
        return False

    def __init__(
        self,
        model: str = "llama3.3",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        config = OllamaConfig(
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        super().__init__(config=config)

    @property
    def client(self) -> Any:
        """Get or create the Ollama async client."""
        if self._client is None:
            try:
                import ollama

                self._client = ollama.AsyncClient(host=self.config.base_url)
            except ImportError as e:
                msg = "ollama package required. Install with: pip install ollama"
                raise ImportError(msg) from e
        return self._client

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Locus messages to Ollama format."""
        ollama_messages: list[dict[str, Any]] = []
        for msg in messages:
            m: dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content or "",
            }
            if msg.tool_calls:
                m["tool_calls"] = [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            ollama_messages.append(m)
        return ollama_messages

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Convert tools to Ollama format (OpenAI-compatible)."""
        if not tools:
            return None
        # Ollama uses OpenAI-compatible tool format
        ollama_tools = []
        for tool in tools:
            if "type" not in tool:
                ollama_tools.append({"type": "function", "function": tool})
            else:
                ollama_tools.append(tool)
        return ollama_tools

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Complete a chat request."""
        ollama_messages = self._convert_messages(messages)
        ollama_tools = self._convert_tools(tools)

        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": ollama_messages,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }
        if ollama_tools:
            params["tools"] = ollama_tools

        response = await self.client.chat(**params)

        # Parse response — ollama returns Message object, not dict
        msg = response.get("message") or response
        if hasattr(msg, "content"):
            # ollama Message object
            content = msg.content
        else:
            content = msg.get("content") if isinstance(msg, dict) else str(msg)
        tool_calls: list[ToolCall] = []

        raw_tool_calls = (
            getattr(msg, "tool_calls", None)
            or (msg.get("tool_calls") if isinstance(msg, dict) else None)
            or []
        )
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(
                ToolCall(
                    id=f"call_{func.get('name', 'unknown')}",
                    name=func.get("name", "unknown"),
                    arguments=args if isinstance(args, dict) else {},
                )
            )

        usage = {}
        prompt_tokens = (
            getattr(response, "prompt_eval_count", None) or response.get("prompt_eval_count")
            if isinstance(response, dict)
            else None
        )
        if prompt_tokens:
            eval_count = getattr(response, "eval_count", None) or (
                response.get("eval_count") if isinstance(response, dict) else 0
            )
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": eval_count or 0}

        done = getattr(response, "done", None) or (
            response.get("done") if isinstance(response, dict) else None
        )

        return ModelResponse(
            message=Message.assistant(content=content, tool_calls=tool_calls),
            usage=usage,
            stop_reason="stop" if done else None,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        """Stream a chat response."""
        ollama_messages = self._convert_messages(messages)

        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": ollama_messages,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
            },
        }

        response = await self.client.chat(**params, stream=True)

        async for chunk in response:
            msg = chunk.get("message", {})
            content = msg.get("content", "")
            if content:
                yield ModelChunkEvent(content=content)
            if chunk.get("done"):
                yield ModelChunkEvent(done=True)
                return

        yield ModelChunkEvent(done=True)
