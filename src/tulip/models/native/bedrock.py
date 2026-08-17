# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Amazon Bedrock, through the Converse API.

Bedrock was the largest hole in the provider table. Sixteen prefixes reach
endpoints that speak the OpenAI wire protocol, and Bedrock speaks its own —
so anyone standardised on AWS could not address a model by name at all, and
the workaround (stand up a LiteLLM gateway in front of Bedrock) means running
a proxy to talk to a service you already have credentials for.

**One API, every model.** Bedrock's older ``invoke_model`` takes a different
request body per vendor: Claude wants ``anthropic_version`` and a messages
array, Titan wants ``inputText``, Llama wants ``prompt``. ``converse`` is the
unified surface AWS added precisely to end that, and it covers tool use and
streaming for every model that supports them. This provider uses ``converse``
only, so ``bedrock:us.amazon.nova-micro-v1:0`` and
``bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0`` run the same code.

**Credentials are boto3's problem, not ours.** No key argument is invented:
the standard chain applies — environment, shared credentials file, profile,
SSO, instance role, IRSA in EKS. Pass ``profile=`` to pick one explicitly.
That is what an AWS shop already has configured, and re-implementing it would
only be a way to get it wrong.

    from tulip import Agent, AgentConfig

    agent = Agent(config=AgentConfig(model="bedrock:us.amazon.nova-micro-v1:0"))
    agent = Agent(config=AgentConfig(model="bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0"))

``boto3`` is an optional dependency (``pip install "tulip-agents[bedrock]"``)
and is imported lazily, so the four-package core install is unchanged for
everyone not on AWS.
"""

from __future__ import annotations

import asyncio
import json as _json
import queue
from collections.abc import AsyncIterator
from typing import Any

from pydantic import Field

from tulip.core.events import ModelChunkEvent
from tulip.core.messages import Message, Role, ToolCall
from tulip.models.base import BaseModel, ModelConfig, ModelResponse


#: Bedrock's own stop reasons. ``guardrail_intervened`` and ``content_filtered``
#: have no counterpart in the other providers and are passed through unchanged
#: rather than flattened into ``stop``: a run that was cut short by an AWS
#: guardrail is not the same event as a model deciding it was finished, and a
#: caller auditing why an action did not happen needs to tell them apart.
_STOP_REASONS = frozenset(
    {
        "end_turn",
        "tool_use",
        "max_tokens",
        "stop_sequence",
        "guardrail_intervened",
        "content_filtered",
    }
)

#: How many stream events may sit between the boto3 reader thread and the
#: consumer. Bounded so a slow consumer applies backpressure instead of letting
#: the queue grow without limit on a long generation.
_STREAM_QUEUE_MAX = 64

_DONE = object()


def _require_boto3() -> Any:
    """Import boto3, or explain exactly how to get it."""
    try:
        import boto3  # noqa: PLC0415 — optional dependency, imported on use
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "Amazon Bedrock needs boto3. Install it with:\n"
            '    pip install "tulip-agents[bedrock]"\n'
            "Credentials come from the standard AWS chain (environment, shared "
            "credentials file, profile, SSO, instance role)."
        ) from exc
    return boto3


class BedrockConfig(ModelConfig):
    """Configuration for Bedrock models."""

    model: str = "us.amazon.nova-lite-v1:0"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    region: str | None = Field(
        default=None,
        description="AWS region. Falls back to the boto3 session's region.",
    )
    profile: str | None = Field(
        default=None,
        description="Named AWS profile. Falls back to the default credential chain.",
    )
    aws_access_key_id: str | None = Field(default=None, description="Explicit access key.")
    aws_secret_access_key: str | None = Field(default=None, description="Explicit secret key.")
    aws_session_token: str | None = Field(default=None, description="Explicit session token.")
    endpoint_url: str | None = Field(
        default=None,
        description="Override the Bedrock endpoint — VPC endpoints, or a local mock.",
    )
    guardrail_id: str | None = Field(default=None, description="Bedrock guardrail identifier.")
    guardrail_version: str | None = Field(default=None, description="Bedrock guardrail version.")
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Retry budget for throttling and transient 5xx, via botocore's adaptive mode.",
    )
    request_timeout: float = Field(default=120.0, gt=0, description="Per-request timeout, seconds.")


class BedrockModel(BaseModel):
    """Amazon Bedrock provider, via the Converse API.

    Example:
        >>> model = BedrockModel(model="us.amazon.nova-micro-v1:0")
        >>> response = await model.complete([Message.user("Hello!")])
    """

    config: BedrockConfig
    _client: Any = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def supports_structured_output(self) -> bool:
        """Converse has no ``response_format``.

        It could be emulated by forcing a single tool via ``toolChoice``, the
        way the Anthropic provider does — but ``toolChoice`` support varies by
        model on Bedrock, and a structured-output path that works on Claude and
        silently fails on Llama is worse than not claiming it. The agent loop
        falls back to prompted JSON, which every Bedrock model handles.
        """
        return False

    def __init__(
        self,
        model: str = "us.amazon.nova-lite-v1:0",
        region: str | None = None,
        profile: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        config = BedrockConfig(
            model=model,
            region=region,
            profile=profile,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        super().__init__(config=config)

    @property
    def client(self) -> Any:
        """The ``bedrock-runtime`` client, built once and reused.

        Deliberately *not* wrapped in :func:`tulip.core.loop_bound`. That helper
        exists for clients whose connection pool binds to the event loop that
        created it — httpx, redis.asyncio, asyncpg. A botocore client is
        synchronous and holds no loop reference, so it survives being used from
        whatever thread :func:`asyncio.to_thread` happens to hand it to, and
        rebuilding it per loop would only pay the (slow) client construction
        cost again.
        """
        if self._client is not None:
            return self._client

        boto3 = _require_boto3()
        from botocore.config import Config  # noqa: PLC0415 — comes with boto3

        session_kwargs: dict[str, Any] = {}
        if self.config.profile:
            session_kwargs["profile_name"] = self.config.profile
        for key in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"):
            if value := getattr(self.config, key, None):
                session_kwargs[key] = value

        session = boto3.Session(**session_kwargs)
        client_kwargs: dict[str, Any] = {
            "config": Config(
                # Adaptive mode backs off on throttling, which is the failure
                # every Bedrock account meets first — on-demand quotas are per
                # model per region and easy to exceed with a parallel agent.
                retries={"max_attempts": self.config.max_retries, "mode": "adaptive"},
                read_timeout=self.config.request_timeout,
                connect_timeout=min(10.0, self.config.request_timeout),
            )
        }
        if self.config.region:
            client_kwargs["region_name"] = self.config.region
        if self.config.endpoint_url:
            client_kwargs["endpoint_url"] = self.config.endpoint_url

        self._client = session.client("bedrock-runtime", **client_kwargs)
        return self._client

    # ---------------------------------------------------------------- convert

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Convert Tulip messages to Converse format.

        Returns ``(system_blocks, messages)`` — Converse takes the system
        prompt as a separate top-level argument, not as a message.
        """
        system: list[dict[str, Any]] = []
        converted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                if msg.content:
                    system.append({"text": msg.content})
                continue

            if msg.role == Role.ASSISTANT:
                content: list[dict[str, Any]] = []
                if msg.tool_calls:
                    # Text and toolUse must not share an assistant turn. Nova
                    # and Claude accept the mix; Llama and Mistral reject the
                    # request outright:
                    #
                    #   ValidationException: messages.N.content: Conversation
                    #   blocks and tool use blocks cannot be provided in the
                    #   same turn.
                    #
                    # So the tool calls win and the preamble is dropped from
                    # the history. What is lost is the model's own "let me look
                    # that up" — never load-bearing for the next turn, because
                    # the tool result that follows carries the actual content.
                    # The alternative is a provider that works on Amazon models
                    # and 400s on Meta ones, which is the worse trade.
                    content.extend(
                        {"toolUse": {"toolUseId": tc.id, "name": tc.name, "input": tc.arguments}}
                        for tc in msg.tool_calls
                    )
                elif msg.content:
                    content.append({"text": msg.content})
                # Converse rejects an empty content list outright, where the
                # OpenAI wire format tolerates it. An assistant turn with
                # neither text nor a tool call is not worth a 400.
                if content:
                    converted.append({"role": "assistant", "content": content})

            elif msg.role == Role.TOOL:
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": msg.tool_call_id or "",
                                    "content": [{"text": str(msg.content or "")}],
                                }
                            }
                        ],
                    }
                )

            elif msg.role == Role.USER:
                converted.append({"role": "user", "content": [{"text": msg.content or ""}]})

        return system, converted

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> dict[str, Any] | None:
        """Convert OpenAI-format tool schemas to a Converse ``toolConfig``."""
        if not tools:
            return None
        specs = []
        for tool in tools:
            func = tool.get("function", tool)
            specs.append(
                {
                    "toolSpec": {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "inputSchema": {
                            "json": func.get("parameters", {"type": "object", "properties": {}})
                        },
                    }
                }
            )
        return {"tools": specs}

    def _params(
        self, messages: list[Message], tools: list[dict[str, Any]] | None, **kwargs: Any
    ) -> dict[str, Any]:
        """Assemble the Converse request."""
        system, converted = self._convert_messages(messages)
        params: dict[str, Any] = {
            "modelId": self.config.model,
            "messages": converted,
            "inferenceConfig": {
                "maxTokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "topP": kwargs.get("top_p", self.config.top_p),
            },
        }
        if system:
            params["system"] = system
        if tool_config := self._convert_tools(tools):
            params["toolConfig"] = tool_config
        if self.config.guardrail_id:
            params["guardrailConfig"] = {
                "guardrailIdentifier": self.config.guardrail_id,
                "guardrailVersion": self.config.guardrail_version or "DRAFT",
            }
        return params

    @staticmethod
    def _usage(response: dict[str, Any]) -> dict[str, int]:
        raw = response.get("usage") or {}
        usage: dict[str, int] = {}
        if "inputTokens" in raw:
            usage["prompt_tokens"] = raw["inputTokens"]
        if "outputTokens" in raw:
            usage["completion_tokens"] = raw["outputTokens"]
        # Present only when the model and region support prompt caching.
        for wire, ours in (
            ("cacheReadInputTokens", "cache_read_input_tokens"),
            ("cacheWriteInputTokens", "cache_creation_input_tokens"),
        ):
            if wire in raw:
                usage[ours] = raw[wire]
        return usage

    # --------------------------------------------------------------- complete

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Complete a chat request through ``converse``.

        botocore is synchronous, so the call runs on a worker thread. Without
        that, every Bedrock request would block the event loop for its whole
        duration — which for an agent means no concurrent tool execution, no
        streaming consumer, and no timeout firing.
        """
        params = self._params(messages, tools, **kwargs)
        response = await asyncio.to_thread(lambda: self.client.converse(**params))

        content: str | None = None
        tool_calls: list[ToolCall] = []
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                content = (content or "") + block["text"]
            elif "toolUse" in block:
                use = block["toolUse"]
                raw_input = use.get("input")
                tool_calls.append(
                    ToolCall(
                        id=use.get("toolUseId", ""),
                        name=use.get("name", ""),
                        arguments=raw_input if isinstance(raw_input, dict) else {},
                    )
                )

        return ModelResponse(
            message=Message.assistant(content=content, tool_calls=tool_calls),
            usage=self._usage(response),
            stop_reason=response.get("stopReason"),
        )

    # ----------------------------------------------------------------- stream

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        """Stream a response through ``converse_stream``.

        botocore's event stream is a *synchronous* iterator, so it is drained on
        a worker thread into a bounded thread-safe queue and read back with
        :func:`asyncio.to_thread`. An ``asyncio.Queue`` would be the wrong tool:
        its ``put_nowait`` is not safe to call from another thread, and the
        resulting corruption shows up as dropped or duplicated chunks under
        load rather than as an error.
        """
        params = self._params(messages, tools, **kwargs)
        events: queue.Queue[Any] = queue.Queue(maxsize=_STREAM_QUEUE_MAX)

        def pump() -> None:
            try:
                for event in self.client.converse_stream(**params)["stream"]:
                    events.put(event)
            except BaseException as exc:  # noqa: BLE001 — re-raised on the consumer side
                events.put(exc)
            finally:
                events.put(_DONE)

        pump_task = asyncio.create_task(asyncio.to_thread(pump))
        # Tool arguments arrive as partial JSON strings across many deltas,
        # keyed by content-block index; they are only parseable once the block
        # closes.
        partial_tools: dict[int, dict[str, Any]] = {}
        usage: dict[str, int] = {}
        stop_reason: str | None = None

        try:
            while True:
                item = await asyncio.to_thread(events.get)
                if item is _DONE:
                    break
                if isinstance(item, BaseException):
                    raise item

                if "contentBlockStart" in item:
                    start = item["contentBlockStart"].get("start", {})
                    if "toolUse" in start:
                        index = item["contentBlockStart"].get("contentBlockIndex", 0)
                        partial_tools[index] = {
                            "id": start["toolUse"].get("toolUseId", ""),
                            "name": start["toolUse"].get("name", ""),
                            "buffer": "",
                        }

                elif "contentBlockDelta" in item:
                    delta = item["contentBlockDelta"].get("delta", {})
                    index = item["contentBlockDelta"].get("contentBlockIndex", 0)
                    if "text" in delta:
                        yield ModelChunkEvent(content=delta["text"])
                    elif "toolUse" in delta and index in partial_tools:
                        partial_tools[index]["buffer"] += delta["toolUse"].get("input", "")

                elif "contentBlockStop" in item:
                    index = item["contentBlockStop"].get("contentBlockIndex", 0)
                    if pending := partial_tools.pop(index, None):
                        yield ModelChunkEvent(
                            tool_calls=[
                                ToolCall(
                                    id=pending["id"],
                                    name=pending["name"],
                                    arguments=_loads_or_empty(pending["buffer"]),
                                )
                            ]
                        )

                elif "messageStop" in item:
                    stop_reason = item["messageStop"].get("stopReason")

                elif "metadata" in item:
                    usage = self._usage(item["metadata"])

            yield ModelChunkEvent(done=True, usage=usage or None, stop_reason=stop_reason)
        finally:
            # A consumer that breaks out early leaves the reader thread parked
            # on a full queue. Drain what is buffered so it can finish and the
            # HTTP connection is released rather than held to timeout.
            if not pump_task.done():
                while True:
                    try:
                        if events.get_nowait() is _DONE:
                            break
                    except queue.Empty:  # noqa: PERF203 — drain loop, not hot path
                        break
            await asyncio.gather(pump_task, return_exceptions=True)

    async def close(self) -> None:
        """Release the underlying client."""
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if close is not None:
                await asyncio.to_thread(close)
            self._client = None

    async def __aenter__(self) -> BedrockModel:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


def _loads_or_empty(buffer: str) -> dict[str, Any]:
    """Parse accumulated tool-argument JSON, tolerating an empty buffer.

    A tool with no arguments streams zero input deltas, so the buffer is ``""``
    — which is not valid JSON. Treating that as a parse failure would turn
    every no-argument tool call into an error.
    """
    if not buffer.strip():
        return {}
    try:
        parsed = _json.loads(buffer)
    except _json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
