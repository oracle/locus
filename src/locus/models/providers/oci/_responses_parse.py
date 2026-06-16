# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Request / response translation for the OCI GenAI Responses API.

Pure functions — no network, no SDK imports — so they're fully unit
testable without an OCI endpoint. :class:`OCIResponsesModel` composes
these with the openai SDK + OCI httpx signer to make the wire call.

The Responses API differs from chat/completions in three meaningful ways:

1. The system role moves out of ``messages`` into the top-level
   ``instructions`` field.
2. ``input`` is a list of typed items, not a list of OpenAI
   chat-completion messages. Roles map differently and tool results
   carry a ``call_id`` reference.
3. The response's ``output`` is a list of typed items (``message`` or
   ``function_call``) — assistant text + tool calls are interleaved
   inside, not separated like chat/completions.

Reference: OpenAI Responses API spec — OCI exposes a compatible
``/openai/v1/responses`` endpoint with the same wire shape.
"""

from __future__ import annotations

import json
from typing import Any

from locus.core.messages import Message, Role, ToolCall


def build_request_body(
    messages: list[Message],
    *,
    model: str,
    tools: list[dict[str, Any]] | None = None,
    previous_response_id: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    stream: bool = False,
    response_format: dict[str, Any] | None = None,
    store: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Responses API POST body from Locus messages.

    Args:
        messages: The slice of state.messages since the last assistant
            turn. On turn 1 includes the system + user message; on
            subsequent turns includes only tool results / the latest
            user input.
        model: OCI model id (e.g. ``openai.gpt-5.5-pro``).
        tools: Tool schemas in OpenAI function-call format.
        previous_response_id: Continuation token from the last turn.
            None on turn 1.
        temperature: Optional sampling temperature.
        max_output_tokens: Optional output token cap.
        stream: True to request SSE streaming.
        response_format: Optional structured-output schema.
        extra: Extra fields merged into the request body (escape hatch
            for caller-specific knobs).

    Returns:
        The complete request body dict.

    System messages are extracted from ``messages`` and joined into the
    ``instructions`` field; they never appear in ``input``. Tool result
    messages (role=tool) become ``function_call_output`` items keyed
    by ``call_id``. User and assistant text messages become ``message``
    items with typed content.
    """
    instructions_parts: list[str] = []
    input_items: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            if msg.content:
                instructions_parts.append(msg.content)
            continue

        if msg.role == Role.TOOL:
            # Tool results reference the call by id.
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id or "",
                    "output": msg.content or "",
                }
            )
            continue

        if msg.role == Role.ASSISTANT:
            # An assistant message can carry text content AND/OR tool
            # calls. Both need to be reconstructed as Responses input
            # items so the server has context for any function_call_output
            # items that follow.
            #
            # Text → ``{type: message, role: assistant, content: [output_text]}``
            # Tool calls → one ``{type: function_call, ...}`` item per call,
            # in the same order. Required in stateless mode (``store=False``):
            # without the function_call item, a subsequent
            # function_call_output has no call_id anchor and the server
            # returns 400.
            if msg.content:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": msg.content}],
                    }
                )
            for tc in msg.tool_calls or []:
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments) if tc.arguments else "{}",
                    }
                )
            continue

        # Default — user message.
        if msg.content:
            input_items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": msg.content}],
                }
            )

    body: dict[str, Any] = {"model": model, "input": input_items}

    if instructions_parts:
        body["instructions"] = "\n\n".join(instructions_parts)
    # In stateless mode (ZDR tenancies, ``store=False``) we also drop
    # any inbound previous_response_id — the server can't honour it.
    if store is False:
        body["store"] = False
    elif previous_response_id is not None:
        body["previous_response_id"] = previous_response_id
    if tools:
        body["tools"] = [_translate_tool_schema(t) for t in tools]
    if temperature is not None:
        body["temperature"] = temperature
    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens
    if stream:
        body["stream"] = True
    if response_format is not None:
        # Responses uses ``response_format`` at the top level too, but
        # nested under ``text.format`` in newer revisions. Keep the
        # chat/completions shape — OCI normalizes both per the spec.
        body["response_format"] = response_format
    if extra:
        body.update(extra)

    return body


def _translate_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI chat-completions tool schema to Responses shape.

    Chat/completions wraps the function in ``{"type": "function",
    "function": {"name", "description", "parameters"}}``. Responses
    flattens it: ``{"type": "function", "name", "description",
    "parameters"}``.
    """
    if tool.get("type") != "function":
        # Pass through unknown tool types (built-in OCI tools, etc.) —
        # OCIResponsesModel will validate them at construction.
        return tool

    fn = tool.get("function", {})
    flat: dict[str, Any] = {"type": "function"}
    if "name" in fn:
        flat["name"] = fn["name"]
    if "description" in fn:
        flat["description"] = fn["description"]
    if "parameters" in fn:
        flat["parameters"] = fn["parameters"]
    return flat


def parse_response(
    response: dict[str, Any],
) -> tuple[Message, dict[str, int], str | None, dict[str, Any]]:
    """Parse an OCI Responses API response into Locus shapes.

    Args:
        response: The decoded JSON body from a Responses POST.

    Returns:
        Tuple of ``(message, usage, stop_reason, provider_state)``:

        - ``message``: ``Message.assistant(...)`` carrying text +
          parsed tool calls extracted from the ``output`` array.
        - ``usage``: ``{"prompt_tokens", "completion_tokens",
          "total_tokens"}`` per Locus convention. Missing fields default
          to 0.
        - ``stop_reason``: Mapped from the response's ``status`` /
          ``stop_reason`` field where present (Responses uses
          ``status`` at the top level — ``"completed"``, ``"incomplete"``,
          ``"failed"``).
        - ``provider_state``: ``{"previous_response_id": <id>}`` —
          the token the agent threads back into the next ``complete()``
          call. Empty dict if the response carried no id (defensive).
    """
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for item in response.get("output", []):
        item_type = item.get("type")
        if item_type == "message":
            # Assistant message — content is a list of typed parts.
            for part in item.get("content", []):
                if part.get("type") in ("output_text", "text"):
                    text = part.get("text")
                    if text:
                        content_parts.append(text)
        elif item_type == "function_call":
            # Tool call.
            call_id = item.get("call_id") or item.get("id") or ""
            name = item.get("name", "")
            raw_args = item.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except (ValueError, TypeError):
                args = {}
            tool_calls.append(ToolCall(id=call_id, name=name, arguments=args))
        # Unknown item types (reasoning, refusal, etc.) are ignored;
        # the model layer surfaces them only when the caller opts in.

    content = "".join(content_parts) if content_parts else None

    # Usage — Responses uses {input_tokens, output_tokens, total_tokens}
    # rather than chat/completions' {prompt_tokens, completion_tokens}.
    # Normalize to Locus's convention.
    raw_usage = response.get("usage", {}) or {}
    usage = {
        "prompt_tokens": int(raw_usage.get("input_tokens", raw_usage.get("prompt_tokens", 0))),
        "completion_tokens": int(
            raw_usage.get("output_tokens", raw_usage.get("completion_tokens", 0))
        ),
        "total_tokens": int(raw_usage.get("total_tokens", 0)),
    }
    if usage["total_tokens"] == 0:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    stop_reason = response.get("status") or response.get("stop_reason")

    response_id = response.get("id")
    provider_state: dict[str, Any] = {"previous_response_id": response_id} if response_id else {}

    msg = Message.assistant(content=content, tool_calls=tool_calls or None)
    return msg, usage, stop_reason, provider_state


def parse_stream_event(event: dict[str, Any]) -> dict[str, Any]:
    """Translate a single Responses SSE event into a Locus chunk shape.

    Returns a dict with optional keys: ``content`` (str delta),
    ``tool_calls`` (list of partial ToolCall dicts), ``done`` (bool),
    ``provider_state`` (dict, set on completion), ``error`` (str).
    The caller maps the dict to a :class:`ModelChunkEvent`.
    """
    etype = event.get("type", "")

    if etype.endswith("output_text.delta"):
        return {"content": event.get("delta", "")}

    if etype.endswith(("function_call.delta", "function_call_arguments.delta")):
        # Argument deltas come in pieces; the caller accumulates by call_id.
        partial = {
            "id": event.get("call_id") or event.get("item_id") or "",
            "arguments_delta": event.get("delta", ""),
        }
        return {"tool_calls": [partial]}

    if etype == "response.completed":
        response = event.get("response", {}) or {}
        rid = response.get("id")
        out: dict[str, Any] = {"done": True}
        if rid:
            out["provider_state"] = {"previous_response_id": rid}
        return out

    if etype == "response.error" or etype.endswith(".error"):
        return {"error": event.get("error", {}).get("message", "Responses API error")}

    # Unknown / structural events (lifecycle, output_item.added, ...) — skip.
    return {}
