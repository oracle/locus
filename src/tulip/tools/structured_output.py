# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A final answer that must match a JSON Schema, delivered as a tool call.

``AgentConfig.output_schema`` coerces the final *text* into a Pydantic model
after the loop, in ``arun`` only. A caller holding a JSON Schema document and
driving ``run()`` — a CLI's ``--json-schema``, an API taking a schema per
request — needs something else: the model hands over the answer as the
arguments of a tool whose parameters *are* the schema, the tool validates
them, and a turn that tries to end without a valid call is sent back.

That is the approach opencode takes (a forced ``StructuredOutput`` tool), and
it travels across providers better than ``response_format``: every model that
calls tools can produce it, and a validation failure comes back as an ordinary
tool error the model can read and fix::

    out = StructuredOutputTool(schema)
    agent = Agent(
        model=...,
        tools=[*tools, out.tool],
        system_prompt=f"{prompt}\\n\\n{out.instructions}",
        **out.agent_options(),
    )
    async for ev in agent.run(task):
        ...
    value = out.value_from(state.tool_executions)  # or from the events

How the turn is held to it: :meth:`StructuredOutputTool.verify` is a
``final_answer_verifier`` that rejects a final answer when the turn has no
successful call, so the model is told to call the tool and gets another
attempt (``final_answer_verifier_max_replans``). The tool is not a terminal
tool, on purpose: the loop stops on a terminal tool whether or not the call
succeeded, which would end the turn on a validation error with nothing to show
for it. The price is one short model call after the successful one.

A schema whose top level is not an object (an array, a string) is wrapped as
``{"value": <schema>}`` because tool parameters must be an object; the value
is unwrapped again on the way out.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from typing import Any

from tulip.core.json_schema import SchemaError, check_schema, validate
from tulip.tools.decorator import Tool
from tulip.tools.output import ToolOutput


__all__ = [
    "STRUCTURED_OUTPUT_TOOL",
    "SchemaError",
    "StructuredOutputTool",
    "load_schema",
]

#: The default tool name. Plain, so a model reads it as what it is.
STRUCTURED_OUTPUT_TOOL = "structured_output"

#: What the tool says once it has accepted a value. The model's next reply
#: ends the turn; asking for a single line keeps that extra call cheap.
ACCEPTED = (
    "Structured output recorded and valid. The task is complete: reply with one "
    "short line confirming it, and do not call this tool again."
)

_MISSING = object()


def load_schema(source: str) -> dict[str, Any]:
    """A schema from inline JSON, or from the file at ``source``.

    Inline wins when ``source`` starts with ``{`` — a path does not, and
    trying the file first would turn a typo in inline JSON into a confusing
    "no such file".

    Raises:
        SchemaError: Unreadable, not JSON, or not a usable schema.
    """
    text = source.strip()
    if not text.startswith("{"):
        try:
            with open(source, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            raise SchemaError(f"cannot read schema file {source!r}: {exc}") from exc
    try:
        schema = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"schema is not valid JSON: {exc}") from exc
    if not isinstance(schema, dict):
        raise SchemaError("a schema must be a JSON object")
    check_schema(schema)
    return schema


class StructuredOutputTool:
    """A tool that takes the final answer as schema-valid arguments.

    Args:
        schema: A JSON Schema (draft 2020-12 subset; see
            :mod:`tulip.core.json_schema`).
        name: The tool's name. Change it only to avoid a clash.
        description: What the model is told the tool is for.
        max_reminders: How many times a turn that tried to end without a
            valid call is sent back (``final_answer_verifier_max_replans``).

    Raises:
        SchemaError: The schema is unusable.
    """

    def __init__(
        self,
        schema: Mapping[str, Any],
        *,
        name: str = STRUCTURED_OUTPUT_TOOL,
        description: str | None = None,
        max_reminders: int = 2,
    ) -> None:
        check_schema(dict(schema))
        self.schema: dict[str, Any] = copy.deepcopy(dict(schema))
        self.name = name
        self.max_reminders = max_reminders
        self.wrapped = self.schema.get("type") != "object"
        parameters = self._parameters()
        self.tool = Tool(
            name=name,
            description=description
            or (
                "Deliver your final answer. Call this exactly once, when the task is "
                "done, with arguments that match the schema. If it reports errors, "
                "fix them and call it again."
            ),
            parameters=parameters,
            fn=self._record,
        )

    def _parameters(self) -> dict[str, Any]:
        if not self.wrapped:
            return self.schema
        # ``$defs`` stay at the top so ``#/$defs/...`` references still resolve
        # from the wrapper's root.
        inner = {k: v for k, v in self.schema.items() if k not in ("$defs", "definitions")}
        wrapper: dict[str, Any] = {
            "type": "object",
            "properties": {"value": inner},
            "required": ["value"],
            "additionalProperties": False,
        }
        for key in ("$defs", "definitions"):
            if key in self.schema:
                wrapper[key] = self.schema[key]
        return wrapper

    # -- the tool body ----------------------------------------------------

    def _record(self, **arguments: Any) -> str:
        errors = self.errors(arguments)
        if errors:
            # Returned as an error rather than raised: the executor keeps only
            # the first line of an exception, and the model needs every path
            # to fix exactly those values.
            return ToolOutput(
                "the structured output does not match the schema:\n- "
                + "\n- ".join(errors[:20])
                + "\nFix these and call the tool again.",
                is_error=True,
            )
        return ACCEPTED

    def errors(self, arguments: Mapping[str, Any]) -> list[str]:
        """Why a call's arguments are not a valid answer (empty when they are)."""
        if self.wrapped:
            if "value" not in arguments:
                return ["$: missing required property 'value'"]
            return validate(arguments["value"], self.schema)
        return validate(dict(arguments), self.schema)

    def value_of(self, arguments: Mapping[str, Any]) -> Any:
        """The answer a call's arguments carry, unwrapped."""
        return arguments.get("value") if self.wrapped else dict(arguments)

    def _last_valid(self, executions: Iterable[Any]) -> Mapping[str, Any] | None:
        """The arguments of the last successful, schema-valid call, if any."""
        for execution in reversed(list(executions)):
            if getattr(execution, "tool_name", None) != self.name:
                continue
            if getattr(execution, "error", None) is not None:
                continue
            arguments: Mapping[str, Any] = getattr(execution, "arguments", None) or {}
            if not self.errors(arguments):
                return arguments
        return None

    def value_from(self, executions: Iterable[Any], default: Any = _MISSING) -> Any:
        """The answer from the last successful call among ``executions``.

        ``executions`` are :class:`~tulip.core.state.ToolExecution` records
        (``state.tool_executions``, ``FinalAnswerContext.tool_executions``) or
        anything with ``tool_name``, ``arguments`` and ``error``.

        Raises:
            LookupError: No successful call, and no ``default``.
        """
        arguments = self._last_valid(executions)
        if arguments is not None:
            return self.value_of(arguments)
        if default is not _MISSING:
            return default
        raise LookupError(f"no successful {self.name} call")

    # -- holding the turn to it ------------------------------------------

    async def verify(self, draft: str, ctx: Any) -> str | None:
        """A ``final_answer_verifier``: no valid call yet means try again."""
        if self._last_valid(ctx.tool_executions) is not None:
            return None
        return (
            f"You have not delivered the result. Call the {self.name} tool with "
            "your answer as arguments matching its schema — the answer only "
            "counts when it arrives through that tool."
        )

    @property
    def instructions(self) -> str:
        """Text for the system prompt, so the model knows how to finish."""
        return (
            f"When the task is done, deliver your final answer by calling the "
            f"`{self.name}` tool with arguments matching its schema. The caller "
            "reads only that call — prose in your reply is not the answer. If the "
            "tool reports schema errors, correct them and call it again."
        )

    def agent_options(self) -> dict[str, Any]:
        """``Agent(...)`` keyword arguments that hold a turn to this tool."""
        return {
            "final_answer_verifier": self.verify,
            "final_answer_verifier_max_replans": self.max_reminders,
        }
