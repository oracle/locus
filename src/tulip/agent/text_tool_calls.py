# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover tool calls a model wrote as text instead of structured calls.

Some models (small self-hosted ones served without a tool parser, the
Hermes/Qwen templates behind a server that does not lift them) put a tool
call in the message body. :func:`parse_text_tool_calls` turns such a body
into :class:`~tulip.core.messages.ToolCall` objects — but only when the
body is unambiguously a call. A final answer that *mentions* a tool
("run ``bash(command="pytest")`` to verify") is prose, and executing it
would run a command nobody asked for. So the accepted shapes are:

- the whole message is a JSON call object, or a list of them:
  ``{"name": "search", "arguments": {"query": "x"}}``;
- a fenced block tagged ``json``, ``tool_call`` or ``tool_code`` whose
  whole content is such JSON;
- a Hermes-style ``<tool_call>{...}</tool_call>`` tag;
- call syntax — ``search(query="x")`` — only when it is the whole message
  or the whole content of a ``tool_call`` / ``tool_code`` fence, one call
  per line (a line may also be a one-line JSON call), with keyword
  arguments that are Python literals declared in the tool's schema.

A call whose name does not resolve against the registry, or whose call
syntax carries positional or undeclared arguments, is not a call: guessing
``read(path)`` into ``read({})`` executes something the model did not ask
for.
"""

from __future__ import annotations

import ast
import json
import re
from typing import TYPE_CHECKING, Any

from tulip.core.messages import ToolCall


if TYPE_CHECKING:
    from tulip.tools.registry import ToolRegistry


__all__ = ["parse_text_tool_calls"]


_FENCE = re.compile(r"```[ \t]*([A-Za-z_]*)[ \t]*\n(.*?)\n?[ \t]*```", re.DOTALL)
_HERMES = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_JSON_FENCES = frozenset({"json", "tool_call", "tool_code"})
_CALL_FENCES = frozenset({"tool_call", "tool_code"})


def _normalize(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


class _Parser:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._lookup = {_normalize(name): name for name in registry.tools}
        self.calls: list[ToolCall] = []
        self._seen: set[str] = set()

    def _resolve(self, raw_name: str) -> str | None:
        return self._lookup.get(_normalize(raw_name))

    def _add(self, name: str, arguments: dict[str, Any]) -> None:
        key = json.dumps([name, arguments], sort_keys=True, default=str)
        if key in self._seen:
            return
        self._seen.add(key)
        self.calls.append(ToolCall(name=name, arguments=arguments))

    def json_calls(self, text: str) -> bool:
        """Add the calls in ``text`` if all of it is JSON call objects."""
        try:
            obj = json.loads(text)
        except ValueError:
            return False
        items = obj if isinstance(obj, list) else [obj]
        found: list[tuple[str, dict[str, Any]]] = []
        for item in items:
            call = self._json_call(item)
            if call is None:
                return False
            found.append(call)
        for name, arguments in found:
            self._add(name, arguments)
        return bool(found)

    def _json_call(self, obj: Any) -> tuple[str, dict[str, Any]] | None:
        if not isinstance(obj, dict):
            return None
        # OpenAI's own wire shape nests the call under "function".
        if isinstance(obj.get("function"), dict):
            obj = obj["function"]
        raw_name = obj.get("name") or obj.get("tool") or obj.get("function")
        if not isinstance(raw_name, str):
            return None
        name = self._resolve(raw_name)
        if name is None:
            return None
        raw_args = obj.get("arguments")
        if raw_args is None:
            raw_args = obj.get("parameters")
        if isinstance(raw_args, str):
            # Several servers double-encode the arguments object.
            try:
                raw_args = json.loads(raw_args)
            except ValueError:
                return None
        if raw_args is None:
            raw_args = {}
        if not isinstance(raw_args, dict):
            return None
        return name, raw_args

    def line_calls(self, text: str) -> bool:
        """Add the calls in ``text`` if every non-blank line is one call.

        A line is a single-line JSON call object or call syntax.
        """
        found: list[tuple[str, dict[str, Any]]] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            call = self._line_json_call(line) or self._syntax_call(line)
            if call is None:
                return False
            found.append(call)
        for name, arguments in found:
            self._add(name, arguments)
        return bool(found)

    def _line_json_call(self, line: str) -> tuple[str, dict[str, Any]] | None:
        try:
            return self._json_call(json.loads(line))
        except ValueError:
            return None

    def _syntax_call(self, line: str) -> tuple[str, dict[str, Any]] | None:
        try:
            node = ast.parse(line, mode="eval").body
        except SyntaxError:
            return None
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return None
        # A positional argument has no parameter name to bind to; inventing
        # one, or dropping it, runs a call the model did not write.
        if node.args:
            return None
        name = self._resolve(node.func.id)
        if name is None:
            return None
        declared = self._declared_params(name)
        arguments: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None or kw.arg not in declared:
                return None
            try:
                arguments[kw.arg] = ast.literal_eval(kw.value)
            except ValueError:
                return None
        return name, arguments

    def _declared_params(self, name: str) -> set[str]:
        tool = self._registry.get(name)
        if tool is None:
            return set()
        schema = tool.to_openai_schema().get("function", {})
        properties = schema.get("parameters", {}).get("properties", {})
        return set(properties)


def parse_text_tool_calls(text: str | None, registry: ToolRegistry | None) -> list[ToolCall]:
    """Tool calls written as the unambiguous shapes the module docstring lists.

    Args:
        text: The assistant message body.
        registry: The tools the agent can call; names resolve against it
            (case-insensitively, ignoring ``_`` and ``-``).

    Returns:
        The calls found, in order and without duplicates; empty when the body
        is prose, even prose that mentions a tool.
    """
    if not text or registry is None or not registry.tools:
        return []
    parser = _Parser(registry)
    body = text.strip()

    if parser.json_calls(body) or parser.line_calls(body):
        return parser.calls

    for match in _FENCE.finditer(body):
        lang, content = match.group(1).lower(), match.group(2).strip()
        if lang in _JSON_FENCES and parser.json_calls(content):
            continue
        if lang in _CALL_FENCES:
            parser.line_calls(content)

    for match in _HERMES.finditer(body):
        parser.json_calls(match.group(1))

    return parser.calls
