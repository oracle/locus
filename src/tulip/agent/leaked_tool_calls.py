# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover a model's own tool-call markup when it arrives as plain text.

A model with native tool calling still sometimes writes its call in its own
training format into the message body, and the server's tool parser does not
lift it: DeepSeek through a router answers with ``<｜DSML｜tool_calls>`` text
and no structured call, and the loop reads that as the final answer — the run
ends with the edit never made. :mod:`tulip.agent.text_tool_calls` does not
help there: it parses only for models *without* native calling, because a
native model writing ``name(args)`` is describing a call, not making one.

Markup like ``<｜DSML｜invoke name="edit">`` is different: it is the model's
call syntax, made of its special tokens, and nobody writes it to describe a
call. So a model family declares the formats it leaks
(:attr:`tulip.models.profiles.ModelProfile.leaked_tool_call_formats`), and
:func:`parse_leaked_tool_calls` recognises those — and only under the same
rules that keep the generic parser safe:

- the whole message, after optional leading prose, is one block of the format
  (anything but whitespace or the format's own end tokens after it means no
  calls);
- every call names a registered tool, its arguments are an object, every key
  is a declared parameter and every required parameter is present;
- one bad call rejects the block: running half of what the model wrote is a
  different action from the one it asked for.

A block the reply opens and never closes is a call cut off — usually by the
output-token limit, in the middle of a large ``write``. It is not a call and
not an answer either: :func:`unfinished_leaked_tool_call` recognises it, so the
loop can ask for the call again instead of ending the run on half of it.

Formats, each as the model's published encoding or the vLLM parser for it
writes it:

``dsml``      DeepSeek V3.2 / V4: ``<｜DSML｜tool_calls>`` (V3.2 says
              ``function_calls``) holding ``<｜DSML｜invoke name="…">`` and
              ``<｜DSML｜parameter name="…" string="true|false">…``
``deepseek``  DeepSeek V3 / V3.1: ``<｜tool▁calls▁begin｜>`` …
              ``<｜tool▁call▁begin｜>function<｜tool▁sep｜>name`` and a json
              fence (V3), or ``name<｜tool▁sep｜>{…}`` (V3.1)
``hermes``    ``<tool_call>{"name": …, "arguments": {…}}</tool_call>``
``qwen_xml``  Qwen3-Coder: ``<tool_call><function=name><parameter=p>v``
              ``</parameter></function></tool_call>``
``kimi``      Kimi K2: ``<|tool_calls_section_begin|><|tool_call_begin|>``
              ``functions.name:0<|tool_call_argument_begin|>{…}<|tool_call_end|>``
``glm``       GLM-4.5 / 4.6: ``<tool_call>name`` and
              ``<arg_key>k</arg_key><arg_value>v</arg_value>`` pairs
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from tulip.core.messages import ToolCall
from tulip.models.profiles import LeakedToolCallFormat


if TYPE_CHECKING:
    from tulip.tools.registry import ToolRegistry


__all__ = [
    "LEAKED_TOOL_CALL_FORMATS",
    "LeakedToolCallFormat",
    "LeakedToolCalls",
    "UnfinishedToolCall",
    "match_leaked_tool_calls",
    "parse_leaked_tool_calls",
    "unfinished_leaked_tool_call",
]

#: Every format :func:`parse_leaked_tool_calls` knows, in the order tried.
LEAKED_TOOL_CALL_FORMATS: tuple[LeakedToolCallFormat, ...] = get_args(LeakedToolCallFormat)

#: A call as written: the raw name and its arguments. ``None`` arguments mean
#: the format's values could not be decoded, which rejects the block.
_RawCall = tuple[str, dict[str, Any] | None]

#: A raw string value from a format whose values are typed by the schema
#: (Qwen's XML, GLM): decoded once the tool, and so the type, is known.
_Untyped = tuple[Literal["untyped"], str]


@dataclass(frozen=True)
class LeakedToolCalls:
    """The calls found in a message, the prose before them, and their format."""

    calls: list[ToolCall]
    prose: str | None
    format: LeakedToolCallFormat


@dataclass(frozen=True)
class UnfinishedToolCall:
    """A call block the message opens and never closes, and the prose before it."""

    prose: str | None
    format: LeakedToolCallFormat


# ------------------------------------------------------------------ helpers --

_WS = re.compile(r"\s*")


def _skip(text: str, pos: int) -> int:
    """``pos`` moved past any whitespace."""
    found = _WS.match(text, pos)
    # ``\s*`` matches the empty string, so there is always a match.
    return found.end() if found is not None else pos  # pragma: no cover


#: DeepSeek's end-of-turn token: it trails a generation that stops after the
#: calls, and is not content.
_DEEPSEEK_EOS = re.compile(r"<[｜|]end[▁_]of[▁_]sentence[｜|]>")


def _rest_is_empty(text: str, pos: int, enders: Iterable[re.Pattern[str]] = ()) -> bool:
    """Whether ``text[pos:]`` is only whitespace and the format's end tokens."""
    rest = text[pos:]
    for ender in enders:
        rest = ender.sub("", rest)
    return not rest.strip()


def _json_object(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    if isinstance(value, str):
        # Several servers double-encode the arguments object.
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


# ------------------------------------------------------------------- dsml --

_B = r"[｜|]"
_DSML = rf"{_B}DSML{_B}"
_DSML_OPEN = re.compile(rf"<{_DSML}(?:tool_calls|function_calls)>")
_DSML_CLOSE = re.compile(rf"</{_DSML}(?:tool_calls|function_calls)>")
_DSML_INVOKE = re.compile(rf'<{_DSML}invoke\s+name="([^"]*)"\s*>')
_DSML_INVOKE_CLOSE = re.compile(rf"</{_DSML}invoke>")
_DSML_PARAM = re.compile(
    rf'<{_DSML}parameter\s+name="([^"]*)"\s+string="(true|false)"\s*>(.*?)</{_DSML}parameter>',
    re.DOTALL,
)


def _dsml(text: str, opened: re.Match[str]) -> list[_RawCall] | None:
    """DeepSeek's DSML block opened by ``opened``.

    A generation cut off after its last parameter has no closing ``invoke`` or
    wrapper tag; the parameters it finished are still the call it made. A
    parameter without its close is not finished, and rejects the block.
    """
    pos = opened.end()
    calls: list[_RawCall] = []
    while True:
        pos = _skip(text, pos)
        if pos >= len(text):
            break
        closed = _DSML_CLOSE.match(text, pos)
        if closed is not None:
            pos = closed.end()
            break
        invoke = _DSML_INVOKE.match(text, pos)
        if invoke is None:
            return None
        pos = invoke.end()
        arguments: dict[str, Any] | None = {}
        while True:
            pos = _skip(text, pos)
            param = _DSML_PARAM.match(text, pos)
            if param is None:
                break
            pos = param.end()
            name, is_string, value = param.groups()
            if arguments is None or name in arguments:
                arguments = None
                continue
            if is_string == "true":
                arguments[name] = value
            else:
                try:
                    arguments[name] = json.loads(value)
                except ValueError:
                    arguments = None
        invoke_closed = _DSML_INVOKE_CLOSE.match(text, pos)
        if invoke_closed is not None:
            pos = invoke_closed.end()
        elif pos < len(text):
            return None
        calls.append((invoke.group(1), arguments))
    if not calls or not _rest_is_empty(text, pos, (_DEEPSEEK_EOS,)):
        return None
    return calls


# --------------------------------------------------------------- deepseek --

_U = r"[▁_]"
_DS_OPEN = re.compile(rf"<{_B}tool{_U}calls{_U}begin{_B}>")
_DS_CLOSE = re.compile(rf"<{_B}tool{_U}calls{_U}end{_B}>")
_DS_CALL = re.compile(
    rf"<{_B}tool{_U}call{_U}begin{_B}>(.*?)<{_B}tool{_U}sep{_B}>(.*?)<{_B}tool{_U}call{_U}end{_B}>",
    re.DOTALL,
)
_DS_V3_BODY = re.compile(r"([^\n]*)\n```json\n(.*)\n```\s*", re.DOTALL)


def _deepseek(text: str, opened: re.Match[str]) -> list[_RawCall] | None:
    """DeepSeek V3 / V3.1's ``<｜tool▁calls▁begin｜>`` block opened by ``opened``."""
    pos = opened.end()
    calls: list[_RawCall] = []
    while True:
        pos = _skip(text, pos)
        call = _DS_CALL.match(text, pos)
        if call is None:
            break
        pos = call.end()
        head, body = call.group(1).strip(), call.group(2)
        v3 = _DS_V3_BODY.fullmatch(body)
        if head == "function" and v3 is not None:
            # V3: the call type, then the name and a json fence.
            calls.append((v3.group(1).strip(), _json_object(v3.group(2))))
        else:
            # V3.1: the name, then the arguments object.
            calls.append((head, _json_object(body.strip())))
    closed = _DS_CLOSE.match(text, pos)
    if closed is not None:
        pos = closed.end()
    if not calls or not _rest_is_empty(text, pos, (_DEEPSEEK_EOS,)):
        return None
    return calls


# ---------------------------------------------------- <tool_call> families --

_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _tool_call_tags(
    text: str, opened: re.Match[str], body: Callable[[str], _RawCall | None]
) -> list[_RawCall] | None:
    """Consecutive ``<tool_call>…</tool_call>`` tags from ``opened``, each read by ``body``."""
    pos = opened.start()
    calls: list[_RawCall] = []
    while True:
        tag = _TOOL_CALL.match(text, pos)
        if tag is None:
            break
        call = body(tag.group(1))
        if call is None:
            return None
        calls.append(call)
        pos = _skip(text, tag.end())
    if not calls or not _rest_is_empty(text, pos):
        return None
    return calls


def _hermes_body(body: str) -> _RawCall | None:
    try:
        obj = json.loads(body)
    except ValueError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    raw_args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(raw_args, str):
        return obj["name"], _json_object(raw_args)
    return obj["name"], raw_args if isinstance(raw_args, dict) else None


_QWEN_FUNCTION = re.compile(r"\s*<function=([^>\n]+)>(.*?)</function>\s*", re.DOTALL)
_QWEN_PARAM = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)


def _qwen_body(body: str) -> _RawCall | None:
    function = _QWEN_FUNCTION.fullmatch(body)
    if function is None:
        return None
    arguments: dict[str, Any] = {}
    pos = 0
    inner = function.group(2)
    while True:
        pos = _skip(inner, pos)
        param = _QWEN_PARAM.match(inner, pos)
        if param is None:
            break
        pos = param.end()
        value = param.group(2)
        # The template puts each value on its own lines; those newlines are
        # layout, not part of the value.
        value = value.removeprefix("\n").removesuffix("\n")
        arguments[param.group(1).strip()] = ("untyped", value)
    if pos < len(inner):
        return None
    return function.group(1).strip(), arguments


_GLM_HEAD = re.compile(r"\s*([^\n<]+?)\s*(?=<arg_key>|$)", re.DOTALL)
_GLM_PAIR = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)


def _glm_body(body: str) -> _RawCall | None:
    head = _GLM_HEAD.match(body)
    if head is None:
        return None
    arguments: dict[str, Any] = {}
    pos = head.end()
    while True:
        pos = _skip(body, pos)
        pair = _GLM_PAIR.match(body, pos)
        if pair is None:
            break
        pos = pair.end()
        arguments[pair.group(1).strip()] = ("untyped", pair.group(2))
    if pos < len(body):
        return None
    return head.group(1).strip(), arguments


# ------------------------------------------------------------------- kimi --

_KIMI_OPEN = re.compile(r"<\|tool_calls?_section_begin\|>")
_KIMI_CLOSE = re.compile(r"<\|tool_calls?_section_end\|>")
_KIMI_CALL = re.compile(
    r"<\|tool_call_begin\|>\s*([^<\s]+):\d+\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*"
    r"<\|tool_call_end\|>",
    re.DOTALL,
)


def _kimi(text: str, opened: re.Match[str]) -> list[_RawCall] | None:
    """Kimi K2's tool-call section opened by ``opened``."""
    pos = opened.end()
    calls: list[_RawCall] = []
    while True:
        pos = _skip(text, pos)
        call = _KIMI_CALL.match(text, pos)
        if call is None:
            break
        pos = call.end()
        # The id is ``functions.<name>``; the name is its last dotted part.
        calls.append((call.group(1).split(".")[-1], _json_object(call.group(2))))
    closed = _KIMI_CLOSE.match(text, pos)
    if closed is not None:
        pos = closed.end()
    if not calls or not _rest_is_empty(text, pos):
        return None
    return calls


# ---------------------------------------------------------------- formats --

_Reader = Callable[[str, re.Match[str]], list[_RawCall] | None]

#: Each format: where a block of it can start, and how to read the block.
_FORMATS: dict[str, tuple[re.Pattern[str], _Reader]] = {
    "dsml": (_DSML_OPEN, _dsml),
    "deepseek": (_DS_OPEN, _deepseek),
    "hermes": (re.compile(r"<tool_call>"), lambda t, s: _tool_call_tags(t, s, _hermes_body)),
    "qwen_xml": (re.compile(r"<tool_call>"), lambda t, s: _tool_call_tags(t, s, _qwen_body)),
    "kimi": (_KIMI_OPEN, _kimi),
    "glm": (re.compile(r"<tool_call>"), lambda t, s: _tool_call_tags(t, s, _glm_body)),
}


_TOOL_CALL_OPEN = re.compile(r"(?:^|\n)[ \t]*(<tool_call>)")
_TOOL_CALL_CLOSE = re.compile(r"</tool_call>")

#: Each format: where a block of it starts (group 1 when the pattern has one),
#: and the tag that closes it. ``<tool_call>`` counts only at the start of a
#: line: unlike the special-token formats it is ordinary text a final answer
#: may quote.
_ENDS: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {
    "dsml": (_DSML_OPEN, _DSML_CLOSE),
    "deepseek": (_DS_OPEN, _DS_CLOSE),
    "hermes": (_TOOL_CALL_OPEN, _TOOL_CALL_CLOSE),
    "qwen_xml": (_TOOL_CALL_OPEN, _TOOL_CALL_CLOSE),
    "kimi": (_KIMI_OPEN, _KIMI_CLOSE),
    "glm": (_TOOL_CALL_OPEN, _TOOL_CALL_CLOSE),
}


# ------------------------------------------------------------- validation --


def _normalize(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def _typed(value: str, schema: Any) -> Any:
    """A schema-typed value written as text: decoded unless the schema says string.

    Qwen's XML and GLM write every value as text and leave the type to the
    tool's schema, as their vLLM parsers do; a value that does not decode
    stays text, and the tool's own validation has the last word.
    """
    declared = schema.get("type") if isinstance(schema, dict) else None
    types = declared if isinstance(declared, list) else [declared]
    if declared is None or "string" in types:
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _validated(raw: list[_RawCall], registry: ToolRegistry) -> list[ToolCall] | None:
    """The calls as :class:`ToolCall` objects, or ``None`` if any is not a valid call."""
    lookup = {_normalize(name): name for name in registry.tools}
    calls: list[ToolCall] = []
    for raw_name, raw_args in raw:
        name = raw_name if raw_name in registry.tools else lookup.get(_normalize(raw_name))
        tool = registry.get(name) if name is not None else None
        if tool is None or name is None or raw_args is None:
            return None
        schema = tool.parameters if isinstance(tool.parameters, dict) else {}
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        arguments: dict[str, Any] = {}
        for key, value in raw_args.items():
            if key not in properties:
                return None
            untyped = isinstance(value, tuple) and len(value) == 2 and value[0] == "untyped"
            arguments[key] = _typed(value[1], properties[key]) if untyped else value
        if any(key not in arguments for key in required):
            return None
        calls.append(ToolCall(name=name, arguments=arguments))
    return calls


# -------------------------------------------------------------------- api --


def match_leaked_tool_calls(
    text: str | None,
    registry: ToolRegistry | None,
    formats: Iterable[str],
) -> LeakedToolCalls | None:
    """The calls ``text`` makes in one of ``formats``, or ``None``.

    Formats are tried in the order given; the first that reads the whole
    message (after its leading prose) as valid calls wins. Unknown format
    names are skipped, so a profile written for a newer SDK still loads.
    """
    if not text or registry is None or not registry.tools:
        return None
    for fmt in formats:
        entry = _FORMATS.get(fmt)
        if entry is None:
            continue
        opener, read = entry
        found = opener.search(text)
        if found is None:
            continue
        raw = read(text, found)
        if raw is None:
            continue
        calls = _validated(raw, registry)
        if calls is None:
            continue
        prose = text[: found.start()].strip() or None
        return LeakedToolCalls(calls=calls, prose=prose, format=cast("LeakedToolCallFormat", fmt))
    return None


def parse_leaked_tool_calls(
    text: str | None,
    registry: ToolRegistry | None,
    formats: Iterable[str],
) -> tuple[list[ToolCall], str | None]:
    """The calls ``text`` makes in one of ``formats``, and the prose before them.

    Returns ``([], None)`` when the message is not such a block — prose, a
    block followed by more text, a call to an unknown tool, or arguments the
    tool does not declare.
    """
    found = match_leaked_tool_calls(text, registry, formats)
    if found is None:
        return [], None
    return found.calls, found.prose


def unfinished_leaked_tool_call(
    text: str | None,
    formats: Iterable[str],
) -> UnfinishedToolCall | None:
    """The call block ``text`` opens in one of ``formats`` and never closes, or ``None``.

    Only for a message :func:`match_leaked_tool_calls` did not read as calls:
    a block whose finished part is a valid call is a call. The last block the
    message opens is the one that counts, so a reply that quotes a closed
    block and then is cut off inside a new one is still unfinished.
    """
    if not text:
        return None
    for fmt in formats:
        ends = _ENDS.get(fmt)
        if ends is None:
            continue
        opener, closer = ends
        opened = list(opener.finditer(text))
        if not opened:
            continue
        last = opened[-1]
        start = last.start(1) if opener.groups else last.start()
        if closer.search(text, last.end()) is None:
            prose = text[:start].strip() or None
            return UnfinishedToolCall(prose=prose, format=cast("LeakedToolCallFormat", fmt))
    return None
