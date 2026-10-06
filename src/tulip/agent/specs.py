# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Agent definitions as files: Markdown with frontmatter, the prompt as body.

A coding harness lets its user define agents without writing code — a
reviewer with read-only tools, a planner on a cheaper model, a test-writer
with its own instructions. Two formats are in wide use and they agree on the
shape: YAML frontmatter for the settings, the Markdown body as the system
prompt. This module reads both, so a team's existing ``.claude/agents`` and
opencode ``agent/`` files work unchanged::

    ---
    name: reviewer
    description: Reviews a diff for bugs. Use after a change, before commit.
    model: anthropic:claude-sonnet-4-6
    tools: read, grep, glob          # or a YAML list
    disallowed_tools: [bash]
    mode: subagent                   # primary | subagent | all
    max_turns: 12
    token_budget: 300000             # tokens in and out, per task
    ---
    You review code changes. Report bugs with file and line...

What each harness accepts as a key:

- ``tools`` — a comma- or space-separated string or a list (Claude Code), or
  a ``{name: bool}`` map (opencode: ``false`` turns a tool off). Names are
  compared with case, ``_`` and ``-`` ignored, so ``WebFetch`` names
  ``web_fetch``; ``*`` and ``?`` match as globs (``mcp__github__*``).
- ``disallowed_tools`` / ``disallowedTools`` — always removed.
- ``max_turns`` / ``maxTurns`` / ``steps`` / ``maxSteps`` — iteration cap.
- ``token_budget`` / ``tokenBudget`` / ``max_tokens`` — tokens (in and out)
  one delegated task may spend before it is stopped.
- ``model: inherit`` — the same as leaving it out.
- ``disable: true`` — the file is skipped.

Anything else in the frontmatter is kept on :attr:`AgentSpec.metadata`, for
the harness to interpret. A spec never grants a tool: :meth:`AgentSpec.select_tools`
picks from the tools the harness offers, so an agent file can narrow what an
agent may do and can never widen it.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


__all__ = [
    "AgentMode",
    "AgentSpec",
    "load_agent_file",
    "load_agent_specs",
    "merge_specs",
    "parse_agent_markdown",
]

logger = logging.getLogger(__name__)

AgentMode = Literal["primary", "subagent", "all"]

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_TURN_KEYS = ("max_turns", "maxTurns", "steps", "maxSteps", "max_iterations")
_DENY_KEYS = ("disallowed_tools", "disallowedTools", "disallowed-tools")
_BUDGET_KEYS = ("token_budget", "tokenBudget", "max_tokens")
_KNOWN_KEYS = {
    "name",
    "description",
    "model",
    "tools",
    "mode",
    "temperature",
    "disable",
    *_TURN_KEYS,
    *_DENY_KEYS,
    *_BUDGET_KEYS,
}


def _norm(name: str) -> str:
    """A tool name as compared: ``WebFetch``, ``web_fetch`` and ``web-fetch`` agree."""
    return re.sub(r"[^a-z0-9*?]", "", name.lower())


def _matches(pattern: str, tool_name: str) -> bool:
    pat, name = _norm(pattern), _norm(tool_name)
    if "*" in pat or "?" in pat:
        return fnmatch.fnmatchcase(name, pat)
    return pat == name


class AgentSpec(BaseModel):
    """One agent definition: who it is, what it may use, what it is told."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    #: When to use this agent. Shown to a parent choosing a subagent type, so
    #: it is written for a model to read.
    description: str = ""
    #: The system prompt. Empty means the harness's default prompt.
    prompt: str = ""
    #: A model id, or ``None`` to use the caller's model. The harness decides
    #: what an id means (a provider prefix, an alias).
    model: str | None = None
    #: Tool names (or globs) this agent may use. ``None`` means every tool
    #: the harness offers it.
    tools: tuple[str, ...] | None = None
    #: Tool names (or globs) it may never use, whatever ``tools`` says.
    disallowed_tools: tuple[str, ...] = ()
    #: ``primary`` drives a session, ``subagent`` is delegated to, ``all`` both.
    mode: AgentMode = "all"
    max_turns: int | None = Field(default=None, ge=1, le=500)
    #: Tokens, in and out, one task of this type may spend. A delegated search
    #: that reads without end costs the parent as much as reading it itself,
    #: and more. ``None`` leaves the child only the parent's own budget.
    token_budget: int | None = Field(default=None, ge=1)
    temperature: float | None = None
    #: The file it was read from, for messages that point at it.
    source: str | None = None
    #: Frontmatter keys this model does not interpret, kept as written.
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME.match(value):
            msg = (
                f"agent name {value!r} must be 1-64 letters, digits, '.', '_' or '-', "
                "starting with a letter or digit"
            )
            raise ValueError(msg)
        return value

    @property
    def is_primary(self) -> bool:
        return self.mode in ("primary", "all")

    @property
    def is_subagent(self) -> bool:
        return self.mode in ("subagent", "all")

    def allows(self, tool_name: str) -> bool:
        """Whether this spec lets the agent use a tool called ``tool_name``."""
        if any(_matches(p, tool_name) for p in self.disallowed_tools):
            return False
        return self.tools is None or any(_matches(p, tool_name) for p in self.tools)

    def select_tools(self, available: Iterable[Any]) -> list[Any]:
        """The tools from ``available`` this agent may use, in their order.

        Only ever a subset: a name the spec lists that ``available`` lacks is
        not conjured up, so a spec can narrow a toolset and never widen it.
        """
        return [t for t in available if self.allows(str(getattr(t, "name", t)))]

    def unknown_tools(self, available: Iterable[Any]) -> list[str]:
        """Names in :attr:`tools` that match nothing in ``available``.

        Usually a typo, or a tool from another harness. Worth a warning: the
        agent will run without it.
        """
        names = [str(getattr(t, "name", t)) for t in available]
        return [p for p in self.tools or () if not any(_matches(p, n) for n in names)]


# ----------------------------------------------------------------- parsing --


def _split_frontmatter(text: str) -> tuple[str | None, str]:
    """``(frontmatter, body)``; ``None`` frontmatter when there is none."""
    text = text.removeprefix("﻿")
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None, text
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "".join(lines[1:i]), "".join(lines[i + 1 :])
    msg = "frontmatter starts with '---' but never closes"
    raise ValueError(msg)


def _scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return None
    if value[0] in "\"'" and value[-1] == value[0] and len(value) >= 2:
        inner = value[1:-1]
        return inner.replace('\\"', '"').replace("\\n", "\n") if value[0] == '"' else inner
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [_scalar(part) for part in _split_inline(inner)] if inner else []
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        out: dict[str, Any] = {}
        for part in _split_inline(inner) if inner else []:
            key, _, val = part.partition(":")
            out[str(_scalar(key))] = _scalar(val)
        return out
    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "~"):
        return None
    for kind in (int, float):
        try:
            return kind(value)
        except ValueError:
            continue
    # A trailing comment, as YAML has it: " #" outside quotes.
    return value.split(" #", 1)[0].rstrip()


def _split_inline(text: str) -> list[str]:
    """Split ``a, "b, c", d`` on top-level commas."""
    parts: list[str] = []
    depth, quote, current = 0, "", []
    for ch in text:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if "".join(current).strip():
        parts.append("".join(current).strip())
    return parts


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _mini_yaml(text: str) -> dict[str, Any]:
    """The YAML subset agent frontmatter uses, for when PyYAML is not installed.

    Mappings and lists by indentation, scalars, inline ``[a, b]`` and
    ``{k: v}``, quoted strings, and ``|`` / ``>`` block text. Enough for every
    agent file either harness documents; anything stranger needs PyYAML.
    """
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    value, _ = _block(lines, 0, 0)
    if not isinstance(value, dict):
        msg = "frontmatter must be a mapping of keys to values"
        raise ValueError(msg)  # noqa: TRY004 — a malformed file, reported like every other parse error
    return value


def _block(lines: list[str], start: int, indent: int) -> tuple[Any, int]:
    """Parse the block at ``indent`` from ``start``; return it and the next line."""
    if start < len(lines) and lines[start].lstrip().startswith("- "):
        items: list[Any] = []
        i = start
        while i < len(lines) and _indent(lines[i]) == indent and lines[i].lstrip().startswith("- "):
            items.append(_scalar(lines[i].lstrip()[2:]))
            i += 1
        return items, i
    mapping: dict[str, Any] = {}
    i = start
    while i < len(lines) and _indent(lines[i]) == indent:
        key, sep, rest = lines[i].strip().partition(":")
        if not sep:
            msg = f"expected 'key: value', got {lines[i].strip()!r}"
            raise ValueError(msg)
        key = str(_scalar(key))
        rest = rest.strip()
        i += 1
        if rest in ("|", ">", "|-", ">-", "|+", ">+"):
            body: list[str] = []
            while i < len(lines) and _indent(lines[i]) > indent:
                body.append(lines[i].strip())
                i += 1
            mapping[key] = ("\n" if rest.startswith("|") else " ").join(body)
        elif rest:
            mapping[key] = _scalar(rest)
        elif i < len(lines) and _indent(lines[i]) > indent:
            mapping[key], i = _block(lines, i, _indent(lines[i]))
        elif i < len(lines) and _indent(lines[i]) == indent and lines[i].lstrip().startswith("- "):
            # YAML allows a list at the key's own indentation.
            mapping[key], i = _block(lines, i, indent)
        else:
            mapping[key] = None
    if i < len(lines) and _indent(lines[i]) > indent:
        msg = f"unexpected indentation at {lines[i].strip()!r}"
        raise ValueError(msg)
    return mapping, i


def _load_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]  # noqa: PLC0415 — optional dependency
    except ImportError:
        return _mini_yaml(text)
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        msg = "frontmatter must be a mapping of keys to values"
        raise ValueError(msg)  # noqa: TRY004 — a malformed file, reported like every other parse error
    return data


def _names(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(p for p in re.split(r"[,\s]+", value) if p)
    if isinstance(value, list | tuple):
        return tuple(str(v).strip() for v in value if str(v).strip())
    msg = f"expected a list of tool names, got {type(value).__name__}"
    raise ValueError(msg)


def _tool_fields(raw: Any) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    """``(allowed, denied)`` from a ``tools`` value in any accepted shape."""
    if raw is None:
        return None, ()
    if isinstance(raw, dict):
        # opencode: every tool is on unless switched off; ``"*": false``
        # switches all off, and the ``true`` entries back on.
        on = tuple(str(k) for k, v in raw.items() if v and k != "*")
        off = tuple(str(k) for k, v in raw.items() if not v and k != "*")
        if raw.get("*") is False:
            return on, off
        return None, off
    names = _names(raw)
    # ``tools: "*"`` (or ``all``) is the default spelled out.
    if names in (("*",), ("all",)):
        return None, ()
    return names, ()


def parse_agent_markdown(
    text: str,
    *,
    name: str | None = None,
    source: str | None = None,
    default_mode: AgentMode = "all",
) -> AgentSpec:
    """Parse one agent file's text into an :class:`AgentSpec`.

    Args:
        text: The file's contents: optional frontmatter, then the prompt.
        name: The name to use when the frontmatter gives none (a loader
            passes the file's stem).
        source: Where the text came from, kept on the spec.
        default_mode: The mode when the frontmatter gives none. A harness
            reading Claude Code's ``.claude/agents`` passes ``"subagent"``,
            since that format has only subagents.

    Raises:
        ValueError: Malformed frontmatter, or a field with an invalid value.
    """
    front, body = _split_frontmatter(text)
    data = _load_yaml(front) if front is not None else {}
    spec_name = data.get("name") or name
    if not spec_name:
        msg = "agent has no name: set 'name' in the frontmatter"
        raise ValueError(msg)
    allowed, denied = _tool_fields(data.get("tools"))
    for key in _DENY_KEYS:
        denied = (*denied, *_names(data.get(key)))
    max_turns = next((data[k] for k in _TURN_KEYS if data.get(k) is not None), None)
    token_budget = next((data[k] for k in _BUDGET_KEYS if data.get(k) is not None), None)
    model = data.get("model")
    if isinstance(model, str) and model.strip().lower() in ("", "inherit", "default"):
        model = None
    metadata = {k: v for k, v in data.items() if k not in _KNOWN_KEYS}
    if data.get("disable"):
        metadata["disable"] = True
    return AgentSpec(
        name=str(spec_name),
        description=str(data.get("description") or "").strip(),
        prompt=body.strip(),
        model=None if model is None else str(model),
        tools=allowed,
        disallowed_tools=denied,
        mode=data.get("mode") or default_mode,
        max_turns=max_turns,
        token_budget=token_budget,
        temperature=data.get("temperature"),
        source=source,
        metadata=metadata,
    )


def load_agent_file(path: str | Path, *, default_mode: AgentMode = "all") -> AgentSpec:
    """Read one ``<name>.md`` agent file. The stem is the default name."""
    file = Path(path)
    return parse_agent_markdown(
        file.read_text(encoding="utf-8"),
        name=file.stem,
        source=str(file),
        default_mode=default_mode,
    )


def _log_error(path: Path, exc: Exception) -> None:
    logger.warning("skipping agent file %s: %s", path, exc)


def load_agent_specs(
    sources: Sequence[str | Path | tuple[str | Path, AgentMode]],
    *,
    on_error: Callable[[Path, Exception], None] | None = None,
) -> dict[str, AgentSpec]:
    """Load every ``*.md`` agent file from ``sources``, later sources winning.

    Pass directories from least to most specific — user-wide, then project —
    so a project's ``reviewer`` replaces the user's. A source may be a
    ``(directory, default_mode)`` pair, for a format whose files carry no
    ``mode``. A directory that does not exist is skipped; so is a file that
    does not parse (``on_error`` hears about it, by default a log warning) or
    that says ``disable: true`` — and a disabled file removes an agent of the
    same name loaded from an earlier source.

    Returns:
        Specs by name, in the order they were first defined.
    """
    report = on_error or _log_error
    specs: dict[str, AgentSpec] = {}
    for source in sources:
        directory, mode = source if isinstance(source, tuple) else (source, "all")
        root = Path(directory)
        if not root.is_dir():
            continue
        for file in sorted(root.glob("*.md")):
            try:
                spec = load_agent_file(file, default_mode=mode)
            except Exception as exc:  # noqa: BLE001 — I/O, validation, PyYAML's own errors: one bad file is not fatal
                report(file, exc)
                continue
            if spec.metadata.get("disable"):
                specs.pop(spec.name, None)
                continue
            specs[spec.name] = spec
    return specs


def merge_specs(*layers: Mapping[str, AgentSpec]) -> dict[str, AgentSpec]:
    """Combine spec maps, later layers replacing earlier ones by name."""
    merged: dict[str, AgentSpec] = {}
    for layer in layers:
        merged.update(layer)
    return merged
