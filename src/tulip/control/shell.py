# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""What a shell command line will actually run.

A gate that decides about shell commands has to know which programs a line
starts. Matching patterns against the raw text gets this wrong in both
directions, and both directions have been observed:

- **Refusing what is harmless.** ``pytest -k shutdown`` and
  ``git commit -m "handle shutdown"`` mention a dangerous word without running
  it. A deny list matched against the text refuses both, and a gate that
  refuses ordinary work gets switched off.
- **Allowing what is not.** ``find . -exec rm -rf {} ;`` starts with a
  read-only program, and so does ``env rm -rf build``. An allow list anchored
  at the start of the line waves both through.

:func:`parse_command` splits a line into the simple commands it runs — across
``;``, ``&&``, ``||``, pipes and newlines, inside ``$(...)``, backticks and
``<(...)``, behind ``sudo``/``env``/``timeout``-style wrappers, after
``find -exec``, ``xargs``, ``sh -c`` and ``eval`` — and says which program each
one is. Quoted text stays a word, so a commit message is never mistaken for a
command.

It is a seatbelt, not a shell. A line it cannot tokenise (an unbalanced quote)
comes back with ``parsed=False``, and callers are expected to fail closed on
that: treat it as unknown rather than as harmless.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field


__all__ = ["ShellCommand", "SimpleCommand", "parse_command"]

#: Deeper than this, nested ``sh -c "sh -c ..."`` stops being unpacked and the
#: line is reported as not parsed. Nobody writes eight levels on purpose.
_MAX_DEPTH = 8

#: Shells whose ``-c`` argument is itself a command line.
SHELLS = frozenset({"sh", "bash", "zsh", "ksh", "dash", "ash", "mksh", "fish", "busybox"})

#: Words that run the command after them: ``sudo rm`` runs ``rm``.
_WRAPPERS = frozenset(
    {
        "sudo",
        "doas",
        "env",
        "nohup",
        "time",
        "command",
        "builtin",
        "exec",
        "nice",
        "ionice",
        "stdbuf",
        "timeout",
        "unbuffer",
        "chronic",
        "setsid",
    }
)

#: Options of a wrapper that take a value, so the value is not taken for the
#: wrapped program's name.
_OPTION_VALUES: dict[str, frozenset[str]] = {
    "sudo": frozenset({"-u", "-g", "-h", "-p", "-C", "-D", "-r", "-t", "-U", "-T", "-R"}),
    "doas": frozenset({"-u", "-C"}),
    "env": frozenset({"-u", "-C", "--unset", "--chdir"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "ionice": frozenset({"-c", "-n", "-p", "--class", "--classdata"}),
    "timeout": frozenset({"-s", "-k", "--signal", "--kill-after"}),
    "xargs": frozenset({"-I", "-n", "-P", "-L", "-l", "-s", "-d", "-E", "-e", "-a", "--max-args"}),
}

#: ``find`` actions that run a program, up to a ``;`` or ``+``.
_FIND_EXEC = frozenset({"-exec", "-execdir", "-ok", "-okdir"})

_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PUNCTUATION = frozenset("();<>|&")
#: Stands in for a substitution's text in the outer line, so the outer line
#: tokenises as the shell would see it before expansion.
_PLACEHOLDER = "__tulip_substitution__"


@dataclass(frozen=True)
class SimpleCommand:
    """One program the line runs, with its arguments.

    Attributes:
        argv: The program and its arguments, with assignments (``A=1``) and
            wrappers (``sudo``, ``env``, ``timeout 5``…) taken off the front.
            ``argv[0]`` is what will actually run.
        words: The words as written, wrappers included, redirections not.
        wrappers: The wrappers taken off, in order.
        redirects: ``(operator, target)`` pairs: ``(">", "out.txt")``.
        connector: The operator that joined this command to the one before
            it: ``""`` for the first, else ``;``, ``&&``, ``||``, ``|``, ``&``.
        pipeline: Commands sharing a number are one pipeline.
        origin: How it was reached: ``""`` at top level, else
            ``"substitution"``, ``"exec"`` (find), ``"xargs"``, ``"shell -c"``
            or ``"eval"``.
    """

    argv: tuple[str, ...]
    words: tuple[str, ...]
    wrappers: tuple[str, ...] = ()
    redirects: tuple[tuple[str, str], ...] = ()
    connector: str = ""
    pipeline: int = 0
    origin: str = ""

    @property
    def name(self) -> str:
        """The program's bare name — ``/usr/bin/rm`` and ``rm`` are both ``rm``."""
        return _basename(self.argv[0]) if self.argv else ""

    @property
    def text(self) -> str:
        """The command as written, words joined by single spaces."""
        return " ".join(self.words)

    @property
    def nested(self) -> bool:
        """Whether something else on the line runs this one."""
        return bool(self.origin)

    @property
    def writes_files(self) -> bool:
        """Whether a redirection writes somewhere other than ``/dev/null``.

        ``2>&1`` duplicates a descriptor and writes nothing new.
        """
        for op, target in self.redirects:
            if ">" not in op:
                continue
            if target == "/dev/null":
                continue
            if op.endswith("&") and (target.isdigit() or target == "-"):
                continue
            return True
        return False


@dataclass(frozen=True)
class ShellCommand:
    """A command line, split into the simple commands it runs.

    Attributes:
        source: The line as given.
        commands: Every simple command, nested ones included, in the order
            they appear.
        parsed: ``False`` when the line (or something nested in it) could not
            be tokenised. Treat such a line as unknown, never as harmless.
        has_substitution: Whether ``$(...)``, backticks or ``<(...)`` occur.
    """

    source: str
    commands: tuple[SimpleCommand, ...] = field(default_factory=tuple)
    parsed: bool = True
    has_substitution: bool = False

    @property
    def top_level(self) -> tuple[SimpleCommand, ...]:
        """The commands the line runs directly, not via another command."""
        return tuple(c for c in self.commands if not c.nested)

    @property
    def is_single(self) -> bool:
        """One plain command: nothing chained, piped, nested or redirected."""
        return (
            self.parsed
            and not self.has_substitution
            and len(self.commands) == 1
            and not self.commands[0].redirects
        )


def parse_command(source: str) -> ShellCommand:
    """Split ``source`` into the simple commands it runs. Never raises."""
    return _parse(source, depth=0, origin="", pipeline_base=0)


# ----------------------------------------------------------------- parsing --


def _parse(source: str, *, depth: int, origin: str, pipeline_base: int) -> ShellCommand:
    if depth > _MAX_DEPTH:
        return ShellCommand(source=source, parsed=False)
    scanned = _scan(source)
    tokens = _tokenise(scanned.outer)
    if tokens is None or not scanned.ok:
        return ShellCommand(
            source=source, parsed=False, has_substitution=bool(scanned.substitutions)
        )

    commands: list[SimpleCommand] = []
    parsed = True
    nested_strings: list[tuple[str, str]] = [(s, "substitution") for s in scanned.substitutions]
    pipeline = pipeline_base
    for words, redirects, connector in _segments(tokens):
        if connector not in ("|", "|&"):
            pipeline += 1
        argv, wrappers, nested_argvs, nested_lines = _unwrap(words)
        if not argv and not wrappers and not words:
            continue
        commands.append(
            SimpleCommand(
                argv=tuple(argv),
                words=tuple(words),
                wrappers=tuple(wrappers),
                redirects=tuple(redirects),
                connector=connector,
                pipeline=pipeline,
                origin=origin,
            )
        )
        for how, nested in nested_argvs:
            pipeline += 1
            inner, inner_wrappers, more_argvs, more_lines = _unwrap(nested)
            commands.append(
                SimpleCommand(
                    argv=tuple(inner),
                    words=tuple(nested),
                    wrappers=tuple(inner_wrappers),
                    pipeline=pipeline,
                    origin=how,
                )
            )
            # ``xargs sh -c '...'`` and ``find -exec sh -c '...'``: one more level.
            nested_lines.extend(more_lines)
            for more_how, more in more_argvs:
                pipeline += 1
                commands.append(
                    SimpleCommand(
                        argv=tuple(more), words=tuple(more), pipeline=pipeline, origin=more_how
                    )
                )
        nested_strings.extend(nested_lines)

    for line, how in nested_strings:
        sub = _parse(line, depth=depth + 1, origin=how, pipeline_base=pipeline + 1000)
        parsed = parsed and sub.parsed
        commands.extend(sub.commands)
        pipeline = max([pipeline, *(c.pipeline for c in sub.commands)])

    return ShellCommand(
        source=source,
        commands=tuple(commands),
        parsed=parsed,
        has_substitution=bool(scanned.substitutions),
    )


@dataclass
class _Scanned:
    outer: str
    substitutions: list[str]
    ok: bool


def _scan(text: str) -> _Scanned:  # noqa: PLR0912, PLR0915 — one pass over a small grammar
    """Lift substitutions out of ``text``, drop comments, honour newlines.

    Quote-aware: inside single quotes nothing is special; inside double quotes
    ``$(...)`` and backticks still run, exactly as in a shell. A heredoc body
    is skipped, except for the substitutions an unquoted delimiter expands.
    """
    out: list[str] = []
    subs: list[str] = []
    ok = True
    quote: str | None = None
    word_start = True
    heredocs: list[tuple[str, bool, bool]] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if quote == "'":
            out.append(c)
            if c == "'":
                quote = None
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            if text[i + 1] == "\n":
                i += 2
                continue
            out.append(text[i : i + 2])
            i += 2
            word_start = False
            continue
        lifted = _substitution_at(text, i, unquoted=quote is None)
        if lifted is not None:
            inner, end = lifted
            if end < 0:
                ok = False
                break
            subs.append(inner)
            out.append(_PLACEHOLDER)
            i = end
            word_start = False
            continue
        if quote == '"':
            if c == '"':
                quote = None
            out.append(c)
            i += 1
            continue
        if c in "'\"":
            quote = c
            out.append(c)
            i += 1
            word_start = False
            continue
        if c == "#" and word_start:
            while i < n and text[i] != "\n":
                i += 1
            continue
        if text.startswith("<<", i) and not text.startswith("<<<", i):
            parsed = _heredoc_header(text, i + 2)
            if parsed is not None:
                delimiter, strip_tabs, expand, end = parsed
                heredocs.append((delimiter, strip_tabs, expand))
                out.append(" << " + shlex.quote(delimiter) + " ")
                i = end
                word_start = True
                continue
        if c == "\n":
            out.append(" ; ")
            i += 1
            for delimiter, strip_tabs, expand in heredocs:
                body, i = _heredoc_body(text, i, delimiter, strip_tabs)
                if expand:
                    subs.extend(_scan_expansions(body))
            heredocs = []
            word_start = True
            continue
        out.append(c)
        word_start = c in " \t" or c in _PUNCTUATION
        i += 1
    if quote is not None:
        ok = False
    return _Scanned(outer="".join(out), substitutions=subs, ok=ok)


def _substitution_at(text: str, i: int, *, unquoted: bool) -> tuple[str, int] | None:
    """``(inner, end)`` for a substitution starting at ``i``, or ``None``.

    ``end`` is -1 when it never closes. ``$((`` is arithmetic and is left alone.
    """
    if text.startswith("$(", i) and not text.startswith("$((", i):
        end = _closing_paren(text, i + 2)
        return (text[i + 2 : end - 1], end) if end > 0 else ("", -1)
    if unquoted and (text.startswith("<(", i) or text.startswith(">(", i)):
        end = _closing_paren(text, i + 2)
        return (text[i + 2 : end - 1], end) if end > 0 else ("", -1)
    if text[i] == "`":
        j = i + 1
        while j < len(text):
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == "`":
                return text[i + 1 : j], j + 1
            j += 1
        return "", -1
    return None


def _closing_paren(text: str, start: int) -> int:
    """Index just past the ``)`` closing a group opened before ``start``; -1 if none."""
    depth = 1
    quote: str | None = None
    j = start
    while j < len(text):
        c = text[j]
        if quote == "'":
            if c == "'":
                quote = None
        elif c == "\\":
            j += 1
        elif quote == '"':
            if c == '"':
                quote = None
        elif c in "'\"":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return -1


def _heredoc_header(text: str, i: int) -> tuple[str, bool, bool, int] | None:
    """``(delimiter, strip_tabs, expands, end)`` for ``<<[-] WORD`` at ``i``."""
    strip_tabs = text.startswith("-", i)
    if strip_tabs:
        i += 1
    while i < len(text) and text[i] in " \t":
        i += 1
    match = re.match(r"""(['"])(.*?)\1|([^\s;&|<>()]+)""", text[i:])
    if match is None:
        return None
    if match.group(1):
        return match.group(2), strip_tabs, False, i + match.end()
    word = match.group(3)
    expands = "\\" not in word and "'" not in word and '"' not in word
    return word.replace("\\", ""), strip_tabs, expands, i + match.end()


def _heredoc_body(text: str, i: int, delimiter: str, strip_tabs: bool) -> tuple[str, int]:
    """The body of a heredoc starting at ``i``, and the index after its end line."""
    lines: list[str] = []
    while i < len(text):
        end = text.find("\n", i)
        line = text[i:] if end < 0 else text[i:end]
        i = len(text) if end < 0 else end + 1
        if (line.lstrip("\t") if strip_tabs else line) == delimiter:
            break
        lines.append(line)
    return "\n".join(lines), i


def _scan_expansions(body: str) -> list[str]:
    """Substitutions in an expanding heredoc body, which runs them like ``"..."``."""
    found: list[str] = []
    i = 0
    while i < len(body):
        if body[i] == "\\":
            i += 2
            continue
        lifted = _substitution_at(body, i, unquoted=False)
        if lifted is None:
            i += 1
            continue
        inner, end = lifted
        if end < 0:
            break
        found.append(inner)
        i = end
    return found


def _tokenise(outer: str) -> list[str] | None:
    lexer = shlex.shlex(outer, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return None


def _is_operator(token: str) -> bool:
    return bool(token) and all(ch in _PUNCTUATION for ch in token)


def _segments(tokens: list[str]) -> list[tuple[list[str], list[tuple[str, str]], str]]:
    """``(words, redirects, connector)`` per simple command, in order."""
    out: list[tuple[list[str], list[tuple[str, str]], str]] = []
    words: list[str] = []
    redirects: list[tuple[str, str]] = []
    connector = ""
    in_exec = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if in_exec and token in (";", "+"):
            # `find -exec rm {} \;`: the `;` ends the -exec, not the command.
            words.append(token)
            in_exec = False
            i += 1
            continue
        if _is_operator(token) and ("<" in token or ">" in token):
            if words and words[-1].isdigit():
                words.pop()  # the descriptor in `2>`, not an argument
            target = tokens[i + 1] if i + 1 < len(tokens) else ""
            redirects.append((token.strip("()"), target))
            i += 2
            continue
        if _is_operator(token):
            if words or redirects:
                out.append((words, redirects, connector))
            words, redirects = [], []
            # `(`/`)` group commands; what joins the next one is the last
            # real operator in the run.
            stripped = token.strip("()")
            connector = stripped or (connector if token.startswith(")") else ";")
            i += 1
            continue
        words.append(token)
        if token in _FIND_EXEC:
            in_exec = True
        i += 1
    if words or redirects:
        out.append((words, redirects, connector))
    return out


def _skip_options(words: list[str], i: int, takes_value: frozenset[str]) -> int:
    """Index of the first word after a wrapper's options, starting at ``i``."""
    while i < len(words):
        word = words[i]
        if word == "--":
            return i + 1
        if not word.startswith("-") or word == "-":
            return i
        i += 2 if word in takes_value else 1
    return i


def _unwrap(  # noqa: PLR0912 — one branch per wrapper family
    words: list[str],
) -> tuple[list[str], list[str], list[tuple[str, list[str]]], list[tuple[str, str]]]:
    """``(argv, wrappers, nested argvs, nested lines)`` for one command's words."""
    wrappers: list[str] = []
    nested_argvs: list[tuple[str, list[str]]] = []
    nested_lines: list[tuple[str, str]] = []
    i = 0
    while i < len(words):
        word = words[i]
        base = _basename(word)
        if _ASSIGNMENT.match(word) or word in ("{", "}", "!"):
            i += 1
            continue
        if base not in _WRAPPERS and base != "xargs":
            break
        wrappers.append(base)
        if base == "env":
            i = _skip_options(words, i + 1, _OPTION_VALUES["env"])
            while i < len(words) and _ASSIGNMENT.match(words[i]):
                i += 1
            continue
        if base == "timeout":
            i = _skip_options(words, i + 1, _OPTION_VALUES["timeout"]) + 1  # the duration
            continue
        if base == "xargs":
            start = _skip_options(words, i + 1, _OPTION_VALUES["xargs"])
            if start < len(words):
                nested_argvs.append(("xargs", words[start:]))
            return [], wrappers, nested_argvs, nested_lines
        i = _skip_options(words, i + 1, _OPTION_VALUES.get(base, frozenset()))
    argv = words[i:]
    if not argv:
        return argv, wrappers, nested_argvs, nested_lines
    name = _basename(argv[0])
    if name == "find":
        j = 1
        while j < len(argv):
            if argv[j] in _FIND_EXEC:
                end = j + 1
                while end < len(argv) and argv[end] not in (";", "+"):
                    end += 1
                if end > j + 1:
                    nested_argvs.append(("exec", argv[j + 1 : end]))
                j = end
            j += 1
    elif name in SHELLS:
        script = _shell_script(argv)
        if script is not None:
            nested_lines.append((script, "shell -c"))
    elif name == "eval" and len(argv) > 1:
        nested_lines.append((" ".join(argv[1:]), "eval"))
    return argv, wrappers, nested_argvs, nested_lines


def _basename(word: str) -> str:
    """``/usr/bin/rm`` → ``rm``: a program is the same program by any path."""
    return word.rsplit("/", 1)[-1]


def _shell_script(argv: list[str]) -> str | None:
    """The script a shell is handed with ``-c`` (``-c``, ``-lc``, ``-ec``…)."""
    for j, word in enumerate(argv[1:], start=1):
        if word.startswith("-") and not word.startswith("--") and "c" in word[1:]:
            return argv[j + 1] if j + 1 < len(argv) else ""
        if not word.startswith("-"):
            return None
    return None
