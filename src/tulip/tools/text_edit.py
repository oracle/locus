# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Find-and-replace that survives the ways a model misquotes a file.

An edit tool takes ``old`` and ``new`` and swaps one for the other. Asking for
an exact match is the safe contract, and on frontier models it is enough. On
open models running unattended it is the single most common reason a run
stalls: the model quotes a line with its trailing spaces dropped, with spaces
where the file has tabs, with the indentation of the enclosing block it
imagined, or with every quote backslash-escaped one level too deep. Each miss
costs a round trip, and a model that misses twice tends to rewrite the whole
file instead.

:func:`apply_edit` tries a cascade of readings, strictest first, and stops at
the first one that finds anything:

``exact``
    The text as given.
``line_trimmed``
    Line by line, ignoring whitespace at either end of each line — indentation
    drift, tabs for spaces, trailing spaces, CRLF.
``whitespace_normalized``
    Token by token, ignoring how much whitespace sits between them — a line the
    model reflowed, ``f(a,b)`` for ``f(a, b)``.
``escape_normalized``
    The text with one level of backslash escaping removed (``\\"`` → ``"``,
    ``\\n`` → newline), for a model that escaped its arguments twice.
``block_anchor``
    Three or more lines whose first and last lines match and whose middle is
    similar enough. The loosest reading, so it runs last and never replaces
    more than one block.

Every reading must be unique. When a strategy finds more than one place, the
edit is refused as ambiguous — the cascade does not fall through to a looser
reading that happens to find one, because that is how an edit lands in the
wrong function. ``replace_all`` changes every place an exact or
whitespace-tolerant reading finds; it never applies to ``block_anchor``.

When the match was not exact, ``new`` is re-indented to the file's
indentation, so a block the model quoted at four spaces lands correctly in a
file indented with tabs.

When nothing matches, :class:`EditMatchError` carries the closest region of
the file, numbered, so the model can quote it correctly on the next try
instead of guessing again.

Pure functions over strings: no I/O, no paths, no tool decorator. File tools
call it between their own read and write.
"""

from __future__ import annotations

import bisect
import difflib
import itertools
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal


#: Minimum similarity of the middle lines for ``block_anchor`` to accept a
#: block, measured on the trimmed lines. High enough that a block with a
#: different body is not mistaken for the one the model meant, low enough that
#: a renamed variable or a reworded comment still matches.
BLOCK_ANCHOR_SIMILARITY = 0.75

#: How far, as a fraction of the quoted block's length, ``block_anchor`` lets
#: the file's block be longer or shorter. A quote missing one line of a twelve
#: line function still finds it; a quote of three lines does not swallow ten.
BLOCK_ANCHOR_LINE_DELTA = 0.25

#: Files longer than this get no closest-region hint: scoring every window of
#: a generated file costs more than the hint is worth.
HINT_MAX_FILE_LINES = 20_000

#: The closest region is shown only when it is at least this similar. Below
#: it, the "closest" text is noise and invites the model to edit the wrong
#: place.
HINT_MIN_SIMILARITY = 0.5

Reason = Literal["empty", "unchanged", "not_found", "ambiguous"]


@dataclass(frozen=True)
class Hit:
    """One place a strategy matched: a character span and what replaces it."""

    start: int
    end: int
    replacement: str


#: A strategy reads ``(content, old, new)`` and returns every place it
#: matches, with ``new`` adapted to that place.
Strategy = Callable[[str, str, str], list[Hit]]


@dataclass(frozen=True)
class EditStrategy:
    """A named reading of ``old``, and whether ``replace_all`` may use it."""

    name: str
    find: Strategy
    allows_replace_all: bool = True


@dataclass(frozen=True)
class EditOutcome:
    """What :func:`apply_edit` did."""

    #: The file content after the edit.
    content: str
    #: The name of the strategy that matched — ``"exact"`` on the common path.
    strategy: str
    #: How many places were replaced. One unless ``replace_all`` was set.
    replacements: int
    #: 1-based line of the start of each replaced region, in the original.
    lines: tuple[int, ...]
    #: Whether ``old`` and ``new`` were given the file's CRLF line ends.
    crlf: bool = False

    @property
    def exact(self) -> bool:
        """Whether ``old`` matched as given."""
        return self.strategy == "exact"


class EditMatchError(ValueError):
    """``old`` could not be placed in the content.

    The message is written for a model to act on. :attr:`reason` is for code
    that wants to word it differently; :attr:`lines` names the places an
    ambiguous match found; :attr:`hint` is the closest region of the file when
    nothing matched, or ``None`` when nothing was close.
    """

    def __init__(
        self,
        reason: Reason,
        message: str,
        *,
        lines: Sequence[int] = (),
        hint: str | None = None,
        strategy: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason: Reason = reason
        self.lines: tuple[int, ...] = tuple(lines)
        self.hint = hint
        self.strategy = strategy


# ----------------------------------------------------------------- helpers --


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _split_old(old: str) -> tuple[list[str], bool]:
    """``old`` as lines, and whether it ended with a newline.

    A quote that ends in a newline means "these whole lines". Splitting would
    leave an empty last line that matches nothing, so it is dropped and
    remembered instead.
    """
    lines = old.split("\n")
    if len(lines) > 1 and lines[-1] == "":
        return lines[:-1], True
    return lines, False


def _line_starts(lines: Sequence[str]) -> list[int]:
    starts = [0]
    for line in lines[:-1]:
        starts.append(starts[-1] + len(line) + 1)
    return starts


def _block_hit(
    content: str,
    lines: Sequence[str],
    starts: Sequence[int],
    first: int,
    last: int,
    replacement: str,
    *,
    whole_lines: bool,
) -> Hit:
    """The span of lines ``first..last``, and ``replacement`` fitted to it.

    When the quote ended with a newline the span takes the file's newline too,
    so ``new``'s own trailing newline replaces it rather than doubling it. At
    the end of a file with no final newline there is none to take, so ``new``
    gives up its own.
    """
    start = starts[first]
    end = starts[last] + len(lines[last])
    if not whole_lines and lines[last].endswith("\r"):
        # The quote stopped before the line break; so does the span, or the
        # file's ``\r`` would go and leave a bare ``\n``.
        end -= 1
    if whole_lines:
        if end < len(content):
            end += 1
        elif replacement.endswith("\n"):
            replacement = replacement[: -2 if replacement.endswith("\r\n") else -1]
    return Hit(start, end, replacement)


def _indent_unit(indents: Sequence[str]) -> int:
    """The width of one indent level among ``indents``, four when unclear."""
    widths = sorted({len(i) for i in indents if i})
    steps = [b - a for a, b in itertools.pairwise(widths)]
    if steps:
        return min(steps)
    if widths and widths[0] % 4 != 0:
        return widths[0]
    return 4


def _convert(rest: str, *, to_tabs: bool, unit: int) -> str:
    if to_tabs:
        return rest.replace(" " * unit, "\t")
    return rest.replace("\t", " " * unit)


def _reindent(
    pairs: Sequence[tuple[str, str]],
    new: str,
    *,
    skip_first: bool = False,
) -> str:
    """``new``, with the indentation the model used mapped onto the file's.

    ``pairs`` are (quoted line, file line) that matched each other. Their
    indents give a mapping — model's ``"    "`` is the file's ``"\\t"`` — and
    each line of ``new`` is re-indented through it: by exact indent where the
    mapping has one, by the longest matching prefix otherwise, with the rest
    converted between tabs and spaces when the two disagree. An inconsistent
    mapping (one quoted indent standing for two file indents) leaves ``new``
    alone: a guess there would be worse than the model's own text.
    """
    mapping: dict[str, str] = {}
    for quoted, actual in pairs:
        if not quoted.strip() or not actual.strip():
            continue
        if mapping.setdefault(_indent(quoted), _indent(actual)) != _indent(actual):
            return new
    if all(k == v for k, v in mapping.items()):
        return new
    model_tabs = any("\t" in k for k in mapping)
    file_tabs = any("\t" in v for v in mapping.values())
    to_tabs = file_tabs and not model_tabs
    to_spaces = model_tabs and not file_tabs
    unit = _indent_unit(list(mapping) if to_tabs else list(mapping.values()))
    keys = sorted(mapping, key=len, reverse=True)
    out: list[str] = []
    for index, line in enumerate(new.split("\n")):
        if (skip_first and index == 0) or not line.strip():
            out.append(line)
            continue
        ws = _indent(line)
        prefix = next((k for k in keys if ws.startswith(k)), None)
        if prefix is None:
            out.append(line)
            continue
        rest = ws[len(prefix) :]
        if to_tabs or to_spaces:
            rest = _convert(rest, to_tabs=to_tabs, unit=unit)
        out.append(mapping[prefix] + rest + line[len(ws) :])
    return "\n".join(out)


def _trim_like(old: str, new: str) -> str:
    """Drop from ``new`` the whitespace ``old`` had at its ends.

    A token-level match covers only the tokens, so leading indentation and a
    trailing newline in the quote are not part of the span. When ``new`` has
    the same ends, they go too, or they would be inserted twice.
    """
    lead = old[: len(old) - len(old.lstrip())]
    trail = old[len(old.rstrip()) :]
    if len(new) >= len(lead) + len(trail) and new.startswith(lead) and new.endswith(trail):
        return new[len(lead) : len(new) - len(trail)]
    return new


# -------------------------------------------------------------- strategies --


def exact(content: str, old: str, new: str) -> list[Hit]:
    """Every non-overlapping occurrence of ``old`` as given."""
    hits: list[Hit] = []
    at = content.find(old)
    while at != -1:
        hits.append(Hit(at, at + len(old), new))
        at = content.find(old, at + len(old))
    return hits


def line_trimmed(content: str, old: str, new: str) -> list[Hit]:
    """Whole lines that equal the quote once each line is stripped."""
    quoted, whole_lines = _split_old(old)
    want = [q.strip() for q in quoted]
    if not any(want):
        return []
    lines = content.split("\n")
    starts = _line_starts(lines)
    n = len(want)
    hits: list[Hit] = []
    i = 0
    while i <= len(lines) - n:
        if all(lines[i + k].strip() == want[k] for k in range(n)):
            block = lines[i : i + n]
            fitted = _reindent(list(zip(quoted, block, strict=True)), new)
            hits.append(
                _block_hit(content, lines, starts, i, i + n - 1, fitted, whole_lines=whole_lines)
            )
            i += n
        else:
            i += 1
    return hits


_TOKEN = re.compile(r"\w+|[^\w\s]")


def whitespace_normalized(content: str, old: str, new: str) -> list[Hit]:
    """The quote's tokens in order, with any amount of whitespace between.

    Two words the quote separated must stay separated (``\\s+``); anything
    next to punctuation may gain or lose whitespace (``\\s*``). The match must
    not start or end inside a word, so ``x=1`` does not land inside
    ``max = 10``.
    """
    tokens = _TOKEN.findall(old)
    if len(tokens) < 2:
        return []
    parts: list[str] = []
    for prev, token in zip([None, *tokens], tokens, strict=False):
        if prev is not None:
            words = prev[-1].isalnum() or prev[-1] == "_"
            parts.append(r"\s+" if words and (token[0].isalnum() or token[0] == "_") else r"\s*")
        parts.append(re.escape(token))
    pattern = "".join(parts)
    if tokens[0][0].isalnum() or tokens[0][0] == "_":
        pattern = r"(?<!\w)" + pattern
    if tokens[-1][-1].isalnum() or tokens[-1][-1] == "_":
        pattern += r"(?!\w)"
    trimmed_new = _trim_like(old, new)
    # The quote's own lines, blank ones at either end dropped but indentation
    # kept: the first line's indent is part of the mapping onto the file.
    quoted = old.split("\n")
    while not quoted[0].strip():
        quoted.pop(0)
    while not quoted[-1].strip():
        quoted.pop()
    lines = content.split("\n")
    starts = _line_starts(lines)
    hits: list[Hit] = []
    for m in re.finditer(pattern, content):
        first = bisect.bisect_right(starts, m.start()) - 1
        last = bisect.bisect_right(starts, max(m.end() - 1, m.start())) - 1
        replacement = trimmed_new
        if last - first + 1 == len(quoted):
            pairs = list(zip(quoted, lines[first : last + 1], strict=True))
            replacement = _reindent(pairs, trimmed_new, skip_first=True)
        hits.append(Hit(m.start(), m.end(), replacement))
    return hits


_ESCAPE = re.compile(r"\\([nrt'\"`\\$])")
_UNESCAPED = {"n": "\n", "r": "\r", "t": "\t"}


def _unescape(text: str) -> str:
    return _ESCAPE.sub(lambda m: _UNESCAPED.get(m.group(1), m.group(1)), text)


def escape_normalized(content: str, old: str, new: str) -> list[Hit]:
    """The quote with one level of backslash escaping removed.

    A model that escaped ``old`` twice escaped ``new`` the same way, so
    ``new`` is unescaped with it — otherwise the edit would write the literal
    backslashes into the file. Only applies when unescaping changes the quote
    and the file does not already contain it as given.
    """
    plain = _unescape(old)
    if plain == old:
        return []
    plain_new = _unescape(new)
    return exact(content, plain, plain_new) or line_trimmed(content, plain, plain_new)


def block_anchor(content: str, old: str, new: str) -> list[Hit]:
    """Blocks whose first and last lines match and whose middle is similar.

    For a quote of three or more lines, where the model has the edges right
    and a line or two of the body wrong — a stale comment, a renamed local.
    The file's block may be up to a quarter longer or shorter. Every block
    above :data:`BLOCK_ANCHOR_SIMILARITY` is returned, so two similar blocks
    are refused as ambiguous rather than the likelier one being guessed.
    """
    quoted, whole_lines = _split_old(old)
    n = len(quoted)
    head, tail = quoted[0].strip(), quoted[-1].strip()
    # Anchors that are a lone brace or blank match everywhere; they say
    # nothing about where the block is.
    if n < 3 or len(head) < 3 or not tail:
        return []
    lines = content.split("\n")
    starts = _line_starts(lines)
    delta = max(1, int(n * BLOCK_ANCHOR_LINE_DELTA))
    middle = "\n".join(q.strip() for q in quoted[1:-1])
    hits: list[Hit] = []
    for i, line in enumerate(lines):
        if line.strip() != head:
            continue
        for j in range(i + max(2, n - 1 - delta), min(len(lines), i + n + delta)):
            if lines[j].strip() != tail:
                continue
            body = "\n".join(x.strip() for x in lines[i + 1 : j])
            if difflib.SequenceMatcher(None, body, middle, autojunk=False).ratio() < (
                BLOCK_ANCHOR_SIMILARITY
            ):
                continue
            block = lines[i : j + 1]
            pairs = (
                list(zip(quoted, block, strict=True))
                if len(block) == n
                else [(quoted[0], block[0]), (quoted[-1], block[-1])]
            )
            hits.append(
                _block_hit(
                    content, lines, starts, i, j, _reindent(pairs, new), whole_lines=whole_lines
                )
            )
            break
    return hits


#: The cascade, strictest first. Pass a different sequence to
#: :func:`apply_edit` to add a reading or leave one out.
DEFAULT_STRATEGIES: tuple[EditStrategy, ...] = (
    EditStrategy("exact", exact),
    EditStrategy("line_trimmed", line_trimmed),
    EditStrategy("whitespace_normalized", whitespace_normalized),
    EditStrategy("escape_normalized", escape_normalized),
    EditStrategy("block_anchor", block_anchor, allows_replace_all=False),
)


# ------------------------------------------------------------------- hint --


def closest_region(content: str, old: str, *, context: int = 2) -> str | None:
    """The part of ``content`` most like ``old``, numbered, or ``None``.

    For the error a model reads when nothing matched: the file's actual text
    where it most likely meant, so the retry quotes that rather than another
    guess. Candidate positions come from lines similar to the quote's lines;
    each candidate window is then scored as a whole.
    """
    lines = content.split("\n")
    quoted, _ = _split_old(old)
    keys = [(k, q.strip()) for k, q in enumerate(quoted) if q.strip()][:8]
    if not keys or len(lines) > HINT_MAX_FILE_LINES:
        return None
    stripped = [line.strip() for line in lines]
    # Votes use the cheap bounds only. Scoring every line properly costs
    # seconds on a long file of similar lines; the windows that collect the
    # most votes are scored properly below.
    votes: defaultdict[int, float] = defaultdict(float)
    matcher = difflib.SequenceMatcher(None, autojunk=False)
    for k, key in keys:
        matcher.set_seq2(key)
        for index, line in enumerate(stripped):
            if not line:
                continue
            matcher.set_seq1(line)
            if matcher.real_quick_ratio() < HINT_MIN_SIMILARITY:
                continue
            score = matcher.quick_ratio()
            if score >= HINT_MIN_SIMILARITY:
                votes[max(0, index - k)] += score
    if not votes:
        return None
    n = len(quoted)
    target = "\n".join(q.strip() for q in quoted)
    best, best_score = -1, 0.0
    for start in sorted(votes, key=votes.__getitem__, reverse=True)[:25]:
        window = "\n".join(stripped[start : start + n])
        score = difflib.SequenceMatcher(None, window, target, autojunk=False).ratio()
        if score > best_score:
            best, best_score = start, score
    if best < 0 or best_score < HINT_MIN_SIMILARITY:
        return None
    end = min(len(lines), best + n)
    lo, hi = max(0, best - context), min(len(lines), end + context)
    shown = "\n".join(f"{i + 1:6d}\t{lines[i].rstrip(chr(13))}" for i in range(lo, hi))
    return (
        f"closest match is lines {best + 1}-{end} ({best_score:.0%} similar)"
        f" — copy the text from there:\n{shown}"
    )


# ---------------------------------------------------------------- cascade --


def _distinct(hits: Sequence[Hit]) -> list[Hit]:
    out: list[Hit] = []
    for hit in sorted(hits, key=lambda h: (h.start, h.end)):
        if out and hit.start == out[-1].start and hit.end == out[-1].end:
            continue
        out.append(hit)
    return out


def _uses_crlf(content: str, old: str) -> bool:
    """Whether to quote ``old`` with CRLF line ends to match the file.

    A model never sends ``\\r``; a file checked out on Windows has one on every
    line. Decided by majority, so one stray CRLF in an LF file changes nothing.
    """
    if "\r" in old or "\n" not in old:
        return False
    crlf = content.count("\r\n")
    return crlf > 0 and crlf * 2 >= content.count("\n")


def apply_edit(
    content: str,
    old: str,
    new: str,
    *,
    replace_all: bool = False,
    strategies: Sequence[EditStrategy] = DEFAULT_STRATEGIES,
) -> EditOutcome:
    """Replace ``old`` with ``new`` in ``content``, tolerating near misses.

    Tries each strategy in order and applies the first that matches. Raises
    :class:`EditMatchError` when ``old`` is empty or equal to ``new``, when
    nothing matches (with the closest region as :attr:`EditMatchError.hint`),
    when a strategy matches more than one place and ``replace_all`` is not
    set (or the strategy does not allow it).
    """
    if not old:
        msg = "old text is empty — quote the text to replace, or write the whole file"
        raise EditMatchError("empty", msg)
    if old == new:
        msg = "old and new text are identical — there is nothing to change"
        raise EditMatchError("unchanged", msg)
    crlf = _uses_crlf(content, old)
    if crlf:
        old = old.replace("\n", "\r\n")
        new = new.replace("\r\n", "\n").replace("\n", "\r\n")
    starts = _line_starts(content.split("\n"))
    for strategy in strategies:
        hits = _distinct(strategy.find(content, old, new))
        if not hits:
            continue
        where = tuple(bisect.bisect_right(starts, h.start) for h in hits)
        overlapping = any(a.end > b.start for a, b in itertools.pairwise(hits))
        if len(hits) > 1 and (not replace_all or not strategy.allows_replace_all or overlapping):
            listed = ", ".join(str(n) for n in where[:10])
            more = "" if len(where) <= 10 else f" and {len(where) - 10} more"
            advice = "include more surrounding lines so it is unique"
            if strategy.allows_replace_all and not overlapping:
                advice += ", or set replace_all to change every one"
            msg = (
                f"old text matches {len(hits)} times (by {strategy.name}) at lines "
                f"{listed}{more} — {advice}"
            )
            raise EditMatchError("ambiguous", msg, lines=where, strategy=strategy.name)
        out = content
        for hit in reversed(hits):
            out = out[: hit.start] + hit.replacement + out[hit.end :]
        return EditOutcome(
            content=out,
            strategy=strategy.name,
            replacements=len(hits),
            lines=where,
            crlf=crlf,
        )
    hint = closest_region(content, old)
    msg = (
        "old text not found — it must match the file's text; indentation, "
        "surrounding whitespace and escaping are forgiven, the words are not"
    )
    if hint:
        msg += f"\n{hint}"
    else:
        msg += ". Read the file again before retrying."
    raise EditMatchError("not_found", msg, hint=hint)


__all__ = [
    "BLOCK_ANCHOR_LINE_DELTA",
    "BLOCK_ANCHOR_SIMILARITY",
    "DEFAULT_STRATEGIES",
    "EditMatchError",
    "EditOutcome",
    "EditStrategy",
    "Hit",
    "Strategy",
    "apply_edit",
    "block_anchor",
    "closest_region",
    "escape_normalized",
    "exact",
    "line_trimmed",
    "whitespace_normalized",
]
