# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The numbering note in ``examples/README.md`` has to stay true.

The note exists because a numbered series that opens at 06 and skips seven
numbers reads as abandoned, when in fact those files never existed. Saying so
costs nothing and breaks nothing — where renumbering would break every link,
bookmark and in-prose "Notebook 26" reference to fix an appearance.

But the note quotes the actual ranges, so adding one notebook makes it wrong,
and a README that confidently states a stale range is worse than one that says
nothing. This recomputes the ranges from the directory and fails if the two
disagree.
"""

from __future__ import annotations

import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _numbers() -> list[int]:
    return sorted(
        int(re.search(r"notebook_(\d+)", path.name).group(1))
        for path in EXAMPLES.glob("notebook_*.py")
    )


def _runs(numbers: list[int]) -> str:
    """``06-09 · 11-40 · …`` — contiguous stretches, the way the note says it."""
    parts, start = [], numbers[0]
    for current, following in zip(numbers, [*numbers[1:], None], strict=True):
        if following != (current + 1):
            parts.append(f"{start:02d}" if start == current else f"{start:02d}-{current:02d}")
            start = following
    return " · ".join(parts)


def test_the_extractor_finds_the_series() -> None:
    """A check that silently matches nothing passes for the wrong reason."""
    assert len(_numbers()) > 50


def test_the_readme_states_the_real_ranges() -> None:
    readme = (EXAMPLES / "README.md").read_text()
    expected = _runs(_numbers())

    assert expected in readme, (
        f"examples/README.md says the series runs something other than {expected!r}. "
        "Adding or removing a notebook changes this; update the note rather than "
        "leaving it to say something that is no longer true."
    )


def test_numbers_are_unique() -> None:
    """The note promises numbers are never reused. That has to hold."""
    numbers = _numbers()

    assert len(numbers) == len(set(numbers))
