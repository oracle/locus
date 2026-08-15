# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Cross-references between notebooks have to point at what they say.

The notebooks name their agents — STEWARD, RIGHTSIZER, LEDGER — and refer to
each other by number and name: *"Notebook 27 (RIGHTSIZER)"*. Three of those
references named an agent the target notebook does not contain, so a reader
who followed one arrived somewhere else and had no way to tell whether the
notebook had changed or they had misread:

    notebook_24  "Notebook 26: there, PILOT routes work centrally"   → STEWARD
    notebook_27  "the worker MARSHAL hands tasks to"                 → STEWARD
    notebook_70  "Notebook 27 (CURATOR)"                             → RIGHTSIZER

Notebook 24 also called PILOT — its *own* war room — "the named on-call
commander built in Notebook 26", when notebook 26 builds a privacy officer.
These are leftovers from renames applied to the notebooks and not to the
sentences pointing at them, and prose is where that rot is invisible: nothing
imports a docstring, so nothing breaks.

What this covers, and what it does not. It catches a reference that names the
target — the parenthesised form unconditionally, the inline form when the
target has a codename to be wrong about. It does **not** catch a notebook
misattributing its own agent to somewhere else, which is how the notebook_24
case survived: PILOT really is notebook 24's agent, so the sentence reads as
prose about itself. Widening the rule to catch it flagged ordinary sentences,
and a check that cries wolf gets deleted, so that one stays a manual find.

The check is narrow on purpose. It reads only the two shapes that make a
checkable claim — ``Notebook NN (NAME)`` and ``Notebook NN: … NAME …`` — and
only for names some notebook actually declares as an agent. A looser rule
flags ``RELEASE GUARD`` (the agent the *referring* notebook builds) and
``HITL`` (a concept, not a codename), and a check that cries wolf gets deleted.
"""

from __future__ import annotations

import pathlib
import re


EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"

#: A notebook's headline: ``Notebook 26: STEWARD — one privacy officer…``
_TITLE = re.compile(r"Notebook (\d{2}):\s*([A-Z][A-Z0-9]{3,})\b")

#: An agent built inside a notebook: ``name="STEWARD Privacy Officer"``.
_DECLARED = re.compile(r"""name\s*=\s*["']([A-Z][A-Z0-9]{3,})\b""")

#: ``Notebook 27 (RIGHTSIZER)`` — a parenthesised claim about that notebook's
#: identity. Checked unconditionally: a name no notebook declares at all is the
#: worst case, not an exempt one. Filtering those out is how the first version
#: of this test passed with ``CURATOR`` still in the tree.
_PAREN_REFERENCE = re.compile(r"Notebook (\d{2})\s*\(\s*([A-Z][A-Z0-9]{3,})\b")

#: ``Notebook 26: there, STEWARD routes work centrally`` — a codename in the
#: same sentence as a pointer. Too loose to check unconditionally (the sentence
#: often names the *referring* notebook's own agent), so this shape is checked
#: only for names some notebook declares, and it stops at a full stop — without
#: that it runs into the next sentence and reads "Notebook 12 covered hook
#: basics. Here we build RELEASE GUARD" as a claim about notebook 12.
_INLINE_REFERENCE = re.compile(r"Notebook (\d{2})\b[^.\n]{0,80}?\b([A-Z][A-Z0-9]{3,})\b")


def _notebooks() -> dict[int, str]:
    return {
        int(re.search(r"notebook_(\d+)", path.name).group(1)): path.read_text(errors="replace")
        for path in sorted(EXAMPLES.glob("notebook_*.py"))
    }


def _declared_names(text: str) -> set[str]:
    """Agent codenames this notebook introduces, by title or construction."""
    return {m.group(2) for m in _TITLE.finditer(text)} | set(_DECLARED.findall(text))


def test_the_extractor_finds_the_series() -> None:
    """A check that silently matches nothing passes for the wrong reason."""
    notebooks = _notebooks()
    assert len(notebooks) > 50, f"only found {len(notebooks)} notebooks — bad glob?"
    assert any(_declared_names(text) for text in notebooks.values()), "no codenames extracted"


def test_a_named_cross_reference_points_at_that_notebook() -> None:
    """``Notebook 27 (CURATOR)`` has to be a notebook 27 that contains CURATOR."""
    notebooks = _notebooks()
    declared = {number: _declared_names(text) for number, text in notebooks.items()}

    wrong = []
    for number, text in notebooks.items():
        claims = [(m, False) for m in _PAREN_REFERENCE.finditer(text)]
        claims += [(m, True) for m in _INLINE_REFERENCE.finditer(text)]
        for match, needs_known in claims:
            target, name = int(match.group(1)), match.group(2)
            if target == number:
                continue
            if needs_known:
                if name in declared[number]:
                    # The sentence names the referring notebook's own agent,
                    # which is prose rather than a claim about the target.
                    continue
                if not declared.get(target):
                    # The target has no codename, so it cannot be called by the
                    # wrong one. This is what keeps "Notebook 19 covered HITL"
                    # — a concept, not an agent — out of the results, without a
                    # blocklist that would need feeding forever.
                    continue
            if target not in declared:
                wrong.append(
                    f"notebook_{number:02d} points at notebook {target:02d}, which does not exist"
                )
            elif name not in declared[target]:
                wrong.append(
                    f"notebook_{number:02d} calls notebook {target:02d} {name!r}, "
                    f"but it is {sorted(declared[target]) or 'unnamed'}"
                )

    assert not wrong, "\n  ".join(["stale cross-references:", *sorted(set(wrong))])
