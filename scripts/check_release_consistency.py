#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Fail when the version is claimed differently in different places.

The version is asserted in four documents that no tool keeps in agreement:
``pyproject.toml`` (what PyPI ships), ``src/tulip/__init__.py`` (what
``tulip.__version__`` reports at runtime), ``CHANGELOG.md`` (what shipped in
it), and ``SECURITY.md`` (which releases get fixes). The release checklist in
CONTRIBUTING.md asks a human to update them together.

That has already failed in production: ``SECURITY.md`` listed 2.1.x and 2.0.x
as the supported versions while the package shipped 2.10.0, so the current
release was absent from its own support table and a reader comparing strings
could read ``2.10`` as older than ``2.1``. It was caught by an outside
reviewer, not by us. For a project whose pitch is putting a gate between an
agent and production, that costs more trust than the underlying facts warrant.

Run it::

    python scripts/check_release_consistency.py          # CI mode
    python scripts/check_release_consistency.py -v       # show what passed

Exit 0 when every claim agrees, 1 otherwise with the specific disagreement.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import tomllib


ROOT = pathlib.Path(__file__).resolve().parent.parent


class ConsistencyError(Exception):
    """One inconsistency, phrased so the fix is obvious from the message."""


def _read(name: str) -> str:
    path = ROOT / name
    if not path.exists():
        raise ConsistencyError(f"{name} is missing — it is part of the release contract")
    return path.read_text(encoding="utf-8")


def pyproject_version() -> str:
    """The version PyPI will ship."""
    data = tomllib.loads(_read("pyproject.toml"))
    try:
        return str(data["project"]["version"])
    except KeyError as exc:  # pragma: no cover — malformed pyproject
        raise ConsistencyError("pyproject.toml has no [project] version") from exc


def dunder_version() -> str:
    """The version ``tulip.__version__`` reports at runtime.

    Parsed rather than imported: importing the package to read one string
    costs ~256 ms and drags in pydantic, and a syntax error anywhere in the
    package would surface here as an unrelated failure.
    """
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        _read("src/tulip/__init__.py"),
        re.MULTILINE,
    )
    if not match:
        raise ConsistencyError("src/tulip/__init__.py declares no __version__")
    return match.group(1)


def check_versions_agree(verbose: bool) -> str:
    """pyproject and __init__ must be the same string."""
    pyproject, dunder = pyproject_version(), dunder_version()
    if pyproject != dunder:
        raise ConsistencyError(
            "the package cannot agree with itself on its own version.\n"
            f"  pyproject.toml        : {pyproject}\n"
            f"  src/tulip/__init__.py : {dunder}\n"
            "  Both must be bumped together — see the release checklist in "
            "CONTRIBUTING.md."
        )
    if verbose:
        print(f"  version           {pyproject}  (pyproject == __init__)")
    return pyproject


def check_changelog(version: str, verbose: bool) -> None:
    """CHANGELOG must describe the version being shipped.

    An `## [Unreleased]` heading is allowed above it — that is the working
    area — but the shipped version needs its own entry, or the release goes
    out with no record of what changed.
    """
    changelog = _read("CHANGELOG.md")
    if not re.search(rf"^## \[{re.escape(version)}\]", changelog, re.MULTILINE):
        found = re.findall(r"^## \[([^\]]+)\]", changelog, re.MULTILINE)[:4]
        raise ConsistencyError(
            f"CHANGELOG.md has no entry for {version}, the version being shipped.\n"
            f"  headings found: {found}\n"
            "  Add a `## [{v}] - YYYY-MM-DD` section describing what changed.".format(v=version)
        )
    if verbose:
        print(f"  CHANGELOG.md      has a [{version}] entry")


def check_security_major(version: str, verbose: bool) -> None:
    """SECURITY.md's support statement must cover the shipped major.

    The table is written in terms of a major line rather than an enumerated
    list of minors, precisely so it does not go stale between releases — three
    minors shipped on 2026-08-16 alone. So the only thing to verify is that the
    major it talks about is the major we ship.
    """
    security = _read("SECURITY.md")
    major = version.split(".", 1)[0]
    if f"`{major}.x`" not in security:
        mentioned = sorted(set(re.findall(r"`(\d+)\.x`", security)))
        raise ConsistencyError(
            f"SECURITY.md does not mention the `{major}.x` line, but the package "
            f"ships {version}.\n"
            f"  majors mentioned: {mentioned or 'none'}\n"
            "  A reader cannot tell whether the release they are running gets "
            "security fixes."
        )
    if verbose:
        print(f"  SECURITY.md       covers the `{major}.x` line")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="show each passing check")
    args = parser.parse_args()

    try:
        version = check_versions_agree(args.verbose)
        check_changelog(version, args.verbose)
        check_security_major(version, args.verbose)
    except ConsistencyError as failure:
        print(f"release consistency: FAILED\n\n{failure}\n", file=sys.stderr)
        return 1

    print(f"release consistency: {version} is claimed consistently in every document.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
