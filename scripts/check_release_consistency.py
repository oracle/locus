#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Fail when the version is claimed differently in different places.

The version is asserted in four documents that no tool keeps in agreement:
``pyproject.toml`` (what PyPI ships), ``src/locus/__init__.py`` (what
``locus.__version__`` reports at runtime), ``CHANGELOG.md`` (what shipped in
it), and ``SECURITY.md`` (which releases get fixes). The release checklist in
CONTRIBUTING.md asks a human to update them together.

It also compares every released CHANGELOG section with the same section at
the release's tag. A PR whose changelog hunk was written under
``## [Unreleased]`` and that merges after the release PR renamed that heading
lands its entries inside the released section without a merge conflict, so
the file (and the release-notes page built from it) claims the wheel contains
changes it does not. That happened with 6.0.0 and again with 6.1.0.

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
import shutil
import subprocess
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
    """The version ``locus.__version__`` reports at runtime.

    Parsed rather than imported: importing the package to read one string
    costs ~256 ms and drags in pydantic, and a syntax error anywhere in the
    package would surface here as an unrelated failure.
    """
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        _read("src/locus/__init__.py"),
        re.MULTILINE,
    )
    if not match:
        raise ConsistencyError("src/locus/__init__.py declares no __version__")
    return match.group(1)


def check_versions_agree(verbose: bool) -> str:
    """pyproject and __init__ must be the same string."""
    pyproject, dunder = pyproject_version(), dunder_version()
    if pyproject != dunder:
        raise ConsistencyError(
            "the package cannot agree with itself on its own version.\n"
            f"  pyproject.toml        : {pyproject}\n"
            f"  src/locus/__init__.py : {dunder}\n"
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


# The oldest release tagged in this repository. The 1.x and 2.x sections came
# over with the history, their tags did not, and some of those sections were
# completed after the release they describe.
FIRST_GUARDED = (3, 0, 0)


def _version_key(heading: str) -> tuple[int, ...] | None:
    """``"2.11.0"`` → ``(2, 11, 0)``; ``None`` for ``Unreleased`` and the like."""
    parts = heading.split(".")
    if not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def changelog_sections(text: str) -> dict[str, list[str]]:
    """The body lines of every ``## [heading]`` section, in file order."""
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        match = re.match(r"^## \[([^\]]+)\]", line)
        if match:
            current = sections.setdefault(match.group(1), [])
        elif current is not None:
            current.append(line)
    return sections


def top_level_entries(lines: list[str]) -> list[str]:
    """The first line of every entry: a ``- `` item at column 0."""
    return [line for line in lines if line.startswith("- ")]


def _git(*args: str) -> str | None:
    """stdout of ``git -C ROOT <args>``, or ``None`` when git cannot answer."""
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(
            [git, "-C", str(ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def check_released_sections(verbose: bool) -> None:
    """A released CHANGELOG section must not gain entries after its tag.

    The wheel on PyPI is built from the tag, so the section at the tag is the
    record of what shipped. An entry that is on ``main`` but not at the tag
    was merged after the release and describes a change the wheel does not
    contain; it belongs under ``## [Unreleased]``. Rewording an entry that was
    already there is fine (links, renames), so the comparison counts entries
    rather than diffing text.

    A section whose tag does not exist yet is the release in flight and is
    skipped. Outside a git checkout, or in one with no tags, there is nothing
    to compare against and the check says so instead of failing.
    """
    if _git("rev-parse", "--is-inside-work-tree") is None:
        print("  CHANGELOG.md      released sections not compared (not a git checkout)")
        return
    sections = changelog_sections(_read("CHANGELOG.md"))
    guarded = [
        heading
        for heading in sections
        if (key := _version_key(heading)) is not None and key >= FIRST_GUARDED
    ]
    compared, untagged = 0, []
    for version in guarded:
        tagged = _git("show", f"refs/tags/v{version}:CHANGELOG.md")
        if tagged is None:
            untagged.append(version)
            continue
        at_tag = changelog_sections(tagged)
        if version not in at_tag:
            raise ConsistencyError(
                f"CHANGELOG.md has a [{version}] section that does not exist at tag v{version}.\n"
                f"  The section was written after the release was tagged, so it is not the\n"
                f"  record of what the {version} wheel contains."
            )
        before, after = top_level_entries(at_tag[version]), top_level_entries(sections[version])
        if len(after) > len(before):
            added = [entry for entry in after if entry not in before] or after[len(before) :]
            listed = "\n".join(f"    {entry}" for entry in added)
            raise ConsistencyError(
                f"CHANGELOG.md's [{version}] section has {len(after) - len(before)} more "
                f"entr{'y' if len(after) - len(before) == 1 else 'ies'} than it had at tag "
                f"v{version}.\n"
                f"  Not at the tag:\n{listed}\n"
                f"  The {version} wheel was built from the tag, so an entry merged after it\n"
                f"  describes a change that wheel does not contain. Move it under\n"
                f"  `## [Unreleased]`."
            )
        compared += 1
    if compared == 0:
        print(
            "  CHANGELOG.md      released sections not compared (no release tags in this checkout)"
        )
        return
    if verbose:
        pending = f"; no tag yet for {', '.join(untagged)}" if untagged else ""
        print(
            f"  CHANGELOG.md      {compared} released sections unchanged since their tags{pending}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="show each passing check")
    args = parser.parse_args()

    try:
        version = check_versions_agree(args.verbose)
        check_changelog(version, args.verbose)
        check_security_major(version, args.verbose)
        check_released_sections(args.verbose)
    except ConsistencyError as failure:
        print(f"release consistency: FAILED\n\n{failure}\n", file=sys.stderr)
        return 1

    print(f"release consistency: {version} is claimed consistently in every document.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
