# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Render the release-notes page from ``CHANGELOG.md`` at build time.

The changelog at the repository root is the one record of what changed in
each release, kept in Keep a Changelog form. ``{{ locus_release_notes }}`` on
a page expands to its sections, ``Unreleased`` first and then every release
newest first with its date, so the site carries release notes that follow
every release without a second file to maintain.

A version heading becomes an ``h2`` whose id is the release tag (``v5.5.0``)
and whose table-of-contents label is the bare version, with the date set
beside it; the ``### Added`` / ``### Changed`` / ``### Fixed`` headings pass
through. Links written for GitHub are rewritten for the site: a path under
``docs/`` becomes a relative link to that page, anything else a link to the
file on GitHub, which ``hooks/build_metadata.py`` then pins to the documented
release (this hook runs first). ``extra.release_notes_since`` in
``mkdocs.yml`` names the earliest release to show; the page says where the
older entries are.
"""

from __future__ import annotations

import os
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RELEASE_NOTES_TOKEN = "{{ locus_release_notes }}"  # noqa: S105 — template placeholder, not a secret

_SDK_REPO = "https://github.com/oracle/locus/"
_CHANGELOG_LINK = f"{_SDK_REPO}blob/main/CHANGELOG.md"

#: ``## [5.5.0] - 2026-11-16`` or ``## [Unreleased]``; the dash between the
#: version and the date has been written as a hyphen and as a dash.
_SECTION = re.compile(
    r"^## \[(?P<version>[^\]]+)\](?:\s*[-–—]\s*(?P<date>\d{4}-\d{2}-\d{2}))?[ \t]*$",
    re.MULTILINE,
)
_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")
#: An inline link's target: ``[text](target)`` or ``[text](target "title")``.
_LINK_TARGET = re.compile(r"\]\((?P<target>[^)\s]+)(?P<title>\s+\"[^\"]*\")?\)")
_HAS_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


@dataclass(frozen=True)
class Section:
    """One changelog section: a version (or ``Unreleased``), its date, its body."""

    version: str
    date: str | None
    body: str

    @property
    def is_unreleased(self) -> bool:
        return self.version.lower() == "unreleased"


def parse_changelog(text: str) -> list[Section]:
    """Split a Keep a Changelog file into its ``## [version]`` sections, in file order."""
    matches = list(_SECTION.finditer(text))
    sections = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip("\n")
        sections.append(Section(match.group("version"), match.group("date"), body.strip()))
    return sections


def version_key(version: str) -> tuple[int, int, int] | None:
    """``major.minor.patch`` as integers, or ``None`` when the heading is not a version."""
    match = _VERSION.search(version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def select(sections: list[Section], since: str | None) -> tuple[list[Section], bool]:
    """The sections to show and whether any release was left out.

    ``Unreleased`` is shown whenever it has entries. A release is shown when it
    is at or above ``since``; with no ``since`` every release is.
    """
    floor = version_key(since) if since else None
    if since and floor is None:
        raise RuntimeError(f"extra.release_notes_since must be a version like 3.0.0, got {since!r}")
    kept: list[Section] = []
    cut = False
    for section in sections:
        if section.is_unreleased:
            if section.body:
                kept.append(section)
            continue
        key = version_key(section.version)
        if floor is not None and key is not None and key < floor:
            cut = True
            continue
        kept.append(section)
    return kept, cut


def rewrite_links(body: str, docs_dir: Path, page_dir: str) -> str:
    """Point the changelog's repository-relative links somewhere that resolves on the site.

    ``docs/<page>.md`` becomes a link relative to the release-notes page when
    the page exists, so MkDocs validates it like any other; every other
    repository path becomes a link to the file on GitHub's main branch, which
    the build-metadata hook pins to the documented release. Links with a
    scheme and in-page anchors are left alone.
    """

    def rewrite(match: re.Match[str]) -> str:
        target = match.group("target")
        title = match.group("title") or ""
        if target.startswith(("#", "//")) or _HAS_SCHEME.match(target):
            return match.group(0)
        path, sep, anchor = target.partition("#")
        if path.startswith("docs/"):
            page = path.removeprefix("docs/")
            if (docs_dir / page).is_file():
                rel = posixpath.relpath(page, page_dir or ".")
                return f"]({rel}{sep}{anchor}{title})"
        return f"]({_SDK_REPO}blob/main/{path}{sep}{anchor}{title})"

    return _LINK_TARGET.sub(rewrite, body)


def render_heading(section: Section) -> str:
    """The page heading for one section: the tag as the id, the version as the TOC label."""
    if section.is_unreleased:
        return "## Unreleased { #unreleased }"
    version = section.version
    attrs = f'#v{version} data-toc-label="{version}"'
    if section.date:
        return f'## {version} <small class="release-date">{section.date}</small> {{ {attrs} }}'
    return f"## {version} {{ {attrs} }}"


def render(text: str, *, since: str | None, docs_dir: Path, page_dir: str = "") -> str:
    """The release notes as Markdown: every kept section, heading then body."""
    sections, cut = select(parse_changelog(text), since)
    if not sections:
        raise RuntimeError("CHANGELOG.md has no '## [version]' sections to render")
    parts = [
        f"{render_heading(section)}\n\n{rewrite_links(section.body, docs_dir, page_dir)}"
        for section in sections
    ]
    if cut:
        parts.append(
            f"Releases before {since} are listed in [`CHANGELOG.md`]({_CHANGELOG_LINK}) on GitHub."
        )
    return "\n\n".join(parts) + "\n"


def changelog_path(config: Any) -> Path:
    """The changelog of the SDK this build documents.

    The same lookup as the SDK checkout in ``hooks/build_metadata.py``:
    ``LOCUS_SDK_DIR``, then a ``.sdk`` sub-checkout, then the repository root
    (the docs live in the SDK repository).
    """
    root = Path(config["config_file_path"]).parent
    candidates = []
    if os.environ.get("LOCUS_SDK_DIR"):
        candidates.append(Path(os.environ["LOCUS_SDK_DIR"]) / "CHANGELOG.md")
    candidates += [root / ".sdk" / "CHANGELOG.md", root / "CHANGELOG.md"]
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        raise RuntimeError(
            "A page uses {{ locus_release_notes }} but no CHANGELOG.md was found at "
            + ", ".join(str(c) for c in candidates)
        )
    return path


def on_page_markdown(markdown: str, *, page: Any, config: Any, **_: Any) -> str:
    """Expand the release-notes token; pages without it never touch the disk."""
    if RELEASE_NOTES_TOKEN not in markdown:
        return markdown
    notes = render(
        changelog_path(config).read_text(encoding="utf-8"),
        since=config.get("extra", {}).get("release_notes_since"),
        docs_dir=Path(config["docs_dir"]),
        page_dir=posixpath.dirname(page.file.src_uri),
    )
    return markdown.replace(RELEASE_NOTES_TOKEN, notes)
