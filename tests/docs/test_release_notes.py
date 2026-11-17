# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Unit: the release-notes hook renders CHANGELOG.md sections onto the page."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import markdown
import pytest
from hooks import release_notes


CHANGELOG = """# Changelog

Intro paragraph that is not a section.

## [Unreleased]

### Added

- **A thing.** See [`DEPRECATION.md`](DEPRECATION.md) and
  [the decision page](docs/decision.md#choices).

## [5.5.0] - 2026-11-16

### Fixed

- A fix, see <https://example.test/> and [a site](https://example.test/x).

## [3.0.0] - 2026-10-22

Back home.

### Changed

- Renamed.

## [2.28.0] - 2026-10-10

### Added

- Old entry linking [a missing page](docs/nope.md).

## [1.0.0] — 2026-06-09

- First.
"""


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "mkdocs.yml").write_text("site_name: locus\n")
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "decision.md").write_text("# Decisions\n")
    return tmp_path


def _config(repo: Path, **extra: str) -> dict:
    return {
        "config_file_path": str(repo / "mkdocs.yml"),
        "docs_dir": str(repo / "docs"),
        "extra": dict(extra),
    }


def _page(src: str = "release-notes.md") -> SimpleNamespace:
    return SimpleNamespace(file=SimpleNamespace(src_uri=src))


# ── parsing ──────────────────────────────────────────────────────────────────


def test_sections_are_split_on_version_headings_in_file_order() -> None:
    sections = release_notes.parse_changelog(CHANGELOG)
    assert [(s.version, s.date) for s in sections] == [
        ("Unreleased", None),
        ("5.5.0", "2026-11-16"),
        ("3.0.0", "2026-10-22"),
        ("2.28.0", "2026-10-10"),
        ("1.0.0", "2026-06-09"),
    ]
    assert sections[2].body.startswith("Back home.")
    assert sections[2].body.endswith("- Renamed.")
    assert sections[0].is_unreleased and not sections[1].is_unreleased


def test_the_file_preamble_is_not_a_section() -> None:
    assert all(s.version != "Changelog" for s in release_notes.parse_changelog(CHANGELOG))


def test_version_key_reads_major_minor_patch() -> None:
    assert release_notes.version_key("5.5.0") == (5, 5, 0)
    assert release_notes.version_key("10.2.33") == (10, 2, 33)
    assert release_notes.version_key("Unreleased") is None


# ── selection ────────────────────────────────────────────────────────────────


def test_since_keeps_releases_at_or_above_the_floor_and_reports_the_cut() -> None:
    kept, cut = release_notes.select(release_notes.parse_changelog(CHANGELOG), "3.0.0")
    assert [s.version for s in kept] == ["Unreleased", "5.5.0", "3.0.0"]
    assert cut is True


def test_without_since_every_release_is_kept() -> None:
    kept, cut = release_notes.select(release_notes.parse_changelog(CHANGELOG), None)
    assert [s.version for s in kept] == ["Unreleased", "5.5.0", "3.0.0", "2.28.0", "1.0.0"]
    assert cut is False


def test_an_empty_unreleased_section_is_not_shown() -> None:
    text = "## [Unreleased]\n\n## [1.0.0] - 2026-06-09\n\n- First.\n"
    kept, _ = release_notes.select(release_notes.parse_changelog(text), None)
    assert [s.version for s in kept] == ["1.0.0"]


def test_a_malformed_since_fails_the_build() -> None:
    with pytest.raises(RuntimeError, match="release_notes_since"):
        release_notes.select(release_notes.parse_changelog(CHANGELOG), "latest")


# ── headings ─────────────────────────────────────────────────────────────────


def test_a_release_heading_carries_the_tag_id_the_date_and_a_bare_toc_label() -> None:
    section = release_notes.Section("5.5.0", "2026-11-16", "")
    assert release_notes.render_heading(section) == (
        '## 5.5.0 <small class="release-date">2026-11-16</small> { #v5.5.0 data-toc-label="5.5.0" }'
    )


def test_the_unreleased_heading_is_plain() -> None:
    section = release_notes.Section("Unreleased", None, "x")
    assert release_notes.render_heading(section) == "## Unreleased { #unreleased }"


def test_the_heading_attributes_are_honoured_by_markdown() -> None:
    """The id is the tag and the table of contents shows the version alone."""
    md = markdown.Markdown(extensions=["toc", "attr_list"])
    html = md.convert(
        release_notes.render_heading(release_notes.Section("5.5.0", "2026-11-16", ""))
    )
    assert 'id="v5.5.0"' in html
    assert md.toc_tokens[0]["id"] == "v5.5.0"
    assert md.toc_tokens[0]["name"] == "5.5.0"


# ── links ────────────────────────────────────────────────────────────────────


def test_repository_links_resolve_on_the_site(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    body = (
        "[a](DEPRECATION.md) [b](docs/decision.md#choices) [c](docs/nope.md) "
        '[d](https://example.test/x "t") [e](#anchor) [f](mailto:x@example.test)'
    )
    assert release_notes.rewrite_links(body, repo / "docs", "") == (
        "[a](https://github.com/oracle/locus/blob/main/DEPRECATION.md) "
        "[b](decision.md#choices) "
        "[c](https://github.com/oracle/locus/blob/main/docs/nope.md) "
        '[d](https://example.test/x "t") [e](#anchor) [f](mailto:x@example.test)'
    )


def test_links_to_docs_pages_are_relative_to_the_page_that_shows_them(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = release_notes.rewrite_links("[b](docs/decision.md)", repo / "docs", "project")
    assert out == "[b](../decision.md)"


# ── the page ─────────────────────────────────────────────────────────────────


def test_the_token_expands_to_the_selected_sections(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    rendered = release_notes.on_page_markdown(
        "# Release notes\n\nIntro.\n\n{{ locus_release_notes }}\n",
        page=_page(),
        config=_config(repo, release_notes_since="3.0.0"),
    )
    assert rendered.startswith("# Release notes\n\nIntro.\n\n## Unreleased { #unreleased }\n\n")
    assert '## 5.5.0 <small class="release-date">2026-11-16</small>' in rendered
    assert "## 3.0.0 " in rendered
    assert "2.28.0" not in rendered.split("Releases before")[0]
    assert "[the decision page](decision.md#choices)" in rendered
    assert "(https://github.com/oracle/locus/blob/main/DEPRECATION.md)" in rendered
    assert rendered.rstrip().endswith(
        "Releases before 3.0.0 are listed in "
        "[`CHANGELOG.md`](https://github.com/oracle/locus/blob/main/CHANGELOG.md) on GitHub."
    )


def test_without_a_cut_there_is_no_pointer_to_older_entries(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    rendered = release_notes.on_page_markdown(
        "{{ locus_release_notes }}", page=_page(), config=_config(repo)
    )
    assert "## 1.0.0 " in rendered and "Releases before" not in rendered


def test_pages_without_the_token_are_returned_unchanged_and_never_read_the_disk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_a, **_k):
        raise AssertionError("no token, no changelog read")

    monkeypatch.setattr(release_notes, "changelog_path", unexpected)
    assert release_notes.on_page_markdown("plain", page=_page(), config={}) == "plain"


def test_the_changelog_is_found_from_the_env_then_the_deploy_path_then_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    monkeypatch.delenv("LOCUS_SDK_DIR", raising=False)
    assert release_notes.changelog_path(_config(repo)) == repo / "CHANGELOG.md"

    (repo / ".sdk").mkdir()
    (repo / ".sdk" / "CHANGELOG.md").write_text("## [9.0.0] - 2027-01-01\n\n- x\n")
    assert release_notes.changelog_path(_config(repo)) == repo / ".sdk" / "CHANGELOG.md"

    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "CHANGELOG.md").write_text("## [8.0.0] - 2027-01-01\n\n- y\n")
    monkeypatch.setenv("LOCUS_SDK_DIR", str(other))
    assert release_notes.changelog_path(_config(repo)) == other / "CHANGELOG.md"


def test_a_missing_changelog_fails_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LOCUS_SDK_DIR", raising=False)
    (tmp_path / "mkdocs.yml").write_text("site_name: locus\n")
    with pytest.raises(RuntimeError, match="no CHANGELOG.md was found"):
        release_notes.changelog_path({"config_file_path": str(tmp_path / "mkdocs.yml")})


def test_a_changelog_without_sections_fails_the_build(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no '## \\[version\\]' sections"):
        release_notes.render("# Changelog\n\nnothing yet\n", since=None, docs_dir=tmp_path)


# ── the repository's own changelog ───────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]


def test_the_repository_changelog_parses_with_a_date_on_every_release() -> None:
    sections = release_notes.parse_changelog((ROOT / "CHANGELOG.md").read_text())
    releases = [s for s in sections if not s.is_unreleased]
    assert releases, "CHANGELOG.md has no releases"
    assert all(s.date for s in releases), [s.version for s in releases if not s.date]
    assert all(release_notes.version_key(s.version) for s in releases)
