# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RAG loaders and the recursive chunker.

``rag/loaders/`` and ``rag/chunkers/`` were empty packages — ``__all__ = []``
— while everything downstream was real: five vector stores, two embedders,
two rerankers. The pipeline had no entrance, which made RAG, the most common
agent use case, the gap most likely to end an evaluation early.
"""

from __future__ import annotations

import pathlib
import random
import string

import pytest

from tulip.rag.chunkers import DEFAULT_SEPARATORS, MARKDOWN_SEPARATORS, recursive_chunks
from tulip.rag.loaders import (
    load_directory,
    load_html,
    load_markdown,
    load_pdf,
    load_text,
)


# --------------------------------------------------------------------------
# Chunker
# --------------------------------------------------------------------------


def test_short_text_is_one_chunk() -> None:
    assert recursive_chunks("hello world", chunk_size=100) == ["hello world"]


def test_empty_text_yields_nothing() -> None:
    assert recursive_chunks("   \n  ", chunk_size=100) == []


def test_paragraphs_are_preferred_over_mid_sentence_cuts() -> None:
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird one."
    chunks = recursive_chunks(text, chunk_size=30, overlap=0)
    assert chunks[0] == "First paragraph here."


def test_no_chunk_exceeds_the_limit() -> None:
    """The size is a limit, not a suggestion.

    Most splitters carry the full overlap window unconditionally, which emits
    chunks larger than the size they were given.
    """
    text = ". ".join(f"Sentence number {n} with some padding" for n in range(40))
    for size, overlap in ((50, 10), (80, 40), (120, 0)):
        for chunk in recursive_chunks(text, chunk_size=size, overlap=overlap):
            assert len(chunk) <= size, f"{len(chunk)} > {size} (overlap={overlap})"


def test_an_indivisible_token_is_hard_split() -> None:
    chunks = recursive_chunks("x" * 250, chunk_size=100, overlap=0)
    assert chunks == ["x" * 100, "x" * 100, "x" * 50]


def test_no_text_is_lost() -> None:
    text = "Alpha para. Two sentences.\n\nBeta para. Also two.\n\nGamma on audits."
    joined = " ".join(recursive_chunks(text, chunk_size=40, overlap=5))
    for word in text.split():
        assert word in joined


def test_overlap_carries_context_between_chunks() -> None:
    text = ". ".join(f"Sentence {n}" for n in range(20))
    with_overlap = recursive_chunks(text, chunk_size=60, overlap=20)
    without = recursive_chunks(text, chunk_size=60, overlap=0)
    assert len(with_overlap) >= len(without)


def test_markdown_separators_keep_sections_with_headings() -> None:
    """A heading must travel with the section it introduces.

    Splitting a section away from its own heading is the single failure
    heading-aware chunking exists to prevent: the retrieved text then has no
    idea what it is about.
    """
    text = "intro\n\n## Refunds\n\nRefund body.\n\n## Deploys\n\nDeploy body."
    chunks = recursive_chunks(text, chunk_size=40, overlap=0, separators=MARKDOWN_SEPARATORS)
    for heading, body in (("## Refunds", "Refund body."), ("## Deploys", "Deploy body.")):
        assert any(heading in c and body in c for c in chunks), (
            f"{heading} was separated from {body}: {chunks}"
        )


def test_a_heading_leads_its_chunk_rather_than_trailing_the_previous_one() -> None:
    text = "## Alpha\n\n" + "a" * 60 + "\n\n## Beta\n\n" + "b" * 60
    chunks = recursive_chunks(text, chunk_size=80, overlap=0, separators=MARKDOWN_SEPARATORS)
    assert any(c.startswith("## Beta") for c in chunks), chunks


@pytest.mark.parametrize(
    "kwargs",
    [{"chunk_size": 0}, {"chunk_size": -1}, {"overlap": -1}, {"chunk_size": 50, "overlap": 50}],
)
def test_invalid_sizes_raise(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        recursive_chunks("some text", **kwargs)


def test_size_limit_holds_across_random_inputs() -> None:
    """Property check — the guarantee should not depend on lucky text."""
    rng = random.Random(7)  # noqa: S311 - fixture data, not crypto
    for _ in range(200):
        size = rng.randint(20, 200)
        overlap = rng.randint(0, size - 1)
        words = [
            "".join(rng.choices(string.ascii_lowercase, k=rng.randint(1, 30))) for _ in range(60)
        ]
        longest = max(len(w) for w in words)
        for chunk in recursive_chunks(" ".join(words), chunk_size=size, overlap=overlap):
            assert len(chunk) <= max(size, longest)


def test_default_separators_end_with_the_terminal_case() -> None:
    """The empty separator is what makes the recursion terminate."""
    assert DEFAULT_SEPARATORS[-1] == ""


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def test_load_text_reads_a_file(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("plain body")
    (doc,) = load_text(f)
    assert doc.content == "plain body"
    assert doc.metadata["source"] == str(f)


def test_ids_are_stable_across_reloads(tmp_path: pathlib.Path) -> None:
    """Random ids would double a corpus every time a pipeline reruns."""
    f = tmp_path / "a.txt"
    f.write_text("body")
    assert load_text(f)[0].id == load_text(f)[0].id


def test_undecodable_bytes_do_not_abort(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "bad.txt"
    f.write_bytes(b"good \xff\xfe bytes")
    assert "good" in load_text(f)[0].content


def test_markdown_strips_front_matter_but_keeps_it(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "page.md"
    f.write_text("---\ntitle: X\ndraft: false\n---\n\n# Heading\n\nBody.")
    (doc,) = load_markdown(f)
    assert "draft" not in doc.content
    assert "Body." in doc.content
    assert "draft: false" in doc.metadata["front_matter"]


def test_markdown_can_keep_front_matter(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "page.md"
    f.write_text("---\ntitle: X\n---\n\nBody.")
    (doc,) = load_markdown(f, strip_front_matter=False)
    assert "title: X" in doc.content


def test_html_extracts_text_and_drops_script_and_style() -> None:
    markup = (
        "<html><head><title>Docs</title><style>p{color:red}</style>"
        "<script>alert(1)</script></head>"
        "<body><p>Hello</p><p>World</p></body></html>"
    )
    (doc,) = load_html(markup=markup)
    assert doc.content == "Hello\nWorld"
    assert "color:red" not in doc.content
    assert "alert" not in doc.content
    assert doc.metadata["title"] == "Docs"


def test_html_preserves_block_boundaries() -> None:
    """Without them a chunker cannot tell a paragraph break from a wrap."""
    (doc,) = load_html(markup="<p>One</p><p>Two</p>")
    assert "\n" in doc.content


def test_html_decodes_entities() -> None:
    (doc,) = load_html(markup="<p>a &amp; b</p>")
    assert doc.content == "a & b"


def test_html_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        load_html()
    with pytest.raises(ValueError, match="exactly one"):
        load_html("x.html", markup="<p>y</p>")


def test_load_directory_dispatches_by_extension(tmp_path: pathlib.Path) -> None:
    (tmp_path / "a.md").write_text("---\nx: 1\n---\n\nMarkdown body.")
    (tmp_path / "b.html").write_text("<p>Html body.</p>")
    (tmp_path / "c.log").write_text("Log body.")
    docs = load_directory(tmp_path)

    bodies = {d.content for d in docs}
    assert "Markdown body." in bodies  # front matter stripped
    assert "Html body." in bodies  # tags stripped
    assert "Log body." in bodies  # unknown suffix read as text


def test_load_directory_respects_a_glob(tmp_path: pathlib.Path) -> None:
    (tmp_path / "a.md").write_text("keep")
    (tmp_path / "b.txt").write_text("drop")
    docs = load_directory(tmp_path, glob="**/*.md")
    assert [d.content for d in docs] == ["keep"]


def test_load_directory_is_recursive_and_ordered(tmp_path: pathlib.Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("second")
    (tmp_path / "a.txt").write_text("first")
    assert [d.content for d in load_directory(tmp_path)] == ["first", "second"]


def test_load_directory_skips_empty_files(tmp_path: pathlib.Path) -> None:
    (tmp_path / "empty.txt").write_text("   ")
    (tmp_path / "real.txt").write_text("body")
    assert [d.content for d in load_directory(tmp_path)] == ["body"]


def test_load_directory_skips_unreadable_files_by_default(
    tmp_path: pathlib.Path,
) -> None:
    """A PDF without pypdf must not abort the whole corpus."""
    (tmp_path / "broken.pdf").write_bytes(b"not really a pdf")
    (tmp_path / "fine.txt").write_text("body")
    assert [d.content for d in load_directory(tmp_path)] == ["body"]


def test_load_directory_can_fail_loudly(tmp_path: pathlib.Path) -> None:
    (tmp_path / "broken.pdf").write_bytes(b"not really a pdf")
    # pypdf raises its own error type; the contract here is only that
    # skip_errors=False propagates rather than swallowing.
    with pytest.raises(Exception):  # noqa: B017
        load_directory(tmp_path, skip_errors=False)


def test_load_pdf_reports_a_missing_dependency_clearly(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "pypdf", None)
    (tmp_path / "x.pdf").write_bytes(b"%PDF-1.4")
    with pytest.raises(ImportError, match="pypdf"):
        load_pdf(tmp_path / "x.pdf")
