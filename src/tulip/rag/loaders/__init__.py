# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Document loaders for RAG ingestion.

Everything downstream of ingestion was already real — five vector stores, two
embedders, two rerankers, MMR — and the pipeline had no entrance: turning a
directory of files into :class:`~tulip.rag.stores.base.Document` objects was
left entirely to the caller. These are that entrance.

    from tulip.rag.loaders import load_directory

    docs = load_directory("docs/", glob="**/*.md")
    await retriever.add_documents([d.content for d in docs])

Text, Markdown, and HTML load on the standard library alone, so the common
cases add no dependency. PDF needs ``pypdf``, and says so at the call rather than at import.

Every loader records where the text came from — ``source``, plus ``page`` for
PDFs — so a retrieved chunk can be traced back to a file and a page rather
than floating free of its provenance.
"""

from __future__ import annotations

import html.parser
import pathlib
import re
import uuid
from typing import Any

from tulip.rag.stores.base import Document


__all__ = [
    "load_directory",
    "load_html",
    "load_markdown",
    "load_pdf",
    "load_text",
]

#: Extension → loader, used by :func:`load_directory`. Anything unlisted is
#: read as plain text rather than skipped: a ``.log``, ``.csv`` or ``.rst`` is
#: still text, and silently dropping files is a worse failure than ingesting
#: one that would have benefited from richer parsing.
_BY_SUFFIX: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}

_MISSING_PYPDF = "PDF loading needs pypdf. Install with: pip install pypdf"

#: YAML front matter at the very top of a Markdown file.
_FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def _doc(content: str, source: pathlib.Path | str, **metadata: Any) -> Document:
    """Build a Document with an id derived from its source and position.

    Deterministic ids mean re-ingesting a file updates the same rows instead
    of duplicating them — random ids silently double a corpus every time a
    pipeline runs twice.
    """
    key = f"{source}:{metadata.get('page', 0)}"
    return Document(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, key)),
        content=content,
        metadata={"source": str(source), **metadata},
    )


def load_text(path: str | pathlib.Path, *, encoding: str = "utf-8") -> list[Document]:
    """Load a plain-text file as one document.

    Args:
        path: File to read.
        encoding: Text encoding. Decoding errors are replaced rather than
            raised — one stray byte should not abort a corpus.
    """
    p = pathlib.Path(path)
    return [_doc(p.read_text(encoding=encoding, errors="replace"), p)]


def load_markdown(
    path: str | pathlib.Path,
    *,
    encoding: str = "utf-8",
    strip_front_matter: bool = True,
) -> list[Document]:
    """Load a Markdown file, dropping YAML front matter by default.

    Front matter is configuration, not prose: embedding it pollutes a corpus
    with ``layout:`` and ``draft: false``. It is removed from the text but
    kept in metadata under ``front_matter``, where a filter can still use it.
    """
    p = pathlib.Path(path)
    text = p.read_text(encoding=encoding, errors="replace")
    metadata: dict[str, Any] = {}

    if strip_front_matter and (match := _FRONT_MATTER.match(text)):
        metadata["front_matter"] = match.group(1)
        text = text[match.end() :]

    return [_doc(text.strip(), p, **metadata)]


class _TextExtractor(html.parser.HTMLParser):
    """Collect visible text, skipping script, style, and head content."""

    _SKIP = {"script", "style", "head", "meta", "link"}
    _BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: str | None = None
        self._skipping = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skipping += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skipping = max(0, self._skipping - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in self._BLOCK:
            # Preserve block boundaries: without them a chunker cannot tell a
            # paragraph break from a word wrap.
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title and self.title is None and data.strip():
            self.title = data.strip()
        if self._skipping:
            return
        if stripped := data.strip():
            self.parts.append(stripped)

    def text(self) -> str:
        joined = " ".join(self.parts)
        collapsed = re.sub(r"[ \t]+", " ", joined)
        return re.sub(r" ?\n ?", "\n", collapsed).strip()


def load_html(
    path: str | pathlib.Path | None = None,
    *,
    markup: str | None = None,
    encoding: str = "utf-8",
) -> list[Document]:
    """Extract visible text from an HTML file or string.

    Uses the standard library's parser rather than requiring BeautifulSoup:
    the job is text extraction, not DOM manipulation, and a dependency for
    that would be a poor trade.

    Args:
        path: File to read. Mutually exclusive with ``markup``.
        markup: HTML source directly, for content already in memory.
        encoding: Text encoding when reading from ``path``.

    Raises:
        ValueError: If neither or both of ``path`` and ``markup`` are given.
    """
    if (path is None) == (markup is None):
        raise ValueError("load_html takes exactly one of path= or markup=")

    if markup is None:
        p = pathlib.Path(str(path))
        markup = p.read_text(encoding=encoding, errors="replace")
        source: str | pathlib.Path = p
    else:
        source = "<string>"

    parser = _TextExtractor()
    parser.feed(markup)
    parser.close()

    metadata = {"title": parser.title} if parser.title else {}
    return [_doc(parser.text(), source, **metadata)]


def load_pdf(path: str | pathlib.Path, *, per_page: bool = True) -> list[Document]:
    """Extract text from a PDF.

    Args:
        path: File to read.
        per_page: One document per page (the default), each carrying its
            ``page`` number so a retrieved chunk can cite one. ``False``
            concatenates the file into a single document.

    Raises:
        ImportError: If ``pypdf`` is not installed, naming the package.
    """
    try:
        from pypdf import PdfReader  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on install shape
        raise ImportError(_MISSING_PYPDF) from exc

    p = pathlib.Path(path)
    reader = PdfReader(str(p))
    pages = [
        (index, (page.extract_text() or "").strip()) for index, page in enumerate(reader.pages, 1)
    ]
    pages = [(index, body) for index, body in pages if body]

    if per_page:
        return [_doc(body, p, page=index) for index, body in pages]
    return [_doc("\n\n".join(body for _, body in pages), p, pages=len(pages))]


def load_directory(
    path: str | pathlib.Path,
    *,
    glob: str = "**/*",
    encoding: str = "utf-8",
    skip_errors: bool = True,
) -> list[Document]:
    """Load every matching file under a directory, dispatching by extension.

    Args:
        path: Directory to walk.
        glob: Pattern relative to ``path``; defaults to everything,
            recursively.
        encoding: Text encoding for text-shaped files.
        skip_errors: Skip a file that fails to load — unreadable, or a PDF
            without ``pypdf`` — rather than aborting the corpus. Set
            ``False`` when a partial ingest would be worse than none.

    Returns:
        Every non-empty document, in sorted path order so a run is
        reproducible.
    """
    root = pathlib.Path(path)
    documents: list[Document] = []

    for file in sorted(root.glob(glob)):
        if not file.is_file():
            continue
        kind = _BY_SUFFIX.get(file.suffix.lower(), "text")
        try:
            if kind == "markdown":
                documents.extend(load_markdown(file, encoding=encoding))
            elif kind == "html":
                documents.extend(load_html(file, encoding=encoding))
            elif kind == "pdf":
                documents.extend(load_pdf(file))
            else:
                documents.extend(load_text(file, encoding=encoding))
        except Exception:
            if not skip_errors:
                raise
            continue

    return [d for d in documents if d.content.strip()]
