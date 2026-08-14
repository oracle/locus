# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Text chunkers — splitters that produce chunks for embedding.

``RAGRetriever`` has always had a built-in splitter, but it breaks on one
separator. When a paragraph is longer than ``chunk_size`` that splitter has
nowhere left to go and emits an oversized chunk, and when the text is mostly
one long block it degenerates to fixed-width cuts through the middle of
sentences — which is exactly where retrieval quality is lost.

:func:`recursive_chunks` walks a descending list of separators instead,
splitting on paragraphs first, then lines, then sentences, then words, and
only falling back to a hard character cut when a single word exceeds the
limit. That keeps semantic units intact whenever the text allows it.

    from tulip.rag.chunkers import recursive_chunks

    chunks = recursive_chunks(document.content, chunk_size=800, overlap=100)

Sizes are in characters, matching ``RAGRetriever.chunk_size``.
"""

from __future__ import annotations

from collections.abc import Sequence


__all__ = [
    "DEFAULT_SEPARATORS",
    "MARKDOWN_SEPARATORS",
    "recursive_chunks",
]

#: Tried in order, coarsest first: paragraph, line, sentence, clause, word.
#: The empty string is the terminal case — split anywhere — reached only when
#: a single token is longer than ``chunk_size``.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", "")

#: Markdown headings first, so a section stays with its own heading rather
#: than being cut adrift from it.
MARKDOWN_SEPARATORS: tuple[str, ...] = (
    "\n## ",
    "\n### ",
    "\n#### ",
    *DEFAULT_SEPARATORS,
)


def _split_on(text: str, separator: str) -> list[str]:
    """Split, keeping the separator attached to the piece it belongs to.

    Dropping it would join ``"a."`` and ``"Next"`` into ``"a.Next"`` on
    reassembly. Punctuation terminates the piece before it, so it trails;
    a Markdown heading *introduces* the piece after it, so it leads —
    otherwise a section is cut adrift from its own heading, which is the
    one thing heading-aware splitting exists to prevent.
    """
    if separator == "":
        return list(text)
    parts = text.split(separator)
    if separator.lstrip("\n").startswith("#"):
        return parts[:1] + [separator.lstrip("\n") + p for p in parts[1:]]
    return [p + separator for p in parts[:-1]] + parts[-1:]


def _merge(pieces: Sequence[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack pieces into chunks, carrying ``overlap`` characters over.

    Overlap is taken from the tail of the previous chunk so a sentence
    straddling a boundary is still retrievable from one side of it.
    """
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            # Carry overlap only as far as it still fits. Taking the full
            # window unconditionally is what makes most splitters emit chunks
            # larger than the size they were given — the limit should mean
            # what it says, so the tail is trimmed to what remains.
            room = max(0, chunk_size - len(piece))
            current = (current[-min(overlap, room) :] if overlap and room else "") + piece
        else:
            current += piece

    if stripped := current.strip():
        chunks.append(stripped)
    return chunks


def recursive_chunks(
    text: str,
    *,
    chunk_size: int = 1000,
    overlap: int | None = None,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[str]:
    """Split ``text`` into chunks, preferring the coarsest separator that fits.

    Args:
        text: Text to split.
        chunk_size: Maximum characters per chunk. A chunk may still exceed it
            only when a single indivisible token does.
        overlap: Characters carried from the end of one chunk into the
            next, so a sentence spanning a boundary stays retrievable.
            Defaults to a fifth of ``chunk_size`` (capped at 200) — a
            fixed default would be invalid for any small chunk_size.
        separators: Candidate separators, coarsest first. Use
            :data:`MARKDOWN_SEPARATORS` for Markdown so sections stay with
            their headings.

    Returns:
        Non-empty chunks in document order.

    Raises:
        ValueError: If ``chunk_size`` is not positive, or ``overlap`` is
            negative or at least ``chunk_size`` — an overlap that large
            cannot make progress and would loop forever.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap is None:
        overlap = min(200, chunk_size // 5)
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size}); "
            f"an overlap that large cannot advance through the text."
        )

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    for index, separator in enumerate(separators):
        if separator and separator not in text:
            continue
        pieces = _split_on(text, separator)
        if len(pieces) == 1:
            continue

        # A piece that is still too large is re-split with the *finer*
        # separators only — retrying the same one would not converge.
        expanded: list[str] = []
        for piece in pieces:
            if len(piece) > chunk_size and index + 1 < len(separators):
                expanded.extend(
                    recursive_chunks(
                        piece,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        separators=separators[index + 1 :],
                    )
                )
            else:
                expanded.append(piece)
        return _merge(expanded, chunk_size, overlap)

    # No separator applied — a single unbroken token longer than chunk_size.
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
