"""Text chunkers — splitters that produce chunks for embedding.

Tulip ships a client-side chunker via :class:`ChunkConfig` on
``RAGRetriever``, which covers the common token/character windowing case.
This package is the home for any additional chunking strategies; none
ship beyond the built-in client-side chunker today.
"""

__all__: list[str] = []
