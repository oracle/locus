# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""RAG Retriever - Combines embedding and vector store for retrieval.

The retriever handles the complete RAG pipeline:
1. Embed query using the embedding provider
2. Search vector store for similar documents
3. Return ranked results for context injection

Supports multimodal content:
- Text documents
- Images (with OCR/description)
- PDFs (with text extraction)
- Audio/Voice (with transcription)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from locus.rag.stores.base import Document, SearchResult


# Pattern that neutralises literal occurrences of the spotlight tag inside
# retrieved content so a poisoned document cannot forge a closing marker.
_SPOTLIGHT_TAG_RE = re.compile(r"</?retrieved_document\s*>", re.IGNORECASE)


def _escape_spotlight(text: str) -> str:
    """Neutralise literal spotlight delimiters embedded in retrieved content."""
    return _SPOTLIGHT_TAG_RE.sub(lambda m: m.group(0).replace("<", "&lt;"), text)


@dataclass
class RetrievalResult:
    """Result from RAG retrieval.

    Attributes:
        documents: Retrieved documents sorted by relevance
        query: Original query text
        total_results: Total number of matches (may be > len(documents))
    """

    documents: list[SearchResult]
    query: str
    total_results: int = 0


@dataclass
class ChunkConfig:
    """Configuration for text chunking.

    Attributes:
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters to overlap between chunks
        separator: Text separator for splitting
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: str = "\n\n"


class RAGRetriever(BaseModel):
    """
    RAG Retriever combining embedding and vector store.

    Provides a unified interface for:
    - Adding documents (with automatic embedding)
    - Retrieving relevant context for queries
    - Chunking large documents

    Example:
        >>> from locus.rag import RAGRetriever, OCIEmbeddings, OracleVectorStore
        >>>
        >>> retriever = RAGRetriever(
        ...     embedder=OCIEmbeddings(model_id="cohere.embed-english-v3.0"),
        ...     store=OracleVectorStore(dsn="..."),
        ... )
        >>>
        >>> # Add documents
        >>> await retriever.add_documents(
        ...     [
        ...         "Python is a programming language.",
        ...         "Oracle Database supports vectors.",
        ...     ]
        ... )
        >>>
        >>> # Retrieve relevant context
        >>> results = await retriever.retrieve("What is Python?", limit=3)
        >>> for r in results.documents:
        ...     print(f"{r.score:.2f}: {r.document.content[:50]}...")

    Example with chunking:
        >>> retriever = RAGRetriever(
        ...     embedder=embedder,
        ...     store=store,
        ...     chunk_size=500,
        ...     chunk_overlap=50,
        ... )
        >>> await retriever.add_document(long_document, metadata={"source": "manual"})
    """

    embedder: Any  # EmbeddingProvider
    store: Any  # VectorStore
    chunk_size: int = Field(default=1000, description="Max characters per chunk")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    reranker: Any = Field(
        default=None,
        description=(
            "Optional ``locus.rag.reranker.Reranker``. When set, ``retrieve()`` "
            "over-fetches ``rerank_candidate_pool`` hits from the vector "
            "store, then has the reranker rescore + trim down to ``limit``. "
            "Closes #216."
        ),
    )
    rerank_candidate_pool: int = Field(
        default=50,
        ge=1,
        description=(
            "Number of vector-store hits to fetch before reranking when "
            "``reranker`` is set. Ignored when the reranker is ``None``."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}

    def _chunk_text(self, text: str, separator: str = "\n\n") -> list[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        # First try to split by separator
        parts = text.split(separator)

        current_chunk = ""
        for part in parts:
            # If adding this part would exceed chunk size
            if len(current_chunk) + len(part) + len(separator) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Keep overlap from end of previous chunk
                    if self.chunk_overlap > 0:
                        overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                        current_chunk = current_chunk[overlap_start:] + separator + part
                    else:
                        current_chunk = part
                else:
                    # Part itself is larger than chunk size, split it
                    for i in range(0, len(part), self.chunk_size - self.chunk_overlap):
                        chunk = part[i : i + self.chunk_size]
                        if chunk.strip():
                            chunks.append(chunk.strip())
                    current_chunk = ""
            elif current_chunk:
                current_chunk += separator + part
            else:
                current_chunk = part

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    async def add_document(
        self,
        content: str,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
    ) -> list[str]:
        """
        Add a document, optionally chunking it.

        Args:
            content: Document text
            doc_id: Optional document ID (auto-generated if not provided)
            metadata: Optional metadata
            chunk: Whether to chunk large documents

        Returns:
            List of document IDs (multiple if chunked)
        """
        base_id = doc_id or uuid4().hex
        base_metadata = metadata or {}

        if chunk and len(content) > self.chunk_size:
            chunks = self._chunk_text(content)
        else:
            chunks = [content]

        # Embed all chunks
        embeddings = await self.embedder.embed_documents(chunks)

        # Create documents
        documents = []
        for i, (chunk_text, emb_result) in enumerate(zip(chunks, embeddings, strict=False)):
            chunk_id = f"{base_id}_{i}" if len(chunks) > 1 else base_id
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "parent_id": base_id,
            }

            doc = Document(
                id=chunk_id,
                content=chunk_text,
                embedding=emb_result.embedding,
                metadata=chunk_metadata,
            )
            documents.append(doc)

        # Store all documents
        added: list[str] = await self.store.add_batch(documents)
        return added

    async def add_documents(
        self,
        contents: list[str],
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
    ) -> list[str]:
        """
        Add multiple documents.

        Args:
            contents: List of document texts
            metadata: Optional metadata (applied to all)
            chunk: Whether to chunk large documents

        Returns:
            List of all document IDs
        """
        all_ids = []
        for content in contents:
            ids = await self.add_document(content, metadata=metadata, chunk=chunk)
            all_ids.extend(ids)
        return all_ids

    async def add_file(
        self,
        file_path: str | Path,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
    ) -> list[str]:
        """
        Add a file (text, PDF, image, or audio).

        Automatically detects content type and processes accordingly:
        - PDFs: Extracts text (with OCR fallback for scanned docs)
        - Images: OCR text extraction + optional description
        - Audio: Speech-to-text transcription
        - Text: Direct processing

        Args:
            file_path: Path to the file
            doc_id: Optional document ID
            metadata: Optional metadata
            chunk: Whether to chunk large documents

        Returns:
            List of document IDs

        Example:
            >>> await retriever.add_file("manual.pdf")
            >>> await retriever.add_file("diagram.png")
            >>> await retriever.add_file("meeting.mp3")
        """
        from locus.rag.multimodal import MultimodalProcessor

        path = Path(file_path)
        processor = MultimodalProcessor()

        # Process the file
        result = await processor.process(path)

        # Create metadata with content type info
        file_metadata = {
            **(metadata or {}),
            "source_file": path.name,
            "content_type": result.content_type.value,
            **result.metadata,
        }

        # Add to store
        base_id = doc_id or uuid4().hex

        if chunk and len(result.text) > self.chunk_size:
            chunks = self._chunk_text(result.text)
        else:
            chunks = [result.text]

        # Embed all chunks
        embeddings = await self.embedder.embed_documents(chunks)

        # Create documents
        documents = []
        for i, (chunk_text, emb_result) in enumerate(zip(chunks, embeddings, strict=False)):
            chunk_id = f"{base_id}_{i}" if len(chunks) > 1 else base_id
            chunk_metadata = {
                **file_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "parent_id": base_id,
            }

            doc = Document(
                id=chunk_id,
                content=chunk_text,
                embedding=emb_result.embedding,
                metadata=chunk_metadata,
                content_type=result.content_type.value,
                raw_content=result.raw_content if i == 0 else None,  # Store raw only in first chunk
            )
            documents.append(doc)

        added: list[str] = await self.store.add_batch(documents)
        return added

    async def add_image(
        self,
        image: bytes | str | Path,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        use_ocr: bool = True,
    ) -> str:
        """
        Add an image document.

        Args:
            image: Image bytes, base64 string, or file path
            doc_id: Optional document ID
            metadata: Optional metadata
            use_ocr: Whether to use OCR for text extraction

        Returns:
            Document ID
        """
        from locus.rag.multimodal import ContentType, ImageProcessor

        processor = ImageProcessor(use_ocr=use_ocr)
        result = await processor.process(image)

        # Embed the extracted text
        embedding_result = await self.embedder.embed(result.text)

        doc = Document(
            id=doc_id or uuid4().hex,
            content=result.text,
            embedding=embedding_result.embedding,
            metadata={**(metadata or {}), **result.metadata},
            content_type=ContentType.IMAGE.value,
            raw_content=result.raw_content,
        )

        doc_added: str = await self.store.add(doc)
        return doc_added

    async def add_pdf(
        self,
        pdf: bytes | str | Path,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
    ) -> list[str]:
        """
        Add a PDF document.

        Args:
            pdf: PDF bytes, base64 string, or file path
            doc_id: Optional document ID
            metadata: Optional metadata
            chunk: Whether to chunk the document

        Returns:
            List of document IDs (multiple if chunked)
        """
        from locus.rag.multimodal import ContentType, PDFProcessor

        processor = PDFProcessor(use_ocr_fallback=True)
        result = await processor.process(pdf)

        return await self.add_document(
            result.text,
            doc_id=doc_id,
            metadata={**(metadata or {}), **result.metadata, "content_type": ContentType.PDF.value},
            chunk=chunk,
        )

    async def add_audio(
        self,
        audio: bytes | str | Path,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add an audio/voice document.

        Args:
            audio: Audio bytes, base64 string, or file path
            doc_id: Optional document ID
            metadata: Optional metadata

        Returns:
            Document ID
        """
        from locus.rag.multimodal import AudioProcessor, ContentType

        processor = AudioProcessor(use_whisper=True)
        result = await processor.process(audio)

        # Embed the transcription
        embedding_result = await self.embedder.embed(result.text)

        doc = Document(
            id=doc_id or uuid4().hex,
            content=result.text,
            embedding=embedding_result.embedding,
            metadata={**(metadata or {}), **result.metadata},
            content_type=ContentType.AUDIO.value,
            raw_content=result.raw_content,
        )

        doc_added: str = await self.store.add(doc)
        return doc_added

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            limit: Maximum documents to return
            threshold: Minimum similarity score (0.0-1.0)
            metadata_filter: Filter by metadata fields

        Returns:
            RetrievalResult with ranked documents
        """
        # Some LLMs (e.g. gpt-5.x via tool calls) JSON-encode floats as
        # strings ("0.5"); coerce here so every store backend sees a real
        # float / None and the threshold comparison below doesn't TypeError.
        if isinstance(threshold, str):
            try:
                threshold = float(threshold)
            except ValueError:
                threshold = None
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                limit = 5
        import time as _time

        from locus.observability.emit import (  # noqa: PLC0415
            EV_RAG_QUERY_COMPLETED,
            EV_RAG_QUERY_STARTED,
            emit,
        )

        await emit(
            EV_RAG_QUERY_STARTED,
            query_preview=query[:160],
            limit=limit,
            store_type=type(self.store).__name__,
            threshold=threshold,
        )
        _started = _time.perf_counter()

        # Embed the query
        query_result = await self.embedder.embed_query(query)

        # If a reranker is wired in, over-fetch from the vector store
        # (cheap embedding search), then have the reranker rescore the
        # wider pool against the query (cross-encoder, more expensive
        # but materially more accurate). Trim back to ``limit`` after.
        store_limit = max(limit, self.rerank_candidate_pool) if self.reranker is not None else limit

        # Search the store
        results = await self.store.search(
            query_embedding=query_result.embedding,
            limit=store_limit,
            threshold=threshold,
            metadata_filter=metadata_filter,
        )

        if self.reranker is not None and results:
            results = await self.reranker.rerank(query, results)
            results = results[:limit]

        await emit(
            EV_RAG_QUERY_COMPLETED,
            hit_count=len(results),
            top_score=results[0].score if results else None,
            duration_ms=(_time.perf_counter() - _started) * 1000,
            store_type=type(self.store).__name__,
            reranker_type=type(self.reranker).__name__ if self.reranker is not None else None,
        )

        return RetrievalResult(
            documents=results,
            query=query,
            total_results=len(results),
        )

    async def retrieve_text(
        self,
        query: str,
        limit: int = 5,
        threshold: float | None = None,
        separator: str = "\n\n---\n\n",
        spotlight: bool = True,
    ) -> str:
        """
        Retrieve and concatenate relevant documents as text.

        Convenience method for injecting context into prompts.

        Args:
            query: Query text
            limit: Maximum documents to return
            threshold: Minimum similarity score
            separator: Text to join documents
            spotlight: When True (default), wrap each document in
                ``<retrieved_document>``...``</retrieved_document>`` markers so the
                LLM can distinguish untrusted retrieved data from trusted
                instructions. Disable only if the caller wraps content itself.

        Returns:
            Concatenated document contents.

        Security note:
            Retrieved content is **untrusted data** — a poisoned document can
            attempt an indirect prompt-injection. The spotlight wrappers let
            you instruct the model (in the system prompt) to treat anything
            inside those tags as data only, never as instructions, and to
            refuse to perform tool calls whose arguments are quoted verbatim
            from retrieved content.
        """
        result = await self.retrieve(query, limit=limit, threshold=threshold)
        contents = [r.document.content for r in result.documents]
        if spotlight:
            contents = [
                f"<retrieved_document>\n{_escape_spotlight(c)}\n</retrieved_document>"
                for c in contents
            ]
        return separator.join(contents)

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        deleted: bool = await self.store.delete(doc_id)
        return deleted

    async def clear(self) -> int:
        """Delete all documents."""
        cleared: int = await self.store.clear()
        return cleared

    async def count(self) -> int:
        """Count documents in store."""
        n: int = await self.store.count()
        return n

    async def close(self) -> None:
        """Close resources."""
        await self.store.close()

    def as_tool(self, name: str = "search_knowledge", description: str | None = None) -> Any:
        """
        Create a tool function for agent use.

        Returns a tool that can be registered with an agent.

        Args:
            name: Tool name
            description: Tool description

        Returns:
            Tool function decorated with @tool
        """
        from locus.rag.tools import create_rag_tool

        return create_rag_tool(self, name=name, description=description)

    def __repr__(self) -> str:
        return f"RAGRetriever(embedder={self.embedder!r}, store={self.store!r})"
