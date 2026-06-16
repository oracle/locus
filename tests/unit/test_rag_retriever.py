# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for RAG retriever."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from locus.rag.retriever import (
    ChunkConfig,
    RAGRetriever,
    RetrievalResult,
)
from locus.rag.stores.base import Document, SearchResult


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_create_retrieval_result(self):
        """Test creating retrieval result."""
        result = RetrievalResult(
            documents=[],
            query="test query",
            total_results=0,
        )
        assert result.query == "test query"
        assert result.total_results == 0
        assert result.documents == []

    def test_create_with_documents(self):
        """Test creating with documents."""
        doc = Document(id="1", content="test", embedding=[0.1, 0.2])
        search_result = SearchResult(document=doc, score=0.95)

        result = RetrievalResult(
            documents=[search_result],
            query="query",
            total_results=1,
        )
        assert len(result.documents) == 1
        assert result.documents[0].score == 0.95


class TestChunkConfig:
    """Tests for ChunkConfig dataclass."""

    def test_default_values(self):
        """Test default chunk config values."""
        config = ChunkConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.separator == "\n\n"

    def test_custom_values(self):
        """Test custom chunk config values."""
        config = ChunkConfig(
            chunk_size=500,
            chunk_overlap=50,
            separator="---",
        )
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.separator == "---"


class TestRAGRetrieverInit:
    """Tests for RAGRetriever initialization."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = MagicMock()
        embedder.embed = AsyncMock()
        embedder.embed_query = AsyncMock()
        embedder.embed_documents = AsyncMock()
        return embedder

    @pytest.fixture
    def mock_store(self):
        """Create a mock store."""
        store = MagicMock()
        store.add = AsyncMock()
        store.add_batch = AsyncMock(return_value=["id1"])
        store.search = AsyncMock(return_value=[])
        store.delete = AsyncMock(return_value=True)
        store.clear = AsyncMock(return_value=0)
        store.count = AsyncMock(return_value=0)
        store.close = AsyncMock()
        return store

    def test_create_retriever(self, mock_embedder, mock_store):
        """Test creating retriever."""
        retriever = RAGRetriever(
            embedder=mock_embedder,
            store=mock_store,
        )
        assert retriever.embedder is mock_embedder
        assert retriever.store is mock_store
        assert retriever.chunk_size == 1000
        assert retriever.chunk_overlap == 200

    def test_create_with_custom_chunk_settings(self, mock_embedder, mock_store):
        """Test creating with custom chunk settings."""
        retriever = RAGRetriever(
            embedder=mock_embedder,
            store=mock_store,
            chunk_size=500,
            chunk_overlap=50,
        )
        assert retriever.chunk_size == 500
        assert retriever.chunk_overlap == 50

    def test_repr(self, mock_embedder, mock_store):
        """Test string representation."""
        retriever = RAGRetriever(
            embedder=mock_embedder,
            store=mock_store,
        )
        repr_str = repr(retriever)
        assert "RAGRetriever" in repr_str


class TestChunkText:
    """Tests for _chunk_text method."""

    @pytest.fixture
    def retriever(self):
        """Create retriever with small chunk size for testing."""
        return RAGRetriever(
            embedder=MagicMock(),
            store=MagicMock(),
            chunk_size=50,
            chunk_overlap=10,
        )

    def test_short_text_no_chunking(self, retriever):
        """Test that short text is not chunked."""
        text = "Short text"
        chunks = retriever._chunk_text(text)
        assert chunks == [text]

    def test_text_split_by_separator(self, retriever):
        """Test text split by separator."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = retriever._chunk_text(text)
        assert len(chunks) > 1

    def test_long_part_split(self):
        """Test that long parts are split."""
        retriever = RAGRetriever(
            embedder=MagicMock(),
            store=MagicMock(),
            chunk_size=20,
            chunk_overlap=5,
        )
        text = "This is a very long single paragraph without separators that exceeds chunk size"
        chunks = retriever._chunk_text(text, separator="\n\n")
        assert len(chunks) > 1

    def test_overlap_applied(self):
        """Test that overlap is applied between chunks."""
        retriever = RAGRetriever(
            embedder=MagicMock(),
            store=MagicMock(),
            chunk_size=30,
            chunk_overlap=10,
        )
        text = "First part.\n\nSecond part.\n\nThird part."
        chunks = retriever._chunk_text(text)
        # Chunks should have some overlap
        assert len(chunks) > 0


class TestRAGRetrieverAddDocument:
    """Tests for add_document method."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = MagicMock()
        mock_result = MagicMock()
        mock_result.embedding = [0.1, 0.2, 0.3]
        embedder.embed_documents = AsyncMock(return_value=[mock_result])
        return embedder

    @pytest.fixture
    def mock_store(self):
        """Create mock store."""
        store = MagicMock()
        store.add_batch = AsyncMock(return_value=["doc_id"])
        return store

    @pytest.fixture
    def retriever(self, mock_embedder, mock_store):
        """Create retriever."""
        return RAGRetriever(
            embedder=mock_embedder,
            store=mock_store,
            chunk_size=1000,
        )

    @pytest.mark.asyncio
    async def test_add_document(self, retriever):
        """Test adding a document."""
        ids = await retriever.add_document("Test content")
        assert ids == ["doc_id"]
        retriever.embedder.embed_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_document_with_id(self, retriever):
        """Test adding a document with specific ID."""
        ids = await retriever.add_document("Content", doc_id="my-id")
        assert ids == ["doc_id"]

    @pytest.mark.asyncio
    async def test_add_document_with_metadata(self, retriever):
        """Test adding a document with metadata."""
        ids = await retriever.add_document(
            "Content",
            metadata={"source": "test"},
        )
        assert ids == ["doc_id"]

    @pytest.mark.asyncio
    async def test_add_document_no_chunk(self, retriever):
        """Test adding without chunking."""
        ids = await retriever.add_document("Content", chunk=False)
        assert ids == ["doc_id"]


class TestRAGRetrieverAddDocuments:
    """Tests for add_documents method."""

    @pytest.fixture
    def retriever(self):
        """Create retriever."""
        embedder = MagicMock()
        mock_result = MagicMock()
        mock_result.embedding = [0.1, 0.2]
        embedder.embed_documents = AsyncMock(return_value=[mock_result])

        store = MagicMock()
        store.add_batch = AsyncMock(return_value=["id"])

        return RAGRetriever(embedder=embedder, store=store)

    @pytest.mark.asyncio
    async def test_add_multiple_documents(self, retriever):
        """Test adding multiple documents."""
        ids = await retriever.add_documents(["Doc 1", "Doc 2"])
        assert len(ids) == 2


class TestRAGRetrieverRetrieve:
    """Tests for retrieve method."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = MagicMock()
        mock_result = MagicMock()
        mock_result.embedding = [0.1, 0.2, 0.3]
        embedder.embed_query = AsyncMock(return_value=mock_result)
        return embedder

    @pytest.fixture
    def mock_store(self):
        """Create mock store."""
        store = MagicMock()
        doc = Document(id="1", content="Test doc", embedding=[0.1, 0.2])
        search_result = SearchResult(document=doc, score=0.9)
        store.search = AsyncMock(return_value=[search_result])
        return store

    @pytest.fixture
    def retriever(self, mock_embedder, mock_store):
        """Create retriever."""
        return RAGRetriever(embedder=mock_embedder, store=mock_store)

    @pytest.mark.asyncio
    async def test_retrieve(self, retriever):
        """Test retrieving documents."""
        result = await retriever.retrieve("test query")
        assert result.query == "test query"
        assert len(result.documents) == 1
        assert result.total_results == 1

    @pytest.mark.asyncio
    async def test_retrieve_with_limit(self, retriever):
        """Test retrieving with limit."""
        result = await retriever.retrieve("query", limit=3)
        retriever.store.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_with_threshold(self, retriever):
        """Test retrieving with threshold."""
        result = await retriever.retrieve("query", threshold=0.5)
        retriever.store.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_with_filter(self, retriever):
        """Test retrieving with metadata filter."""
        result = await retriever.retrieve("query", metadata_filter={"type": "doc"})
        retriever.store.search.assert_called_once()


class TestRAGRetrieverRetrieveText:
    """Tests for retrieve_text method."""

    @pytest.fixture
    def retriever(self):
        """Create retriever."""
        embedder = MagicMock()
        mock_result = MagicMock()
        mock_result.embedding = [0.1, 0.2]
        embedder.embed_query = AsyncMock(return_value=mock_result)

        store = MagicMock()
        doc1 = Document(id="1", content="First doc", embedding=[0.1])
        doc2 = Document(id="2", content="Second doc", embedding=[0.2])
        store.search = AsyncMock(
            return_value=[
                SearchResult(document=doc1, score=0.9),
                SearchResult(document=doc2, score=0.8),
            ]
        )

        return RAGRetriever(embedder=embedder, store=store)

    @pytest.mark.asyncio
    async def test_retrieve_text(self, retriever):
        """Test retrieving as text."""
        text = await retriever.retrieve_text("query")
        assert "First doc" in text
        assert "Second doc" in text

    @pytest.mark.asyncio
    async def test_retrieve_text_custom_separator(self, retriever):
        """Test retrieving with custom separator."""
        text = await retriever.retrieve_text("query", separator=" | ")
        assert " | " in text


class TestRAGRetrieverOperations:
    """Tests for other retriever operations."""

    @pytest.fixture
    def retriever(self):
        """Create retriever."""
        embedder = MagicMock()
        store = MagicMock()
        store.delete = AsyncMock(return_value=True)
        store.clear = AsyncMock(return_value=5)
        store.count = AsyncMock(return_value=10)
        store.close = AsyncMock()
        return RAGRetriever(embedder=embedder, store=store)

    @pytest.mark.asyncio
    async def test_delete_document(self, retriever):
        """Test deleting a document."""
        result = await retriever.delete_document("doc_id")
        assert result is True
        retriever.store.delete.assert_called_once_with("doc_id")

    @pytest.mark.asyncio
    async def test_clear(self, retriever):
        """Test clearing all documents."""
        count = await retriever.clear()
        assert count == 5
        retriever.store.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_count(self, retriever):
        """Test counting documents."""
        count = await retriever.count()
        assert count == 10
        retriever.store.count.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, retriever):
        """Test closing retriever."""
        await retriever.close()
        retriever.store.close.assert_called_once()

    def test_as_tool(self, retriever):
        """Test creating tool from retriever."""
        tool = retriever.as_tool(name="my_search")
        assert tool.name == "my_search"


class TestRAGRetrieverAddFile:
    """Tests for add_file method."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = MagicMock()
        mock_result = MagicMock()
        mock_result.embedding = [0.1, 0.2]
        embedder.embed_documents = AsyncMock(return_value=[mock_result])
        return embedder

    @pytest.fixture
    def mock_store(self):
        """Create mock store."""
        store = MagicMock()
        store.add_batch = AsyncMock(return_value=["doc_1"])
        return store

    @pytest.mark.asyncio
    async def test_add_file_text(self, mock_embedder, mock_store):
        """Test adding a text file."""
        from locus.rag.multimodal import ContentType, ProcessedContent

        mock_result = ProcessedContent(
            text="File content here",
            content_type=ContentType.TEXT,
            metadata={"encoding": "utf-8"},
        )

        with patch("locus.rag.multimodal.MultimodalProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process = AsyncMock(return_value=mock_result)
            mock_processor_cls.return_value = mock_processor

            retriever = RAGRetriever(
                embedder=mock_embedder,
                store=mock_store,
            )
            ids = await retriever.add_file("/path/to/file.txt")

            assert len(ids) == 1
            mock_processor.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_file_with_metadata(self, mock_embedder, mock_store):
        """Test adding file with additional metadata."""
        from locus.rag.multimodal import ContentType, ProcessedContent

        mock_result = ProcessedContent(
            text="Content",
            content_type=ContentType.TEXT,
            metadata={},
        )

        with patch("locus.rag.multimodal.MultimodalProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process = AsyncMock(return_value=mock_result)
            mock_processor_cls.return_value = mock_processor

            retriever = RAGRetriever(
                embedder=mock_embedder,
                store=mock_store,
            )
            await retriever.add_file(
                "/path/to/doc.pdf",
                metadata={"category": "manuals"},
            )

            docs = mock_store.add_batch.call_args[0][0]
            assert docs[0].metadata["category"] == "manuals"

    @pytest.mark.asyncio
    async def test_add_file_with_chunking(self, mock_embedder, mock_store):
        """Test adding file with chunking enabled."""
        from locus.rag.multimodal import ContentType, ProcessedContent

        # Create long content that will be chunked
        long_content = "A" * 2000

        mock_result = ProcessedContent(
            text=long_content,
            content_type=ContentType.TEXT,
            metadata={},
        )

        mock_embedder.embed_documents = AsyncMock(
            return_value=[MagicMock(embedding=[0.1]), MagicMock(embedding=[0.2])]
        )
        mock_store.add_batch = AsyncMock(return_value=["doc_0", "doc_1"])

        with patch("locus.rag.multimodal.MultimodalProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process = AsyncMock(return_value=mock_result)
            mock_processor_cls.return_value = mock_processor

            retriever = RAGRetriever(
                embedder=mock_embedder,
                store=mock_store,
                chunk_size=1000,
            )
            ids = await retriever.add_file("/path/to/large.txt")

            assert len(ids) == 2

    @pytest.mark.asyncio
    async def test_add_file_with_custom_id(self, mock_embedder, mock_store):
        """Test adding file with custom document ID."""
        from locus.rag.multimodal import ContentType, ProcessedContent

        mock_result = ProcessedContent(
            text="Content",
            content_type=ContentType.PDF,
            metadata={},
        )

        with patch("locus.rag.multimodal.MultimodalProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process = AsyncMock(return_value=mock_result)
            mock_processor_cls.return_value = mock_processor

            retriever = RAGRetriever(
                embedder=mock_embedder,
                store=mock_store,
            )
            await retriever.add_file(
                "/path/to/doc.pdf",
                doc_id="custom_id",
            )

            docs = mock_store.add_batch.call_args[0][0]
            assert docs[0].id == "custom_id"


class TestRAGRetrieverAddImage:
    """Tests for add_image method."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = MagicMock()
        mock_result = MagicMock()
        mock_result.embedding = [0.1, 0.2]
        embedder.embed = AsyncMock(return_value=mock_result)
        return embedder

    @pytest.fixture
    def mock_store(self):
        """Create mock store."""
        store = MagicMock()
        store.add = AsyncMock(return_value="img_1")
        return store

    @pytest.mark.asyncio
    async def test_add_image_bytes(self, mock_embedder, mock_store):
        """Test adding image bytes."""
        from locus.rag.multimodal import ContentType, ProcessedContent

        mock_result = ProcessedContent(
            text="Image description",
            content_type=ContentType.IMAGE,
            metadata={"format": "png"},
            raw_content=b"image bytes",
        )

        with patch("locus.rag.multimodal.ImageProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process = AsyncMock(return_value=mock_result)
            mock_processor_cls.return_value = mock_processor

            retriever = RAGRetriever(
                embedder=mock_embedder,
                store=mock_store,
            )
            doc_id = await retriever.add_image(b"fake image bytes")

            assert doc_id == "img_1"
            mock_embedder.embed.assert_called_once_with("Image description")

    @pytest.mark.asyncio
    async def test_add_image_with_custom_id(self, mock_embedder, mock_store):
        """Test adding image with custom ID."""
        from locus.rag.multimodal import ContentType, ProcessedContent

        mock_result = ProcessedContent(
            text="Description",
            content_type=ContentType.IMAGE,
            metadata={},
            raw_content=b"bytes",
        )

        with patch("locus.rag.multimodal.ImageProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process = AsyncMock(return_value=mock_result)
            mock_processor_cls.return_value = mock_processor

            retriever = RAGRetriever(
                embedder=mock_embedder,
                store=mock_store,
            )
            await retriever.add_image(b"bytes", doc_id="my_image_id")

            call_args = mock_store.add.call_args[0][0]
            assert call_args.id == "my_image_id"

    @pytest.mark.asyncio
    async def test_add_image_with_metadata(self, mock_embedder, mock_store):
        """Test adding image with metadata."""
        from locus.rag.multimodal import ContentType, ProcessedContent

        mock_result = ProcessedContent(
            text="Description",
            content_type=ContentType.IMAGE,
            metadata={"width": 800},
            raw_content=b"bytes",
        )

        with patch("locus.rag.multimodal.ImageProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process = AsyncMock(return_value=mock_result)
            mock_processor_cls.return_value = mock_processor

            retriever = RAGRetriever(
                embedder=mock_embedder,
                store=mock_store,
            )
            await retriever.add_image(b"bytes", metadata={"source": "camera"})

            call_args = mock_store.add.call_args[0][0]
            assert call_args.metadata["source"] == "camera"
            assert call_args.metadata["width"] == 800

    @pytest.mark.asyncio
    async def test_add_image_without_ocr(self, mock_embedder, mock_store):
        """Test adding image without OCR."""
        from locus.rag.multimodal import ContentType, ProcessedContent

        mock_result = ProcessedContent(
            text="No OCR",
            content_type=ContentType.IMAGE,
            metadata={},
            raw_content=b"bytes",
        )

        with patch("locus.rag.multimodal.ImageProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process = AsyncMock(return_value=mock_result)
            mock_processor_cls.return_value = mock_processor

            retriever = RAGRetriever(
                embedder=mock_embedder,
                store=mock_store,
            )
            await retriever.add_image(b"bytes", use_ocr=False)

            mock_processor_cls.assert_called_once_with(use_ocr=False)


class TestRAGRetrieverAddPdf:
    """Tests for add_pdf method."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = MagicMock()
        mock_result = MagicMock()
        mock_result.embedding = [0.1]
        embedder.embed_documents = AsyncMock(return_value=[mock_result])
        return embedder

    @pytest.fixture
    def mock_store(self):
        """Create mock store."""
        store = MagicMock()
        store.add_batch = AsyncMock(return_value=["pdf_1"])
        return store

    @pytest.mark.asyncio
    async def test_add_pdf(self, mock_embedder, mock_store):
        """Test adding PDF document."""
        from locus.rag.multimodal import ContentType, ProcessedContent

        mock_result = ProcessedContent(
            text="PDF content here",
            content_type=ContentType.PDF,
            metadata={"pages": 5},
        )

        with patch("locus.rag.multimodal.PDFProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process = AsyncMock(return_value=mock_result)
            mock_processor_cls.return_value = mock_processor

            retriever = RAGRetriever(
                embedder=mock_embedder,
                store=mock_store,
            )
            ids = await retriever.add_pdf(b"fake pdf bytes")

            assert len(ids) == 1
            mock_processor_cls.assert_called_once_with(use_ocr_fallback=True)

    @pytest.mark.asyncio
    async def test_add_pdf_with_chunking(self, mock_embedder, mock_store):
        """Test adding large PDF with chunking."""
        from locus.rag.multimodal import ContentType, ProcessedContent

        # Long content for chunking
        long_content = "A" * 2000

        mock_result = ProcessedContent(
            text=long_content,
            content_type=ContentType.PDF,
            metadata={},
        )

        mock_embedder.embed_documents = AsyncMock(
            return_value=[MagicMock(embedding=[0.1]), MagicMock(embedding=[0.2])]
        )
        mock_store.add_batch = AsyncMock(return_value=["pdf_0", "pdf_1"])

        with patch("locus.rag.multimodal.PDFProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process = AsyncMock(return_value=mock_result)
            mock_processor_cls.return_value = mock_processor

            retriever = RAGRetriever(
                embedder=mock_embedder,
                store=mock_store,
                chunk_size=1000,
            )
            ids = await retriever.add_pdf(b"pdf bytes")

            assert len(ids) == 2

    @pytest.mark.asyncio
    async def test_add_pdf_no_chunk(self, mock_embedder, mock_store):
        """Test adding PDF without chunking."""
        from locus.rag.multimodal import ContentType, ProcessedContent

        mock_result = ProcessedContent(
            text="Short PDF",
            content_type=ContentType.PDF,
            metadata={},
        )

        with patch("locus.rag.multimodal.PDFProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process = AsyncMock(return_value=mock_result)
            mock_processor_cls.return_value = mock_processor

            retriever = RAGRetriever(
                embedder=mock_embedder,
                store=mock_store,
            )
            ids = await retriever.add_pdf(b"pdf bytes", chunk=False)

            assert len(ids) == 1


class TestRAGRetrieverAddAudio:
    """Tests for add_audio method."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = MagicMock()
        mock_result = MagicMock()
        mock_result.embedding = [0.1]
        embedder.embed = AsyncMock(return_value=mock_result)
        return embedder

    @pytest.fixture
    def mock_store(self):
        """Create mock store."""
        store = MagicMock()
        store.add = AsyncMock(return_value="audio_1")
        return store

    @pytest.mark.asyncio
    async def test_add_audio(self, mock_embedder, mock_store):
        """Test adding audio document."""
        from locus.rag.multimodal import ContentType, ProcessedContent

        mock_result = ProcessedContent(
            text="Transcribed audio content",
            content_type=ContentType.AUDIO,
            metadata={"duration": 120},
            raw_content=b"audio bytes",
        )

        with patch("locus.rag.multimodal.AudioProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process = AsyncMock(return_value=mock_result)
            mock_processor_cls.return_value = mock_processor

            retriever = RAGRetriever(
                embedder=mock_embedder,
                store=mock_store,
            )
            doc_id = await retriever.add_audio(b"fake audio bytes")

            assert doc_id == "audio_1"
            mock_embedder.embed.assert_called_once_with("Transcribed audio content")
            mock_processor_cls.assert_called_once_with(use_whisper=True)

    @pytest.mark.asyncio
    async def test_add_audio_with_metadata(self, mock_embedder, mock_store):
        """Test adding audio with metadata."""
        from locus.rag.multimodal import ContentType, ProcessedContent

        mock_result = ProcessedContent(
            text="Transcript",
            content_type=ContentType.AUDIO,
            metadata={"duration": 60},
            raw_content=b"bytes",
        )

        with patch("locus.rag.multimodal.AudioProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process = AsyncMock(return_value=mock_result)
            mock_processor_cls.return_value = mock_processor

            retriever = RAGRetriever(
                embedder=mock_embedder,
                store=mock_store,
            )
            await retriever.add_audio(b"bytes", metadata={"speaker": "Alice"})

            call_args = mock_store.add.call_args[0][0]
            assert call_args.metadata["speaker"] == "Alice"
            assert call_args.metadata["duration"] == 60

    @pytest.mark.asyncio
    async def test_add_audio_with_custom_id(self, mock_embedder, mock_store):
        """Test adding audio with custom ID."""
        from locus.rag.multimodal import ContentType, ProcessedContent

        mock_result = ProcessedContent(
            text="Transcript",
            content_type=ContentType.AUDIO,
            metadata={},
            raw_content=b"bytes",
        )

        with patch("locus.rag.multimodal.AudioProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process = AsyncMock(return_value=mock_result)
            mock_processor_cls.return_value = mock_processor

            retriever = RAGRetriever(
                embedder=mock_embedder,
                store=mock_store,
            )
            await retriever.add_audio(b"bytes", doc_id="my_audio_id")

            call_args = mock_store.add.call_args[0][0]
            assert call_args.id == "my_audio_id"


class TestRAGRetrieverChunkingEdgeCases:
    """Edge case tests for chunking."""

    def test_chunk_very_large_part(self):
        """Test chunking when a part is larger than chunk_size."""
        retriever = RAGRetriever(
            embedder=MagicMock(),
            store=MagicMock(),
            chunk_size=20,
            chunk_overlap=5,
        )
        # Create text with a very long part (no separator)
        text = "A" * 100
        chunks = retriever._chunk_text(text)

        # Should split into multiple chunks
        assert len(chunks) > 1

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        retriever = RAGRetriever(
            embedder=MagicMock(),
            store=MagicMock(),
        )
        chunks = retriever._chunk_text("")
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_chunk_no_overlap(self):
        """Test chunking with zero overlap."""
        retriever = RAGRetriever(
            embedder=MagicMock(),
            store=MagicMock(),
            chunk_size=50,
            chunk_overlap=0,
        )
        text = "First part.\n\nSecond part.\n\nThird part."
        chunks = retriever._chunk_text(text)
        assert len(chunks) >= 1

    def test_chunk_text_exact_size(self):
        """Test chunking text exactly at chunk_size."""
        retriever = RAGRetriever(
            embedder=MagicMock(),
            store=MagicMock(),
            chunk_size=100,
            chunk_overlap=10,
        )
        text = "A" * 100
        chunks = retriever._chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text
