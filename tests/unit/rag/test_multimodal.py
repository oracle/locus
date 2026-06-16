# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for multimodal content processing."""

from pathlib import Path

import pytest

from locus.rag.multimodal import (
    AudioProcessor,
    ContentType,
    ImageProcessor,
    MultimodalProcessor,
    PDFProcessor,
    ProcessedContent,
    TextProcessor,
    process_content,
)


class TestContentType:
    """Tests for ContentType enum."""

    def test_content_types(self):
        """Test all content types exist."""
        assert ContentType.TEXT.value == "text"
        assert ContentType.IMAGE.value == "image"
        assert ContentType.PDF.value == "pdf"
        assert ContentType.AUDIO.value == "audio"
        assert ContentType.HTML.value == "html"
        assert ContentType.MARKDOWN.value == "markdown"


class TestProcessedContent:
    """Tests for ProcessedContent dataclass."""

    def test_create_processed_content(self):
        """Test creating processed content."""
        result = ProcessedContent(
            text="Extracted text",
            content_type=ContentType.PDF,
            metadata={"pages": 5},
            raw_content=b"pdf bytes",
        )

        assert result.text == "Extracted text"
        assert result.content_type == ContentType.PDF
        assert result.metadata["pages"] == 5
        assert result.raw_content == b"pdf bytes"


class TestTextProcessor:
    """Tests for text processor."""

    @pytest.fixture
    def processor(self):
        return TextProcessor()

    def test_supports(self, processor):
        """Test supported content types."""
        assert processor.supports(ContentType.TEXT)
        assert processor.supports(ContentType.MARKDOWN)
        assert processor.supports(ContentType.HTML)
        assert not processor.supports(ContentType.IMAGE)

    @pytest.mark.asyncio
    async def test_process_string(self, processor):
        """Test processing string content."""
        result = await processor.process("Hello world")

        assert result.text == "Hello world"
        assert result.content_type == ContentType.TEXT
        assert result.metadata["length"] == 11

    @pytest.mark.asyncio
    async def test_process_bytes(self, processor):
        """Test processing bytes content."""
        result = await processor.process(b"Hello world")

        assert result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_process_html(self, processor):
        """Test HTML stripping."""
        html = "<html><body><p>Hello</p><script>evil()</script></body></html>"
        result = await processor.process(html, content_type=ContentType.HTML)

        assert "Hello" in result.text
        assert "script" not in result.text.lower()
        assert "evil" not in result.text


class TestImageProcessor:
    """Tests for image processor."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor(use_ocr=False)

    def test_supports(self, processor):
        """Test supported content types."""
        assert processor.supports(ContentType.IMAGE)
        assert not processor.supports(ContentType.TEXT)

    def test_detect_format_png(self, processor):
        """Test PNG format detection."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        assert processor._detect_format(png_header) == "png"

    def test_detect_format_jpeg(self, processor):
        """Test JPEG format detection."""
        jpeg_header = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        assert processor._detect_format(jpeg_header) == "jpeg"

    def test_detect_format_gif(self, processor):
        """Test GIF format detection."""
        gif_header = b"GIF89a" + b"\x00" * 100
        assert processor._detect_format(gif_header) == "gif"

    @pytest.mark.asyncio
    async def test_process_without_ocr(self, processor):
        """Test processing image without OCR."""
        # Create a minimal PNG
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        result = await processor.process(png_data)

        assert result.content_type == ContentType.IMAGE
        assert "Image:" in result.text
        assert result.metadata["format"] == "png"
        assert result.raw_content == png_data


class TestPDFProcessor:
    """Tests for PDF processor."""

    @pytest.fixture
    def processor(self):
        return PDFProcessor(use_ocr_fallback=False)

    def test_supports(self, processor):
        """Test supported content types."""
        assert processor.supports(ContentType.PDF)
        assert not processor.supports(ContentType.TEXT)

    @pytest.mark.asyncio
    async def test_process_invalid_pdf(self, processor):
        """Test processing invalid PDF content."""
        result = await processor.process(b"not a pdf")

        assert result.content_type == ContentType.PDF
        assert "extraction failed" in result.text.lower() or "PDF:" in result.text
        assert result.metadata.get("extraction_method") in ("failed", None)


class TestAudioProcessor:
    """Tests for audio processor."""

    @pytest.fixture
    def processor(self):
        return AudioProcessor(use_whisper=False)

    def test_supports(self, processor):
        """Test supported content types."""
        assert processor.supports(ContentType.AUDIO)
        assert not processor.supports(ContentType.TEXT)

    def test_detect_format_wav(self, processor):
        """Test WAV format detection."""
        wav_header = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 100
        assert processor._detect_format(wav_header, "test.wav") == "wav"

    def test_detect_format_mp3(self, processor):
        """Test MP3 format detection."""
        mp3_header = b"ID3" + b"\x00" * 100
        assert processor._detect_format(mp3_header, "test.mp3") == "mp3"

    def test_detect_format_from_extension(self, processor):
        """Test format detection from file extension."""
        assert processor._detect_format(b"\x00" * 100, "audio.m4a") == "m4a"


class TestMultimodalProcessor:
    """Tests for unified multimodal processor."""

    @pytest.fixture
    def processor(self):
        return MultimodalProcessor(use_ocr=False, use_whisper=False)

    def test_detect_content_type_from_bytes(self, processor):
        """Test content type detection from bytes."""
        # PNG
        assert (
            processor.detect_content_type(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100) == ContentType.IMAGE
        )

        # PDF
        assert processor.detect_content_type(b"%PDF-1.4" + b"\x00" * 100) == ContentType.PDF

        # WAV
        assert (
            processor.detect_content_type(b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 100)
            == ContentType.AUDIO
        )

        # Unknown defaults to TEXT
        assert processor.detect_content_type(b"unknown content") == ContentType.TEXT

    def test_detect_content_type_from_path(self, processor):
        """Test content type detection from path."""
        assert processor.detect_content_type(Path("doc.pdf")) == ContentType.PDF
        assert processor.detect_content_type(Path("image.png")) == ContentType.IMAGE
        assert processor.detect_content_type(Path("audio.mp3")) == ContentType.AUDIO
        assert processor.detect_content_type(Path("page.html")) == ContentType.HTML

    @pytest.mark.asyncio
    async def test_process_text(self, processor):
        """Test processing text content."""
        result = await processor.process("Hello world", ContentType.TEXT)

        assert result.text == "Hello world"
        assert result.content_type == ContentType.TEXT

    @pytest.mark.asyncio
    async def test_process_with_auto_detection(self, processor):
        """Test processing with automatic type detection."""
        # Text content
        result = await processor.process("Plain text content")

        assert result.content_type == ContentType.TEXT
        assert result.text == "Plain text content"


class TestProcessContentFunction:
    """Tests for process_content convenience function."""

    @pytest.mark.asyncio
    async def test_process_text(self):
        """Test processing text."""
        result = await process_content("Hello world")

        assert result.text == "Hello world"
        assert result.content_type == ContentType.TEXT

    @pytest.mark.asyncio
    async def test_process_with_explicit_type(self):
        """Test processing with explicit content type."""
        html = "<p>Hello</p>"
        result = await process_content(html, content_type=ContentType.HTML)

        assert "Hello" in result.text
        assert result.content_type == ContentType.HTML
