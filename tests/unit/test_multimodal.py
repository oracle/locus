# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for multimodal content processing."""

import base64
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from locus.rag.multimodal import (
    AudioProcessor,
    ContentType,
    ImageProcessor,
    MultimodalProcessor,
    PDFProcessor,
    ProcessedContent,
    TextProcessor,
)


class TestContentType:
    """Tests for ContentType enum."""

    def test_content_types(self):
        """Test all content types exist."""
        assert ContentType.TEXT == "text"
        assert ContentType.IMAGE == "image"
        assert ContentType.PDF == "pdf"
        assert ContentType.AUDIO == "audio"
        assert ContentType.HTML == "html"
        assert ContentType.MARKDOWN == "markdown"


class TestProcessedContent:
    """Tests for ProcessedContent dataclass."""

    def test_basic_creation(self):
        """Test creating ProcessedContent."""
        content = ProcessedContent(
            text="Hello world",
            content_type=ContentType.TEXT,
        )

        assert content.text == "Hello world"
        assert content.content_type == ContentType.TEXT
        assert content.metadata == {}
        assert content.chunks is None
        assert content.raw_content is None

    def test_with_all_fields(self):
        """Test creating ProcessedContent with all fields."""
        content = ProcessedContent(
            text="Test text",
            content_type=ContentType.IMAGE,
            metadata={"width": 100, "height": 200},
            chunks=["chunk1", "chunk2"],
            raw_content=b"raw bytes",
        )

        assert content.text == "Test text"
        assert content.content_type == ContentType.IMAGE
        assert content.metadata == {"width": 100, "height": 200}
        assert content.chunks == ["chunk1", "chunk2"]
        assert content.raw_content == b"raw bytes"


class TestTextProcessor:
    """Tests for TextProcessor."""

    @pytest.fixture
    def processor(self):
        return TextProcessor()

    def test_supports_text(self, processor):
        """Test supports text content type."""
        assert processor.supports(ContentType.TEXT) is True
        assert processor.supports(ContentType.MARKDOWN) is True
        assert processor.supports(ContentType.HTML) is True
        assert processor.supports(ContentType.IMAGE) is False
        assert processor.supports(ContentType.PDF) is False

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
        result = await processor.process(b"Hello bytes")

        assert result.text == "Hello bytes"
        assert result.metadata["length"] == 11

    @pytest.mark.asyncio
    async def test_process_path(self, processor):
        """Test processing Path content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Content from file")
            temp_path = Path(f.name)

        try:
            result = await processor.process(temp_path)
            assert result.text == "Content from file"
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_process_html_strips_tags(self, processor):
        """Test HTML processing strips tags."""
        html = "<html><head><script>alert('x')</script></head><body><p>Hello</p></body></html>"
        result = await processor.process(html, content_type=ContentType.HTML)

        assert result.content_type == ContentType.HTML
        assert "<p>" not in result.text
        assert "Hello" in result.text
        assert "alert" not in result.text

    @pytest.mark.asyncio
    async def test_process_html_strips_style(self, processor):
        """Test HTML processing strips style tags."""
        html = "<div><style>body { color: red; }</style>Content</div>"
        result = await processor.process(html, content_type=ContentType.HTML)

        assert "color:" not in result.text
        assert "Content" in result.text


class TestImageProcessor:
    """Tests for ImageProcessor."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor(use_ocr=True, use_vision_llm=False)

    def test_supports_image(self, processor):
        """Test supports image content type."""
        assert processor.supports(ContentType.IMAGE) is True
        assert processor.supports(ContentType.TEXT) is False
        assert processor.supports(ContentType.PDF) is False

    def test_detect_format_png(self, processor):
        """Test PNG format detection."""
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        assert processor._detect_format(png_bytes) == "png"

    def test_detect_format_jpeg(self, processor):
        """Test JPEG format detection."""
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        assert processor._detect_format(jpeg_bytes) == "jpeg"

    def test_detect_format_gif87a(self, processor):
        """Test GIF87a format detection."""
        gif_bytes = b"GIF87a" + b"\x00" * 100
        assert processor._detect_format(gif_bytes) == "gif"

    def test_detect_format_gif89a(self, processor):
        """Test GIF89a format detection."""
        gif_bytes = b"GIF89a" + b"\x00" * 100
        assert processor._detect_format(gif_bytes) == "gif"

    def test_detect_format_webp(self, processor):
        """Test WebP format detection."""
        webp_bytes = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 100
        assert processor._detect_format(webp_bytes) == "webp"

    def test_detect_format_unknown(self, processor):
        """Test unknown format detection."""
        unknown_bytes = b"\x00\x00\x00\x00" + b"\x00" * 100
        assert processor._detect_format(unknown_bytes) == "unknown"

    @pytest.mark.asyncio
    async def test_process_image_bytes(self, processor):
        """Test processing image bytes."""
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        with patch.object(processor, "_extract_text_ocr", new_callable=AsyncMock) as mock_ocr:
            mock_ocr.return_value = None

            result = await processor.process(png_bytes)

            assert result.content_type == ContentType.IMAGE
            assert result.metadata["format"] == "png"
            assert result.metadata["size_bytes"] == len(png_bytes)
            assert result.raw_content == png_bytes

    @pytest.mark.asyncio
    async def test_process_image_with_ocr_text(self, processor):
        """Test processing image with OCR result."""
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        with patch.object(processor, "_extract_text_ocr", new_callable=AsyncMock) as mock_ocr:
            mock_ocr.return_value = "Extracted text from image"

            result = await processor.process(png_bytes)

            assert "OCR Text" in result.text
            assert "Extracted text from image" in result.text
            assert result.metadata["ocr_text"] == "Extracted text from image"

    @pytest.mark.asyncio
    async def test_process_image_from_path(self, processor):
        """Test processing image from Path."""
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            temp_path = Path(f.name)

        try:
            with patch.object(processor, "_extract_text_ocr", new_callable=AsyncMock) as mock_ocr:
                mock_ocr.return_value = None

                result = await processor.process(temp_path)

                assert result.metadata["filename"] == temp_path.name
                assert result.metadata["format"] == "png"
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_process_image_from_base64(self, processor):
        """Test processing base64 encoded image."""
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        b64_content = base64.b64encode(png_bytes).decode()

        with patch.object(processor, "_extract_text_ocr", new_callable=AsyncMock) as mock_ocr:
            mock_ocr.return_value = None

            result = await processor.process(b64_content)

            assert result.metadata["format"] == "png"

    @pytest.mark.asyncio
    async def test_process_with_vision_llm(self):
        """Test processing with vision LLM."""
        mock_model = MagicMock()
        processor = ImageProcessor(use_ocr=False, use_vision_llm=True, vision_model=mock_model)
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        with patch.object(
            processor, "_get_vision_description", new_callable=AsyncMock
        ) as mock_vision:
            mock_vision.return_value = "A beautiful sunset"

            result = await processor.process(png_bytes)

            assert "Image Description" in result.text
            assert "A beautiful sunset" in result.text
            assert result.metadata["description"] == "A beautiful sunset"

    @pytest.mark.asyncio
    async def test_extract_text_ocr_import_error(self, processor):
        """Test OCR handles ImportError."""
        with patch.dict("sys.modules", {"pytesseract": None}):
            result = await processor._extract_text_ocr(b"\x00" * 100)
            # Should return None, not raise
            assert result is None

    @pytest.mark.asyncio
    async def test_get_vision_description_no_model(self, processor):
        """Test vision description without model."""
        result = await processor._get_vision_description(b"\x00" * 100)
        assert result is None


class TestPDFProcessor:
    """Tests for PDFProcessor."""

    @pytest.fixture
    def processor(self):
        return PDFProcessor(use_ocr_fallback=True)

    def test_supports_pdf(self, processor):
        """Test supports PDF content type."""
        assert processor.supports(ContentType.PDF) is True
        assert processor.supports(ContentType.TEXT) is False
        assert processor.supports(ContentType.IMAGE) is False

    @pytest.mark.asyncio
    async def test_process_pdf_bytes(self, processor):
        """Test processing PDF bytes."""
        pdf_bytes = b"%PDF-1.4" + b"\x00" * 100

        with patch.object(processor, "_extract_with_pypdf", new_callable=AsyncMock) as mock_pypdf:
            mock_pypdf.return_value = "--- Page 1 ---\nContent"

            result = await processor.process(pdf_bytes)

            assert result.content_type == ContentType.PDF
            assert result.metadata["extraction_method"] == "pypdf"
            assert "Content" in result.text

    @pytest.mark.asyncio
    async def test_process_pdf_from_path(self, processor):
        """Test processing PDF from Path."""
        pdf_bytes = b"%PDF-1.4" + b"\x00" * 100

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            temp_path = Path(f.name)

        try:
            with patch.object(
                processor, "_extract_with_pypdf", new_callable=AsyncMock
            ) as mock_pypdf:
                mock_pypdf.return_value = "Text from PDF"

                result = await processor.process(temp_path)

                assert result.metadata["filename"] == temp_path.name
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_process_pdf_from_base64(self, processor):
        """Test processing base64 encoded PDF."""
        pdf_bytes = b"%PDF-1.4" + b"\x00" * 100
        b64_content = base64.b64encode(pdf_bytes).decode()

        with patch.object(processor, "_extract_with_pypdf", new_callable=AsyncMock) as mock_pypdf:
            mock_pypdf.return_value = "Text"

            result = await processor.process(b64_content)

            assert result.metadata["filename"] == "document.pdf"

    @pytest.mark.asyncio
    async def test_process_pdf_ocr_fallback(self, processor):
        """Test OCR fallback when pypdf fails."""
        pdf_bytes = b"%PDF-1.4" + b"\x00" * 100

        with (
            patch.object(processor, "_extract_with_pypdf", new_callable=AsyncMock) as mock_pypdf,
            patch.object(processor, "_extract_with_ocr", new_callable=AsyncMock) as mock_ocr,
        ):
            mock_pypdf.return_value = None
            mock_ocr.return_value = "OCR extracted text"

            result = await processor.process(pdf_bytes)

            assert result.metadata["extraction_method"] == "ocr"
            assert "OCR extracted text" in result.text

    @pytest.mark.asyncio
    async def test_process_pdf_extraction_failed(self, processor):
        """Test handling when all extraction fails."""
        pdf_bytes = b"%PDF-1.4" + b"\x00" * 100

        with (
            patch.object(processor, "_extract_with_pypdf", new_callable=AsyncMock) as mock_pypdf,
            patch.object(processor, "_extract_with_ocr", new_callable=AsyncMock) as mock_ocr,
        ):
            mock_pypdf.return_value = None
            mock_ocr.return_value = None

            result = await processor.process(pdf_bytes)

            assert result.metadata["extraction_method"] == "failed"
            assert "text extraction failed" in result.text

    @pytest.mark.asyncio
    async def test_extract_with_pypdf_import_error(self, processor):
        """Test pypdf handles ImportError."""
        with patch.dict("sys.modules", {"pypdf": None, "PyPDF2": None}):
            result = await processor._extract_with_pypdf(b"%PDF" + b"\x00" * 100)
            assert result is None


class TestAudioProcessor:
    """Tests for AudioProcessor."""

    @pytest.fixture
    def processor(self):
        return AudioProcessor(use_whisper=True, whisper_model="base")

    def test_supports_audio(self, processor):
        """Test supports audio content type."""
        assert processor.supports(ContentType.AUDIO) is True
        assert processor.supports(ContentType.TEXT) is False
        assert processor.supports(ContentType.IMAGE) is False

    def test_detect_format_wav(self, processor):
        """Test WAV format detection."""
        wav_bytes = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 100
        assert processor._detect_format(wav_bytes, "audio") == "wav"

    def test_detect_format_mp3_id3(self, processor):
        """Test MP3 with ID3 format detection."""
        mp3_bytes = b"ID3" + b"\x00" * 100
        assert processor._detect_format(mp3_bytes, "audio") == "mp3"

    def test_detect_format_mp3_sync(self, processor):
        """Test MP3 sync bytes format detection."""
        mp3_bytes = b"\xff\xfb" + b"\x00" * 100
        assert processor._detect_format(mp3_bytes, "audio") == "mp3"

    def test_detect_format_flac(self, processor):
        """Test FLAC format detection."""
        flac_bytes = b"fLaC" + b"\x00" * 100
        assert processor._detect_format(flac_bytes, "audio") == "flac"

    def test_detect_format_ogg(self, processor):
        """Test OGG format detection."""
        ogg_bytes = b"OggS" + b"\x00" * 100
        assert processor._detect_format(ogg_bytes, "audio") == "ogg"

    def test_detect_format_m4a(self, processor):
        """Test M4A format detection."""
        m4a_bytes = b"\x00\x00\x00\x00ftypM4A " + b"\x00" * 100
        assert processor._detect_format(m4a_bytes, "audio") == "m4a"

    def test_detect_format_from_extension(self, processor):
        """Test format detection from extension."""
        unknown_bytes = b"\x00" * 100
        assert processor._detect_format(unknown_bytes, "audio.aac") == "aac"

    def test_detect_format_unknown(self, processor):
        """Test unknown format detection."""
        unknown_bytes = b"\x00" * 100
        assert processor._detect_format(unknown_bytes, "audio") == "unknown"

    @pytest.mark.asyncio
    async def test_process_audio_bytes(self, processor):
        """Test processing audio bytes."""
        wav_bytes = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 100

        with patch.object(processor, "_transcribe_whisper", new_callable=AsyncMock) as mock_whisper:
            mock_whisper.return_value = "Hello world"

            result = await processor.process(wav_bytes)

            assert result.content_type == ContentType.AUDIO
            assert result.metadata["format"] == "wav"
            assert result.text == "Hello world"
            assert result.metadata["transcription_method"] == "whisper"

    @pytest.mark.asyncio
    async def test_process_audio_from_path(self, processor):
        """Test processing audio from Path."""
        wav_bytes = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 100

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            temp_path = Path(f.name)

        try:
            with patch.object(
                processor, "_transcribe_whisper", new_callable=AsyncMock
            ) as mock_whisper:
                mock_whisper.return_value = "Transcribed text"

                result = await processor.process(temp_path)

                assert result.metadata["filename"] == temp_path.name
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_process_audio_from_base64(self, processor):
        """Test processing base64 encoded audio."""
        wav_bytes = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 100
        b64_content = base64.b64encode(wav_bytes).decode()

        with patch.object(processor, "_transcribe_whisper", new_callable=AsyncMock) as mock_whisper:
            mock_whisper.return_value = "Transcribed"

            result = await processor.process(b64_content)

            assert result.metadata["filename"] == "audio"

    @pytest.mark.asyncio
    async def test_process_audio_transcription_unavailable(self, processor):
        """Test handling when transcription fails."""
        wav_bytes = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 100

        with patch.object(processor, "_transcribe_whisper", new_callable=AsyncMock) as mock_whisper:
            mock_whisper.return_value = None

            result = await processor.process(wav_bytes)

            assert result.metadata["transcription_method"] == "unavailable"
            assert "transcription unavailable" in result.text

    @pytest.mark.asyncio
    async def test_transcribe_whisper_import_error(self, processor):
        """Test whisper handles ImportError."""
        with patch.dict("sys.modules", {"whisper": None}):
            result = await processor._transcribe_whisper(b"\x00" * 100, "wav")
            assert result is None


class TestMultimodalProcessor:
    """Tests for MultimodalProcessor."""

    @pytest.fixture
    def processor(self):
        return MultimodalProcessor(use_ocr=True, use_whisper=True)

    def test_initialization(self, processor):
        """Test MultimodalProcessor initialization."""
        assert ContentType.TEXT in processor.processors
        assert ContentType.MARKDOWN in processor.processors
        assert ContentType.HTML in processor.processors
        assert ContentType.IMAGE in processor.processors
        assert ContentType.PDF in processor.processors
        assert ContentType.AUDIO in processor.processors

    def test_detect_content_type_from_path_txt(self, processor):
        """Test content type detection from path."""
        path = Path("test_files") / "document.txt"
        assert processor.detect_content_type(path) == ContentType.TEXT

    def test_detect_content_type_from_path_pdf(self, processor):
        """Test content type detection from PDF path."""
        path = Path("test_files") / "document.pdf"
        assert processor.detect_content_type(path) == ContentType.PDF

    def test_detect_content_type_from_path_png(self, processor):
        """Test content type detection from PNG path."""
        path = Path("test_files") / "image.png"
        assert processor.detect_content_type(path) == ContentType.IMAGE

    def test_detect_content_type_from_path_jpg(self, processor):
        """Test content type detection from JPG path."""
        path = Path("test_files") / "image.jpg"
        assert processor.detect_content_type(path) == ContentType.IMAGE

    def test_detect_content_type_from_path_mp3(self, processor):
        """Test content type detection from MP3 path."""
        path = Path("test_files") / "audio.mp3"
        assert processor.detect_content_type(path) == ContentType.AUDIO

    def test_detect_content_type_from_path_html(self, processor):
        """Test content type detection from HTML path."""
        path = Path("test_files") / "page.html"
        assert processor.detect_content_type(path) == ContentType.HTML

    def test_detect_content_type_from_path_md(self, processor):
        """Test content type detection from Markdown path."""
        path = Path("test_files") / "readme.md"
        assert processor.detect_content_type(path) == ContentType.MARKDOWN

    def test_detect_content_type_from_string_path(self, processor):
        """Test content type detection from string path."""
        assert processor.detect_content_type("test_files/doc.pdf") == ContentType.PDF

    def test_detect_content_type_from_bytes_png(self, processor):
        """Test content type detection from PNG bytes."""
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        assert processor.detect_content_type(png_bytes) == ContentType.IMAGE

    def test_detect_content_type_from_bytes_pdf(self, processor):
        """Test content type detection from PDF bytes."""
        pdf_bytes = b"%PDF-1.4" + b"\x00" * 100
        assert processor.detect_content_type(pdf_bytes) == ContentType.PDF

    def test_detect_content_type_unknown(self, processor):
        """Test content type detection for unknown bytes."""
        unknown_bytes = b"\x00" * 100
        result = processor.detect_content_type(unknown_bytes)
        # Unknown bytes default to TEXT
        assert result == ContentType.TEXT
