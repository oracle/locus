# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for RAG multimodal processing."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from locus.rag.multimodal import (
    ContentType,
    MultimodalProcessor,
    ProcessedContent,
    TextProcessor,
    process_content,
)


class TestContentType:
    """Tests for ContentType enum."""

    def test_text_type(self):
        """Text content type."""
        assert ContentType.TEXT.value == "text"

    def test_image_type(self):
        """Image content type."""
        assert ContentType.IMAGE.value == "image"

    def test_audio_type(self):
        """Audio content type."""
        assert ContentType.AUDIO.value == "audio"

    def test_pdf_type(self):
        """PDF content type."""
        assert ContentType.PDF.value == "pdf"

    def test_html_type(self):
        """HTML content type."""
        assert ContentType.HTML.value == "html"

    def test_markdown_type(self):
        """Markdown content type."""
        assert ContentType.MARKDOWN.value == "markdown"


class TestProcessedContent:
    """Tests for ProcessedContent dataclass."""

    def test_create_text_content(self):
        """Create text content."""
        content = ProcessedContent(
            text="Hello world",
            content_type=ContentType.TEXT,
        )

        assert content.content_type == ContentType.TEXT
        assert content.text == "Hello world"
        assert content.metadata == {}
        assert content.chunks is None
        assert content.raw_content is None

    def test_create_with_metadata(self):
        """Create content with metadata."""
        content = ProcessedContent(
            text="Hello",
            content_type=ContentType.TEXT,
            metadata={"length": 5, "language": "en"},
        )

        assert content.metadata == {"length": 5, "language": "en"}

    def test_create_image_content(self):
        """Create image content."""
        content = ProcessedContent(
            text="Image description",
            content_type=ContentType.IMAGE,
            metadata={"width": 800, "height": 600},
            raw_content=b"fake image bytes",
        )

        assert content.content_type == ContentType.IMAGE
        assert content.metadata["width"] == 800
        assert content.raw_content == b"fake image bytes"

    def test_create_with_chunks(self):
        """Create content with chunks."""
        content = ProcessedContent(
            text="Full text",
            content_type=ContentType.TEXT,
            chunks=["chunk1", "chunk2", "chunk3"],
        )

        assert content.chunks == ["chunk1", "chunk2", "chunk3"]


class TestTextProcessor:
    """Tests for TextProcessor."""

    @pytest.fixture
    def processor(self):
        """Create text processor."""
        return TextProcessor()

    def test_supports_text(self, processor):
        """Supports TEXT content type."""
        assert processor.supports(ContentType.TEXT) is True

    def test_supports_markdown(self, processor):
        """Supports MARKDOWN content type."""
        assert processor.supports(ContentType.MARKDOWN) is True

    def test_supports_html(self, processor):
        """Supports HTML content type."""
        assert processor.supports(ContentType.HTML) is True

    def test_does_not_support_image(self, processor):
        """Does not support IMAGE content type."""
        assert processor.supports(ContentType.IMAGE) is False

    @pytest.mark.asyncio
    async def test_process_string(self, processor):
        """Process string content."""
        result = await processor.process("Hello world")

        assert result.content_type == ContentType.TEXT
        assert result.text == "Hello world"
        assert result.metadata["length"] == 11

    @pytest.mark.asyncio
    async def test_process_bytes(self, processor):
        """Process bytes content."""
        result = await processor.process(b"Hello bytes")

        assert result.text == "Hello bytes"

    @pytest.mark.asyncio
    async def test_strip_html(self, processor):
        """Strip HTML tags from content."""
        html = "<html><body><p>Hello <b>world</b></p></body></html>"
        result = await processor.process(html, content_type=ContentType.HTML)

        assert "<" not in result.text
        assert "Hello" in result.text
        assert "world" in result.text


class TestMultimodalProcessor:
    """Tests for MultimodalProcessor."""

    def test_init_default(self):
        """Initialize with defaults."""
        processor = MultimodalProcessor()
        assert processor is not None
        assert ContentType.TEXT in processor.processors
        assert ContentType.IMAGE in processor.processors

    def test_init_without_ocr(self):
        """Initialize without OCR."""
        processor = MultimodalProcessor(use_ocr=False)
        assert processor is not None

    def test_init_without_whisper(self):
        """Initialize without whisper."""
        processor = MultimodalProcessor(use_whisper=False)
        assert processor is not None

    def test_detect_content_type_text(self):
        """Detect text content type."""
        processor = MultimodalProcessor()
        content_type = processor.detect_content_type("plain text content")
        assert content_type == ContentType.TEXT

    def test_detect_content_type_png_bytes(self):
        """Detect PNG from magic bytes."""
        processor = MultimodalProcessor()
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        content_type = processor.detect_content_type(png_header)
        assert content_type == ContentType.IMAGE

    def test_detect_content_type_jpeg_bytes(self):
        """Detect JPEG from magic bytes."""
        processor = MultimodalProcessor()
        jpeg_header = b"\xff\xd8" + b"\x00" * 100
        content_type = processor.detect_content_type(jpeg_header)
        assert content_type == ContentType.IMAGE

    def test_detect_content_type_pdf_bytes(self):
        """Detect PDF from magic bytes."""
        processor = MultimodalProcessor()
        pdf_header = b"%PDF-1.4" + b"\x00" * 100
        content_type = processor.detect_content_type(pdf_header)
        assert content_type == ContentType.PDF

    @pytest.mark.asyncio
    async def test_process_text(self):
        """Process plain text."""
        processor = MultimodalProcessor()

        result = await processor.process("Hello world", content_type=ContentType.TEXT)

        assert result.content_type == ContentType.TEXT
        assert result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_process_unsupported_type(self):
        """Process unsupported content type raises error."""
        processor = MultimodalProcessor()
        # Remove a processor to simulate unsupported type
        del processor.processors[ContentType.AUDIO]

        with pytest.raises(ValueError, match="No processor"):
            await processor.process(b"audio data", content_type=ContentType.AUDIO)


class TestProcessContent:
    """Tests for process_content helper function."""

    @pytest.mark.asyncio
    async def test_process_string(self):
        """Process string content."""
        result = await process_content("Hello world")

        assert result.content_type == ContentType.TEXT
        assert result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_process_with_type_hint(self):
        """Process content with type hint."""
        result = await process_content(
            "Some text content",
            content_type=ContentType.MARKDOWN,
        )

        assert result.content_type == ContentType.MARKDOWN


class TestImageProcessor:
    """Tests for ImageProcessor."""

    @pytest.fixture
    def processor(self):
        """Create image processor."""
        from locus.rag.multimodal import ImageProcessor

        return ImageProcessor(use_ocr=False, use_vision_llm=False)

    def test_supports_image(self, processor):
        """Test supports IMAGE content type."""
        assert processor.supports(ContentType.IMAGE) is True

    def test_does_not_support_text(self, processor):
        """Test does not support TEXT."""
        assert processor.supports(ContentType.TEXT) is False

    def test_detect_format_png(self, processor):
        """Test detecting PNG format."""
        png_header = b"\x89PNG\r\n\x1a\n"
        assert processor._detect_format(png_header) == "png"

    def test_detect_format_jpeg(self, processor):
        """Test detecting JPEG format."""
        jpeg_header = b"\xff\xd8\xff"
        assert processor._detect_format(jpeg_header) == "jpeg"

    def test_detect_format_gif87(self, processor):
        """Test detecting GIF87a format."""
        gif_header = b"GIF87a"
        assert processor._detect_format(gif_header) == "gif"

    def test_detect_format_gif89(self, processor):
        """Test detecting GIF89a format."""
        gif_header = b"GIF89a"
        assert processor._detect_format(gif_header) == "gif"

    def test_detect_format_webp(self, processor):
        """Test detecting WebP format."""
        webp_header = b"RIFF" + b"\x00" * 4 + b"WEBP"
        assert processor._detect_format(webp_header) == "webp"

    def test_detect_format_unknown(self, processor):
        """Test unknown format."""
        assert processor._detect_format(b"unknown") == "unknown"

    @pytest.mark.asyncio
    async def test_process_png_bytes(self, processor):
        """Test processing PNG bytes."""
        # Minimal PNG bytes (header + minimal data)
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        result = await processor.process(png_data)

        assert result.content_type == ContentType.IMAGE
        assert result.metadata["format"] == "png"
        assert result.raw_content == png_data


class TestPDFProcessor:
    """Tests for PDFProcessor."""

    @pytest.fixture
    def processor(self):
        """Create PDF processor."""
        from locus.rag.multimodal import PDFProcessor

        return PDFProcessor(use_ocr_fallback=False)

    def test_supports_pdf(self, processor):
        """Test supports PDF content type."""
        assert processor.supports(ContentType.PDF) is True

    def test_does_not_support_text(self, processor):
        """Test does not support TEXT."""
        assert processor.supports(ContentType.TEXT) is False


class TestAudioProcessor:
    """Tests for AudioProcessor."""

    @pytest.fixture
    def processor(self):
        """Create audio processor."""
        from locus.rag.multimodal import AudioProcessor

        return AudioProcessor(use_whisper=False)

    def test_supports_audio(self, processor):
        """Test supports AUDIO content type."""
        assert processor.supports(ContentType.AUDIO) is True

    def test_does_not_support_text(self, processor):
        """Test does not support TEXT."""
        assert processor.supports(ContentType.TEXT) is False


class TestMultimodalProcessorDetection:
    """Tests for content type detection in MultimodalProcessor."""

    @pytest.fixture
    def processor(self):
        """Create multimodal processor."""
        return MultimodalProcessor()

    def test_detect_gif_bytes(self, processor):
        """Test detecting GIF from magic bytes."""
        gif_header = b"GIF89a" + b"\x00" * 100
        content_type = processor.detect_content_type(gif_header)
        assert content_type == ContentType.IMAGE

    def test_detect_string_as_text(self, processor):
        """Test detecting string content as TEXT."""
        html = "<!DOCTYPE html><html><body>Hello</body></html>"
        content_type = processor.detect_content_type(html)
        # String content is detected as TEXT by default
        assert content_type == ContentType.TEXT

    def test_detect_markdown_content(self, processor):
        """Test detecting markdown content."""
        markdown = "# Heading\n\nParagraph with **bold** text."
        content_type = processor.detect_content_type(markdown)
        # May detect as TEXT or MARKDOWN depending on implementation
        assert content_type in [ContentType.TEXT, ContentType.MARKDOWN]


class TestMultimodalProcessorCustomProcessors:
    """Tests for MultimodalProcessor with custom processors."""

    def test_add_custom_processor(self):
        """Test adding a custom processor to the processors dict."""
        processor = MultimodalProcessor()

        # Create mock processor

        custom_proc = MagicMock()
        custom_proc.supports = MagicMock(side_effect=lambda ct: ct == ContentType.TEXT)
        custom_proc.process = AsyncMock(
            return_value=ProcessedContent(
                text="Custom processed",
                content_type=ContentType.TEXT,
            )
        )

        # Add directly to processors dict
        processor.processors[ContentType.TEXT] = custom_proc

        assert ContentType.TEXT in processor.processors
        assert processor.processors[ContentType.TEXT] is custom_proc


class TestTextProcessorHtmlStripping:
    """Additional tests for TextProcessor HTML stripping."""

    @pytest.fixture
    def processor(self):
        """Create text processor."""
        return TextProcessor()

    @pytest.mark.asyncio
    async def test_strip_complex_html(self, processor):
        """Test stripping complex HTML structures."""
        html = """
        <html>
            <head><title>Test</title></head>
            <body>
                <div class="container">
                    <h1>Header</h1>
                    <p>Paragraph with <a href="#">link</a></p>
                    <ul>
                        <li>Item 1</li>
                        <li>Item 2</li>
                    </ul>
                </div>
            </body>
        </html>
        """
        result = await processor.process(html, content_type=ContentType.HTML)

        assert "<html>" not in result.text
        assert "<div" not in result.text
        assert "Header" in result.text
        assert "Paragraph" in result.text

    @pytest.mark.asyncio
    async def test_process_markdown(self, processor):
        """Test processing markdown content."""
        markdown = "# Title\n\n**Bold** and *italic* text."
        result = await processor.process(markdown, content_type=ContentType.MARKDOWN)

        assert result.content_type == ContentType.MARKDOWN
        assert "Title" in result.text

    @pytest.mark.asyncio
    async def test_process_plain_text(self, processor):
        """Test processing plain text."""
        text = "This is plain text with\nmultiple lines."
        result = await processor.process(text)

        assert result.text == text
        assert result.content_type == ContentType.TEXT


class TestProcessedContentChunking:
    """Tests for ProcessedContent with chunks."""

    def test_content_with_many_chunks(self):
        """Test content with multiple chunks."""
        chunks = [f"Chunk {i}" for i in range(10)]
        content = ProcessedContent(
            text="Full text",
            content_type=ContentType.TEXT,
            chunks=chunks,
        )

        assert len(content.chunks) == 10
        assert content.chunks[0] == "Chunk 0"
        assert content.chunks[9] == "Chunk 9"

    def test_content_with_raw_content(self):
        """Test content preserving raw content."""
        raw = b"binary data here"
        content = ProcessedContent(
            text="Description of binary",
            content_type=ContentType.IMAGE,
            raw_content=raw,
            metadata={"size": len(raw)},
        )

        assert content.raw_content == raw
        assert content.metadata["size"] == 16


class TestMultimodalProcessorBatchProcessing:
    """Tests for batch processing in MultimodalProcessor."""

    @pytest.mark.asyncio
    async def test_process_auto_detect_type(self):
        """Test automatic content type detection."""
        processor = MultimodalProcessor()

        # String should be auto-detected as TEXT
        result = await processor.process("Just some text content")

        assert result.content_type == ContentType.TEXT
        assert "text" in result.text.lower()

    @pytest.mark.asyncio
    async def test_process_with_explicit_type(self):
        """Test processing with explicitly specified type."""
        processor = MultimodalProcessor()

        result = await processor.process(
            "Some markdown-ish content",
            content_type=ContentType.MARKDOWN,
        )

        assert result.content_type == ContentType.MARKDOWN


class TestImageProcessorFormats:
    """Additional tests for ImageProcessor format detection."""

    @pytest.fixture
    def processor(self):
        """Create image processor."""
        from locus.rag.multimodal import ImageProcessor

        return ImageProcessor(use_ocr=False, use_vision_llm=False)

    def test_detect_format_bmp(self, processor):
        """Test detecting BMP format."""
        bmp_header = b"BM" + b"\x00" * 50
        # BMP might be detected as unknown or bmp depending on implementation
        format_detected = processor._detect_format(bmp_header)
        assert format_detected in ["bmp", "unknown"]

    @pytest.mark.asyncio
    async def test_process_image_with_metadata(self, processor):
        """Test image processing returns metadata."""
        jpeg_data = b"\xff\xd8\xff" + b"\x00" * 100
        result = await processor.process(jpeg_data)

        assert result.content_type == ContentType.IMAGE
        assert "format" in result.metadata
        assert result.raw_content == jpeg_data


class TestTextProcessorPathInput:
    """Tests for TextProcessor with Path input."""

    @pytest.fixture
    def processor(self):
        """Create text processor."""
        return TextProcessor()

    @pytest.mark.asyncio
    async def test_process_path_input(self, processor, tmp_path):
        """Test processing a file path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content from file")

        result = await processor.process(test_file)

        assert result.text == "Content from file"
        assert result.content_type == ContentType.TEXT

    @pytest.mark.asyncio
    async def test_process_bytes_input(self, processor):
        """Test processing bytes input."""
        result = await processor.process(b"Bytes content")
        assert result.text == "Bytes content"


class TestImageProcessorInputTypes:
    """Tests for ImageProcessor with different input types."""

    @pytest.fixture
    def processor(self):
        """Create image processor without OCR."""
        from locus.rag.multimodal import ImageProcessor

        return ImageProcessor(use_ocr=False, use_vision_llm=False)

    @pytest.mark.asyncio
    async def test_process_path_input(self, processor, tmp_path):
        """Test processing image from path."""
        # Create a fake PNG file
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        test_file = tmp_path / "test.png"
        test_file.write_bytes(png_header)

        result = await processor.process(test_file)

        assert result.content_type == ContentType.IMAGE
        assert result.metadata["filename"] == "test.png"
        assert result.metadata["format"] == "png"

    @pytest.mark.asyncio
    async def test_process_base64_input(self, processor):
        """Test processing base64 encoded image."""
        import base64

        jpeg_data = b"\xff\xd8\xff" + b"\x00" * 100
        b64_data = base64.b64encode(jpeg_data).decode()

        result = await processor.process(b64_data)

        assert result.content_type == ContentType.IMAGE
        assert result.metadata["format"] == "jpeg"

    @pytest.mark.asyncio
    async def test_process_with_custom_filename(self, processor):
        """Test processing with custom filename kwarg."""
        jpeg_data = b"\xff\xd8\xff" + b"\x00" * 100

        result = await processor.process(jpeg_data, filename="custom.jpg")

        assert result.metadata["filename"] == "custom.jpg"


class TestPDFProcessor:
    """Tests for PDFProcessor."""

    @pytest.fixture
    def processor(self):
        """Create PDF processor."""
        from locus.rag.multimodal import PDFProcessor

        return PDFProcessor(use_ocr_fallback=False)

    def test_supports_pdf(self, processor):
        """Test supports PDF type."""
        assert processor.supports(ContentType.PDF) is True
        assert processor.supports(ContentType.TEXT) is False

    @pytest.mark.asyncio
    async def test_process_bytes_fallback(self, processor):
        """Test processing invalid PDF bytes shows fallback."""
        # Invalid PDF bytes
        fake_pdf = b"not a real pdf"

        result = await processor.process(fake_pdf)

        assert result.content_type == ContentType.PDF
        assert "text extraction failed" in result.text
        assert result.metadata["extraction_method"] == "failed"

    @pytest.mark.asyncio
    async def test_process_with_filename(self, processor):
        """Test processing with filename kwarg."""
        fake_pdf = b"%PDF-1.4 fake"

        result = await processor.process(fake_pdf, filename="report.pdf")

        assert result.metadata["filename"] == "report.pdf"


class TestAudioProcessorDetectFormat:
    """Tests for AudioProcessor format detection."""

    @pytest.fixture
    def processor(self):
        """Create audio processor."""
        from locus.rag.multimodal import AudioProcessor

        return AudioProcessor(use_whisper=False)

    def test_detect_wav_format(self, processor):
        """Test detecting WAV format."""
        wav_header = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 100
        assert processor._detect_format(wav_header, "audio.wav") == "wav"

    def test_detect_mp3_format_id3(self, processor):
        """Test detecting MP3 with ID3 header."""
        mp3_header = b"ID3" + b"\x00" * 100
        assert processor._detect_format(mp3_header, "audio.mp3") == "mp3"

    def test_detect_mp3_format_sync(self, processor):
        """Test detecting MP3 with sync bytes."""
        mp3_header = b"\xff\xfb" + b"\x00" * 100
        assert processor._detect_format(mp3_header, "audio.mp3") == "mp3"

    def test_detect_flac_format(self, processor):
        """Test detecting FLAC format."""
        flac_header = b"fLaC" + b"\x00" * 100
        assert processor._detect_format(flac_header, "audio.flac") == "flac"

    def test_detect_ogg_format(self, processor):
        """Test detecting OGG format."""
        ogg_header = b"OggS" + b"\x00" * 100
        assert processor._detect_format(ogg_header, "audio.ogg") == "ogg"

    def test_detect_m4a_format(self, processor):
        """Test detecting M4A format."""
        m4a_header = b"\x00\x00\x00\x00ftypM4A " + b"\x00" * 100
        assert processor._detect_format(m4a_header, "audio.m4a") == "m4a"

    def test_detect_unknown_uses_extension(self, processor):
        """Test unknown format falls back to extension."""
        unknown_data = b"\x00\x00\x00\x00" * 10
        assert processor._detect_format(unknown_data, "audio.aac") == "aac"

    def test_detect_unknown_no_extension(self, processor):
        """Test unknown format with no extension."""
        unknown_data = b"\x00\x00\x00\x00" * 10
        assert processor._detect_format(unknown_data, "audiofile") == "unknown"


class TestAudioProcessorProcess:
    """Tests for AudioProcessor process method."""

    @pytest.fixture
    def processor(self):
        """Create audio processor without whisper."""
        from locus.rag.multimodal import AudioProcessor

        return AudioProcessor(use_whisper=False)

    @pytest.mark.asyncio
    async def test_process_bytes(self, processor):
        """Test processing audio bytes."""
        wav_data = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 100

        result = await processor.process(wav_data)

        assert result.content_type == ContentType.AUDIO
        assert "transcription unavailable" in result.text
        assert result.metadata["format"] == "wav"

    @pytest.mark.asyncio
    async def test_process_path(self, processor, tmp_path):
        """Test processing audio from path."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"ID3" + b"\x00" * 100)

        result = await processor.process(audio_file)

        assert result.content_type == ContentType.AUDIO
        assert result.metadata["filename"] == "test.mp3"

    @pytest.mark.asyncio
    async def test_process_base64(self, processor):
        """Test processing base64 audio."""
        import base64

        wav_data = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 100
        b64_data = base64.b64encode(wav_data).decode()

        result = await processor.process(b64_data)

        assert result.content_type == ContentType.AUDIO


class TestMultimodalProcessorDetectContentType:
    """Tests for MultimodalProcessor content type detection."""

    @pytest.fixture
    def processor(self):
        """Create multimodal processor."""
        return MultimodalProcessor()

    def test_detect_from_path(self, processor, tmp_path):
        """Test detecting type from Path object."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        content_type = processor.detect_content_type(pdf_file)
        assert content_type == ContentType.PDF

    def test_detect_from_path_string(self, processor):
        """Test detecting type from path string."""
        content_type = processor.detect_content_type("/path/to/image.png")
        assert content_type == ContentType.IMAGE

    def test_detect_from_html_path(self, processor):
        """Test detecting HTML from path."""
        content_type = processor.detect_content_type("page.html")
        assert content_type == ContentType.HTML

    def test_detect_from_markdown_path(self, processor):
        """Test detecting markdown from path."""
        content_type = processor.detect_content_type("readme.md")
        # MD may be detected as text or markdown
        assert content_type in [ContentType.TEXT, ContentType.MARKDOWN]

    def test_detect_from_audio_path(self, processor):
        """Test detecting audio from path."""
        content_type = processor.detect_content_type("recording.mp3")
        assert content_type == ContentType.AUDIO


class TestMultimodalProcessorMimeDetection:
    """Tests for _detect_mime_from_bytes."""

    @pytest.fixture
    def processor(self):
        """Create multimodal processor."""
        return MultimodalProcessor()

    def test_detect_png(self, processor):
        """Test detecting PNG MIME type."""
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mime = processor._detect_mime_from_bytes(png_data)
        assert mime == "image/png"

    def test_detect_jpeg(self, processor):
        """Test detecting JPEG MIME type."""
        jpeg_data = b"\xff\xd8" + b"\x00" * 100
        mime = processor._detect_mime_from_bytes(jpeg_data)
        assert mime == "image/jpeg"

    def test_detect_gif(self, processor):
        """Test detecting GIF MIME type."""
        gif_data = b"GIF89a" + b"\x00" * 100
        mime = processor._detect_mime_from_bytes(gif_data)
        assert mime == "image/gif"

    def test_detect_pdf(self, processor):
        """Test detecting PDF MIME type."""
        pdf_data = b"%PDF-1.4" + b"\x00" * 100
        mime = processor._detect_mime_from_bytes(pdf_data)
        assert mime == "application/pdf"

    def test_detect_wav(self, processor):
        """Test detecting WAV MIME type."""
        wav_data = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 100
        mime = processor._detect_mime_from_bytes(wav_data)
        assert mime == "audio/wav"

    def test_detect_mp3(self, processor):
        """Test detecting MP3 MIME type."""
        mp3_data = b"ID3" + b"\x00" * 100
        mime = processor._detect_mime_from_bytes(mp3_data)
        assert mime == "audio/mpeg"

    def test_detect_unknown(self, processor):
        """Test unknown bytes return None."""
        unknown_data = b"UNKN" + b"\x00" * 100
        mime = processor._detect_mime_from_bytes(unknown_data)
        assert mime is None


class TestMultimodalProcessorRaisesOnUnsupported:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_raises_on_removed_processor(self):
        """Test raises ValueError when no processor available."""
        processor = MultimodalProcessor()
        # Remove a processor to simulate unsupported type
        del processor.processors[ContentType.AUDIO]

        with pytest.raises(ValueError, match="No processor"):
            await processor.process(b"audio data", content_type=ContentType.AUDIO)
