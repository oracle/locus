# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Multimodal content processing for RAG.

Supports processing of various content types:
- Text documents
- Images (PNG, JPEG, etc.)
- PDFs
- Audio/Voice files

Each content type is converted to text for embedding.
"""

from __future__ import annotations

import base64
import io
import mimetypes
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class ContentType(str, Enum):
    """Supported content types."""

    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    AUDIO = "audio"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class ProcessedContent:
    """Result of content processing.

    Attributes:
        text: Extracted/generated text for embedding
        content_type: Original content type
        metadata: Additional metadata from processing
        chunks: If content was chunked, the individual chunks
        raw_content: Original binary content (for storage)
    """

    text: str
    content_type: ContentType
    metadata: dict[str, Any] = field(default_factory=dict)
    chunks: list[str] | None = None
    raw_content: bytes | None = None


@runtime_checkable
class ContentProcessor(Protocol):
    """Protocol for content processors."""

    def supports(self, content_type: ContentType) -> bool:
        """Check if this processor supports the content type."""
        ...

    async def process(
        self,
        content: bytes | str | Path,
        **kwargs: Any,
    ) -> ProcessedContent:
        """Process content and extract text."""
        ...


class TextProcessor:
    """Process plain text content."""

    def supports(self, content_type: ContentType) -> bool:
        return content_type in (ContentType.TEXT, ContentType.MARKDOWN, ContentType.HTML)

    async def process(
        self,
        content: bytes | str | Path,
        **kwargs: Any,
    ) -> ProcessedContent:
        """Process text content."""
        if isinstance(content, Path):
            text = content.read_text()
        elif isinstance(content, bytes):
            text = content.decode("utf-8")
        else:
            text = content

        # Basic HTML stripping if needed
        content_type = kwargs.get("content_type", ContentType.TEXT)
        if content_type == ContentType.HTML:
            text = self._strip_html(text)

        return ProcessedContent(
            text=text,
            content_type=content_type,
            metadata={"length": len(text)},
        )

    def _strip_html(self, html: str) -> str:
        """Strip HTML tags (basic implementation)."""
        import re

        # Remove script and style elements
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        html = re.sub(r"<[^>]+>", " ", html)
        # Clean up whitespace
        html = re.sub(r"\s+", " ", html)
        return html.strip()


class ImageProcessor:
    """Process image content using vision models or OCR.

    Can use:
    - OCI Vision AI for image understanding
    - Tesseract OCR for text extraction
    - Vision LLMs for description generation
    """

    def __init__(
        self,
        use_ocr: bool = True,
        use_vision_llm: bool = False,
        vision_model: Any | None = None,
    ):
        self.use_ocr = use_ocr
        self.use_vision_llm = use_vision_llm
        self.vision_model = vision_model

    def supports(self, content_type: ContentType) -> bool:
        return content_type == ContentType.IMAGE

    async def process(
        self,
        content: bytes | str | Path,
        **kwargs: Any,
    ) -> ProcessedContent:
        """Process image and extract text/description."""
        # Load image bytes
        if isinstance(content, Path):
            image_bytes = content.read_bytes()
            filename = content.name
        elif isinstance(content, str):
            # Assume base64 encoded
            image_bytes = base64.b64decode(content)
            filename = "image"
        else:
            image_bytes = content
            filename = kwargs.get("filename", "image")

        # Detect image format
        image_format = self._detect_format(image_bytes)
        texts = []
        metadata = {
            "filename": filename,
            "format": image_format,
            "size_bytes": len(image_bytes),
        }

        # OCR extraction
        if self.use_ocr:
            ocr_text = await self._extract_text_ocr(image_bytes)
            if ocr_text:
                texts.append(f"[OCR Text]: {ocr_text}")
                metadata["ocr_text"] = ocr_text

        # Vision LLM description
        if self.use_vision_llm and self.vision_model:
            description = await self._get_vision_description(image_bytes)
            if description:
                texts.append(f"[Image Description]: {description}")
                metadata["description"] = description

        # Fallback: basic image info
        if not texts:
            texts.append(
                f"[Image: {filename}, format={image_format}, size={len(image_bytes)} bytes]"
            )

        return ProcessedContent(
            text="\n".join(texts),
            content_type=ContentType.IMAGE,
            metadata=metadata,
            raw_content=image_bytes,
        )

    def _detect_format(self, data: bytes) -> str:
        """Detect image format from magic bytes."""
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "png"
        if data[:2] == b"\xff\xd8":
            return "jpeg"
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return "gif"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "webp"
        return "unknown"

    async def _extract_text_ocr(self, image_bytes: bytes) -> str | None:
        """Extract text using OCR."""
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            return text.strip() if text.strip() else None
        except ImportError:
            return None
        except Exception:  # noqa: BLE001 — best-effort extraction; return None on any failure
            return None

    async def _get_vision_description(self, image_bytes: bytes) -> str | None:
        """Get image description from vision model."""
        if not self.vision_model:
            return None

        try:
            # Encode image as base64
            b64_image = base64.b64encode(image_bytes).decode()

            # Call vision model (implementation depends on model type)
            # This is a placeholder for the actual implementation
            return None
        except Exception:  # noqa: BLE001 — best-effort extraction; return None on any failure
            return None


class PDFProcessor:
    """Process PDF documents.

    Extracts text from PDFs using:
    - PyPDF2/pypdf for text extraction
    - OCR fallback for scanned PDFs
    """

    def __init__(self, use_ocr_fallback: bool = True):
        self.use_ocr_fallback = use_ocr_fallback

    def supports(self, content_type: ContentType) -> bool:
        return content_type == ContentType.PDF

    async def process(
        self,
        content: bytes | str | Path,
        **kwargs: Any,
    ) -> ProcessedContent:
        """Process PDF and extract text."""
        # Load PDF bytes
        if isinstance(content, Path):
            pdf_bytes = content.read_bytes()
            filename = content.name
        elif isinstance(content, str):
            # Assume base64 encoded
            pdf_bytes = base64.b64decode(content)
            filename = "document.pdf"
        else:
            pdf_bytes = content
            filename = kwargs.get("filename", "document.pdf")

        metadata = {
            "filename": filename,
            "size_bytes": len(pdf_bytes),
        }

        # Try pypdf extraction
        text = await self._extract_with_pypdf(pdf_bytes)

        if text:
            metadata["extraction_method"] = "pypdf"
            metadata["page_count"] = text.count("\n--- Page") + 1
        elif self.use_ocr_fallback:
            # Fallback to OCR
            text = await self._extract_with_ocr(pdf_bytes)
            if text:
                metadata["extraction_method"] = "ocr"

        if not text:
            text = f"[PDF: {filename}, size={len(pdf_bytes)} bytes - text extraction failed]"
            metadata["extraction_method"] = "failed"

        return ProcessedContent(
            text=text,
            content_type=ContentType.PDF,
            metadata=metadata,
            raw_content=pdf_bytes,
        )

    async def _extract_with_pypdf(self, pdf_bytes: bytes) -> str | None:
        """Extract text using pypdf."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(pdf_bytes))
            texts = []

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    texts.append(f"--- Page {i + 1} ---\n{page_text}")

            return "\n\n".join(texts) if texts else None
        except ImportError:
            try:
                # Try older PyPDF2
                from PyPDF2 import PdfReader

                reader = PdfReader(io.BytesIO(pdf_bytes))
                texts = []

                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        texts.append(f"--- Page {i + 1} ---\n{page_text}")

                return "\n\n".join(texts) if texts else None
            except ImportError:
                return None
        except Exception:  # noqa: BLE001 — best-effort extraction; return None on any failure
            return None

    async def _extract_with_ocr(self, pdf_bytes: bytes) -> str | None:
        """Extract text using OCR (for scanned PDFs)."""
        try:
            import pdf2image
            import pytesseract

            images = pdf2image.convert_from_bytes(pdf_bytes)
            texts = []

            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                if text.strip():
                    texts.append(f"--- Page {i + 1} ---\n{text}")

            return "\n\n".join(texts) if texts else None
        except ImportError:
            return None
        except Exception:  # noqa: BLE001 — best-effort extraction; return None on any failure
            return None


class AudioProcessor:
    """Process audio/voice content.

    Uses speech-to-text to extract transcription:
    - OCI Speech AI
    - OpenAI Whisper (local or API)
    - Other STT services
    """

    def __init__(
        self,
        use_whisper: bool = True,
        whisper_model: str = "base",
    ):
        self.use_whisper = use_whisper
        self.whisper_model = whisper_model
        self._whisper: Any = None

    def supports(self, content_type: ContentType) -> bool:
        return content_type == ContentType.AUDIO

    async def process(
        self,
        content: bytes | str | Path,
        **kwargs: Any,
    ) -> ProcessedContent:
        """Process audio and extract transcription."""
        # Load audio bytes
        if isinstance(content, Path):
            audio_bytes = content.read_bytes()
            filename = content.name
        elif isinstance(content, str):
            # Assume base64 encoded
            audio_bytes = base64.b64decode(content)
            filename = "audio"
        else:
            audio_bytes = content
            filename = kwargs.get("filename", "audio")

        # Detect audio format
        audio_format = self._detect_format(audio_bytes, filename)
        metadata = {
            "filename": filename,
            "format": audio_format,
            "size_bytes": len(audio_bytes),
        }

        # Transcribe
        text = None
        if self.use_whisper:
            text = await self._transcribe_whisper(audio_bytes, audio_format)
            if text:
                metadata["transcription_method"] = "whisper"

        if not text:
            text = f"[Audio: {filename}, format={audio_format}, size={len(audio_bytes)} bytes - transcription unavailable]"
            metadata["transcription_method"] = "unavailable"

        return ProcessedContent(
            text=text,
            content_type=ContentType.AUDIO,
            metadata=metadata,
            raw_content=audio_bytes,
        )

    def _detect_format(self, data: bytes, filename: str) -> str:
        """Detect audio format."""
        # Check magic bytes
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return "wav"
        if data[:3] == b"ID3" or (data[:2] == b"\xff\xfb"):
            return "mp3"
        if data[:4] == b"fLaC":
            return "flac"
        if data[:4] == b"OggS":
            return "ogg"
        if data[4:12] == b"ftypM4A ":
            return "m4a"

        # Fallback to extension
        ext = Path(filename).suffix.lower().lstrip(".")
        return ext or "unknown"

    async def _transcribe_whisper(self, audio_bytes: bytes, audio_format: str) -> str | None:
        """Transcribe using OpenAI Whisper."""
        try:
            import os
            import tempfile

            import whisper

            # Load model (cached)
            if self._whisper is None:
                self._whisper = whisper.load_model(self.whisper_model)

            # Write to temp file (Whisper requires file path)
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name

            try:
                result: dict[str, Any] = self._whisper.transcribe(temp_path)
                text: str = result["text"].strip()
                return text
            finally:
                os.unlink(temp_path)

        except ImportError:
            return None
        except Exception:  # noqa: BLE001 — best-effort extraction; return None on any failure
            return None


class MultimodalProcessor:
    """
    Unified processor for all content types.

    Example:
        >>> processor = MultimodalProcessor()
        >>> result = await processor.process(Path("doc.pdf"))
        >>> print(result.text)

        >>> result = await processor.process(image_bytes, content_type=ContentType.IMAGE)
    """

    def __init__(
        self,
        use_ocr: bool = True,
        use_whisper: bool = True,
    ):
        self.processors: dict[ContentType, ContentProcessor] = {
            ContentType.TEXT: TextProcessor(),
            ContentType.MARKDOWN: TextProcessor(),
            ContentType.HTML: TextProcessor(),
            ContentType.IMAGE: ImageProcessor(use_ocr=use_ocr),
            ContentType.PDF: PDFProcessor(use_ocr_fallback=use_ocr),
            ContentType.AUDIO: AudioProcessor(use_whisper=use_whisper),
        }

    def detect_content_type(self, content: bytes | str | Path) -> ContentType:
        """Detect content type from content or path."""
        if isinstance(content, Path):
            mime_type, _ = mimetypes.guess_type(str(content))
        elif isinstance(content, str) and not content.startswith("data:"):
            # Assume it's a path string
            mime_type, _ = mimetypes.guess_type(content)
        # Try to detect from bytes
        elif isinstance(content, bytes):
            mime_type = self._detect_mime_from_bytes(content)
        else:
            mime_type = None

        if mime_type:
            if mime_type.startswith("image/"):
                return ContentType.IMAGE
            if mime_type == "application/pdf":
                return ContentType.PDF
            if mime_type.startswith("audio/"):
                return ContentType.AUDIO
            if mime_type == "text/html":
                return ContentType.HTML
            if mime_type == "text/markdown":
                return ContentType.MARKDOWN

        return ContentType.TEXT

    def _detect_mime_from_bytes(self, data: bytes) -> str | None:
        """Detect MIME type from magic bytes."""
        # Images
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if data[:2] == b"\xff\xd8":
            return "image/jpeg"
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"

        # PDF
        if data[:4] == b"%PDF":
            return "application/pdf"

        # Audio
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return "audio/wav"
        if data[:3] == b"ID3" or (data[:2] == b"\xff\xfb"):
            return "audio/mpeg"

        return None

    async def process(
        self,
        content: bytes | str | Path,
        content_type: ContentType | None = None,
        **kwargs: Any,
    ) -> ProcessedContent:
        """
        Process content of any supported type.

        Args:
            content: Content to process (bytes, string, or path)
            content_type: Explicit content type (auto-detected if None)
            **kwargs: Additional processor options

        Returns:
            ProcessedContent with extracted text
        """
        if content_type is None:
            content_type = self.detect_content_type(content)

        processor = self.processors.get(content_type)
        if processor is None:
            raise ValueError(f"No processor for content type: {content_type}")

        return await processor.process(content, content_type=content_type, **kwargs)


# Convenience function
async def process_content(
    content: bytes | str | Path,
    content_type: ContentType | None = None,
    **kwargs: Any,
) -> ProcessedContent:
    """
    Process any content type and extract text for embedding.

    Args:
        content: Content to process
        content_type: Optional content type hint

    Returns:
        ProcessedContent with extracted text

    Example:
        >>> result = await process_content(Path("document.pdf"))
        >>> embeddings = await embedder.embed(result.text)
    """
    processor = MultimodalProcessor()
    return await processor.process(content, content_type, **kwargs)
