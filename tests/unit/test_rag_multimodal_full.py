# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage tests for ``locus.rag.multimodal``.

Existing tests in ``test_rag_retriever.py`` mock the processor classes
wholesale. These tests exercise the actual processor logic by stubbing
``pytesseract``, ``PIL.Image``, ``pypdf``, ``PyPDF2``, ``pdf2image``,
and ``whisper`` at the module-import boundary so we cover both the
"package available" and "package missing" paths.

Coverage targets:
- ``TextProcessor`` — bytes / str / Path inputs + HTML stripping
- ``ImageProcessor`` — OCR success, OCR import-missing, Vision-LLM
  branch, image-format magic-byte detection, base64 input
- ``PDFProcessor`` — pypdf success, PyPDF2 fallback, both-missing,
  OCR fallback for scanned PDFs, all-extraction-fails string
- ``AudioProcessor`` — whisper success, whisper import-missing,
  whisper-disabled path, format detection across magic bytes
- ``MultimodalProcessor`` — content-type detection, mime-from-bytes
  for every supported family, dispatch to the right processor,
  unsupported content type raises
- ``process_content`` convenience entrypoint
"""

from __future__ import annotations

import base64
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from locus.rag.multimodal import (
    AudioProcessor,
    ContentProcessor,
    ContentType,
    ImageProcessor,
    MultimodalProcessor,
    PDFProcessor,
    ProcessedContent,
    TextProcessor,
    process_content,
)


# ---------------------------------------------------------------------------
# Module-stub helpers
# ---------------------------------------------------------------------------


def _stub_pytesseract(monkeypatch: pytest.MonkeyPatch, *, output: str = "OCR text") -> None:
    fake = types.ModuleType("pytesseract")
    fake.image_to_string = lambda image: output  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pytesseract", fake)


def _stub_pil_image(monkeypatch: pytest.MonkeyPatch) -> None:
    pil_pkg = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    image_mod.open = lambda buf: MagicMock(name="image")  # type: ignore[attr-defined]
    pil_pkg.Image = image_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", pil_pkg)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_mod)


def _stub_pypdf(monkeypatch: pytest.MonkeyPatch, *, pages: list[str] | None = None) -> None:
    pages_text = pages or ["page one"]
    fake = types.ModuleType("pypdf")

    class _Page:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _PdfReader:
        def __init__(self, _stream: Any) -> None:
            self.pages = [_Page(t) for t in pages_text]

    fake.PdfReader = _PdfReader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdf", fake)


def _stub_pypdf2(monkeypatch: pytest.MonkeyPatch, *, pages: list[str] | None = None) -> None:
    """Same shape as pypdf — install under ``PyPDF2`` for the fallback path."""
    pages_text = pages or ["legacy page"]
    fake = types.ModuleType("PyPDF2")

    class _Page:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _PdfReader:
        def __init__(self, _stream: Any) -> None:
            self.pages = [_Page(t) for t in pages_text]

    fake.PdfReader = _PdfReader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PyPDF2", fake)


def _stub_pdf2image(monkeypatch: pytest.MonkeyPatch, *, page_count: int = 2) -> None:
    fake = types.ModuleType("pdf2image")
    fake.convert_from_bytes = lambda b: [MagicMock(name=f"page_{i}") for i in range(page_count)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pdf2image", fake)


def _stub_whisper(monkeypatch: pytest.MonkeyPatch, *, text: str = "hi") -> None:
    fake = types.ModuleType("whisper")
    model = MagicMock()
    model.transcribe.return_value = {"text": text}
    fake.load_model = lambda name: model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "whisper", fake)


# ---------------------------------------------------------------------------
# TextProcessor
# ---------------------------------------------------------------------------


class TestTextProcessor:
    def test_supports_text_md_html(self) -> None:
        p = TextProcessor()
        assert p.supports(ContentType.TEXT)
        assert p.supports(ContentType.MARKDOWN)
        assert p.supports(ContentType.HTML)
        assert not p.supports(ContentType.IMAGE)

    @pytest.mark.asyncio
    async def test_processes_str(self) -> None:
        result = await TextProcessor().process("hello")
        assert result.text == "hello"
        assert result.metadata["length"] == 5

    @pytest.mark.asyncio
    async def test_processes_bytes(self) -> None:
        result = await TextProcessor().process(b"hello")
        assert result.text == "hello"

    @pytest.mark.asyncio
    async def test_processes_path(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("file content")
        result = await TextProcessor().process(f)
        assert result.text == "file content"

    @pytest.mark.asyncio
    async def test_strips_html(self) -> None:
        # Script tags, style tags, and bare tags are removed; whitespace collapsed.
        html = (
            "<html><head><style>x{}</style></head>"
            "<body><script>alert(1)</script><p>Hello <b>world</b></p></body></html>"
        )
        result = await TextProcessor().process(html, content_type=ContentType.HTML)
        assert "<script>" not in result.text
        assert "<style>" not in result.text
        assert "Hello world" in result.text


# ---------------------------------------------------------------------------
# ImageProcessor
# ---------------------------------------------------------------------------


PNG_HEADER = b"\x89PNG\r\n\x1a\n"
JPEG_HEADER = b"\xff\xd8" + b"\x00" * 8
GIF_HEADER = b"GIF89a" + b"\x00" * 4
WEBP_HEADER = b"RIFF" + b"\x00" * 4 + b"WEBP"


class TestImageProcessor:
    def test_supports_image_only(self) -> None:
        p = ImageProcessor()
        assert p.supports(ContentType.IMAGE)
        assert not p.supports(ContentType.PDF)

    @pytest.mark.asyncio
    async def test_extracts_ocr_text_when_dependencies_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_pytesseract(monkeypatch, output="hello world")
        _stub_pil_image(monkeypatch)
        result = await ImageProcessor().process(PNG_HEADER + b"\x00" * 100)
        assert "hello world" in result.text
        assert result.metadata["format"] == "png"

    @pytest.mark.asyncio
    async def test_falls_back_to_metadata_when_ocr_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Block both pytesseract and PIL so the OCR import raises.
        monkeypatch.setitem(sys.modules, "pytesseract", None)
        result = await ImageProcessor().process(PNG_HEADER + b"\x00" * 5)
        # Falls through to ``[Image: ..., format=png, ...]`` placeholder.
        assert "format=png" in result.text

    @pytest.mark.asyncio
    async def test_handles_str_base64_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_pytesseract(monkeypatch, output="b64-ocr")
        _stub_pil_image(monkeypatch)
        b64 = base64.b64encode(PNG_HEADER + b"\x00").decode()
        result = await ImageProcessor().process(b64)
        assert "b64-ocr" in result.text

    @pytest.mark.asyncio
    async def test_path_input(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_pytesseract(monkeypatch, output="from disk")
        _stub_pil_image(monkeypatch)
        f = tmp_path / "img.png"
        f.write_bytes(PNG_HEADER + b"\x00" * 4)
        result = await ImageProcessor().process(f)
        assert result.metadata["filename"] == "img.png"

    @pytest.mark.asyncio
    async def test_ocr_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pytesseract raises at runtime → ``except Exception`` swallows it.
        fake = types.ModuleType("pytesseract")

        def boom(image: Any) -> str:
            raise RuntimeError("tesseract crashed")

        fake.image_to_string = boom  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pytesseract", fake)
        _stub_pil_image(monkeypatch)
        proc = ImageProcessor()
        result = await proc._extract_text_ocr(PNG_HEADER + b"\x00")
        assert result is None

    @pytest.mark.asyncio
    async def test_vision_llm_branch_runs_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Vision LLM is a placeholder (returns None), but the branch
        # still runs when ``use_vision_llm=True`` and a model is set.
        proc = ImageProcessor(use_ocr=False, use_vision_llm=True, vision_model=MagicMock())
        result = await proc.process(PNG_HEADER + b"\x00")
        # No OCR, no vision result → fallback metadata string.
        assert "format=png" in result.text

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (PNG_HEADER, "png"),
            (JPEG_HEADER, "jpeg"),
            (GIF_HEADER, "gif"),
            (WEBP_HEADER, "webp"),
            (b"\x00\x01\x02\x03", "unknown"),
        ],
    )
    def test_format_detection(self, data: bytes, expected: str) -> None:
        assert ImageProcessor()._detect_format(data) == expected


# ---------------------------------------------------------------------------
# PDFProcessor
# ---------------------------------------------------------------------------


PDF_HEADER = b"%PDF-1.4\n" + b"\x00" * 100


class TestPDFProcessor:
    def test_supports_pdf_only(self) -> None:
        p = PDFProcessor()
        assert p.supports(ContentType.PDF)
        assert not p.supports(ContentType.IMAGE)

    @pytest.mark.asyncio
    async def test_extracts_with_pypdf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_pypdf(monkeypatch, pages=["page one", "page two"])
        result = await PDFProcessor().process(PDF_HEADER)
        assert "page one" in result.text
        assert "page two" in result.text
        assert result.metadata["extraction_method"] == "pypdf"

    @pytest.mark.asyncio
    async def test_falls_back_to_pypdf2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "pypdf", None)
        _stub_pypdf2(monkeypatch, pages=["legacy"])
        result = await PDFProcessor().process(PDF_HEADER)
        assert "legacy" in result.text

    @pytest.mark.asyncio
    async def test_falls_back_to_ocr_when_text_extraction_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # pypdf yields a reader where every ``extract_text`` returns ``""``
        # (falsy) — ``_extract_with_pypdf`` skips them all and returns None
        # → OCR fallback runs.
        _stub_pypdf(monkeypatch, pages=["", ""])
        _stub_pdf2image(monkeypatch, page_count=1)
        _stub_pytesseract(monkeypatch, output="ocr text from pdf")
        result = await PDFProcessor().process(PDF_HEADER)
        assert "ocr text from pdf" in result.text
        assert result.metadata["extraction_method"] == "ocr"

    @pytest.mark.asyncio
    async def test_all_extraction_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No pypdf, no PyPDF2, no pdf2image — final string is the
        # ``[PDF: …]`` placeholder + ``extraction_method=failed``.
        monkeypatch.setitem(sys.modules, "pypdf", None)
        monkeypatch.setitem(sys.modules, "PyPDF2", None)
        monkeypatch.setitem(sys.modules, "pdf2image", None)
        result = await PDFProcessor().process(PDF_HEADER)
        assert result.metadata["extraction_method"] == "failed"
        assert "text extraction failed" in result.text

    @pytest.mark.asyncio
    async def test_pypdf_runtime_failure_falls_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # pypdf imports fine but raises at runtime → ``except Exception``
        # swallows and returns None.
        fake = types.ModuleType("pypdf")

        class _Boom:
            def __init__(self, _: Any) -> None:
                raise RuntimeError("corrupt pdf")

        fake.PdfReader = _Boom  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pypdf", fake)
        result = await PDFProcessor()._extract_with_pypdf(PDF_HEADER)
        assert result is None

    @pytest.mark.asyncio
    async def test_str_path_input(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # str input → treated as base64.
        _stub_pypdf(monkeypatch, pages=["from b64"])
        b64 = base64.b64encode(PDF_HEADER).decode()
        result = await PDFProcessor().process(b64)
        assert result.metadata["filename"] == "document.pdf"

    @pytest.mark.asyncio
    async def test_path_input(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_pypdf(monkeypatch, pages=["from disk"])
        f = tmp_path / "doc.pdf"
        f.write_bytes(PDF_HEADER)
        result = await PDFProcessor().process(f)
        assert result.metadata["filename"] == "doc.pdf"


# ---------------------------------------------------------------------------
# AudioProcessor
# ---------------------------------------------------------------------------


WAV_HEADER = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 16
MP3_HEADER = b"ID3" + b"\x00" * 100
FLAC_HEADER = b"fLaC" + b"\x00" * 16
OGG_HEADER = b"OggS" + b"\x00" * 16
M4A_HEADER = b"\x00" * 4 + b"ftypM4A " + b"\x00" * 16


class TestAudioProcessor:
    def test_supports_audio_only(self) -> None:
        p = AudioProcessor()
        assert p.supports(ContentType.AUDIO)
        assert not p.supports(ContentType.IMAGE)

    @pytest.mark.asyncio
    async def test_transcribes_with_whisper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_whisper(monkeypatch, text="hello there")
        result = await AudioProcessor().process(WAV_HEADER)
        assert "hello there" in result.text
        assert result.metadata["transcription_method"] == "whisper"

    @pytest.mark.asyncio
    async def test_whisper_missing_falls_back_to_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "whisper", None)
        result = await AudioProcessor().process(WAV_HEADER)
        assert result.metadata["transcription_method"] == "unavailable"

    @pytest.mark.asyncio
    async def test_whisper_runtime_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = types.ModuleType("whisper")

        def boom(name: str) -> Any:
            raise RuntimeError("model load fail")

        fake.load_model = boom  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "whisper", fake)
        result = await AudioProcessor()._transcribe_whisper(WAV_HEADER, "wav")
        assert result is None

    @pytest.mark.asyncio
    async def test_use_whisper_disabled(self) -> None:
        result = await AudioProcessor(use_whisper=False).process(WAV_HEADER)
        assert result.metadata["transcription_method"] == "unavailable"

    @pytest.mark.asyncio
    async def test_str_input_treated_as_base64(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_whisper(monkeypatch, text="from b64")
        b64 = base64.b64encode(WAV_HEADER).decode()
        result = await AudioProcessor().process(b64)
        assert result.metadata["filename"] == "audio"

    @pytest.mark.asyncio
    async def test_path_input(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_whisper(monkeypatch, text="from disk")
        f = tmp_path / "voice.wav"
        f.write_bytes(WAV_HEADER)
        result = await AudioProcessor().process(f)
        assert result.metadata["filename"] == "voice.wav"

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (WAV_HEADER, "wav"),
            (MP3_HEADER, "mp3"),
            (FLAC_HEADER, "flac"),
            (OGG_HEADER, "ogg"),
            (M4A_HEADER, "m4a"),
        ],
    )
    def test_format_detection_magic_bytes(self, data: bytes, expected: str) -> None:
        assert AudioProcessor()._detect_format(data, "x") == expected

    def test_format_detection_falls_back_to_extension(self) -> None:
        # Unknown magic bytes — fall back to filename extension.
        assert AudioProcessor()._detect_format(b"\x00" * 16, "audio.opus") == "opus"

    def test_format_detection_no_extension(self) -> None:
        assert AudioProcessor()._detect_format(b"\x00" * 16, "audio") == "unknown"


# ---------------------------------------------------------------------------
# MultimodalProcessor — content type detection
# ---------------------------------------------------------------------------


class TestMultimodalProcessor:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (PNG_HEADER, ContentType.IMAGE),
            (JPEG_HEADER, ContentType.IMAGE),
            (GIF_HEADER, ContentType.IMAGE),
            (b"%PDF-1.4", ContentType.PDF),
            (WAV_HEADER, ContentType.AUDIO),
            (MP3_HEADER, ContentType.AUDIO),
        ],
    )
    def test_detect_from_bytes(self, data: bytes, expected: ContentType) -> None:
        assert MultimodalProcessor().detect_content_type(data) == expected

    def test_detect_unknown_bytes_defaults_to_text(self) -> None:
        assert MultimodalProcessor().detect_content_type(b"plain text bytes") == ContentType.TEXT

    def test_detect_from_path_with_known_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        assert MultimodalProcessor().detect_content_type(f) == ContentType.PDF

    def test_detect_from_html_path(self, tmp_path: Path) -> None:
        f = tmp_path / "page.html"
        assert MultimodalProcessor().detect_content_type(f) == ContentType.HTML

    def test_detect_from_string_path_with_image_extension(self) -> None:
        assert MultimodalProcessor().detect_content_type("photo.jpg") == ContentType.IMAGE

    def test_detect_data_uri_falls_to_text(self) -> None:
        # Strings starting with ``data:`` are skipped over by the
        # path-based mime guess; without a magic-byte match it lands on TEXT.
        assert (
            MultimodalProcessor().detect_content_type("data:image/png;base64,xx")
            == ContentType.TEXT
        )

    @pytest.mark.asyncio
    async def test_dispatches_to_correct_processor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Monkey-patch the IMAGE processor so we don't need the real OCR deps.
        proc = MultimodalProcessor()
        called: dict[str, Any] = {}

        async def fake_process(content: Any, **kwargs: Any) -> ProcessedContent:
            called["content_type"] = kwargs.get("content_type")
            return ProcessedContent(text="image-text", content_type=ContentType.IMAGE)

        # Replace the IMAGE processor with a stub that satisfies ContentProcessor.
        class _StubImage:
            def supports(self, _: ContentType) -> bool:
                return True

            async def process(self, content: Any, **kwargs: Any) -> ProcessedContent:
                return await fake_process(content, **kwargs)

        proc.processors[ContentType.IMAGE] = _StubImage()
        result = await proc.process(PNG_HEADER + b"\x00")
        assert result.text == "image-text"
        assert called["content_type"] == ContentType.IMAGE

    @pytest.mark.asyncio
    async def test_explicit_content_type_overrides_detection(self) -> None:
        proc = MultimodalProcessor()
        # Force MARKDOWN — even if the bytes look like PDF.
        result = await proc.process(b"hello", content_type=ContentType.MARKDOWN)
        assert result.content_type == ContentType.MARKDOWN

    @pytest.mark.asyncio
    async def test_unsupported_content_type_raises(self) -> None:
        proc = MultimodalProcessor()
        proc.processors.pop(ContentType.PDF)
        with pytest.raises(ValueError, match="No processor for content type"):
            await proc.process(b"%PDF-", content_type=ContentType.PDF)

    @pytest.mark.asyncio
    async def test_process_content_convenience_function(self) -> None:
        # Smoke test the module-level ``process_content`` shortcut.
        result = await process_content("hello", content_type=ContentType.TEXT)
        assert result.text == "hello"

    def test_processors_share_runtime_protocol(self) -> None:
        proc = MultimodalProcessor()
        for p in proc.processors.values():
            assert isinstance(p, ContentProcessor)
