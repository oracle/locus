# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Speech provider protocol — text-to-speech + speech-to-text.

A single provider type covers both directions because most production
backends (OpenAI Audio, Google STT/TTS, OCI Speech) ship them as one
SDK and you typically wire them together.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class SynthesizedAudio(BaseModel):
    """Output of ``speak(text)`` — the audio bytes + their content type."""

    text: str = Field(description="The text that was synthesized")
    audio_bytes: bytes = Field(description="Raw audio bytes")
    content_type: str = Field(
        default="audio/mpeg",
        description="MIME type of the audio (audio/mpeg, audio/wav, …)",
    )

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class SpeechTranscript(BaseModel):
    """Output of ``transcribe(audio_bytes)`` — recognized text + metadata."""

    text: str = Field(description="The recognized text")
    language: str | None = Field(
        default=None,
        description="ISO language code when the provider exposes it",
    )
    duration_seconds: float | None = Field(
        default=None,
        description="Audio duration when known",
    )

    model_config = {"frozen": True}


@runtime_checkable
class BaseSpeechProvider(Protocol):
    """Protocol every speech provider must implement.

    A provider may implement only one direction (e.g. TTS-only) by
    raising ``NotImplementedError`` from the unsupported method;
    callers can detect via the ``capabilities`` attribute.
    """

    capabilities: frozenset[str]
    """``{"tts", "stt"}`` (either or both)."""

    async def speak(
        self,
        text: str,
        *,
        voice: str | None = None,
        **kwargs: Any,
    ) -> SynthesizedAudio:
        """Synthesize ``text`` to audio bytes."""
        ...

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        content_type: str = "audio/mpeg",
        **kwargs: Any,
    ) -> SpeechTranscript:
        """Transcribe raw audio bytes to text."""
        ...


class OpenAISpeechProvider:
    """Speech provider backed by OpenAI's audio APIs.

    ``speak`` uses ``audio.speech.create`` (TTS, default ``tts-1``).
    ``transcribe`` uses ``audio.transcriptions.create`` (Whisper, default
    ``whisper-1``).
    """

    capabilities: frozenset[str] = frozenset({"tts", "stt"})

    def __init__(
        self,
        *,
        tts_model: str = "tts-1",
        stt_model: str = "whisper-1",
        default_voice: str = "alloy",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._tts_model = tts_model
        self._stt_model = stt_model
        self._default_voice = default_voice
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    async def speak(
        self,
        text: str,
        *,
        voice: str | None = None,
        **kwargs: Any,
    ) -> SynthesizedAudio:
        client = self._get_client()
        resp = await client.audio.speech.create(
            model=self._tts_model,
            voice=voice or self._default_voice,
            input=text,
            response_format=kwargs.pop("response_format", "mp3"),
            **kwargs,
        )
        # The OpenAI SDK exposes the bytes via `.content`.
        audio = resp.content if hasattr(resp, "content") else bytes(resp)
        return SynthesizedAudio(
            text=text,
            audio_bytes=audio,
            content_type="audio/mpeg",
        )

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        content_type: str = "audio/mpeg",
        **kwargs: Any,
    ) -> SpeechTranscript:
        import io

        client = self._get_client()
        # The SDK accepts a file-like — wrap our bytes.
        ext = "mp3" if "mpeg" in content_type else "wav"
        buf = io.BytesIO(audio_bytes)
        buf.name = f"audio.{ext}"
        resp = await client.audio.transcriptions.create(
            model=self._stt_model,
            file=buf,
            **kwargs,
        )
        return SpeechTranscript(
            text=getattr(resp, "text", ""),
            language=getattr(resp, "language", None),
            duration_seconds=getattr(resp, "duration", None),
        )


__all__ = [
    "BaseSpeechProvider",
    "OpenAISpeechProvider",
    "SpeechTranscript",
    "SynthesizedAudio",
]
