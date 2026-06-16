# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage tests for the image- and speech-generation providers.

Both providers wrap the OpenAI SDK lazily; we mock ``openai.AsyncOpenAI``
so the tests never reach the network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from locus.providers.image import (
    BaseImageGenerationProvider,
    ImageResult,
    OpenAIImageProvider,
)
from locus.providers.speech import (
    BaseSpeechProvider,
    OpenAISpeechProvider,
    SpeechTranscript,
    SynthesizedAudio,
)


pytest.importorskip("openai")


# ---------------------------------------------------------------------------
# OpenAIImageProvider
# ---------------------------------------------------------------------------


class TestOpenAIImageProvider:
    def test_protocol_runtime_check(self) -> None:
        assert isinstance(OpenAIImageProvider(), BaseImageGenerationProvider)

    @pytest.mark.asyncio
    async def test_generate_returns_image_results(self) -> None:
        # Build a fake openai response.
        entries = [
            SimpleNamespace(url="https://x.example/a.png", b64_json=None, revised_prompt=None),
            SimpleNamespace(url=None, b64_json="aGVsbG8=", revised_prompt="rewritten"),
        ]
        fake_client = MagicMock()
        fake_client.images.generate = AsyncMock(return_value=SimpleNamespace(data=entries))
        with patch("openai.AsyncOpenAI", return_value=fake_client) as mock_ctor:
            provider = OpenAIImageProvider(model="dall-e-3", api_key="sk-x")  # noqa: S106
            results = await provider.generate("two robots", n=2)
        mock_ctor.assert_called_once_with(api_key="sk-x", base_url=None)
        assert len(results) == 2
        assert isinstance(results[0], ImageResult)
        assert results[0].url == "https://x.example/a.png"
        assert results[1].b64_png == "aGVsbG8="
        assert results[1].revised_prompt == "rewritten"

    @pytest.mark.asyncio
    async def test_get_client_caches(self) -> None:
        fake_client = MagicMock()
        fake_client.images.generate = AsyncMock(return_value=SimpleNamespace(data=[]))
        with patch("openai.AsyncOpenAI", return_value=fake_client) as mock_ctor:
            provider = OpenAIImageProvider()
            await provider.generate("hi", n=1)
            await provider.generate("hi", n=1)
        # AsyncOpenAI should have been constructed only once.
        assert mock_ctor.call_count == 1


# ---------------------------------------------------------------------------
# OpenAISpeechProvider
# ---------------------------------------------------------------------------


class TestOpenAISpeechProvider:
    def test_protocol_runtime_check(self) -> None:
        assert isinstance(OpenAISpeechProvider(), BaseSpeechProvider)

    def test_capabilities_set(self) -> None:
        assert OpenAISpeechProvider().capabilities == frozenset({"tts", "stt"})

    @pytest.mark.asyncio
    async def test_speak_returns_synthesized_audio(self) -> None:
        fake_client = MagicMock()
        fake_resp = SimpleNamespace(content=b"audio-bytes")
        fake_client.audio.speech.create = AsyncMock(return_value=fake_resp)
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            provider = OpenAISpeechProvider(default_voice="nova")
            out = await provider.speak("hello world")
        assert isinstance(out, SynthesizedAudio)
        assert out.audio_bytes == b"audio-bytes"
        assert out.text == "hello world"
        # default voice is honored
        kwargs = fake_client.audio.speech.create.call_args.kwargs
        assert kwargs["voice"] == "nova"
        assert kwargs["model"] == "tts-1"
        assert kwargs["response_format"] == "mp3"

    @pytest.mark.asyncio
    async def test_speak_voice_override(self) -> None:
        fake_client = MagicMock()
        fake_client.audio.speech.create = AsyncMock(return_value=SimpleNamespace(content=b"x"))
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            provider = OpenAISpeechProvider()
            await provider.speak("hi", voice="onyx")
        assert fake_client.audio.speech.create.call_args.kwargs["voice"] == "onyx"

    @pytest.mark.asyncio
    async def test_speak_falls_back_to_bytes_when_no_content_attr(self) -> None:
        # The OpenAI SDK exposes ``.content`` on real responses, but old
        # SDK versions returned a ``bytes``-like object. Confirm we fall
        # through to the ``bytes(resp)`` path.

        class _BytesLike:
            def __bytes__(self) -> bytes:
                return b"bytes-style-response"

        fake_client = MagicMock()
        fake_client.audio.speech.create = AsyncMock(return_value=_BytesLike())
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            provider = OpenAISpeechProvider()
            out = await provider.speak("hi")
        assert out.audio_bytes == b"bytes-style-response"

    @pytest.mark.asyncio
    async def test_transcribe_returns_speech_transcript(self) -> None:
        fake_client = MagicMock()
        fake_resp = SimpleNamespace(text="recognized words", language="en", duration=2.5)
        fake_client.audio.transcriptions.create = AsyncMock(return_value=fake_resp)
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            provider = OpenAISpeechProvider()
            out = await provider.transcribe(b"audio-bytes", content_type="audio/mpeg")
        assert isinstance(out, SpeechTranscript)
        assert out.text == "recognized words"
        assert out.language == "en"
        assert out.duration_seconds == 2.5
        # File-like wrapper has the right name (mp3).
        kwargs = fake_client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["file"].name.endswith(".mp3")

    @pytest.mark.asyncio
    async def test_transcribe_uses_wav_extension_for_wav_audio(self) -> None:
        fake_client = MagicMock()
        fake_client.audio.transcriptions.create = AsyncMock(
            return_value=SimpleNamespace(text="", language=None, duration=None)
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            provider = OpenAISpeechProvider()
            await provider.transcribe(b"x", content_type="audio/wav")
        kwargs = fake_client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["file"].name.endswith(".wav")

    @pytest.mark.asyncio
    async def test_transcribe_handles_missing_attrs(self) -> None:
        fake_client = MagicMock()
        # Response with no attributes — getattr falls back to defaults.
        fake_client.audio.transcriptions.create = AsyncMock(return_value=SimpleNamespace())
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            provider = OpenAISpeechProvider()
            out = await provider.transcribe(b"x")
        assert out.text == ""
        assert out.language is None
        assert out.duration_seconds is None

    @pytest.mark.asyncio
    async def test_get_client_caches_across_calls(self) -> None:
        fake_client = MagicMock()
        fake_client.audio.speech.create = AsyncMock(return_value=SimpleNamespace(content=b""))
        fake_client.audio.transcriptions.create = AsyncMock(return_value=SimpleNamespace(text=""))
        with patch("openai.AsyncOpenAI", return_value=fake_client) as mock_ctor:
            provider = OpenAISpeechProvider(api_key="sk-x")  # noqa: S106
            await provider.speak("a")
            await provider.transcribe(b"b")
        assert mock_ctor.call_count == 1
