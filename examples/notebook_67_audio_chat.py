#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 67: Payments-support voice assistant — voice in, voice out.

Notebook 66 was text in, voice out (Agent plus dedicated TTS). This is
the next step: a single multimodal chat call to an audio-capable
OpenAI model that takes a .wav as the user message and replies with
both text and audio in one shot — the shape of a 24/7 payments support
line where a cardholder phones in about a declined charge and gets
spoken guidance back.

Pipeline::

                         (synth via notebook 66 if absent)
                                       │
                                       ▼
                          ./notebook_67_question.wav
                                       │
                                       ▼
              POST /v1/chat/completions
              model=gpt-audio
              modalities=["text","audio"]
              messages[-1].content = [{type:"input_audio", ...}]
                                       │
                                       │ {choices[0].message.audio.data, .transcript}
                                       ▼
                          ./notebook_67_answer.wav
                          (+ printed transcript)

- One model call replaces three (transcribe → chat → synthesise),
  cutting latency for a payments line that must answer in seconds.
- A plain OpenAI client — no realtime websocket plumbing required.
- gpt-audio returns a PCM-16 audio block, wrapped in a WAV header for
  portability (re-encode to mp3 with ffmpeg if you need it).

Prerequisites: an OpenAI API key with access to an audio-capable model
(gpt-audio for chat, gpt-4o-mini-tts to synthesise the cardholder's
question on first run).

Run it
    TULIP_MODEL_PROVIDER=openai \\
    OPENAI_API_KEY=sk-... \\
    python examples/notebook_67_audio_chat.py

    afplay notebook_67_answer.wav   # macOS

Note: with TULIP_MODEL_PROVIDER=mock (or no OPENAI_API_KEY) the
notebook runs fully offline — it skips the network and produces a
short simulated PCM-16 reply so you can read the event flow before
wiring real credentials.
"""

from __future__ import annotations

import asyncio
import base64
import math
import os
import struct
import wave
from pathlib import Path


CHAT_MODEL = "gpt-audio"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"

ROOT = Path(__file__).resolve().parent
QUESTION_WAV = ROOT / "notebook_67_question.wav"
ANSWER_MP3 = ROOT / "notebook_67_answer.mp3"

SAMPLE_RATE = 24000  # gpt-audio returns mono PCM-16 at 24 kHz

# The cardholder's question — synthesised once on first run, reused thereafter.
QUESTION_TEXT = (
    "Hi, payments support? My card was just declined twice trying to pay for "
    "my order, but my bank says I have funds. What should I do?"
)

# Frames the model as the payments-support assistant: practical, safe advice.
SUPPORT_SYSTEM = (
    "You are the payments support assistant for an online merchant. Give "
    "calm, practical guidance in two or three sentences. Never ask the caller "
    "to read out their full card number, CVV, or one-time passcode, and "
    "always point them to the bank number on the back of their card to "
    "approve a flagged charge."
)

# Canned spoken reply used in offline/mock mode so the flow runs end-to-end.
OFFLINE_TRANSCRIPT = (
    "I'm sorry about the trouble. A double decline with funds available "
    "usually means your bank flagged the charge for verification rather than "
    "a balance problem, so please call the number on the back of your card to "
    "approve it and then retry the payment. For your safety, never share your "
    "full card number or security code with anyone who calls you."
)


def _is_offline() -> bool:
    """True when we should skip the network (mock provider or no key)."""
    provider = os.environ.get("TULIP_MODEL_PROVIDER", "").lower() or "mock"
    return provider == "mock" or not os.environ.get("OPENAI_API_KEY")


def _build_audio_client():
    """An OpenAI async client for both /v1/audio and /v1/chat endpoints."""
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        msg = "OPENAI_API_KEY is required for the audio endpoints"
        raise RuntimeError(msg)
    return openai.AsyncOpenAI(api_key=api_key)


def _synth_tone_pcm16(seconds: float = 0.6, freq: float = 320.0) -> bytes:
    """A short, quiet sine tone as mono PCM-16 — a stand-in for real speech."""
    n = int(SAMPLE_RATE * seconds)
    amp = 6000  # well below the 32767 ceiling, so it stays soft
    frames = (
        struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)))
        for i in range(n)
    )
    return b"".join(frames)


def _write_wav_pcm16(pcm: bytes, path: Path) -> int:
    """Write mono PCM-16 @ 24 kHz into a portable WAV container."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return path.stat().st_size


async def _ensure_question_audio(client) -> bytes:
    """Synthesise the caller's question once; reuse it on subsequent runs."""
    if QUESTION_WAV.exists():
        return QUESTION_WAV.read_bytes()
    if client is None:  # offline: write a placeholder tone instead of TTS
        print(f"→ offline: writing placeholder caller audio → {QUESTION_WAV.name}")
        _write_wav_pcm16(_synth_tone_pcm16(), QUESTION_WAV)
        return QUESTION_WAV.read_bytes()
    print(f"→ synthesising caller audio with {TTS_MODEL!r} (one-time)")
    speech = await client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=QUESTION_TEXT,
        response_format="wav",
    )
    audio = await speech.aread()
    QUESTION_WAV.write_bytes(audio)
    print(f"  wrote {len(audio):,} bytes → {QUESTION_WAV}")
    return audio


async def _voice_reply(client, audio_b64: str) -> tuple[str, str]:
    """One multimodal chat call: audio in, transcript + PCM-16 audio out.

    Returns ``(transcript, pcm16_base64)``. In offline mode the network
    call is skipped and a canned reply is returned instead, preserving
    the same return shape the live endpoint produces.
    """
    if client is None:
        print(f"→ offline: simulating {CHAT_MODEL!r} reply (no network call)")
        pcm_b64 = base64.b64encode(_synth_tone_pcm16(seconds=1.0)).decode("ascii")
        return OFFLINE_TRANSCRIPT, pcm_b64

    print(f"→ asking {CHAT_MODEL!r} (caller audio in, spoken advice + text out)")
    response = await client.chat.completions.create(
        model=CHAT_MODEL,
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "pcm16"},
        messages=[
            {"role": "system", "content": SUPPORT_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    }
                ],
            },
        ],
    )
    msg = response.choices[0].message
    transcript = getattr(msg.audio, "transcript", "") if msg.audio else (msg.content or "")
    pcm_b64 = msg.audio.data if msg.audio else None
    if not pcm_b64:
        msg_err = "gpt-audio returned no audio block — check the response shape"
        raise RuntimeError(msg_err)
    return transcript, pcm_b64


async def main() -> None:
    print("Notebook 67: Payments-support voice assistant")
    print("=" * 60)

    offline = _is_offline()
    client = None if offline else _build_audio_client()

    # Step 1: make sure we have the caller's input wav.
    audio_in = await _ensure_question_audio(client)
    audio_b64 = base64.b64encode(audio_in).decode("ascii")

    # Step 2: one multimodal chat-completions call does transcribe + advise
    # + synthesise in a single round-trip.
    print(f"\n→ cardholder asks: {QUESTION_TEXT!r}")
    transcript, pcm_b64 = await _voice_reply(client, audio_b64)

    print(f"\n← support transcript:\n{transcript.strip()}\n")

    # Step 3: write the spoken advice (PCM16 in a WAV wrapper).
    out_wav = ANSWER_MP3.with_suffix(".wav")
    out_size = _write_wav_pcm16(base64.b64decode(pcm_b64), out_wav)
    print(f"✓ wrote {out_size:,} bytes → {out_wav}")
    print("  Play it on macOS:        afplay notebook_67_answer.wav")
    print("  Linux (aplay):           aplay notebook_67_answer.wav")
    print("  Re-encode to mp3:        ffmpeg -i notebook_67_answer.wav notebook_67_answer.mp3")


if __name__ == "__main__":
    asyncio.run(main())
