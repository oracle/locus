#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 62: Voice in, voice out chat through one audio-capable model call.

Notebook 61 was text in, voice out (Agent plus dedicated TTS). This is
the next step: a single multimodal chat call to an audio-capable model
on OCI Generative AI that takes a .wav as the user message and replies
with both text and audio in one shot.

Pipeline::

                         (synth via notebook 60 if absent)
                                       │
                                       ▼
                          ./notebook_62_question.wav
                                       │
                                       ▼
              POST /openai/v1/chat/completions
              model=openai.gpt-audio
              modalities=["text","audio"]
              messages[0].content = [{type:"input_audio", ...}]
                                       │
                                       │ {choices[0].message.audio.data, .transcript}
                                       ▼
                          ./notebook_62_answer.wav
                          (+ printed transcript)

- One model call replaces three (transcribe → chat → synthesise),
  cutting latency for voice agents.
- Same OCI v1 signer and base URL as the rest of the notebooks — no
  realtime websocket plumbing required.
- gpt-audio returns a PCM-16 audio block, wrapped in a WAV header for
  portability (re-encode to mp3 with ffmpeg if you need it).

Prerequisites: an audio-capable model on OCI Generative AI
(openai.gpt-audio for chat, openai.gpt-4o-mini-tts to synthesise the
question on first run).

Run it
    LOCUS_MODEL_PROVIDER=oci \\
    LOCUS_OCI_PROFILE=MY_PROFILE \\
    LOCUS_OCI_REGION=us-chicago-1 \\
    LOCUS_OCI_AUTH_TYPE=security_token \\
    LOCUS_OCI_COMPARTMENT=ocid1.compartment.oc1..…  \\
    python examples/notebook_67_audio_chat.py

    afplay notebook_62_answer.wav   # macOS

Note: this notebook does not run under LOCUS_MODEL_PROVIDER=mock —
it builds an OCI signer directly, so it needs real OCI credentials.
The smoke test for mock environments is `python -m py_compile <file>`.
"""

from __future__ import annotations

import asyncio
import base64
import os
import wave
from pathlib import Path


CHAT_MODEL = "openai.gpt-audio"
TTS_MODEL = "openai.gpt-4o-mini-tts"
TTS_VOICE = "alloy"

ROOT = Path(__file__).resolve().parent
QUESTION_WAV = ROOT / "notebook_62_question.wav"
ANSWER_MP3 = ROOT / "notebook_62_answer.mp3"

# Spoken question — synthesised once on first run, reused thereafter.
QUESTION_TEXT = "What's the elevator pitch for the locus SDK? Two sentences, friendly tone."


def _build_oci_audio_client():
    """Reuse the OCI v1 signer for both /openai/v1 audio and chat endpoints."""
    import httpx
    import oci  # noqa: PLC0415 — optional [oci] extra
    import openai

    from locus.models.providers.oci._signing import OCIRequestSigner
    from locus.models.providers.oci.openai_compat import build_oci_openai_base_url

    profile = os.environ.get("LOCUS_OCI_PROFILE", "DEFAULT")
    region = os.environ.get("LOCUS_OCI_REGION", "us-chicago-1")
    compartment_id = os.environ.get("LOCUS_OCI_COMPARTMENT")

    cfg = oci.config.from_file(profile_name=profile)
    if os.environ.get("LOCUS_OCI_AUTH_TYPE") == "security_token":
        token_file = os.path.expanduser(cfg["security_token_file"])
        key_file = os.path.expanduser(cfg["key_file"])
        with open(token_file, encoding="utf-8") as fh:
            token = fh.read().strip()
        private_key = oci.signer.load_private_key_from_file(key_file)
        signer = oci.auth.signers.SecurityTokenSigner(token, private_key)
    else:
        signer = oci.signer.Signer.from_config(cfg)

    http_client = httpx.AsyncClient(
        auth=OCIRequestSigner(signer, compartment_id=compartment_id),
        timeout=httpx.Timeout(120.0, connect=10.0),
    )
    return openai.AsyncOpenAI(
        api_key="not-used",
        base_url=build_oci_openai_base_url(region),
        http_client=http_client,
    )


async def _ensure_question_audio(client) -> bytes:
    """Synthesise the question once; reuse it on subsequent runs."""
    if QUESTION_WAV.exists():
        return QUESTION_WAV.read_bytes()
    print(f"→ synthesising question audio with {TTS_MODEL!r} (one-time)")
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


def _wav_to_mp3_pcm16_passthrough(pcm16_b64: str, out_path: Path) -> int:
    """Wrap gpt-audio's base64 PCM-16 mono 24 kHz block in a WAV header.

    No codec install required. Re-encode to mp3 with ffmpeg if you need
    a smaller file or a different container.
    """
    pcm = base64.b64decode(pcm16_b64)
    wav_path = out_path.with_suffix(".wav")
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(24000)
        wf.writeframes(pcm)
    return wav_path.stat().st_size


async def main() -> None:
    print("Notebook 62: Voice in, voice out chat on OCI Generative AI")
    print("=" * 60)

    client = _build_oci_audio_client()

    # Step 1: make sure we have an input wav.
    audio_in = await _ensure_question_audio(client)
    audio_b64 = base64.b64encode(audio_in).decode("ascii")

    # Step 2: one multimodal chat-completions call does transcribe + chat
    # + synthesise in a single round-trip.
    print(f"\n→ asking {CHAT_MODEL!r} (audio in, audio + text out)")
    response = await client.chat.completions.create(
        model=CHAT_MODEL,
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "pcm16"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    }
                ],
            }
        ],
    )
    msg = response.choices[0].message
    transcript = getattr(msg.audio, "transcript", "") if msg.audio else (msg.content or "")
    pcm_b64 = msg.audio.data if msg.audio else None

    print(f"\n← transcript:\n{transcript.strip()}\n")

    if not pcm_b64:
        msg_err = "gpt-audio returned no audio block — check the response shape"
        raise RuntimeError(msg_err)

    # Step 3: write the audio reply (PCM16 in a WAV wrapper).
    out_size = _wav_to_mp3_pcm16_passthrough(pcm_b64, ANSWER_MP3)
    out_wav = ANSWER_MP3.with_suffix(".wav")
    print(f"✓ wrote {out_size:,} bytes → {out_wav}")
    print("  Play it on macOS:        afplay notebook_62_answer.wav")
    print("  Linux (aplay):           aplay notebook_62_answer.wav")
    print("  Re-encode to mp3:        ffmpeg -i notebook_62_answer.wav notebook_62_answer.mp3")


if __name__ == "__main__":
    asyncio.run(main())
