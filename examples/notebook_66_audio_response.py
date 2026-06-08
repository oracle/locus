#!/usr/bin/env python3
# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Notebook 66: Voice output — turn an agent's reply into speech.

A real agent often needs to talk, not just type. This notebook pairs a
regular chat-completions agent (text in, text out) with OpenAI's
audio.speech endpoint so the response can be spoken aloud.

Pipeline::

    user prompt ──▶ Agent (chat model)
                       │
                       │  reply text
                       ▼
                 OpenAI /v1/audio/speech
                 (gpt-4o-mini-tts)
                       │
                       │  mp3 bytes
                       ▼
                 ./notebook_66_response.mp3

- Bring-your-own-voice via the voice= parameter (alloy, ash, ballad,
  coral, echo, sage, shimmer, verse).
- Output is a normal MP3 you can pipe into a frontend <audio> element,
  an IVR system, or a podcast feed.

Prerequisites: an OpenAI API key with access to a TTS model. The
notebook uses gpt-4o-mini-tts for synthesis.

Run it
    TULIP_MODEL_PROVIDER=openai \\
    OPENAI_API_KEY=sk-... \\
    python examples/notebook_66_audio_response.py

    afplay notebook_66_response.mp3   # macOS
    # or open it in any media player

Note: this notebook does not run under TULIP_MODEL_PROVIDER=mock —
it calls a real TTS endpoint, so it needs real credentials.
The smoke test for mock environments is `python -m py_compile <file>`.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from config import get_model

from tulip.agent import Agent, AgentConfig


PROMPT = (
    "Give me a 25-word elevator pitch for the tulip SDK aimed at a senior "
    "platform engineer. Speak it in the second person."
)
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
OUT_PATH = Path(__file__).resolve().parent / "notebook_66_response.mp3"


def _build_audio_client():
    """An OpenAI async client for /v1/audio/speech.

    Tulip's chat model wraps chat completions; for audio.speech.create
    we use a plain ``openai.AsyncOpenAI`` against the same key.
    """
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        msg = "OPENAI_API_KEY is required for the TTS endpoint"
        raise RuntimeError(msg)
    return openai.AsyncOpenAI(api_key=api_key)


async def main() -> None:
    print("Notebook 66: Voice output via OpenAI text-to-speech")
    print("=" * 60)

    # Step 1: a regular Tulip Agent answers the prompt as text.
    agent = Agent(
        config=AgentConfig(
            agent_id="elevator-pitch",
            model=get_model(max_tokens=600),
            system_prompt=(
                "You are a senior developer-relations engineer. Reply in "
                "natural spoken English, no markdown, no bullet points."
            ),
            max_iterations=2,
        )
    )
    print(f"\n→ asking the agent: {PROMPT!r}")
    result = agent.run_sync(PROMPT)
    reply = (result.message or "").strip()
    if not reply:
        msg = "Agent returned no text — check provider creds + max_tokens"
        raise RuntimeError(msg)
    print(f"\n← agent reply ({len(reply)} chars):\n{reply}\n")

    # Step 2: synthesise speech through the audio.speech endpoint.
    print(f"→ synthesising speech with model={TTS_MODEL!r} voice={TTS_VOICE!r}")
    client = _build_audio_client()
    speech = await client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=reply,
        response_format="mp3",
    )
    audio_bytes = await speech.aread()
    OUT_PATH.write_bytes(audio_bytes)

    print(f"\n✓ wrote {len(audio_bytes):,} bytes of mp3 → {OUT_PATH}")
    print("  Play it on macOS:        afplay notebook_66_response.mp3")
    print("  Linux (mpg123):          mpg123 notebook_66_response.mp3")
    print("  Browser (file:// URL):   open notebook_66_response.mp3")


if __name__ == "__main__":
    asyncio.run(main())
