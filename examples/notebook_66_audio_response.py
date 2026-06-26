#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Notebook 66: Spoken cloud status advisory — voice output.

A cloud platform team often needs to talk, not just type — recorded
advisories for the on-call hotline, status-page audio, IVR announcements
about a region degradation. This notebook pairs a regular
chat-completions agent (text in, text out) with OpenAI's audio.speech
endpoint so an incident advisory can be spoken aloud. A regional
degradation — say an availability zone losing capacity while autoscaling
backs off — is exactly the kind of fast-moving event a recorded spoken
advisory is meant to keep on-call engineers ahead of.

Pipeline::

    advisory request ──▶ Agent (chat model)
                            │
                            │  advisory text
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
  the on-call IVR, or a status-page audio feed.

Prerequisites for live speech: an OpenAI API key with access to a TTS
model. The notebook uses gpt-4o-mini-tts for synthesis.

Run it
    TULIP_MODEL_PROVIDER=openai \\
    OPENAI_API_KEY=sk-... \\
    python examples/notebook_66_audio_response.py

    afplay notebook_66_response.mp3   # macOS
    # or open it in any media player

Offline: under TULIP_MODEL_PROVIDER=mock (or with no OPENAI_API_KEY) the
agent still drafts the advisory text against the mock model, and the
notebook prints the synthesis step it *would* run instead of calling the
real TTS endpoint — so it runs end-to-end with zero credentials.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from config import get_model

from tulip.agent import Agent, AgentConfig


PROMPT = (
    "Write a 60-word spoken cloud status advisory for on-call engineers about "
    "an ongoing degradation in the us-east-1 region: one availability zone is "
    "losing compute capacity and autoscaling is backing off, so new instance "
    "launches are failing. Tell them to fail workloads over to us-west-2 and "
    "to watch the status page for the next update."
)
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
OUT_PATH = Path(__file__).resolve().parent / "notebook_66_response.mp3"


def _audio_client():
    """An OpenAI async client for /v1/audio/speech, or None if offline.

    Tulip's chat model wraps chat completions; for audio.speech.create
    we use a plain ``openai.AsyncOpenAI`` against the same key. When no
    ``OPENAI_API_KEY`` is set (e.g. the mock-model walkthrough) we return
    None so the caller can describe the synthesis step without a network
    call.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    import openai

    return openai.AsyncOpenAI(api_key=api_key)


async def main() -> None:
    print("Notebook 66: Spoken cloud status advisory via OpenAI text-to-speech")
    print("=" * 60)

    # Step 1: a regular Tulip Agent drafts the advisory as text.
    agent = Agent(
        config=AgentConfig(
            agent_id="cloud-status-advisory",
            model=get_model(max_tokens=600),
            system_prompt=(
                "You are a cloud platform on-call lead recording a short voice "
                "advisory. Reply in natural spoken English, no markdown, no "
                "bullet points. Calm, clear, and specific."
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
    print(f"\n← advisory text ({len(reply)} chars):\n{reply}\n")

    # Step 2: synthesise speech through the audio.speech endpoint.
    client = _audio_client()
    if client is None:
        print(
            "→ offline: skipping synthesis (no OPENAI_API_KEY). Would call "
            f"audio.speech.create model={TTS_MODEL!r} voice={TTS_VOICE!r}"
        )
        print("  Set OPENAI_API_KEY to write a real mp3 to", OUT_PATH)
        return

    print(f"→ synthesising speech with model={TTS_MODEL!r} voice={TTS_VOICE!r}")
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
