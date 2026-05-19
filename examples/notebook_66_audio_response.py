#!/usr/bin/env python3
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 61: Voice output — turn an agent's reply into speech.

A real agent often needs to talk, not just type. This notebook pairs a
regular chat-completions agent (text in, text out) with an audio-capable
model on OCI Generative AI so the response can be spoken aloud.

Pipeline::

    user prompt ──▶ Agent (OCIOpenAIModel · chat model)
                       │
                       │  reply text
                       ▼
                 OCI /openai/v1/audio/speech
                 (openai.gpt-4o-mini-tts)
                       │
                       │  mp3 bytes
                       ▼
                 ./notebook_66_response.mp3

- Same OCI v1 transport as the rest of the notebooks — one signer, one
  base URL, one set of credentials. No separate audio service to
  configure.
- Bring-your-own-voice via the voice= parameter (alloy, ash, ballad,
  coral, echo, sage, shimmer, verse).
- Output is a normal MP3 you can pipe into a frontend <audio> element,
  an IVR system, or a podcast feed.

Prerequisites: an audio-capable model on OCI Generative AI. The
notebook uses openai.gpt-4o-mini-tts for synthesis.

Run it
    LOCUS_MODEL_PROVIDER=oci \\
    LOCUS_OCI_PROFILE=MY_PROFILE \\
    LOCUS_OCI_REGION=us-chicago-1 \\
    LOCUS_OCI_AUTH_TYPE=security_token \\
    LOCUS_OCI_COMPARTMENT=ocid1.compartment.oc1..…  \\
    python examples/notebook_66_audio_response.py

    afplay notebook_66_response.mp3   # macOS
    # or open it in any media player

Note: this notebook does not run under LOCUS_MODEL_PROVIDER=mock —
it builds an OCI signer directly, so it needs real OCI credentials.
The smoke test for mock environments is `python -m py_compile <file>`.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from config import get_model

from locus.agent import Agent, AgentConfig


PROMPT = (
    "Give me a 25-word elevator pitch for the locus SDK aimed at a senior "
    "platform engineer. Speak it in the second person."
)
TTS_MODEL = "openai.gpt-4o-mini-tts"
TTS_VOICE = "alloy"
OUT_PATH = Path(__file__).resolve().parent / "notebook_66_response.mp3"


def _build_oci_audio_client():
    """Reuse the OCI v1 signer to talk to /openai/v1/audio/speech.

    OCIOpenAIModel wraps chat completions; for audio.speech.create we
    attach the same signer to a fresh openai.AsyncOpenAI so the audio
    endpoint goes through the same authenticated transport.
    """
    import httpx
    import openai

    from locus.models.providers.oci._signing import OCIRequestSigner
    from locus.models.providers.oci.openai_compat import build_oci_openai_base_url

    profile = os.environ.get("LOCUS_OCI_PROFILE", "DEFAULT")
    region = os.environ.get("LOCUS_OCI_REGION", "us-chicago-1")
    compartment_id = os.environ.get("LOCUS_OCI_COMPARTMENT")

    # Build the signer the same way OCIOpenAIModel does internally.
    import oci  # noqa: PLC0415 — optional dep, lives behind the [oci] extra

    cfg = oci.config.from_file(profile_name=profile)
    auth_type = os.environ.get("LOCUS_OCI_AUTH_TYPE", "api_key")
    if auth_type == "security_token":
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
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    return openai.AsyncOpenAI(
        api_key="not-used",
        base_url=build_oci_openai_base_url(region),
        http_client=http_client,
    )


async def main() -> None:
    print("Notebook 61: Voice output via OCI Generative AI text-to-speech")
    print("=" * 60)

    # Step 1: a regular Locus Agent answers the prompt as text.
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

    # Step 2: synthesise speech through the OCI v1 audio.speech endpoint.
    print(f"→ synthesising speech with model={TTS_MODEL!r} voice={TTS_VOICE!r}")
    client = _build_oci_audio_client()
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
