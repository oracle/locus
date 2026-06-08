# Voice Output

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

- A plain OpenAI client — no separate audio service to configure.
- Bring-your-own-voice via the `voice=` parameter (alloy, ash, ballad,
  coral, echo, sage, shimmer, verse).
- Output is a normal MP3 you can pipe into a frontend `<audio>`
  element, an IVR system, or a podcast feed.

Prerequisites: an OpenAI API key with access to a TTS model. The
notebook uses `gpt-4o-mini-tts` for synthesis.

Run it:

    TULIP_MODEL_PROVIDER=openai \
    OPENAI_API_KEY=sk-... \
    python examples/notebook_66_audio_response.py

    afplay notebook_66_response.mp3   # macOS

This notebook does not run under `TULIP_MODEL_PROVIDER=mock` — it
calls a real TTS endpoint, so it needs real credentials.

## Source

```python
--8<-- "examples/notebook_66_audio_response.py"
```
