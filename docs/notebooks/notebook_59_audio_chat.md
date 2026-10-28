# Voice Chat

The previous notebook, [Voice Output](notebook_58_audio_response.md), is
text in, voice out (an Agent plus a dedicated TTS call). This notebook
takes the next step: a single multimodal chat call to an audio-capable
OpenAI model that takes a `.wav` as the user message and replies with
both text and audio in the same response — the shape of a 24/7 payments
support line where a cardholder phones in about a declined charge and
gets spoken guidance back.

{{ locus_diagram audio-chat }}

- One model call replaces three (transcribe → chat → synthesise),
  cutting latency for a payments line that must answer in seconds.
- A plain OpenAI client — no realtime WebSocket connection is required.
- `gpt-audio` returns a PCM-16 audio block, wrapped in a WAV header for
  portability (re-encode to MP3 with ffmpeg if you need it).
- The assistant is framed to never ask the caller to read out their full
  card number, CVV, or one-time passcode, and to point them at the bank
  number on the back of the card to approve a flagged charge.

Prerequisites: an OpenAI API key with access to an audio-capable model
(`gpt-audio` for chat, `gpt-4o-mini-tts` to synthesise the cardholder's
question on first run).

Run it:

    LOCUS_MODEL_PROVIDER=openai \
    OPENAI_API_KEY=sk-... \
    python examples/notebook_59_audio_chat.py

    afplay notebook_59_answer.wav   # macOS

With `LOCUS_MODEL_PROVIDER=mock` (or no `OPENAI_API_KEY`) the notebook
runs fully offline: it skips the network and produces a short simulated
PCM-16 reply, so you can follow the three steps and check the output
file before adding real credentials.

## Source

````python
--8<-- "examples/notebook_59_audio_chat.py"
````
