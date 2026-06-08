# Voice Chat

Notebook 66 was text in, voice out (Agent plus dedicated TTS). This is
the next step: a single multimodal chat call to an audio-capable
OpenAI model that takes a `.wav` as the user message and replies with
both text and audio in one shot.

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
              messages[0].content = [{type:"input_audio", ...}]
                                       │
                                       │ {choices[0].message.audio.data, .transcript}
                                       ▼
                          ./notebook_67_answer.wav
                          (+ printed transcript)

- One model call replaces three (transcribe → chat → synthesise),
  cutting latency for voice agents.
- A plain OpenAI client — no realtime websocket plumbing required.
- `gpt-audio` returns a PCM-16 audio block, wrapped in a WAV header for
  portability (re-encode to mp3 with ffmpeg if you need it).

Prerequisites: an OpenAI API key with access to an audio-capable model
(`gpt-audio` for chat, `gpt-4o-mini-tts` to synthesise the question on
first run).

Run it:

    TULIP_MODEL_PROVIDER=openai \
    OPENAI_API_KEY=sk-... \
    python examples/notebook_67_audio_chat.py

    afplay notebook_67_answer.wav   # macOS

This notebook does not run under `TULIP_MODEL_PROVIDER=mock` — it
calls a real audio endpoint, so it needs real credentials.

## Source

```python
--8<-- "examples/notebook_67_audio_chat.py"
```
