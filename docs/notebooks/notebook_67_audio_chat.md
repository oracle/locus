# Voice Chat

Notebook 61 was text in, voice out (Agent plus dedicated TTS). This is
the next step: a single multimodal chat call to an audio-capable model
on OCI Generative AI that takes a `.wav` as the user message and
replies with both text and audio in one shot.

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
- `gpt-audio` returns a PCM-16 audio block, wrapped in a WAV header for
  portability (re-encode to mp3 with ffmpeg if you need it).

Prerequisites: an audio-capable model on OCI Generative AI
(`openai.gpt-audio` for chat, `openai.gpt-4o-mini-tts` to synthesise
the question on first run).

Run it:

    LOCUS_MODEL_PROVIDER=oci \
    LOCUS_OCI_PROFILE=MY_PROFILE \
    LOCUS_OCI_REGION=us-chicago-1 \
    LOCUS_OCI_AUTH_TYPE=security_token \
    LOCUS_OCI_COMPARTMENT=ocid1.compartment.oc1..…  \
    python examples/notebook_67_audio_chat.py

    afplay notebook_62_answer.wav   # macOS

This notebook does not run under `LOCUS_MODEL_PROVIDER=mock` — it
builds an OCI signer directly, so it needs real OCI credentials.

## Source

```python
--8<-- "examples/notebook_67_audio_chat.py"
```
