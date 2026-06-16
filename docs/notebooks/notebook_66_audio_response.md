# Voice Output

A real agent often needs to talk, not just type. This notebook pairs a
regular chat-completions agent (text in, text out) with an
audio-capable model on OCI Generative AI so the response can be spoken
aloud.

Pipeline::

    user prompt ──▶ Agent (OCIChatCompletionsModel · chat model)
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
- Bring-your-own-voice via the `voice=` parameter (alloy, ash, ballad,
  coral, echo, sage, shimmer, verse).
- Output is a normal MP3 you can pipe into a frontend `<audio>`
  element, an IVR system, or a podcast feed.

Prerequisites: an audio-capable model on OCI Generative AI. The
notebook uses `openai.gpt-4o-mini-tts` for synthesis.

Run it:

    LOCUS_MODEL_PROVIDER=oci \
    LOCUS_OCI_PROFILE=MY_PROFILE \
    LOCUS_OCI_REGION=us-chicago-1 \
    LOCUS_OCI_AUTH_TYPE=security_token \
    LOCUS_OCI_COMPARTMENT=ocid1.compartment.oc1..…  \
    python examples/notebook_66_audio_response.py

    afplay notebook_66_response.mp3   # macOS

This notebook does not run under `LOCUS_MODEL_PROVIDER=mock` — it
builds an OCI signer directly, so it needs real OCI credentials.

## Source

```python
--8<-- "examples/notebook_66_audio_response.py"
```
