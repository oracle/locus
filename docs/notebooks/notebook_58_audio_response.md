# Voice Output

An agent often needs to speak as well as write. This notebook pairs a
regular chat-completions agent (text in, text out) with the SDK's OpenAI
speech provider (`OpenAISpeechProvider`), which calls OpenAI's
`/v1/audio/speech` endpoint so a cloud status advisory can be spoken
aloud — ready for the on-call hotline, a status-page audio feed, or an
IVR announcement about a region degradation. The provider is a plain
OpenAI client, so there is no separate audio service to configure.

{{ locus_diagram audio-response }}

Two details of the speech step:

- Select the voice with `default_voice=` on the provider, or per call
  with `speak(..., voice=...)` (alloy, ash, ballad, coral, echo, sage,
  shimmer, verse).
- Output is a normal MP3 you can pipe into a frontend `<audio>`
  element, the on-call IVR, or a status-page audio feed.

Prerequisites for live speech: an OpenAI API key with access to a TTS
model. The notebook uses `gpt-4o-mini-tts` for synthesis.

Run it:

    LOCUS_MODEL_PROVIDER=openai \
    OPENAI_API_KEY=sk-... \
    python examples/notebook_58_audio_response.py

    afplay notebook_58_response.mp3   # macOS

Under `LOCUS_MODEL_PROVIDER=mock`, or with a chat provider other than
OpenAI (for example the OCI Generative AI default) and no
`OPENAI_API_KEY`, the agent still drafts the advisory text and the
notebook prints the synthesis step it would run instead of calling the
TTS endpoint; with the mock model it therefore runs end-to-end without
any credentials.

## Source

````python
--8<-- "examples/notebook_58_audio_response.py"
````
