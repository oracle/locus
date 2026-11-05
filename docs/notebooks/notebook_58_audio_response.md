# Audio response

An agent often needs to speak as well as write: a recorded advisory for
the on-call hotline, a status-page audio feed, or an interactive voice
response (IVR) announcement about a region degradation. This notebook
pairs a regular `Agent` (text in, text out) with the SDK's
`OpenAISpeechProvider`, which calls OpenAI's `/v1/audio/speech` endpoint,
so a cloud status advisory drafted by the agent is spoken aloud.

The provider wraps the OpenAI Python client, so an OpenAI API key is all
the speech step needs and there is no separate audio service to configure.
Setting `speech_provider=` on `AgentConfig` registers the provider as
`speak` and `transcribe` tools (notebook 57). The `speak` tool returns a
one-line summary to the model rather than the audio, so a program that
needs the audio itself calls the provider directly, as this one does.

What this notebook covers:

- Drafting the advisory with a regular agent and reading the reply back
  as text.
- Building `OpenAISpeechProvider` once with the text-to-speech (TTS) model
  and a default voice; the voice can also be chosen per call with
  `speak(..., voice=...)` (for example `alloy`, `coral` or `verse`; the
  catalogue is OpenAI's and the SDK passes the name through unchanged).
- Writing the returned bytes to a standard MP3 file that can be served to
  a web `<audio>` element, the on-call IVR or a status-page audio feed.
- An offline branch that still drafts the text but skips synthesis when
  no OpenAI key is set.

## Prerequisites

Live speech needs an OpenAI API key with access to a text-to-speech model
(the notebook uses `gpt-4o-mini-tts`) and the `openai` extra
(`pip install "locus-sdk[openai]"`):

```bash
export OPENAI_API_KEY=sk-...
```

The request goes to the agent, whose reply is the advisory text. The
provider's `speak()` sends that text to `/v1/audio/speech` and returns a
`SynthesizedAudio` whose `audio_bytes` the notebook writes to a file. The
dashed branch is the run without an OpenAI key: the text is still
drafted, but the `speak()` call is printed rather than made and no file
is written:

{{ locus_diagram audio-response }}

## Key lines

The provider is built once with the TTS model and the default voice, so
the synthesis step in `main` is a single `speak()` call; when
`OPENAI_API_KEY` is unset the helper returns `None`:

{{ locus_source examples/notebook_58_audio_response.py::_speech_provider }}

## Run it

```bash
python examples/notebook_58_audio_response.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one. The speech step is
separate from that choice: it reads `OPENAI_API_KEY` directly and builds the
`OpenAISpeechProvider` only when the key is set. Without the key the agent
still drafts the advisory, and the script prints the `speak()` call it would
make and returns without writing a file.

To hear the advisory, supply the key; the script then writes
`notebook_58_response.mp3` next to the example. Forcing the chat model to
OpenAI as well keeps the spoken text a real advisory on a machine with no
OCI profile, where `examples/config.py` would otherwise select the mock
model and its fixed test sentence:

```bash
LOCUS_MODEL_PROVIDER=openai \
OPENAI_API_KEY=sk-... \
python examples/notebook_58_audio_response.py

afplay examples/notebook_58_response.mp3   # macOS
```

## See also

- [Concepts: Multi-modal providers](../concepts/multi-modal-providers.md) —
  the four provider protocols, the tools they register, and why the
  `speak` tool returns a summary string rather than audio.
- [Multi-modal providers](notebook_57_multimodal_providers.md) — the four
  provider protocols and the tools the SDK registers for them on
  `AgentConfig`; the speech provider is shown there in a wiring snippet
  (notebook 57).
- [Audio chat](notebook_59_audio_chat.md) — one audio-capable chat call
  with speech in and speech out (notebook 59).

## Source

??? note "Full source — `examples/notebook_58_audio_response.py`"

    ````python
    --8<-- "examples/notebook_58_audio_response.py"
    ````
