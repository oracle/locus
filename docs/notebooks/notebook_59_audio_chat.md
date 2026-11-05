# Audio chat

The previous notebook, [Audio response](notebook_58_audio_response.md), is
text in, voice out: an `Agent` drafts the reply and the SDK's
`OpenAISpeechProvider` speaks it through a separate text-to-speech (TTS)
call. This notebook takes the next step: a single multimodal chat call to
an audio-capable OpenAI model that takes a `.wav` as the user message and
replies with both text and audio in the same response — the shape of a
24/7 payments support line where a cardholder phones in about a declined
charge and gets spoken guidance back.

What this notebook covers:

- One model call replaces three (transcribe → chat → synthesize), cutting
  latency for a payments line that must answer in seconds.
- A plain OpenAI client (`openai.AsyncOpenAI`) is all the notebook uses:
  no realtime WebSocket connection and no model class are involved.
- The call asks for 16-bit pulse-code modulation (PCM-16) audio, and the
  notebook wraps the returned block in a WAV header for portability
  (re-encode to MP3 with ffmpeg when needed).
- The assistant is framed to never ask the caller to read out their full
  card number, card verification value (CVV) or one-time passcode, and to
  point them at the bank number on the back of the card to approve a
  flagged charge.

## Prerequisites

A live run needs an OpenAI API key with access to `gpt-audio`, which
answers the call, and, on the first run only, to `gpt-4o-mini-tts`, which
synthesizes the cardholder's question. It also needs the `openai` package
(`pip install "locus-sdk[openai]"`), imported only when the live client
is built; without a key the notebook still runs offline, as described
under [Run it](#run-it), using nothing beyond the standard library.

The caller's question is a WAV file. Only when that file is not yet on
disk does the notebook synthesize it first, through `/v1/audio/speech`
(the dashed branch); the file then goes into one `/v1/chat/completions`
request as an `input_audio` content part with the `text` and `audio`
modalities, and the reply's `message.audio` block becomes the spoken
answer plus its transcript:

{{ locus_diagram audio-chat }}

## Key lines

The caller's question is synthesized once through the speech endpoint
(`client.audio.speech.create` with `response_format="wav"`) and reused
from disk on later runs; offline, a placeholder tone is written instead:

{{ locus_source examples/notebook_59_audio_chat.py::_ensure_question_audio }}

One chat-completions request, under the payments-support system prompt,
carries the question as an `input_audio` content part and asks for the
`text` and `audio` modalities in `pcm16` format, so the reply's
`message.audio` block holds both the spoken answer (`data`, base64
PCM-16) and its `transcript`; offline, a canned reply preserves the same
return shape:

{{ locus_source examples/notebook_59_audio_chat.py::_voice_reply }}

The `pcm16` output is raw mono 16-bit samples at 24 kHz, which the
notebook wraps in a WAV container for portability:

{{ locus_source examples/notebook_59_audio_chat.py::_write_wav_pcm16 }}

## Run it

```bash
python examples/notebook_59_audio_chat.py
```

This notebook does not take its model from `examples/config.py` and has no
OCI Generative AI path: it reads `LOCUS_MODEL_PROVIDER` and
`OPENAI_API_KEY` itself, and the chat call always goes to OpenAI. With the
variable unset or set to `mock`, or with no key, the script runs fully
offline: nothing is sent, a question `.wav` not yet on disk is written as
a placeholder tone, and the reply is a canned transcript plus a one-second
simulated PCM-16 tone, so the question audio, the printed transcript and
the written answer file can be checked before adding real credentials.

Both files are written next to the example: `examples/notebook_59_question.wav`
on the first run and `examples/notebook_59_answer.wav` on every run; the
play hints the script prints at the end assume `examples/` is the working
directory. An existing question file is reused as is, so delete the
placeholder left by an offline run before the first live run, or the tone
rather than a synthesized question is sent as the caller's audio.

The live path runs when `LOCUS_MODEL_PROVIDER` is set to any value other
than `mock` (`openai`, matching the endpoint the call actually uses) and
`OPENAI_API_KEY` is present with access to `gpt-audio` and, for the first
run, `gpt-4o-mini-tts`:

```bash
LOCUS_MODEL_PROVIDER=openai \
OPENAI_API_KEY=sk-... \
python examples/notebook_59_audio_chat.py

afplay examples/notebook_59_answer.wav   # macOS
```

## See also

- [Audio response](notebook_58_audio_response.md) — the preceding step: a
  `Agent` drafts the text and `OpenAISpeechProvider` speaks it
  (notebook 58).
- [Concepts: Multi-modal providers](../concepts/multi-modal-providers.md) —
  the `BaseSpeechProvider` protocol and the built-in `OpenAISpeechProvider`
  (`speak` and `transcribe`); the audio-in, audio-out chat call made here
  has no wrapper and goes through the OpenAI client directly.

## Source

??? note "Full source — `examples/notebook_59_audio_chat.py`"

    ````python
    --8<-- "examples/notebook_59_audio_chat.py"
    ````
