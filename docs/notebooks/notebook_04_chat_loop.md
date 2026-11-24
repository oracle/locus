# A chat loop

The other examples in this series are batch scripts: each asks one thing,
prints the answer, and exits. That is the right shape for teaching a
primitive and the wrong shape for the first thing most people want to do,
which is have a conversation.

This is the missing read-eval-print loop (REPL). It is deliberately small — a
loop, a checkpointer, and one `thread_id` — because the interesting part is
not the loop, it is that continuity comes from the checkpointer rather than
from anything you have to hold in your own code. Every turn is saved; the next
turn resumes from the saved state. Swap `MemoryCheckpointer` for a
[Redis or Postgres checkpointer](../concepts/checkpointers.md#picking-a-backend)
and the same conversation survives a restart.

What this notebook covers:

- A `thread_id` names the conversation; the checkpointer stores it.
- The loop keeps no history of its own — the agent's state is the history.
- Each answer is printed with the tool calls the agent made to reach it, so
  you can see the agent act rather than infer it from the answer.
- `/reset` starts a fresh thread, which shows the boundary a `thread_id`
  draws: same agent, no shared memory.

## Key lines

One agent and one checkpointer. `checkpoint_every_n_iterations=1` saves the
state after each iteration's tool results are in, so on a durable backend a
process exit mid-turn leaves a state that `Agent.continue_turn` picks up from
the last completed iteration instead of re-running finished tool calls. On a
`thread_id` with a checkpointer that can delete single checkpoints, such as
`MemoryCheckpointer`, this cadence is already the default; it is explicit
here so it is visible:

{{ locus_source examples/notebook_04_chat_loop.py::build_agent }}

Each turn is a single `arun` call that passes the new message and the
`thread_id`, nothing else, because the saved state is the history; once the
call returns, the turn's `result.tool_executions` are printed ahead of the
answer:

{{ locus_source examples/notebook_04_chat_loop.py::say }}

The loop reads one line at a time in a worker thread, so `input()` does not
block the event loop, and `/reset` switches to a new `thread_id` so the
previous turns are no longer visible:

{{ locus_source examples/notebook_04_chat_loop.py::interactive_run }}

## Run it

```bash
python examples/notebook_04_chat_loop.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

When standard input is not a terminal, as in a continuous-integration job, the
script replays a scripted three-turn conversation instead of prompting:

```bash
echo "" | python examples/notebook_04_chat_loop.py
```

Against the mock model every answer is the same placeholder, so that run shows
the loop and the tool call but not the memory. With a live model, the third
turn's question can only be answered because the first turn was saved and
reloaded with the `thread_id`.

## See also

- [Checkpointers](../concepts/checkpointers.md) — the backends and the
  checkpoint cadence.
- [Persist conversations across restarts](../how-to/persist-conversations.md)
  — the same thread on Redis or Postgres.
- [Conversation memory](notebook_05_agent_memory.md) — next in the series:
  the same checkpointer, with the saved state loaded back and inspected
  (notebook 05).

## Output

Running it against a live model — OCI Generative AI, `openai.gpt-4.1` — prints the scripted three-turn conversation: a greeting, a `get_weather` call for the city named a turn earlier, and the name from the first turn repeated back on the third; the model's wording and the timings differ from run to run:

```text
(no terminal detected — replaying a scripted conversation)

  you  : My name is Marta and I'm looking at flights to Lisbon.
  agent: Hi Marta! Lisbon is a fantastic destination. Do you need information on flights, things to do, or perhaps the current weather in Lisbon?

  you  : What's the weather there?
    · get_weather({'city': 'Lisbon'})
  agent: Lisbon is currently 18°C with light rain.

  you  : What did I say my name was?
  agent: You said your name is Marta.

The third answer is only possible because the first turn was saved.

============================================================
Next: Notebook 05 — Conversation memory: remembering what you told it
============================================================
```
<!-- notebook-output:end -->

## Source

??? note "Full source — `examples/notebook_04_chat_loop.py`"

    ````python
    --8<-- "examples/notebook_04_chat_loop.py"
    ````
