# Conversation memory

Give an agent a checkpointer and every conversation turn is saved under
its `thread_id`. Call the agent again with that `thread_id` — or attach a
new agent to the same store with the same `thread_id` — and the next turn
starts from the saved conversation, with every earlier message in front
of the model. With a durable backend, that holds across a process restart
too.

This notebook uses `MemoryCheckpointer`, an in-memory store, so it runs
with no setup; its state lasts only as long as the process. For a
conversation that survives a restart, swap in a durable checkpointer —
`FileCheckpointer`, or `redis_checkpointer()` /
`postgresql_checkpointer()` — see [Checkpointers](../concepts/checkpointers.md).

What this notebook covers:

- Building a `MemoryCheckpointer` and passing it to the agent.
- Keying conversations with `thread_id` (one per conversation).
- Pinning a checkpoint after every tool-calling iteration with
  `checkpoint_every_n_iterations=1`.
- A second agent attaching to the same store and picking up the same
  `thread_id`.
- Loading the saved `AgentState` and inspecting it field by field.
- Running many independent threads against a single store.

A checkpointer is the second of the four memory layers an agent can carry
— short-term memory, the thread: agent state saved under a `thread_id` and
loaded back on the next call.

{{ locus_diagram memory-layers }}

## Key lines

One checkpointer on the agent and the same `thread_id` on both calls are
enough for the second turn to see the first:

{{ locus_source examples/notebook_05_agent_memory.py::example_conversation_memory }}

`checkpoint_every_n_iterations=1` pins a checkpoint after each
tool-calling iteration — the default for a run with a `thread_id` on a
store that can delete a single checkpoint, such as `MemoryCheckpointer`,
and the setting to add on one that cannot — so a process killed mid-turn
loses at most the iteration in flight and `Agent.continue_turn(thread_id)`
picks the turn up from the last save:

{{ locus_source examples/notebook_05_agent_memory.py::example_checkpointing_with_tools }}

`checkpointer.load(thread_id)` returns the latest saved `AgentState`: the
whole conversation's messages, plus the last turn's iteration count, tool
history and [Reflexion](../concepts/reasoning.md#reflexion) confidence
history — the three that start afresh on each turn — open for inspection:

{{ locus_source examples/notebook_05_agent_memory.py::example_inspect_checkpoint }}

## Run it

```bash
python examples/notebook_05_agent_memory.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## See also

- [Checkpointers](../concepts/checkpointers.md) — every backend, and how
  to wire the two checkpointer shapes.
- [Memory](../concepts/memory.md) — the four layers in the diagram above.
- [A chat loop](notebook_04_chat_loop.md) — the same checkpointer behind
  an interactive loop (notebook 04).
- [Oracle agent memory](notebook_44_oracle_agent_memory.md) — long-term
  memory backed by Oracle Database 26ai (notebook 44).

## Output

Running it against a live model — OCI Generative AI, `openai.gpt-4.1` — prints the agent recalling a name and destination across turns, a note saved through a tool and read back, a second agent on the same store answering from the first agent's thread, two threads that keep their topics apart, and the fields and messages of a saved checkpoint; the model's wording and the timings differ from run to run:

```text
=== Part 1: Conversation memory ===

You: Hi! My name is Sam and I'm planning a trip to Japan next spring.
Assistant: Hi, Sam! That sounds so exciting—spring is a beautiful time to visit Japan, especially with the cherry blossoms. Do you already have some places or cities in mind that you’d like to visit? Or would you like some recommendations on where to go and what to see?

You: What's my name, and where am I going?
Assistant: Your name is Sam, and you're planning a trip to Japan next spring! If you have any more details to share or need help planning your itinerary, just let me know.

=== Part 2: Checkpoint after each iteration ===

You: Save a note: buy oat milk and coffee filters on the way home.
Assistant: Note saved: buy oat milk and coffee filters on the way home.
Tool calls: 1

You: What notes do we have so far?
Assistant: You have 1 note so far:
- buy oat milk and coffee filters on the way home.

=== Part 3: Reattaching to the same store ===

You: Remember: my favorite color is teal.
Assistant: Got it! Your favorite color is teal. I'll remember that for future conversations. If you want me to use it in any suggestions or ideas, just let me know!

[New agent — same store]
You: What's my favorite color?
Assistant: Your favorite color is teal!

=== Part 4: Multiple conversations ===

Thread 'chat_travel': What are we talking about?
Assistant: We're talking about your upcoming hiking trip in the Alps! You mentioned you're planning to go hiking there, and I was just asking for a bit more detail about your plans—like which part of the Alps you'll be visiting, how long your trip is, and what kind of hikes you're interested in. If you'd like, we can talk about anything from trail recommendations and packing tips to local food and weather conditions. Just let me know what you'd like help with!

Thread 'chat_music': What are we talking about?
Assistant: We’re talking about you starting to learn how to play the guitar! You mentioned that you just started, and I was offering some help or tips if you need them. If you want to talk about something else, or if you have any questions about learning guitar, just let me know!

=== Part 5: Inspecting a saved checkpoint ===

Thread ID: inspect_chat
Agent ID: chat_inspector
Iteration: 2
Message count: 10
Tool calls so far: 1
Confidence: 0.15
Confidence history: [0.15]

Messages:
  0. [system] You are a friendly assistant. Whenever the person ...
  1. [user] My name is Sam and I live in Portland.
  2. [assistant] None
  3. [tool] Recorded detail #1: [name] Their name is Sam.
  4. [tool] Recorded detail #2: [location] They live in Portla...
  5. [assistant] Nice to meet you, Sam! Portland is a great city—do...
  6. [user] On weekends I like to go rock climbing.
  7. [assistant] None
  8. [tool] Recorded detail #3: [hobby] They like to go rock c...
  9. [assistant] That sounds like an awesome way to spend your week...

Details recorded by record_detail: ['[name] Their name is Sam.', '[location] They live in Portland.', '[hobby] They like to go rock climbing on weekends.']

============================================================
Next: Notebook 06 — Streaming a Reply
============================================================
```
<!-- notebook-output:end -->

## Source

??? note "Full source — `examples/notebook_05_agent_memory.py`"

    ````python
    --8<-- "examples/notebook_05_agent_memory.py"
    ````
