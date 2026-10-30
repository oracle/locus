Build a resumable conversation with Locus checkpointing.

1. Write `memory_agent.py` exposing `make_agent(checkpointer) -> Agent` whose model is a
   `FunctionModel` replying `user turns seen: <n>` where `<n>` counts the user messages in the
   conversation so far, and `demo(base_dir: str) -> list[int]` that uses
   `FileCheckpointer(base_dir=...)`, runs one agent on `thread_id="case-1"`, a second agent
   instance (same checkpointer) on `case-1` again, then on `case-other`, and returns the three
   counts. `python memory_agent.py` prints `TURNS 1 2 1` using a temporary directory.
2. Write `test_memory_agent.py` asserting `demo(tmp_path)` returns `[1, 2, 1]`.
3. Run the tests and the script; both must pass.
