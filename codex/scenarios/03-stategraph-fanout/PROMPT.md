Build a map-reduce workflow with Locus's `StateGraph`.

1. Write `triage_graph.py` exposing `build() -> StateGraph` and `run(tickets: list[str]) -> dict`.
   The graph fans out over the tickets with `Send` (one `score` node execution per ticket, each
   producing a string `"<ticket>:<score>"` where the score is the length of the ticket id), and
   a `merge` node collects every fan-out result into `state["report"]`, a list with one entry
   per ticket. Finish at `END`. No model is needed; nodes are plain functions.
2. `python triage_graph.py` prints `REPORT <n>` with the number of entries, then the Mermaid
   text from `graph.get_mermaid()`.
3. Write `test_triage_graph.py`: `run(["t-1", "t-2", "t-3"])["report"]` has three entries, each
   of the form `ticket:score`; the Mermaid text names the `score` and `merge` nodes.
4. Run the tests and the script; both must pass.
