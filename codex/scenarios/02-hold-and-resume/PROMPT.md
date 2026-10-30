Build the human-in-the-loop flow with Locus: a held refund that a person approves, after which
the run resumes and the refund happens exactly once.

1. Write `flow.py`. An `Agent` with one gated tool `issue_refund(order_id: str, amount_usd: float)`
   that appends to a module-level list and returns a string. Gate it with `gate_tool`:
   `ControlPolicy(require_verification_score=0.0, require_human_for=frozenset({"payment"}))`, an
   `Action` of kind `payment` derived from the arguments, an `AuditTrail`,
   `on_refusal="interrupt"`, and `approval=FileApprovals("approvals.json")`. Give the agent
   `checkpointer=FileCheckpointer("checkpoints")`. The model is a `FunctionModel` that calls
   `issue_refund(order_id="ORD-1", amount_usd=250.0)` on its first turn and answers `Done.`
   once a tool result is in the messages. Running `python flow.py` must: run the agent on
   `thread_id="t1"`, catch the `InterruptEvent`, print `HELD <approval_id>`, decide the approval
   with `store.decide(approval_id, "approved", by="alice")`, resume with
   `agent.resume("approved", thread_id="t1", perform_dangling=True)`, then print `REFUNDS <n>`
   where `<n>` is how many times the refund body ran, and `TRAIL <bool>` from
   `trail.verify()`. Remove `approvals.json` and `checkpoints` at the start of `main()` so the
   script is repeatable.
2. Write `test_flow.py` that drives the same flow inside a temporary directory and asserts the
   refund ran once and the trail verifies.
3. Run the tests and `python flow.py`; both must pass.
