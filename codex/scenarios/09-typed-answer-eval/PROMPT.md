Build a triage agent with Locus: a typed answer, a termination rule and an evaluation, offline.

1. Write `triage.py`: a Pydantic `Verdict(priority: str, reason: str)`, a tool
   `escalate(reason: str) -> str`, and `build_agent(model) -> Agent` with
   `output_schema=Verdict`, `tools=[escalate]` and
   `termination=MaxIterations(6) | ToolCalled("escalate")`.
2. Write `test_triage.py`: (a) on a `ScriptedModel` that answers with Verdict JSON,
   `result.parsed_as(Verdict).priority` is what the JSON said; (b) on a `ScriptedModel` that calls
   `escalate`, `result.stop_reason == "terminal_tool"`; (c) an `EvalRunner` over an `EvalCase`
   with `expected_tools=["escalate"]` reports one pass on the escalating model.
3. Run the tests; all must pass.
