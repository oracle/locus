Build an incident commander with Locus's orchestrator shape, offline.

1. Write `incident_commander.py` exposing `build(coordinator_model) -> Orchestrator` with two
   `Specialist`s, `logs` (type `log_analyst`) and `metrics` (type `metrics_analyst`), each on a
   `ScriptedModel` with one finding, and `run(task: str)`, which builds the orchestrator with a
   `FunctionModel` coordinator that returns the routing JSON block (both specialists) when the
   prompt lists the available specialists and a one-line summary otherwise, executes the task
   and returns the `OrchestratorResult`.
2. Write `test_incident_commander.py`: the result is a success, both specialist types appear in
   `specialist_results`, and the summary is non-empty.
3. Run the tests; all must pass.
