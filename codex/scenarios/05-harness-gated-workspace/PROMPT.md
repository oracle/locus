Build a gated coding-harness workspace with Locus.

1. Write `workspace_tools.py` exposing `build(files: dict[str, str]) -> tuple[Harness, AuditTrail]`:
   `build_harness(MemoryBackend(files), tools=["read", "write", "ls"], wrap=...)` where `wrap`
   applies `gate_tool` with `ControlPolicy(require_verification_score=0.0,
   require_human_for=frozenset({"workspace.write"}))` and the trail to every tool.
2. Write `test_workspace_tools.py`: `read` of an existing file returns its content; `ls`
   lists it; a `write` to a new path returns a JSON refusal containing `held_for_approval`
   and the file is not created; `harness.preview("write", {...})` returns a unified diff that
   contains the new line; the trail verifies and records the held write.
3. Run the tests; all must pass.
