Build a customer-support agent with Locus.

1. Write `support_agent.py` with two typed tools: `lookup_order(order_id: str)` returning a
   dict with `status` and `amount_usd` (make `ORD-7842` delivered, 42.50), and
   `issue_refund(order_id: str, amount_usd: float)` that records the refund in a module-level
   list and returns a dict. Gate the refund with `gate_tool` from `locus.control`: a
   `ControlPolicy` that decides by label (`require_verification_score=0.0`), holds refunds
   tagged `high_value` (amount over 100) for a human and denies refunds tagged `prohibited`
   (amount of 0 or less); derive the `Action` from the real arguments; record every decision
   on a module-level `AuditTrail`. Expose `build_agent(model)` so tests can pass a model double.
2. Write `test_support_agent.py`: offline pytest tests with `ScriptedModel` and
   `AgentTestClient` from `locus.testing` that cover the allow, hold and deny outcomes as
   recorded on the trail, assert the refund body ran only on allow, and assert
   `trail.verify()` is `True`.
3. Run the tests; all must pass.
