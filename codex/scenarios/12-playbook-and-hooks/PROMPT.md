Enforce a plan and observe tool calls with Locus, offline.

1. Write `refund_playbook.py`: tools `lookup_order(order_id: str)` and
   `issue_refund(order_id: str, amount_usd: float)`; a `Playbook` with two steps, `verify`
   expecting `lookup_order` and `pay` expecting `issue_refund`; `run(model) -> tuple[AgentResult,
   PlaybookEnforcerHook]` that builds the agent with `hooks=[PlaybookEnforcerHook(playbook=...),
   AuditLog()]` where `AuditLog` is your own `HookProvider` recording every tool name it sees in
   `on_before_tool_call` on a list attribute `seen`.
2. Write `test_refund_playbook.py`: on a `ScriptedModel` that calls `issue_refund` first, then
   `lookup_order`, then `issue_refund`, then answers, the enforcer records one `unexpected_tool`
   violation on step `verify`, both steps complete, and `seen` lists the three tool calls in order.
3. Run the tests; all must pass.
