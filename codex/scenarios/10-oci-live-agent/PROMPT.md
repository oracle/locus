Build and run a Locus agent on OCI Generative AI. `OCI_PROFILE` and `OCI_REGION` are already set
in the environment; never read or print any credential.

1. Write `run.py` exposing `build_agent(model) -> Agent` with one tool
   `lookup_order(order_id: str)` returning `{"order_id": ..., "status": "delivered",
   "amount_usd": 42.50}`, and a `main()` that builds the agent on `"oci:openai.gpt-5.5"`,
   asks `What happened to order ORD-7842?` once, and prints the answer.
2. Write `test_run.py` with an offline test of `build_agent` on a `ScriptedModel` that calls
   `lookup_order` then answers; assert the tool call with `AgentTestClient`.
3. Run the tests, then run `python run.py` once (one live model call) and include the answer in
   your report.
