# Notebook 34: emergent incident routing — the model picks the specialist protocol

The default router (Notebook 58) is deterministic: the LLM fills a
``GoalFrame``, then ``_rank_key`` picks a protocol via tuple
comparison. Reproducible, auditable, rule-based — exactly what an
on-call lead wants from an incident router.

This notebook covers the opt-in second mode. When multiple protocols
pass the filter — say a latency regression that could go to the
service-mesh specialist or the capacity-planning specialist — an
``LLMProtocolPicker`` asks the model to make the last-mile choice and
records its rationale on the ``router.protocol.selected`` event. Moving
*only* the disambiguation step to the model keeps the routing audit
trail intact, which matters when the router itself can fan work out to
agents that touch production infrastructure.

- Filtering, policy gating, capability binding, and builder dispatch
  stay rule-based. Only the disambiguation step moves to the model.
- The picker short-circuits when only one candidate survives the
  filter — no extra LLM call, no extra token spend.
- If the picker raises or returns an unknown protocol id, the compiler
  falls back to ``_rank_key`` and emits
  ``router.protocol.picker_fallback`` so the degradation is
  observable.
- The same five on-call requests run through both routers side by
  side. Rows marked ``≠`` are where the two modes disagreed.

Run it:
    .venv/bin/python examples/notebook_34_emergent_routing.py

It needs a live model: the router's first step fills a structured
``GoalFrame``, which the bundled mock model cannot produce, so under mock the
notebook explains that and exits. Set LOCUS_MODEL_PROVIDER=openai (or
anthropic, or any registry provider with LOCUS_MODEL_ID) and the matching
credentials.

Prerequisites:

- Notebook 35 (structured output).
- Notebook 58 (cognitive router — the default rule-based path).

## Source

````python
--8<-- "examples/notebook_34_emergent_routing.py"
````
