# Emergent Routing

[Notebook 53](notebook_58_cognitive_router.md) covers the default
deterministic router: the LLM fills a typed `GoalFrame`, then
`_rank_key` picks a protocol via a four-element tuple comparison.

This notebook covers the opt-in second mode. When multiple protocols
pass the filter, an `LLMProtocolPicker` asks the model to make the
last-mile choice and records its rationale on the
`router.protocol.selected` event.

## When to reach for it

The rule-based ranker is the right default — reproducible, auditable,
free of model latency. Use the emergent picker when:

- Custom protocols are registered alongside the built-ins and the
  cost/complexity heuristic doesn't capture their actual fit.
- You want the model's **rationale** captured as part of the audit
  trail.
- The frame's `primary_goal` is one where multiple protocols qualify
  (e.g. `COMPARE` → both `specialist_fanout` and `debate`) and the
  pick depends on something the frame alone doesn't encode.

## What stays rule-based

The picker is strictly limited to disambiguation. The compiler still:

1. **Filters candidates** by `handles`, `risk_max`, and
   `requires_capabilities` before the picker sees anything.
2. **Short-circuits** when only one candidate survives — no LLM call,
   no extra tokens.
3. **Falls back** to `_rank_key` if the picker raises or returns an
   unknown id; emits `router.protocol.picker_fallback` so the
   degradation is observable.
4. **Runs PolicyGate** after the pick — same risk/approval gating
   regardless of which mode chose the protocol.

## Run

```bash
python examples/notebook_34_emergent_routing.py
```

The default provider is the bundled mock model. Set
`TULIP_MODEL_PROVIDER` (openai / anthropic) and credentials to use a
live model. Set
`TULIP_MODEL_PROVIDER=mock` for offline runs.

You'll see five prompts dispatched through both routers side by side.
Rows marked `≠` are where the two modes disagreed — the picker's
rationale (on the SSE event) explains why.

## See also

- [Notebook 53 — cognitive router (default rule-based path)](notebook_58_cognitive_router.md)
- [Concepts: cognitive router](../concepts/router.md) — the
  filter-then-pick invariant plus observability schema.
- [Notebook 27 — orchestrator pattern](notebook_26_orchestrator_pattern.md)
  — for emergent coordination *inside* a protocol.

## Source

```python
--8<-- "examples/notebook_34_emergent_routing.py"
```
