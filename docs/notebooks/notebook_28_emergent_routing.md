# Emergent routing

This notebook builds two incident routers over one tool registry and one
protocol registry: the default rule-based router that Notebook 29
introduces, and an emergent router whose compiler carries an
`LLMProtocolPicker`. The same five on-call requests go through both, and
the table marks the rows where the two modes chose a different protocol.

The default router is deterministic: the model fills a `GoalFrame`, then
`_rank_key` picks a protocol by tuple comparison. The result is
reproducible, auditable and rule-based, which is what an on-call lead
expects from an incident router.

The emergent router is the opt-in second mode. When more than one protocol passes the filter — for example, a `COMPARE` goal at LOW risk, which both `debate` and `specialist_fanout` handle — the `LLMProtocolPicker` asks the model to make the last-mile choice and returns a one-sentence rationale with its pick.

On a `Router.dispatch()` the compiler records that rationale on the `router.protocol.selected` event. Moving *only* the disambiguation step to the model keeps the routing audit trail intact, which matters when a routed protocol fans work out to agents that act on production systems.

What this notebook covers:

- Filtering, policy gating, capability binding and builder dispatch stay
  rule-based. Only the disambiguation step moves to the model.
- The compiler skips the picker when only one candidate survives the
  filter: no extra model call, no extra token spend.
- If the picker raises or returns an unknown protocol id, the compiler
  falls back to `_rank_key`; when it has a run id, as it does on every
  `Router.dispatch()`, it also emits `router.protocol.picker_fallback`, so
  the degradation is observable.
- The same five on-call requests run through both routers side by side.
  Rows marked `≠` are where the two modes disagreed.

## Prerequisites

- [Notebook 07 (structured output)](notebook_07_structured_output.md).
- [Notebook 29 (cognitive router, the default rule-based path)](notebook_29_cognitive_router.md).

## Key lines

Both routers come from one tool registry, one `CapabilityIndex` and one
`ProtocolRegistry`; the only difference is the `LLMProtocolPicker` handed
to the emergent compiler. Each router's extractor comes from
`_build_extractor`, an `Agent` with `output_schema=GoalFrame`, the
structured-output step that needs a live model:

{{ locus_source examples/notebook_28_emergent_routing.py::build_routers }}

The picker plugs into `CognitiveCompiler` through `protocol_picker`; the
protocol filter, the `PolicyGate` and builder dispatch are unchanged, and
the default router leaves it `None`:

{{ locus_source examples/notebook_28_emergent_routing.py::_build_compiler }}

Routing is the subject, so each request goes through `Router.compile()`,
which chooses and builds the topology without running it, and the table
reads the chosen `protocol_id` off the compiled runnable. Both routers are
given the same extracted frame, so a row marked `≠` comes from the picker
alone. `compile()` passes no run id to the compiler, so this path emits no
router events: the picker's rationale appears on `router.protocol.selected`
(see the [event catalogue](../concepts/sse-events.md#router-goal-routing))
only when the request goes through `dispatch()` or
`router.compiler.compile(frame, run_id=...)`:

{{ locus_source examples/notebook_28_emergent_routing.py::_selected }}

## Run it

```bash
python examples/notebook_28_emergent_routing.py
```

The router's first step fills a structured `GoalFrame`, so the notebook
needs a live model that can produce JSON. Model choice comes from
`examples/config.py`: OCI Generative AI when `~/.oci/config` has a
usable profile; otherwise set `LOCUS_MODEL_PROVIDER` to `openai` or
`anthropic` with `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the
environment, or to any other registry provider together with
`LOCUS_MODEL_ID` and its own credentials. There is no offline run: under
the mock model, `check_structured_output_capable()` prints a notice with
example provider settings and exits cleanly (status 0) before the
routers are built.

To force a provider and model:

```bash
LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_28_emergent_routing.py
```

## Source

??? note "Full source — `examples/notebook_28_emergent_routing.py`"

    ````python
    --8<-- "examples/notebook_28_emergent_routing.py"
    ````
