# Advanced Hooks

The agent-hooks notebook covered hook basics. This one focuses on the safety
properties Locus enforces on the event objects hooks see, and on the
control lever a hook pulls mid-flight: `event.cancel` to block a
destructive tool call. A second lever, `event.retry`, is writable on the
after-events (`on_after_model_call`, `on_after_tool_call`) and is not
exercised here.

The scenario is RELEASE GUARD — the change-gating layer that sits between a
deploy-ops agent and the production cluster. A deploy agent that can delete a
namespace or destroy an environment on its own has an unbounded blast radius;
RELEASE GUARD is the bound on that agency.

What you'll learn:

- Most fields on hook event objects are read-only. Mutating
  `event.tool_name` raises `AttributeError` — that's the framework
  protecting the agent's invariants (and your change log).
- `event.arguments` and `event.cancel` *are* writable.
- Setting `event.cancel = "<reason>"` in `on_before_tool_call` skips the
  call and feeds the reason back as the tool's result — here, blocking a
  `delete_namespace` call during an active change freeze.
- Priority ordering is reversed on "after" callbacks so cleanup unwinds
  LIFO.

Run it:

```
.venv/bin/python examples/notebook_10_hooks_advanced.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

Prerequisite: the agent-hooks notebook.

## Source

````python
--8<-- "examples/notebook_10_hooks_advanced.py"
````
