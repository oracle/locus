# Plugins

Plugins bundle hooks (and optionally tools) into one reusable object.
Drop a plugin onto an agent and every relevant hook method runs
automatically.

- `Plugin` base class — subclass it, give it a `name`, decorate any
  method with `@hook` and the agent picks it up.
- `@hook` decorator — marks methods like `on_before_model_call` and
  `on_before_tool_call` for auto-discovery.
- `callback_handler` — a plain function that receives every event;
  the lighter-weight alternative when you don't need a class.
- `Agent.cancel()` — stop a running agent from another thread; the
  next step returns `stop_reason="cancelled"`.

## Run it

OCI GenAI is the default — the notebook harness picks it when `~/.oci/config` exists (set `OCI_PROFILE` to choose a non-default profile):

```bash
LOCUS_MODEL_ID=openai.gpt-4.1 python examples/notebook_47_plugins.py
```

Offline:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_47_plugins.py
```

## Prerequisites

- An OCI profile with GenAI access, or `LOCUS_MODEL_PROVIDER` set to
  `openai` / `anthropic` / `mock`.

## Source

```python
--8<-- "examples/notebook_47_plugins.py"
```
