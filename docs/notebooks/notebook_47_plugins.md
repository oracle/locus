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

The bundled mock model is the default; set `TULIP_MODEL_PROVIDER` for a live provider:

```bash
TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_47_plugins.py
```

Offline:

```bash
TULIP_MODEL_PROVIDER=mock python examples/notebook_47_plugins.py
```

## Prerequisites

- An OpenAI or Anthropic API key, or `TULIP_MODEL_PROVIDER` set to
  `openai` / `anthropic` / `mock`.

## Source

```python
--8<-- "examples/notebook_47_plugins.py"
```
