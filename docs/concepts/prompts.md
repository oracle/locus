# Prompts

The model in an agent
sees three sources of prompt content, in this order, every iteration:

1. The **system prompt** — `Agent(system_prompt=...)`. Stable across
   the whole run; describes the agent's role, constraints, and tools.
2. The **conversation history** — accumulated `state.messages`,
   including the user's prompt, every model response, and every tool
   result. Grows as the loop iterates.
3. **Reflexion / Grounding output** — when those reasoning add-ons
   are enabled, their judgments are appended to the message stream
   and the next Think sees them.

You don't usually configure 2 and 3 directly. You configure 1.

## A first system prompt

{{ locus_model_tabs }}

```python
agent = Agent(
    model="{{ locus_model }}",
    tools=[lookup_order, issue_refund],
    system_prompt=(
        "You are a support agent. "
        "Look up the order before acting on it. "
        "Confirm the order id with the customer before calling issue_refund."
    ),
)
```

System prompts live next to the agent definition. They're short on
purpose: every token counts toward your context window, and long
prompts are usually a sign that *more constraints* belong in
[playbooks](playbooks.md) (declarative step plans) or
[tools](tools.md) (typed signatures the model has to obey).

## What goes in the system prompt

- **Role.** *"You are X."* One sentence.
- **Goal.** *"Your job is to Y."* One sentence.
- **Constraints.** *"Never Z."* / *"Always W before V."* — short
  bullets.
- **Tone, when it matters.** Customer-facing agents → say so.

What does **not** belong in the system prompt:

- Tool documentation. The `@tool` decorator already exposes a typed
  contract to the model; duplicating it in prose doesn't help.
- Long examples. Fold them into a `Skill`'s `instructions=` (loaded
  only when the skill activates) or into a `Playbook`.
- Fields that change per request. Pass those as the user's `prompt`
  argument to `agent.run()`.

## System prompt vs. user prompt

```python
agent.run_sync(
    "Customer 4821 reports a duplicate charge on ord-4821, placed 2026-05-04.",
    thread_id="th-4821",
)
```

Everything in the `agent.run_sync(...)` argument is the **user
prompt** — request-specific data the agent should act on. The system
prompt sets identity once; the user prompt drives this particular
turn.

If you find yourself wanting to "reset" the agent's role mid-thread,
that's a sign you want a different agent — not a different prompt.
Use [Handoff](multi-agent/handoff.md).

## Prompt templates

For agents whose system prompts vary per tenant or per environment,
build the prompt string before constructing the agent. Plain Python
f-strings are usually enough; for richer templating use Jinja:

```python
from jinja2 import Template

template = Template("""
You are the support agent for {{ tenant.name }}.
Your refund authority covers {{ tenant.refund_cap }} and below.
Always escalate to a human before refunding above it.
""")

agent = Agent(
    model=...,
    system_prompt=template.render(tenant=tenant_record).strip(),
)
```

For prompts that need a model — say, summarising long conversation
histories on demand — see [Conversation Management](conversation-management.md).

## Prompt caching

Long, stable system prompts cost real money on every iteration. The
`AnthropicModel` can mark the stable prefix for caching so the provider
charges fewer tokens on cache hits — pass `prompt_cache=True`:

```python
from locus.models import AnthropicModel

agent = Agent(
    model=AnthropicModel(model="claude-sonnet-5-5", prompt_cache=True),
    system_prompt=very_long_prompt,
)
```

When `prompt_cache=True` and the provider returns cache token counts,
they surface on `result.metrics` (`cache_creation_input_tokens` /
`cache_read_input_tokens`). See [Models](models.md) for the per-provider
matrix.

## When the model misbehaves

If the agent picks the wrong tool, the system prompt is rarely the
fix — start with the **tool docstring**. Tools are how the model
discovers what's available; their docstrings are part of the
contract the model sees.

If the agent loops on a wrong premise, the fix is
[Reflexion](reasoning.md), not a more elaborate system prompt.

If the agent does the right thing 80% of the time and goes off-script
20%, the fix is a [Playbook](playbooks.md) — a declarative,
enforceable step plan — not a longer system prompt.

## See also

- [Tools](tools.md) — typed contracts the model honours.
- [Skills](skills.md) — filesystem-first capability disclosure.
- [Playbooks](playbooks.md) — declarative step plans the agent must
  follow.
- [Reasoning](reasoning.md) — Reflexion / Grounding / Causal.
