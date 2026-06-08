# Prompts

The model in an Tulip agent
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

```python
agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search_flights, book_flight],
    system_prompt=(
        "You are a travel concierge. "
        "Search before booking. "
        "Confirm the flight number with the user before calling book_flight."
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
- Long examples. Use `examples=` on a `Skill` or fold into a
  `Playbook`.
- Fields that change per request. Pass those as the user's `prompt`
  argument to `agent.run()`.

## System prompt vs. user prompt

```python
agent.run_sync(
    "Book a flight from JFK to NRT on 2026-05-04 for customer C-42.",
    thread_id="th-c42",
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
You are the procurement officer for {{ tenant.name }}.
Your spending limit is {{ tenant.limit_usd }} USD per quarter.
Always run compliance review before approving over $50,000.
""")

agent = Agent(
    model=...,
    system_prompt=template.render(tenant=tenant_record).strip(),
)
```

For prompts that need a model — say, summarising long conversation
histories on demand — see [Conversation Management](conversation-management.md).

## Prompt caching

Long, stable system prompts cost real money on every iteration.
Anthropic and OpenAI both support prompt caching: tag the part of the
prompt that doesn't change per turn and the provider charges fewer
tokens on cache hits.

```python
from tulip.models import OpenAIModel

agent = Agent(
    model=OpenAIModel("gpt-4o", cache_system_prompt=True),
    system_prompt=very_long_prompt,
)
```

OpenAI's V1 transport inherits prompt caching from the underlying
provider models when supported. See
[Models](models.md) for the per-provider matrix.

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
