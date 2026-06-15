# tulip examples — the cybersecurity agent SDK

Every runnable file in this directory is a security workflow. The
snippets below are the smallest possible shapes; the numbered
`notebook_*.py` files build them out into SOC triage, IOC enrichment,
phishing analysis, vulnerability research, and incident response.

## Quick Start

```python
from tulip.agent import Agent
from tulip.models import get_model

model = get_model("openai:gpt-4o")  # or "anthropic:claude-sonnet-4-6"

agent = Agent(
    model=model,
    system_prompt="You are a SOC triage analyst. Cite the evidence behind every verdict.",
)

# Synchronous
result = agent.run_sync("Triage: outbound beaconing from 192.0.2.14 to a domain registered yesterday.")
print(result.message)  # a one-paragraph verdict with the evidence that backs it
```

A model is just a string: the prefix before the colon (`openai:`,
`anthropic:`) selects the provider; the rest is the model id.
See [`docs/concepts/models.md`](../docs/concepts/models.md) for the full
provider story.

## With Tools

```python
from tulip.tools import tool

@tool
def domain_reputation(domain: str) -> str:
    """Return registrar age, category, and reputation for a domain."""
    return f"{domain}: registered 2 days ago, category 'newly observed', reputation 'suspicious'"

@tool
def ioc_lookup(indicator: str) -> str:
    """Look up an IP / domain / hash against threat intelligence."""
    return f"{indicator}: 3 vendor detections, last seen in a phishing campaign"

agent = Agent(
    model=model,
    tools=[domain_reputation, ioc_lookup],
    system_prompt="You are a SOC triage analyst. Use the tools, then cite what you found.",
)

result = agent.run_sync("Users got mail linking to login.phish.example.net — phishing or legit?")
```

For tools where a duplicate call would hurt — isolating a host, paging an
on-call, filing a ticket — declare `@tool(idempotent=True)`: the loop
keys every invocation on `(name, args)` and refuses to fire the same one
twice, even across retries.

## Streaming

```python
import asyncio

async def main():
    async for event in agent.run("Triage alert A-101: impossible-travel login from 198.51.100.7."):
        if event.event_type == "think":
            print(event.reasoning)
        elif event.event_type == "tool_complete":
            print(f"Tool {event.tool_name}: {event.result}")

asyncio.run(main())
```

## Multi-Agent (Swarm)

```python
from tulip.multiagent import create_swarm, create_swarm_agent

analyst = create_swarm_agent(
    name="Analyst",
    capabilities=["enrich", "correlate"],
    system_prompt="You enrich indicators and correlate them across alerts.",
)

reporter = create_swarm_agent(
    name="Reporter",
    capabilities=["write", "summarize"],
    system_prompt="You write clear, evidence-backed incident summaries.",
)

swarm = create_swarm(agents=[analyst, reporter], model=model)
result = await swarm.execute("Investigate the impossible-travel alert and write the incident summary.")
print(result.summary)
```

## With Hooks

```python
from tulip.hooks import LoggingHook, GuardrailsHook

agent = Agent(
    model=model,
    hooks=[LoggingHook(), GuardrailsHook()],  # audit trail + prompt-injection / secret-leak guardrails
)
```
