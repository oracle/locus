# tulip examples — build agents with Tulip

Every runnable file in this directory is a self-contained agent you can run.
They span the domains where agents act — support, payments, infrastructure,
and data — and a **security** track (SOC triage, IOC — indicator of
compromise — enrichment, phishing analysis, vulnerability research, incident
response) sits alongside them as one fully worked domain. The snippets below
are the smallest possible shapes; the numbered `notebook_*.py` files build
them out.

## How to run

Every numbered notebook runs **offline by default** — with no API key set,
`config.py` hands the agents a bundled mock model, so you can run the whole
suite credential-free:

```bash
pip install "tulip-agents[openai,anthropic]"   # or: pip install -e . from the repo root
python examples/notebook_06_basic_agent.py     # runs on the mock model, no keys needed
```

To run against a real provider, set the provider and its key:

```bash
export TULIP_MODEL_PROVIDER=openai   ; export OPENAI_API_KEY=sk-...
# or
export TULIP_MODEL_PROVIDER=anthropic; export ANTHROPIC_API_KEY=sk-ant-...
export TULIP_MODEL_ID=gpt-4o         # optional — provider-specific model id
```

The `tulip.control` gate notebooks (83–87) are fully offline by design and
need no provider at all.

### Governing an agent you did not build on Tulip

`notebook_88_framework_interop.py` builds a real LangChain agent, runs it
through LangGraph's own ReAct loop, and watches a $4,000,000 refund execute —
then wraps that one tool and runs the identical agent again, where the money
does not move. It needs the per-framework bridges, which live outside this SDK
so that installing Tulip never pulls in a competitor's package:

```bash
pip install "tulip-frameworks[langchain,langgraph,crewai]"
python examples/notebook_88_framework_interop.py
```

Without them the file still runs, reports which bridge is missing, and exits 0.

## Quick Start

```python
from tulip.agent import Agent
from tulip.models import get_model

model = get_model("openai:gpt-4o")  # or "anthropic:claude-sonnet-4-6"

agent = Agent(
    model=model,
    system_prompt="You are a concise assistant. Cite the evidence behind every answer.",
)

# Synchronous
result = agent.run_sync("Summarize the trade-offs of moving checkout to a queue.")
print(result.text)  # a one-paragraph answer with the evidence that backs it
```

A model is just a string: the prefix before the colon (`openai:`,
`anthropic:`) selects the provider; the rest is the model id.
See [the models guide](https://tulipagents.ai/concepts/models/) for the full
provider story, including any OpenAI-compatible endpoint via `base_url`.

## With Tools

```python
from tulip.tools import tool


@tool
def lookup_order(order_id: str) -> str:
    """Return status, items, and payment state for an order."""
    return f"{order_id}: delivered 2026-07-12, 1 item ($49.00), payment settled"


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order — a consequential action worth gating in production."""
    return f"refund of ${amount:.2f} queued for {order_id}"


agent = Agent(
    model=model,
    tools=[lookup_order, issue_refund],
    system_prompt="You are a support agent. Look the order up first, then cite what you found.",
)

result = agent.run_sync("Customer says ord-4821 arrived broken — are they owed a refund?")
```

For tools where a duplicate call would hurt — issuing a refund, paging an
on-call, filing a ticket — declare `@tool(idempotent=True)`: the loop
keys every invocation on `(name, args)` and refuses to fire the same one
twice, even across retries.

## Streaming

```python
import asyncio


async def main():
    async for event in agent.run("Work out why order ord-4821 was charged twice."):
        if event.event_type == "think":
            print(event.reasoning)
        elif event.event_type == "tool_complete":
            print(f"Tool {event.tool_name}: {event.result}")


asyncio.run(main())
```

## Multi-Agent (Swarm)

```python
import asyncio

from tulip.multiagent import create_swarm, create_swarm_agent

analyst = create_swarm_agent(
    name="Analyst",
    capabilities=["lookup", "correlate"],
    system_prompt="You pull order, payment, and shipping records and correlate them.",
)

reporter = create_swarm_agent(
    name="Reporter",
    capabilities=["write", "summarize"],
    system_prompt="You write clear, evidence-backed case summaries.",
)

swarm = create_swarm(agents=[analyst, reporter], model=model)


async def main():
    result = await swarm.execute(
        "Work out why order ord-4821 was charged twice and write the case summary."
    )
    print(result.summary)


asyncio.run(main())
```

## With Hooks

```python
from tulip.hooks.builtin import LoggingHook, GuardrailsHook

agent = Agent(
    model=model,
    hooks=[
        LoggingHook(),
        GuardrailsHook(),
    ],  # audit trail + content guardrails (PII redaction, secret-egress blocks)
)
```
