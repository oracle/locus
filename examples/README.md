# Multi-Agent Reasoning Orchestrator SDK Examples

## Quick Start

```python
from tulip.agent import Agent
from tulip.models import get_model

model = get_model("openai:gpt-4o")  # or "anthropic:claude-sonnet-4-6", "anthropic:claude-sonnet-4-6"

agent = Agent(
    model=model,
    system_prompt="You are a helpful assistant.",
)

# Synchronous
result = agent.run_sync("What is the capital of France?")
print(result.message)  # "Paris."
```

A model is just a string: the prefix before the colon (`openai:`,
`anthropic:`) selects the provider; the rest is the model id.
See [`docs/concepts/models.md`](../docs/concepts/models.md) for the full
provider story.

## With Tools

```python
from tulip.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny, 72°F in {city}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression.

    NOTE: Never pass untrusted/model-generated strings to eval(). Use a safe
    AST-based evaluator in real applications — see examples/notebook_04 for
    a concrete example.
    """
    import ast
    import operator as op

    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}
    def _eval(n):
        if isinstance(n, ast.Expression): return _eval(n.body)
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](_eval(n.left), _eval(n.right))
        raise ValueError("bad expr")
    return str(_eval(ast.parse(expression, mode="eval")))

agent = Agent(
    model=model,
    tools=[get_weather, calculate],
    system_prompt="Use tools when needed.",
)

result = agent.run_sync("What's the weather in Tokyo?")
```

## Streaming

```python
import asyncio

async def main():
    async for event in agent.run("Tell me about Python"):
        if event.event_type == "think":
            print(event.reasoning)
        elif event.event_type == "tool_complete":
            print(f"Tool {event.tool_name}: {event.result}")

asyncio.run(main())
```

## Multi-Agent (Swarm)

```python
from tulip.multiagent import create_swarm, create_swarm_agent

researcher = create_swarm_agent(
    name="Researcher",
    capabilities=["search", "analyze"],
    system_prompt="You research topics thoroughly.",
)

writer = create_swarm_agent(
    name="Writer",
    capabilities=["write", "summarize"],
    system_prompt="You write clear, concise content.",
)

swarm = create_swarm(agents=[researcher, writer], model=model)
result = await swarm.execute("Research and summarize AI trends")
print(result.summary)
```

## With Hooks

```python
from tulip.hooks import LoggingHook, GuardrailsHook

agent = Agent(
    model=model,
    hooks=[LoggingHook(), GuardrailsHook()],
)
```
