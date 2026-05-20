# Locus Examples

## Quick Start

```python
from locus.agent import Agent
from locus.models import OCIChatCompletionsModel

model = OCIChatCompletionsModel(
    model="openai.gpt-5",
    profile="MY_PROFILE",  # any profile in ~/.oci/config
)

agent = Agent(
    model=model,
    system_prompt="You are a helpful assistant.",
)

# Synchronous
result = agent.run_sync("What is the capital of France?")
print(result.message)  # "Paris."
```

`OCIChatCompletionsModel` uses OCI GenAI's OpenAI-compatible `/openai/v1`
endpoint — real SSE streaming, day-0 model support, no GenAI Project
OCID required. For Cohere R-series models, use `OCIModel` instead — see
[`docs/how-to/oci-models.md`](../docs/how-to/oci-models.md) for the full
transport story and the production `auth_type=` modes
(`instance_principal` / `resource_principal`) for OCI VMs and Functions.

## With Tools

```python
from locus.tools import tool

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
from locus.multiagent import create_swarm, create_swarm_agent

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
from locus.hooks import LoggingHook, GuardrailsHook

agent = Agent(
    model=model,
    hooks=[LoggingHook(), GuardrailsHook()],
)
```
