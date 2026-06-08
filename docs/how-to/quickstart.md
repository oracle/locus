# Quickstart

A working Tulip agent in five
minutes.

## 1. Install

```bash
pip install "tulip-agents[openai]"
```

This installs the SDK plus the OpenAI provider. For other
providers add the corresponding extra:

```bash
pip install "tulip-agents[openai]"        # OpenAI directly
pip install "tulip-agents[anthropic]"     # Anthropic directly
pip install "pip install "tulip-agents[sdk]"           # everything
```

## 2. Configure your provider

Set the API key for whichever provider you're using:

```bash
export OPENAI_API_KEY=sk-...          # OpenAI
export ANTHROPIC_API_KEY=sk-ant-...   # Anthropic
```

the model's `base_url` — no key needed. See
[Models](../concepts/models.md) for the per-provider matrix. With no key
set, Tulip falls back to a bundled mock model so the quickstart still
runs offline.

## 3. Your first agent

Save this as `hello_agent.py`:

```python
from tulip.agent import Agent
from tulip.tools.decorator import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b

@tool
def search_books(topic: str) -> list[str]:
    """Search the catalogue for books on a topic."""
    return [f"{topic} for Beginners", f"Advanced {topic}"]

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[add, search_books],
    system_prompt="You are a helpful assistant.",
)

result = agent.run_sync("What's 17 + 25, and recommend two books on Rust.")
print(result.message)
```

Run:

```bash
python hello_agent.py
```

You should see something like:

```text
17 + 25 is 42. Two books on Rust I'd recommend: "Rust for Beginners"
and "Advanced Rust".
```

## 4. Stream the events

For UIs and real-time logging, switch to async and consume the typed
event stream:

```python
import asyncio
from tulip.core.events import (
    ThinkEvent, ToolStartEvent, ToolCompleteEvent, TerminateEvent,
)

async def main():
    async for event in agent.run("What's 17 + 25?"):
        match event:
            case ThinkEvent(reasoning=r) if r:
                print(f"💭 {r}")
            case ToolStartEvent(tool_name=n, arguments=a):
                print(f"🔧 {n}({a})")
            case ToolCompleteEvent(result=r):
                print(f"   ↳ {r}")
            case TerminateEvent(final_message=m):
                print(f"\n✅ {m}")

asyncio.run(main())
```

See [Streaming](../concepts/streaming.md) for the full event taxonomy.

## 5. Persist conversations across restarts

For real applications you'll want state to survive a restart. Wire a
checkpointer and a `thread_id`:

```python
from tulip.memory.backends.file import FileCheckpointer

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[...],
    system_prompt="...",
    checkpointer=FileCheckpointer(directory="./threads"),
)

# Day 1
agent.run_sync("I'm planning a trip to Tokyo.", thread_id="user-c42")

# Day 2 — same thread_id, conversation continues
agent.run_sync("What were we talking about?", thread_id="user-c42")
```

For vendor-neutral durability, swap to `S3Backend(bucket=..., namespace=...)`.
See [Conversation Management](../concepts/conversation-management.md).

## 6. Make it production-grade

Add idempotency to side-effecting tools, Reflexion to catch wrong
premises, and termination algebra to stop when the work is done:

```python
from tulip.memory.backends import S3Backend
from tulip.core.termination import (
    MaxIterations, ToolCalled, ConfidenceMet,
)

@tool(idempotent=True)
def submit_order(item_id: str, qty: int) -> dict:
    return shop.submit(item_id, qty)

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search_catalog, submit_order],
    system_prompt="...",
    reflexion=True,
    checkpointer=S3Backend(bucket="tulip-threads", namespace="..."),
    termination=(
        ToolCalled("submit_order") & ConfidenceMet(0.9)
    ) | MaxIterations(8),
)
```

Each piece in detail:

- **`@tool(idempotent=True)`** → [Idempotency](../concepts/idempotency.md)
- **`reflexion=True`** → [Reasoning](../concepts/reasoning.md)
- **`checkpointer=...`** → [Checkpointers](../concepts/checkpointers.md)
- **`termination=...`** → [Termination](../concepts/termination.md)

## 7. Multi-agent

When one agent isn't enough — pick the coordination shape that fits
the problem:

| Shape | When |
|---|---|
| [Composition](../concepts/multi-agent/composition.md) | linear chain, fan-out + merge |
| [Orchestrator + Specialists](../concepts/multi-agent/orchestrator.md) | one router, parallel experts |
| [Swarm](../concepts/multi-agent/swarm.md) | open-ended research, peer-to-peer |
| [Handoff](../concepts/multi-agent/handoff.md) | escalation desks |
| [StateGraph](../concepts/multi-agent/graph.md) | review-loops, retry-until-confidence |
| [Functional API](../concepts/multi-agent/functional.md) | map/reduce over agents |
| [A2A](../concepts/multi-agent/a2a.md) | cross-process meshes |

## 8. Deploy

`AgentServer` is a drop-in FastAPI app:

```python
from tulip.server import AgentServer

server = AgentServer(agent=agent)
server.run(host="0.0.0.0", port=8080)
```

`POST /invoke`, `POST /stream`, `GET /threads/{id}`. Deploys
anywhere FastAPI runs — see [Deploy](deploy.md).

## Where to next

- **Read deeper.** [Agent Loop](../concepts/agent-loop.md) is the
  architectural reference for how all of this fits together.
- **Browse examples.** Forty progressive notebooks at
  [`examples/`](https://github.com/tuliplabs-ai/sdk-python/tree/main/examples).
  Each is a single runnable file that adds one idea on top of the
  previous.
- **Steer it.** [Hooks](../concepts/hooks.md) give you logging,
  telemetry, retry, guardrails, and steering as one-line additions.
