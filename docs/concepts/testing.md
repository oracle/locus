# Testing agents

Testing an agent means controlling the one part you do not own: the model.
`locus.testing` ships two model doubles for that (plus `CompromisedModel`, for
a rogue-agent suite), so a test is deterministic, needs no API key, and never
touches the network — plus a client that runs the agent and hands back a trace
you can assert on.

```python
from locus.agent import Agent
from locus.testing import ScriptedModel, text, tool_call


async def test_refund_is_looked_up_first():
    model = ScriptedModel([
        tool_call("lookup_order", order_id="ord-4821"),
        text("That order is eligible for a refund."),
    ])
    agent = Agent(model=model, tools=[lookup_order, issue_refund])

    result = await agent.arun("Is ord-4821 refundable?")

    assert [t.tool_name for t in result.tool_executions] == ["lookup_order"]
    assert model.call_count == 2
```

## `ScriptedModel` — a fixed sequence

Each call returns the next turn. Build turns with `text()` for a plain answer
and `tool_call()` for a tool invocation; a bare string is shorthand for
`text()`.

```python
ScriptedModel([
    tool_call("issue_refund", order_id="ord-4821", amount=49.0),
    "Refunded.",
])
```

A tool whose argument is named like one of `tool_call()`'s own parameters
(`name`, `content`, `call_id`, `arguments`) takes them as a mapping; the
harness `write` tool is one:

```python
tool_call("write", arguments={"path": "app.py", "content": "print(1)\n"})
```

Arguments are keywords, so the call reads like the tool it names, and the
tool-call id is derived from the name rather than randomised — assertions
never depend on a value that changes per run.

!!! note "Running out of turns is an error"
    If the agent asks for more turns than you scripted, `ScriptedModel`
    raises rather than inventing a reply. An agent that keeps going has
    usually failed to terminate, and answering quietly would hide the bug the
    test exists to find.

    When the number of turns genuinely is not the point, pass
    `repeat_last=True` and the final turn is returned indefinitely.

## `FunctionModel` — decide per turn

A fixed script cannot express "call the tool, then answer using its result",
because the second turn depends on the first. `FunctionModel` takes a callable
that sees the conversation so far:

```python
from locus.testing import FunctionModel, tool_call


def handler(messages, tools):
    if any(m.role == "tool" for m in messages):
        return "The order was refunded."
    return tool_call("issue_refund", order_id="ord-4821")


agent = Agent(model=FunctionModel(handler), tools=[issue_refund])
```

The handler may return a `ModelResponse` or a plain string.

## Assert on what the agent sent

The interesting failures are usually in the *inputs* the agent produced, not
the final string — a tool that was never bound, a second turn that lost the
tool result. Both doubles record every call:

| Attribute | What it holds |
|---|---|
| `call_count` | How many times the agent called the model |
| `received_messages` | The `messages` list for each call, in order |
| `offered_tools` | Tool names bound on each call — `[]` when none |
| `last_prompt` | Content of the most recent user turn |

```python
async def test_the_agent_binds_only_the_read_tool():
    model = ScriptedModel([text("ok")])
    await Agent(model=model, tools=[lookup_order]).arun("status?")

    assert model.offered_tools[0] == ["lookup_order"]
    assert "issue_refund" not in model.offered_tools[0]
```

That style catches a class of bug an output assertion cannot: the agent
answering plausibly while never having been given the tool it claimed to use.

## `AgentTestClient` — assert on what the agent did

The doubles answer "what did the model see?". `AgentTestClient` answers the
other half — "what did the agent do?" — so a test stops rebuilding the same
comprehensions over `result.tool_executions`. It wraps the agent you built,
owns no configuration, and hands back an `AgentTrace`:

```python
from locus.agent import Agent
from locus.testing import AgentTestClient, ScriptedModel, text, tool_call


def test_refund_is_looked_up_first_via_client():
    model = ScriptedModel([
        tool_call("lookup_order", order_id="ord-4821"),
        text("That order is eligible for a refund."),
    ])
    client = AgentTestClient(Agent(model=model, tools=[lookup_order, issue_refund]))

    trace = client.run("Is ord-4821 refundable?")

    trace.assert_tools_called("lookup_order").assert_model_calls(2)
    trace.assert_tool_called("lookup_order", order_id="ord-4821")
    trace.assert_tool_not_called("issue_refund")
    trace.assert_succeeded()
    assert trace.message == "That order is eligible for a refund."
```

`client.run()` is blocking — it calls `Agent.run_sync()` — so the test above
is a plain `def`. In a pytest-asyncio suite, `await client.arun(prompt)`
returns the same `AgentTrace`.

| Assertion | Passes when |
|---|---|
| `assert_tool_called(name, **args)` | The agent called `name` — with these argument values, if given; only the arguments you name are compared |
| `assert_tool_not_called(name)` | The agent never called `name` |
| `assert_tools_called(*names)` | The agent called exactly these tools, in exactly this order |
| `assert_model_calls(count)` | The agent called the model exactly `count` times |
| `assert_tool_offered(name)` | `name` was offered to the model on the first turn |
| `assert_succeeded()` | The run reported no error and no tool raised |

A call counts once the agent sends it, whether or not the tool body runs. A
call that a hook cancelled, or that `gate_tool` refused with its default
`on_refusal="return"`, still appears in the trace, and `assert_succeeded()`
still passes. `assert_tool_not_called` fails for such a call just as it does
for one that ran, so it is the wrong check for a gate. To test a gate, assert
on the refusal the call returned (`trace.result.tool_executions`) or on the
audit trail. The exception is a call that pauses the run — `ask_user`, or a
hold under `gate_tool(..., on_refusal="interrupt")`: the run stops before that
call is recorded, so it is not in the trace and `assert_tool_not_called`
passes.

Each assertion returns the trace, so they chain, and a failure reports what
did happen — `expected tool 'issue_refund' to be called, but the agent called:
['lookup_order']` — rather than a bare `AssertionError`. When no helper fits,
the trace is plain data: `message` (the final answer; unlike `AgentResult`,
the trace has no `.text` alias), `tool_names`, `tool_calls` as
`(name, arguments)` pairs, `failed_tools`, `model_calls`, and `result` and
`model` for the underlying `AgentResult` and double.

`assert_tool_offered` is positive only — there is no "not offered"
counterpart — so the never-bound check from the previous section stays a raw
assertion. `trace.model` is the double the agent ran against, so it reads the
same `offered_tools` list:

```python
def test_client_offers_only_the_read_tool():
    model = ScriptedModel([text("ok")])
    trace = AgentTestClient(Agent(model=model, tools=[lookup_order])).run("status?")

    trace.assert_tool_offered("lookup_order")
    assert "issue_refund" not in trace.model.offered_tools[0]
```

## Streaming

`stream()` is implemented on both doubles, chunking whatever `complete()`
would have returned, so one double serves a streaming and a non-streaming
agent alike.

```python
chunks = [event async for event in model.stream(messages, None)]
assert chunks[-1].done is True
```

## What the doubles do not do

They return exactly what you scripted. Neither validates arguments against a
tool's schema, enforces token limits, or reproduces a provider's quirks — a
failing test should be telling you about your agent, not about a mock's
opinion of your JSON.

For the behaviours that *are* provider-specific — reasoning models returning
empty content at small token budgets, structured-output support varying by
model — test against a real endpoint. Any
[OpenAI-compatible provider](providers/openai-compatible.md) works, including
a local Ollama or vLLM server, so that does not require a vendor account
either.

→ [Agent](agent.md) · [Evaluation](evaluation.md) · [Hooks](hooks.md)
