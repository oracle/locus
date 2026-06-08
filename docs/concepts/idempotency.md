# Idempotency

> The single most important word in production agents is **once**.

The model is *allowed* to retry. The side effect *isn't*. Tulip makes that distinction a
one-keyword decision on the tool, enforced inside the ReAct loop.
This is an SDK-specific primitive — none of LangChain / LangGraph
/ CrewAI / Strands ship it.

If you ever plan to run an agent that **books**, **charges**,
**emails**, **pages**, or **writes**, this is the most important
single page on the docs site.

## When to use `idempotent=True`

| Situation | `idempotent=True`? |
|---|---|
| Side-effecting tool with real-world cost (charge, email, page, book) | **yes — always** |
| Database write you can't trivially roll back | **yes** |
| External service that's already idempotent on its end | yes — the SDK dedupes the round-trip too |
| Read-only catalogue lookup | no — re-reads are cheap, leave it to the model |
| Tool that *intentionally* generates a new entity each call (e.g. `mint_uuid`) | no — that breaks the contract |

## How it works

Inside a single agent run, the SDK hashes the tool's
`(name, arguments)` tuple as the model emits each call. **The first
call with a given key hits the function body** and the result is
recorded. **Every subsequent call with the same key short-circuits
to the cached response** without invoking the body.

```python
from tulip.tools import tool
@tool(idempotent=True)
def transfer(from_acct: str, to_acct: str, amount: float) -> dict:
    """Transfer funds. Re-fires within a run return the cached receipt."""
    return ledger.transfer(from_acct, to_acct, amount)
```

The argument hash is the trust boundary:

- **Same call**: the model re-emits `transfer("A", "B", 100)` after
  seeing the receipt → cache hit, body skipped.
- **Different call**: the model emits `transfer("A", "B", 200)` →
  different key, body runs.

Caching is keyed on the **canonical JSON form** of the arguments, so
key order, default values, and whitespace don't matter.

## Why this matters

### Booking, billing, payments

The model that calls `book_flight` twice in one run is more common
than you think. Sometimes it sees an ambiguous tool result and tries
again "to be sure". Sometimes the network glitches and the model
believes the call failed. Without idempotency, you charge the
customer twice and they're on the phone with their bank.

```python
@tool(idempotent=True)
def book_flight(flight_id: str, customer_id: str) -> dict:
    return billing.charge_and_book(flight_id, customer_id)
```

The customer gets billed once. Always.

### Outbound side-effects

`email_cfo`, `page_oncall`, `submit_po`, `slack_alert` — anything
that touches a human or a downstream system. **One and done**.

### Database writes you can't roll back

Insert into a journal table, append to a Kafka topic, sign a JWT —
operations where retrying isn't free. Idempotent tools turn the
"exactly once" problem into a "not-our-problem-after-the-first-call"
guarantee.

### Replays after checkpoint resume

When a checkpointer resumes a stalled run, the model may decide to
re-issue tool calls it's already seen. Idempotent tools see the
cache pre-populated from the checkpoint and skip the side effect on
replay. (This requires `tool_executions` to be restored from the
checkpoint; the SDK's [native checkpointers](checkpointers.md) handle
it.)

## What it is *not*

| Concept | Idempotency is… | Idempotency is *not*… |
|---|---|---|
| Scope | within a single agent run | cross-run — restart and the cache is gone (use a [checkpointer](checkpointers.md)) |
| Failure | one fire per identical call | retry — if the body raises, the exception propagates as the cached "result" |
| Boundary | per-agent | network — two different agents both calling `transfer(a, b, 100)` each fire once |

If you need cross-run idempotency, configure a checkpointer + an
idempotent server-side endpoint. The combo gives you "the side
effect runs at most once across all replays of all agents".

## Practical recipe — vendor PO approval

A canonical multi-agent idempotency shape: an agent (or three of
them, debating) loops over a vendor decision, then writes once.

```python
@tool(idempotent=True)
def submit_po(vendor_id: str, line_items: list[dict]) -> dict:
    return procurement.submit(vendor_id, line_items)

@tool(idempotent=True)
def email_cfo(po_id: str, summary: str) -> str:
    return mail.send(to="cfo@org.com", subject=f"PO {po_id}", body=summary)
```

The agent can iterate ten times reasoning about whether to approve.
The PO ships once. The CFO email lands once. The model can fail
mid-run and a checkpointer-backed resume re-issues the same calls;
the side effects still fire exactly once.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Tool re-fires despite `idempotent=True` | Argument changed between calls. Check that the model isn't mutating ids / amounts between turns. |
| Idempotent cache survives across runs unexpectedly | It shouldn't — only the checkpointer persists state. If you're seeing this, you're loading state from a checkpoint and don't want to. |
| Body raised first time, cache returns the exception | This is by design — the failure is part of the "result" of the first call. The model sees the failure and can react. To re-attempt, the model must change an argument. |
| Read-only lookup tagged `idempotent=True` | Harmless but wasteful — the cache hit savings are negligible vs the read itself. Leave it off. |

## Source and notebook

- [`@tool` decorator with idempotency hook](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/tools/decorator.py)
- [`_find_matching_execution`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/loop/nodes.py#L114) — where the dedup actually happens, in the ReAct loop's Execute node.
- [`notebook_07_agent_with_tools.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_07_agent_with_tools.py) — walks through the `@tool` decorator end-to-end (idempotency covered in the agent-loop walkthrough).

## See also

- [Tools](tools.md) — the full `@tool` decorator surface.
- [Checkpointers](checkpointers.md) — durable runs where idempotency interacts with replay.
