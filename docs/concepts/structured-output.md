# Structured output

When you need a typed object rather than free text, set `output_schema=`
to a Pydantic model and the SDK parses the agent's final answer into a
typed instance.

{{ locus_model_tabs }}

```python
from pydantic import BaseModel, Field
from locus.agent import Agent
from locus.tools import tool

class TicketTriage(BaseModel):
    ticket_id: str = Field(description="The ticket id (e.g. t-8804)")
    confidence: float = Field(ge=0.0, le=1.0)
    severity: str
    needs_human: bool

class TriageList(BaseModel):
    tickets: list[TicketTriage]

@tool
def lookup_ticket(ticket_id: str) -> str:
    """Return one ticket's subject, age and customer notes."""
    return "results"

agent = Agent(
    model="{{ locus_model }}",
    tools=[lookup_ticket],
    output_schema=TriageList,
    system_prompt="Triage the top three tickets from this queue.",
)

result = agent.run_sync("Score the three open tickets for customer 4821.")

picks: TriageList = result.parsed   # type: ignore[assignment]
for t in picks.tickets:
    print(t.ticket_id, t.confidence, t.severity, t.needs_human)
```

`output_schema` must be a `pydantic.BaseModel` subclass — including
nested models, lists, optionals, discriminated unions, and any
`@field_validator` / `@model_validator` you attach. The schema always
rides the system prompt as a JSON instruction; on providers with native
structured output (OpenAI, and OpenAI-compatible transports such as
OCI's `/openai/v1`) a model call that offers no tools also carries it as
a strict `response_format`, and either way the final answer is
extracted and validated client-side. The drawing below follows the
schema from `output_schema=` to `result.parsed`, repair included.

{{ locus_diagram structured-output-flow }}

## What ends up on `AgentResult`

| Attribute | Type | Meaning |
|---|---|---|
| `result.parsed` | <code>BaseModel &#124; None</code> | The parsed instance. `None` if every retry failed. |
| `result.parse_error` | <code>str &#124; None</code> | Last Pydantic validation error, when `parsed is None`. |
| `result.message` | `str` | The canonical JSON dump of `parsed` (when set), otherwise the raw final assistant message. |

### Typed access

`result.parsed` is `BaseModel | None`. For typed access without casting,
call `result.parsed_as(YourSchema)` — runtime-checked and raises
`ValueError` (no parsed output) or `TypeError` (wrong concrete type):

```python
picks = result.parsed_as(TriageList)   # TriageList, narrowed by mypy
for t in picks.tickets:
    print(t.ticket_id)
```

## Repair on validation failure

If the model's first answer fails validation, the SDK re-prompts up to
`output_schema_retries` times (default 2) with the Pydantic
`ValidationError` details inlined so the model can fix the response. The
repair call offers no tools and carries the full response format,
`{"type": "json_schema", "json_schema": {"name": …, "schema": …,
"strict": True}}` (`strict` follows `output_schema_strict`), so a
provider that constrains decoding does so here even for a tool-using
agent; a model whose `complete()` does not accept the keyword is called
again without it. The repair exchange is metered — its token usage is
counted in the run's totals — but its messages are never written to the
returned conversation state.

```python
agent = Agent(
    model="{{ locus_example_model }}",
    output_schema=TriageList,
    output_schema_retries=3,        # default 2; set 0 to disable
    output_schema_strict=True,      # default; set False if your provider
                                    # rejects strict json_schema mode
)
```

When `output_schema_retries=0`, the first response is the final attempt.

## Provider compatibility

| Provider | Native mode | Mechanism | Prompted fallback |
|---|---|---|---|
| `oci:` | strict, on the `/openai/v1` transport | `response_format` (`json_schema`, strict) | yes |
| `openai:` | strict | `response_format` (`json_schema`, strict) | yes |
| `anthropic:` | repair call only (tool-use) | forced `respond_with_schema` tool | yes |

`oci:` ids run on the `/openai/v1` transport (`OCIChatCompletionsModel`),
which inherits OpenAI's native mode, so the model in the examples above
receives the strict `response_format`. The exceptions are `cohere.*` ids
and dedicated AI cluster endpoint OCIDs (`ocid1.generativeaiendpoint…`):
the SDK serves those through the OCI SDK transport (`OCIModel`), which
declares no native mode, so they get the prompted schema and the
client-side repair loop only. See
[OCI Generative AI models](../how-to/oci-models.md) and
[Dedicated AI Clusters](../how-to/oci-dac.md) for the transport each id
takes.

Anthropic models do not declare native mode, so the loop sends them the
prompted schema only; when the repair call carries `response_format`,
the SDK translates it into the idiomatic tool-use pattern: a single
`respond_with_schema` tool whose `input_schema` is your Pydantic schema,
with `tool_choice` pinned to it.
Anthropic's API guarantees the tool's arguments match the schema, and
the SDK surfaces those arguments as the message content for downstream
parsing — **the synthetic tool never reaches your agent's tool list**.

Strict mode adds two guarantees on supporting providers: (1) the model
**cannot** emit a JSON object that violates the schema, and (2) you do
not pay tokens for retries on simple shape violations. For non-strict
providers the prompted fallback validates client-side and replays.

## Streaming partial objects

For streaming UIs, you often want to render the model in flight as
fields populate — not wait for the full response. `StructuredStream`
wraps any agent event iterator and yields incrementally validated
Pydantic instances. Partials come from token chunks, so start the run
with `stream_tokens=True` — without it no `ModelChunkEvent` is emitted,
the loop body never runs, and nothing raises. Even with the flag, a
model that has no `stream()` method, or whose `server_stateful`
attribute is `True`, emits no chunks, so only `stream.final` can be
filled:

```python
from locus.streaming import StructuredStream

agent = Agent(
    model="{{ locus_example_model }}",
    output_schema=TriageList,
)

stream = StructuredStream(
    agent.run("Score the 3 open tickets.", stream_tokens=True),
    schema=TriageList,
)
async for partial in stream:
    ui.render(partial)               # may have 0, 1, 2, then 3 tickets
final: TriageList | None = stream.final
```

Each `ModelChunkEvent` is appended to a buffer; the SDK auto-closes any
unbalanced braces / brackets / strings, runs the result through
`schema.model_validate`, and yields the parsed instance if it succeeds.
By default identical consecutive partials are deduplicated; pass
`emit_unchanged=True` to surface every parseable chunk.

A partial is only yielded when **all required fields** are present —
optional fields may still be `None` or absent. When the run ends with a
`TerminateEvent`, `stream.final` is parsed from the run's final message
(or from the token buffer if that message is empty), even if no partial
ever validated. If that parse fails, it falls back to the last partial,
and it is `None` when neither validated. A run that pauses for input or
approval — an `InterruptEvent` from `ask_user`, or from a call held
under `on_refusal="interrupt"` — ends without a `TerminateEvent`, so
`stream.final` stays `None` even if partials arrived. A run that raises
passes the exception out of the `async for` loop, and `stream.final`
stays `None`.

## Composing with tools

`output_schema` only affects the **final answer**, not the iterations
that use tools. The agent can call any tool during the loop; once it
emits a non-tool response, the SDK parses that response into the schema:

```python
agent = Agent(
    model="{{ locus_example_model }}",
    tools=[lookup_ticket, lookup_order, lookup_customer],
    output_schema=TriageList,
    system_prompt=(
        "Investigate the tickets with the available tools, then return "
        "your scored triage as a JSON object."
    ),
)
```

## Source

`src/locus/core/structured.py` — parser, JSON extractor, response-format
builder, validation-error formatter.

`src/locus/agent/runtime_loop.py:_structure_output` — repair-on-failure loop.

## Notebooks

- [`notebook_07_structured_output.py`](https://github.com/oracle/locus/blob/main/examples/notebook_07_structured_output.py)
  covers both the standalone `parse_structured()` parser (useful for
  non-Agent flows) and the Agent `output_schema=` integration above.
- [`notebook_26_debate_with_judge.py`](https://github.com/oracle/locus/blob/main/examples/notebook_26_debate_with_judge.py)
  — typed `Verdict` as the workflow boundary artifact.
- [`notebook_77_incident_response.py`](https://github.com/oracle/locus/blob/main/examples/notebook_77_incident_response.py)
  — typed `Postmortem` as the terminal artifact of an incident graph.
- [`notebook_78_concession_approval.py`](https://github.com/oracle/locus/blob/main/examples/notebook_78_concession_approval.py)
  — typed `ConcessionDecision` from a risk-tiered approval flow.
- [`notebook_79_contract_review.py`](https://github.com/oracle/locus/blob/main/examples/notebook_79_contract_review.py)
  — typed `ContractDecision` from a parallel-review + negotiation loop.
