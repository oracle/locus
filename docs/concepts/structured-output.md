# Structured output

Sometimes you want the model to *fill a shape*, not write prose. Set
`output_schema=` to a Pydantic model and Tulip parses the agent's final answer into a typed
instance for you.

```python
from pydantic import BaseModel, Field
from tulip.agent import Agent
class Vendor(BaseModel):
    name: str = Field(description="Legal name of the vendor")
    score: float = Field(ge=0.0, le=1.0)
    region: str

class VendorList(BaseModel):
    vendors: list[Vendor]

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search_vendors],
    output_schema=VendorList,
    system_prompt="Pick three vendors for our cloud-hosting RFP.",
)

result = agent.run_sync("Top three for $2M of cloud spend.")

picks: VendorList = result.parsed   # type: ignore[assignment]
for v in picks.vendors:
    print(v.name, v.score, v.region)
```

`output_schema` must be a `pydantic.BaseModel` subclass — including
nested models, lists, optionals, discriminated unions, and any
`@field_validator` / `@model_validator` you attach. The schema flows to
the provider as a strict `response_format` when supported (OpenAI,
OpenAI-compatible); otherwise the SDK falls back to prompted JSON +
extraction + validation.

## What ends up on `AgentResult`

| Attribute | Type | Meaning |
|---|---|---|
| `result.parsed` | `BaseModel \| None` | The parsed instance. `None` if every retry failed. |
| `result.parse_error` | `str \| None` | Last Pydantic validation error, when `parsed is None`. |
| `result.message` | `str` | The canonical JSON dump of `parsed` (when set), otherwise the raw final assistant message. |

### Typed access

`result.parsed` is `BaseModel | None`. For typed access without casting,
call `result.parsed_as(YourSchema)` — runtime-checked and raises
`ValueError` (no parsed output) or `TypeError` (wrong concrete type):

```python
picks = result.parsed_as(VendorList)   # VendorList, narrowed by mypy
for v in picks.vendors:
    print(v.name)
```

## Repair on validation failure

If the model's first answer fails validation, the SDK re-prompts up to
`output_schema_retries` times (default 2) with the Pydantic
`ValidationError` details inlined so the model can fix the response. On
supporting providers the repair call also ships
`response_format={"type": "json_schema", "strict": True}` for
constrained decoding.

```python
agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    output_schema=VendorList,
    output_schema_retries=3,        # default 2; set 0 to disable
    output_schema_strict=True,      # default; set False if your provider
                                    # rejects strict json_schema mode
)
```

When `output_schema_retries=0`, the first response is the final attempt.

## Provider compatibility

| Provider | Native mode | Mechanism | Prompted fallback | Tested |
|---|---|---|---|---|
| `openai:` (gpt-4o, gpt-4.1, gpt-5*, o-series) | ✓ strict | `response_format={"type":"json_schema","strict":true}` | ✓ | yes |
| `anthropic:claude-*` | ✓ tool-use | synthetic `respond_with_schema` tool + pinned `tool_choice` | ✓ | unit-mocked |

For Anthropic, the SDK translates `response_format` into the idiomatic
tool-use pattern: a single `respond_with_schema` tool whose
`input_schema` is your Pydantic schema, with `tool_choice` pinned to it.
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
Pydantic instances:

```python
from tulip.streaming import StructuredStream

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    output_schema=VendorList,
)

stream = StructuredStream(agent.run("Top 3 vendors."), schema=VendorList)
async for partial in stream:
    ui.render(partial)               # may have 0, 1, 2, then 3 vendors
final: VendorList | None = stream.final
```

Each `ModelChunkEvent` is appended to a buffer; the SDK auto-closes any
unbalanced braces / brackets / strings, runs the result through
`schema.model_validate`, and yields the parsed instance if it succeeds.
By default identical consecutive partials are deduplicated; pass
`emit_unchanged=True` to surface every parseable chunk.

A partial is only yielded when **all required fields** are present —
optional fields may still be `None` or absent. If the stream ends
without a single valid partial, `stream.final` is `None`.

## Composing with tools

`output_schema` only affects the **final answer**, not the iterations
that use tools. The agent can call any tool during the loop; once it
emits a non-tool response, the SDK parses that response into the schema:

```python
agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search_vendors, fetch_pricing, soc2_check],
    output_schema=VendorList,
    system_prompt=(
        "Research vendors with the available tools, then return your "
        "ranked picks as a JSON object."
    ),
)
```

## Source

`src/tulip/core/structured.py` — parser, JSON extractor, response-format
builder, validation-error formatter.

`src/tulip/agent/agent.py:_structure_output` — repair-on-failure loop.

## Notebooks

- [`notebook_35_structured_output.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_35_structured_output.py)
  covers both the standalone `parse_structured()` parser (useful for
  non-Agent flows) and the Agent `output_schema=` integration above.
- [`notebook_32_debate_with_judge.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_32_debate_with_judge.py)
  — typed `Verdict` as the workflow boundary artifact.
- [`notebook_63_incident_response.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_63_incident_response.py)
  — typed `Postmortem` as the terminal artifact of an incident graph.
- [`notebook_64_procurement_approval.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_64_procurement_approval.py)
  — typed `PurchaseOrder` from a tiered approval flow.
- [`notebook_65_contract_review.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_65_contract_review.py)
  — typed `ContractDecision` from a parallel-review + negotiation loop.
