# OCIResponsesModel

`OCIResponsesModel` is the client for Oracle Cloud Infrastructure
(OCI) Generative AI's Responses endpoint
(`/openai/v1/responses`). Use it to reach Responses-only OCI models
(such as `openai.gpt-5.5-pro`) or to let OCI hold the conversation
thread so multi-turn payloads stay small. Two modes:

- **`store=False`** — Zero Data Retention (ZDR) safe. `store=false` goes
  out on every request, the agent sends the full history each turn and
  OCI keeps nothing. Works in every tenancy, including those with ZDR
  enforced.
- **`store=True`** — the class default. OCI holds the thread; the SDK
  passes `previous_response_id` between turns and only the latest turn
  goes on the wire. Requires a non-ZDR tenancy.

The model reports its mode as `server_stateful`, and the runtime loop
reads that flag. With `store=True` the loop sends only the messages
added since the last assistant turn and skips the steps that shape a
request from client-side history — `ConversationManager`, message-pair
validation and the observation pack — because the history they work on
is held by OCI; it also calls `complete()` rather than `stream()`, so an
agent reply arrives whole rather than token by token. With `store=False`
the loop takes the same client-side path as with
`OCIChatCompletionsModel`, streaming included.
Memory, Reflexion, [GSAR](../concepts/gsar.md) (typed grounding),
grounding, hooks, idempotency, checkpointing and termination conditions
do not depend on the transport. An `output_schema` travels in the
system prompt and the reply is parsed afterwards, because
`OCIResponsesModel` does not advertise native structured output. See
the [OCI Responses concept page](../concepts/oci-responses.md) for the
per-primitive table.

What this notebook covers:

- Part 1: stateless multi-turn (`store=False`)
- Part 2: server-stateful multi-turn with continuation id (`store=True`)
- Part 3: tool round-trip — model emits `function_call`, the SDK posts
  back `function_call_output`
- Part 4: streaming through `OCIResponsesModel.stream`

## Prerequisites

```bash
export OCI_PROFILE=<your-profile>
export OCI_REGION=us-chicago-1
export OCI_COMPARTMENT=ocid1.compartment.oc1..…
```

The `oci:` prefix picks a transport from the model id alone: Dedicated
AI Cluster endpoint OCIDs and `cohere.` ids go to `OCIModel` over the
OCI Python SDK, every other id goes to `OCIChatCompletionsModel`.
`OCIResponsesModel`, the transport this notebook uses, is the third one
and the prefix never selects it, which is why the example constructs
the class itself:

{{ locus_diagram oci-transports }}

## Key lines

`_make_model` constructs `OCIResponsesModel` from the profile, region and
compartment in Prerequisites; the `store` flag is the only difference
between the two modes:

{{ locus_source examples/notebook_34_oci_responses.py::_make_model }}

With `store=True`, OCI holds the thread: the second call sends only the
new user message and hands the continuation id back through
`provider_state`:

{{ locus_source examples/notebook_34_oci_responses.py::part2_server_state_multiturn }}

The tool round-trip runs through a plain `Agent`: the model emits a
`function_call` item, the SDK runs the tool client-side and posts back a
`function_call_output`, and the agent code is the same as with any other
model:

{{ locus_source examples/notebook_34_oci_responses.py::part3_tool_roundtrip }}

## Run it

```bash
python examples/notebook_34_oci_responses.py
```

The script calls OCI Generative AI directly and does not use the shared
`examples/config.py` harness, so there is no mock fallback. It reads the
variables from Prerequisites: `OCI_PROFILE` names a profile in
`~/.oci/config` with Generative AI access and is the only one the script
requires; `OCI_REGION` defaults to `us-chicago-1`; `OCI_COMPARTMENT` is a
compartment with Generative AI access and, when it is unset,
`OCIResponsesModel` falls back to `OCI_COMPARTMENT_ID` and then to the
profile's tenancy OCID. Without `OCI_PROFILE` the script prints its
banner and the Part 1 heading, writes a skip notice to stderr and exits
with status 77, which `scripts/run_notebooks.py` reports as a skip
rather than a failure; it exits the same way when the Responses endpoint
is not served in the chosen region and the first call answers 404. Part 2
runs only when `OCI_RESPONSES_STORE=1`; otherwise it prints a skip line
and the remaining parts continue.

To also exercise `store=True` in a tenancy without Zero Data Retention:

```bash
OCI_RESPONSES_STORE=1 python examples/notebook_34_oci_responses.py
```

## See also

- [Notebook 01 — the three OCI client classes side-by-side](notebook_01_oci_transports.md)
- [Notebook 33 — OCIChatCompletionsModel and the chat-completions endpoint](notebook_33_oci_openai_chat.md)
- [Concepts: OCI Responses](../concepts/oci-responses.md)
- [OCI Responses API — Oracle docs](https://docs.oracle.com/iaas/Content/generative-ai/responses-api.htm)
- [OCI OpenAI-compatible endpoints — Oracle docs](https://docs.oracle.com/iaas/Content/generative-ai/openai-compatible-api.htm)

## Source

??? note "Full source — `examples/notebook_34_oci_responses.py`"

    ````python
    --8<-- "examples/notebook_34_oci_responses.py"
    ````
