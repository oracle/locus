# OCI transports

The SDK ships three model classes for Oracle Cloud Infrastructure (OCI)
Generative AI. They are not interchangeable: the right one depends on
which model family you call and whether you want OCI to hold the
conversation history. This notebook builds an agent against each class,
runs the same prompt through all three and prints a decision table.

Read top to bottom; the first match wins.

| Want… | Use |
| --- | --- |
| A Cohere model (`cohere.command-r*`, `cohere.command-a*`) or a Dedicated AI Cluster endpoint | `OCIModel` |
| A Responses-only model (for example `openai.gpt-5.5-pro`), or OCI to hold the thread | `OCIResponsesModel` |
| Everything else (OpenAI, Meta, Mistral, xAI, Gemini) | `OCIChatCompletionsModel` (default) |

The runtime reads `model.server_stateful`. When it is `True`, the SDK
sends only the messages added since the last assistant turn and threads
OCI's continuation id through `AgentState.provider_state`. On that path
the runtime skips `ConversationManager` and the observation pack: both
reshape the history a request carries, and there is none to reshape
when the server holds the thread. It also calls `complete()` rather
than `stream()`, so a reply arrives whole rather than token by token.

Memory, reflexion, grounding, [GSAR](../concepts/gsar.md) (typed
grounding), hooks, idempotency, checkpointing and termination conditions
do not depend on the transport. Structured output does: the runtime
attaches a native `response_format` only when the agent has no tools and
the model advertises `supports_structured_output`, which
`OCIChatCompletionsModel` does. `OCIModel` reports `False` and
`OCIResponsesModel` does not define the property, so with those two the
schema instruction the runtime appends to the system prompt is the only
constraint; on every transport the reply is parsed afterwards, with a
re-prompt on a validation failure.

What this notebook covers:

- `OCIResponsesModel`, `OCIModel` and `OCIChatCompletionsModel` built
  side by side, each running the same prompt through a plain `Agent`
- `model.server_stateful`, the flag that tells the runtime to send only
  the latest turn and thread OCI's continuation id
- the transport the `oci:` prefix selects from a model id, and the one
  it never selects
- the decision table the script prints at the end

## Prerequisites

```bash
export OCI_PROFILE=<your-profile>
export OCI_REGION=us-chicago-1
export OCI_COMPARTMENT=ocid1.compartment.oc1..…   # a compartment with Generative AI access
```

This notebook constructs each class explicitly; when a model string
carries the `oci:` prefix, the registry picks the transport from the
model id: `cohere.` ids and Dedicated AI Cluster endpoints go to
`OCIModel` over the OCI Python SDK, every other id goes to
`OCIChatCompletionsModel`, and `OCIResponsesModel` is the one transport
the prefix never selects:

{{ locus_diagram oci-transports }}

## Key lines

`OCIResponsesModel` reaches a Responses-only model and lets OCI hold the
thread; `model.server_stateful` reports whether the runtime sends only
the latest turn:

{{ locus_source examples/notebook_01_oci_transports.py::example_responses_transport }}

`OCIModel` goes through the OCI Python SDK to the proprietary chat
endpoint, the path every Cohere chat model (Command R and Command A)
requires:

{{ locus_source examples/notebook_01_oci_transports.py::example_sdk_transport }}

`OCIChatCompletionsModel` is the default, one class for every other
family on the OpenAI-compatible endpoint, and the agent code around it
does not change:

{{ locus_source examples/notebook_01_oci_transports.py::example_v1_transport }}

## Run it

```bash
python examples/notebook_01_oci_transports.py
```

The script calls OCI Generative AI directly and does not use the shared
`examples/config.py` harness, so there is no mock fallback. It reads the
three variables from Prerequisites: `OCI_PROFILE` names a profile in
`~/.oci/config` with Generative AI access, `OCI_REGION` defaults to
`us-chicago-1`, and `OCI_COMPARTMENT` is a compartment with Generative AI
access. Without `OCI_PROFILE` the script prints its banner, writes a skip
notice to stderr and exits with status 77, which `scripts/run_notebooks.py`
reports as a skip rather than a failure.

The `--transport` flag runs a single transport instead of all three:

```bash
python examples/notebook_01_oci_transports.py --transport responses
python examples/notebook_01_oci_transports.py --transport sdk
python examples/notebook_01_oci_transports.py --transport v1
```

## See also

- [Notebook 33 — `OCIChatCompletionsModel` deep dive](notebook_33_oci_openai_chat.md)
- [Notebook 34 — `OCIResponsesModel` and the Responses endpoint](notebook_34_oci_responses.md)
- [Notebook 35 — OCI Dedicated AI Cluster](notebook_35_oci_dac.md)
- [Concepts: OCI Generative AI](../concepts/providers/oci.md) — the V1, SDK and Responses transports
- [Concepts: OCI Responses](../concepts/oci-responses.md) — when to opt in, what changes against `OCIChatCompletionsModel`, continuation state and checkpointing
- [How-to: OCI models](../how-to/oci-models.md) — transport table, configuration recipes and Zero Data Retention (ZDR) tenancies
- [Hooks](../concepts/hooks.md) — run on every transport

### Oracle reference docs

- [OCI Generative AI — Chat](https://docs.oracle.com/iaas/Content/generative-ai/use-playground-chat.htm) — SDK transport
- [OCI OpenAI-compatible endpoints](https://docs.oracle.com/iaas/Content/generative-ai/openai-compatible-api.htm) — V1 transport
- [OCI Responses API](https://docs.oracle.com/iaas/Content/generative-ai/responses-api.htm) — Responses transport

## Source

??? note "Full source — `examples/notebook_01_oci_transports.py`"

    ````python
    --8<-- "examples/notebook_01_oci_transports.py"
    ````
