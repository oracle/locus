# OCI v1 chat

`OCIChatCompletionsModel` is the default Oracle Cloud Infrastructure
(OCI) Generative AI client class. It targets the OpenAI-compatible
endpoint at `/openai/v1/chat/completions` and is the right choice for
almost every agent you build on OCI: one class covers the OpenAI, Meta,
Mistral, xAI and Google Gemini families, it needs no Project OCID, it
streams token by token, and structured output is native.

Cohere models (Command R and Command A) are the exception. OCI does not
expose them on this endpoint, so the `oci:` prefix routes every
`cohere.` id to `OCIModel`, the OCI SDK transport covered in notebook
01; the last part of this notebook shows what happens when a Cohere id
is sent through this class anyway.

What this notebook covers:

- Basic completion (`meta.llama-3.3-70b-instruct`)
- Streaming responses (`ModelChunkEvent`)
- Tool calling end-to-end
- Structured output: Pydantic schema → typed `result.parsed`
- Swapping model families without changing the model class

## Prerequisites

```bash
export OCI_PROFILE=<your-profile>
export OCI_REGION=us-chicago-1
export OCI_COMPARTMENT=ocid1.compartment.oc1..…
```

This notebook constructs `OCIChatCompletionsModel` directly rather than
through the `oci:` prefix, and the drawing shows where the class sits
among the OCI transports: the prefix sends every id that is not a
`cohere.` id or a Dedicated AI Cluster endpoint to this class, while
`OCIResponsesModel`, the subject of notebook 34, is the one transport
the prefix never selects.

{{ locus_diagram oci-transports }}

## Key lines

Every part builds its model through one helper; the class takes the
model id, the `~/.oci/config` profile, the inference region and the
compartment, and nothing else changes between families:

{{ locus_source examples/notebook_33_oci_openai_chat.py::_make_model }}

Structured output is native on this transport: `OCIChatCompletionsModel`
inherits `supports_structured_output` from `OpenAIModel`, so for an agent
without tools the runtime attaches the class passed as `output_schema=`
as a JSON-schema `response_format`, on top of the schema instruction it
appends to the system prompt, and the reply is parsed into
`result.parsed` as a typed `Weather` instance:

{{ locus_source examples/notebook_33_oci_openai_chat.py::part4_structured_output }}

The loop keeps the class and changes one string per model id: the first
three ids belong to families this endpoint serves, while the fourth is a
Cohere model that OCI does not expose on `/openai/v1`, so that request
fails and the `except` branch reports it as unavailable instead of
stopping the run, the same branch that covers a model the tenancy does
not offer:

{{ locus_source examples/notebook_33_oci_openai_chat.py::part5_model_swap }}

## Run it

```bash
python examples/notebook_33_oci_openai_chat.py
```

The script calls OCI Generative AI directly and does not use the shared `examples/config.py` harness, so there is no mock fallback. It reads the three variables from Prerequisites:

- `OCI_PROFILE` names a profile in `~/.oci/config` with Generative AI access.
- `OCI_REGION` defaults to `us-chicago-1`.
- `OCI_COMPARTMENT` is the compartment sent with each request; when it is unset the class falls back to `OCI_COMPARTMENT_ID` and then to the profile's tenancy OCID.

Without `OCI_PROFILE` the script prints its banner and the Part 1 heading, writes a skip notice to stderr and exits with status 77, which `scripts/run_notebooks.py` reports as a skip rather than a failure.

The walkthrough calls `meta.llama-3.3-70b-instruct` and `openai.gpt-5`, and Part 5 adds an xAI and a Cohere model id.

## See also

- [Notebook 01 — the three OCI client classes side-by-side](notebook_01_oci_transports.md)
- [Notebook 34 — OCIResponsesModel and the Responses endpoint](notebook_34_oci_responses.md)
- [How-to: OCI models](../how-to/oci-models.md) — transport table and configuration recipes
- [OCI OpenAI-compatible endpoints — Oracle docs](https://docs.oracle.com/iaas/Content/generative-ai/openai-compatible-api.htm)

## Source

??? note "Full source — `examples/notebook_33_oci_openai_chat.py`"

    ````python
    --8<-- "examples/notebook_33_oci_openai_chat.py"
    ````
