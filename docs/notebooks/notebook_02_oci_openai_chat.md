# OCIChatCompletionsModel

Deep dive on the default Oracle Cloud Infrastructure (OCI) Generative
AI client class. `OCIChatCompletionsModel` targets the OpenAI-compatible
endpoint at `/openai/v1/chat/completions` and is the right choice for
almost every agent you'll build on OCI: one class covers OpenAI, Meta,
Mistral, xAI, Gemini, and non-R-series Cohere; no Project OCID; native
streaming; native structured output.

## What this covers

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

## Run

```bash
python examples/notebook_02_oci_openai_chat.py
```

## See also

- [Notebook 01 — the three OCI client classes side-by-side](notebook_01_oci_transports.md)
- [Notebook 03 — OCIResponsesModel and the Responses endpoint](notebook_03_oci_responses.md)
- [OCI OpenAI-compatible endpoints — Oracle docs](https://docs.oracle.com/iaas/Content/generative-ai/openai-compatible-api.htm)

## Source

```python
--8<-- "examples/notebook_02_oci_openai_chat.py"
```
