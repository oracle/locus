# OCI Dedicated AI Cluster

A Dedicated AI Cluster (DAC) is Oracle Cloud Infrastructure (OCI)
Generative AI's provisioned-capacity serving mode: instead of
pay-per-token inference against a shared model id, you address a
dedicated endpoint by its OCID
(`ocid1.generativeaiendpoint.oc1.<region>....`) and OCI routes
inference to your cluster. This notebook wires the SDK to one.

The shape is the same as any other OCI model — only the model id
changes. Pass the endpoint OCID and the SDK routes it through `OCIModel`,
the transport built on the OCI Generative AI Python SDK, because the
OpenAI-compatible transport does not support `DedicatedServingMode`,
which is what a DAC requires. Both `complete()` and `stream()` work
end-to-end: `stream()` reads Server-Sent Events (SSE) from the cluster
and, if the endpoint rejects streaming, falls back to a single chunk
carrying the whole reply.

What this notebook covers:

- Part 1: how the `oci:` prefix picks the SDK transport (`OCIModel`) for
  a DAC OCID.
- Part 2: build an `Agent` against a DAC endpoint.
- Part 3: a single `complete()` round-trip.
- Part 4: SSE `stream()` with deltas printed inline.
- Part 5: a tool-using `Agent` on top of the DAC, including the
  Qwen-style `<tool_call>` text-block caveat.

## Prerequisites

- `oci` SDK installed (`pip install -e ".[oci]"`).
- An OCI profile with permission to invoke the DAC endpoint.
- The DAC endpoint OCID, compartment OCID, and region.

```bash
export OCI_DAC_ENDPOINT_OCID=ocid1.generativeaiendpoint.oc1.uk-london-1....
export OCI_DAC_COMPARTMENT_ID=ocid1.compartment.oc1....
export OCI_DAC_REGION=uk-london-1
export OCI_PROFILE=MY_DAC_PROFILE
export OCI_AUTH_TYPE=api_key   # or security_token
```

The transport map below is the routing that Part 1 summarises (Part 1
names the Cohere rule by its `command-r` family; the registry matches
every `cohere.` id): the `oci:` prefix reads the model id and sends a
Dedicated AI Cluster endpoint OCID (`ocid1.generativeaiendpoint…`) to
`OCIModel` over the OCI Generative AI Python SDK with
`DedicatedServingMode`, sends `cohere.` ids down the same SDK path with
`OnDemandServingMode`, and sends every other id to
`OCIChatCompletionsModel` on the OpenAI-compatible endpoint, while
`OCIResponsesModel` is opt-in and the prefix never selects it.

{{ locus_diagram oci-transports }}

## Key lines

`_dac_model` builds the `OCIModel` instance that Parts 3 and 4 call
directly: the endpoint OCID goes in as `model_id`, with the compartment,
the `~/.oci/config` profile, the auth type and the regional inference
endpoint alongside it. `OCIModel` turns an OCID-shaped `model_id` into
`DedicatedServingMode(endpoint_id=...)` and sends it in the request
together with the compartment. The request shape is the generic one:
`OCIModel` selects the Cohere request only for `cohere.command` ids,
never for an endpoint OCID:

{{ locus_source examples/notebook_35_oci_dac.py::_dac_model }}

One `complete()` round-trip against the cluster, with the reply, token
usage and stop reason read off the response exactly as for a public
model:

{{ locus_source examples/notebook_35_oci_dac.py::example_complete }}

The tool-using `Agent` on top of the DAC: `get_model` with the `oci:`
prefix resolves the endpoint OCID to `OCIModel`, the `@tool` definition
is converted to the OCI SDK's `FunctionDefinition` by the generic
request builder `OCIModel` uses for every id other than
`cohere.command*`, and the docstring records what to do when the
deployed model answers with a `<tool_call>` text block instead of a
structured tool call — the `Agent` parses such a block only when it is
built with `text_tool_calls="on"`:

{{ locus_source examples/notebook_35_oci_dac.py::example_agent_with_tool }}

## Run it

```bash
python examples/notebook_35_oci_dac.py
```

The script calls a Dedicated AI Cluster directly and does not use the
shared `examples/config.py` harness, so there is no mock fallback. It
reads the variables from Prerequisites: `OCI_DAC_ENDPOINT_OCID` and
`OCI_DAC_COMPARTMENT_ID` name the endpoint and its compartment,
`OCI_DAC_REGION` (default `us-chicago-1`) selects the regional inference
endpoint, and `OCI_PROFILE` (default `DEFAULT`) together with
`OCI_AUTH_TYPE` (`api_key` by default, or `security_token` for a profile
created by `oci session authenticate`) selects the `~/.oci/config`
profile that may invoke it. Without the two DAC variables the script
still exits cleanly and makes no network call: Part 1 prints the routing
rules, Part 2 prints the wiring snippet, and Parts 3 to 5 print a skip
notice.

## See also

- [Concepts: OCI Generative AI](../concepts/providers/oci.md) — the V1, SDK and Responses transports.
- [How-to: OCI DAC endpoints](../how-to/oci-dac.md) — configuration, streaming and auth for DAC endpoints.
- [OCI Generative AI — concepts (Dedicated AI Cluster)](https://docs.oracle.com/iaas/Content/generative-ai/concepts.htm) — Oracle reference.

## Source

??? note "Full source — `examples/notebook_35_oci_dac.py`"

    ````python
    --8<-- "examples/notebook_35_oci_dac.py"
    ````
