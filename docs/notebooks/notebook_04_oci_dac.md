# OCI Dedicated AI Cluster

A Dedicated AI Cluster (DAC) is Oracle Cloud Infrastructure (OCI)
Generative AI's provisioned-capacity serving mode: instead of
pay-per-token inference against a shared model id, you address a
dedicated endpoint by its OCID
(`ocid1.generativeaiendpoint.oc1.<region>....`) and OCI routes
inference to your cluster. This notebook wires Locus to one.

The shape is the same as any other OCI model — only the model id
changes. Pass the endpoint OCID and Locus routes it through `OCIModel`
(the OCI SDK client). The OpenAI-compatible client cannot speak
`DedicatedServingMode`, which is what a DAC requires. Both `complete()`
and SSE `stream()` work end-to-end.

## What this covers

- Part 1: how Locus picks the OCI SDK client for a DAC OCID.
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
```

Without those env vars Parts 2-5 print the wiring snippet and skip, so
the file still runs cleanly in CI.

Difficulty: Intermediate

## See also

- [Concepts: OCI GenAI](../concepts/providers/oci.md) — V1 vs SDK transports.
- [How-to: OCI DAC endpoints](../how-to/oci-dac.md) — config recipes.
- [OCI Generative AI — concepts (Dedicated AI Cluster)](https://docs.oracle.com/iaas/Content/generative-ai/concepts.htm) — Oracle reference.

## Source

```python
--8<-- "examples/notebook_04_oci_dac.py"
```
