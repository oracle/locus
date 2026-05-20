# OCIResponsesModel

`OCIResponsesModel` is the Locus client for Oracle Cloud Infrastructure
(OCI) Generative AI's Responses endpoint
(`/openai/v1/responses`). Use it to reach Responses-only OCI models
(such as `openai.gpt-5.5-pro`) or to let OCI hold the conversation
thread so multi-turn payloads stay small. Two modes:

- **`store=False`** — Zero Data Retention (ZDR) safe. The agent sends
  the full history each turn; OCI keeps nothing. Works in every
  tenancy, including those with ZDR enforced.
- **`store=True`** — OCI holds the thread. Locus passes
  `previous_response_id` between turns and only the latest turn goes on
  the wire. Requires a non-ZDR tenancy.

`ConversationManager` is the one Locus primitive that bypasses on this
path. Memory, Reflexion, GSAR, grounding, hooks, idempotency,
checkpointing, output schema, streaming, and termination conditions all
apply — see [concepts/oci-responses.md](../concepts/oci-responses.md)
for the full matrix.

## What this covers

- Part 1: stateless multi-turn (`store=False`)
- Part 2: server-stateful multi-turn with continuation id (`store=True`)
- Part 3: tool round-trip — model emits `function_call`, Locus posts
  back `function_call_output`
- Part 4: streaming

## Prerequisites

```bash
export OCI_PROFILE=<your-profile>
export OCI_REGION=us-chicago-1
export OCI_COMPARTMENT=ocid1.compartment.oc1..…
# Optional — set to 1 to also exercise store=True (skip in ZDR tenants):
export OCI_RESPONSES_STORE=0
```

## Run

```bash
python examples/notebook_03_oci_responses.py
```

## See also

- [Notebook 01 — the three OCI client classes side-by-side](notebook_01_oci_transports.md)
- [Notebook 02 — OCIChatCompletionsModel deep dive](notebook_02_oci_openai_chat.md)
- [Concepts: OCI Responses](../concepts/oci-responses.md)
- [OCI Responses API — Oracle docs](https://docs.oracle.com/iaas/Content/generative-ai/responses-api.htm)
- [OCI OpenAI-compatible endpoints — Oracle docs](https://docs.oracle.com/iaas/Content/generative-ai/openai-compatible-api.htm)

## Source

```python
--8<-- "examples/notebook_03_oci_responses.py"
```
