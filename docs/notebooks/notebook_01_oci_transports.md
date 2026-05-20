# OCI Transports

Locus ships three model classes for Oracle Cloud Infrastructure (OCI)
Generative AI. They are not interchangeable — the right one depends on
which model family you call and whether you want OCI to hold the
conversation history.

## Decision rule

Read top-to-bottom. The first match wins.

| Want… | Use |
|---|---|
| A Responses-only model (e.g. `openai.gpt-5.5-pro`), or OCI to hold the thread | `OCIResponsesModel` |
| A Cohere R-series model (`cohere.command-r*`) | `OCIModel` |
| Everything else (OpenAI, Meta, Mistral, xAI, Gemini, non-R Cohere) | `OCIChatCompletionsModel` (default) |

The runtime reads `model.server_stateful`. When it is `True`, Locus
sends only the latest turn and threads OCI's continuation id through
`AgentState.provider_state`. `ConversationManager` is the one Locus
primitive that bypasses on the Responses path (no client-side history
to trim). Memory, reflexion, GSAR, grounding, tool hooks, idempotency,
checkpointing, streaming, structured output, and termination conditions
all apply identically across the three classes.

## Prerequisites

```bash
export OCI_PROFILE=<your-profile>
export OCI_REGION=us-chicago-1
export OCI_COMPARTMENT=ocid1.compartment.oc1..…   # compartment with GenAI access
```

## Run

```bash
python examples/notebook_01_oci_transports.py            # all three
python examples/notebook_01_oci_transports.py --transport responses
python examples/notebook_01_oci_transports.py --transport sdk
python examples/notebook_01_oci_transports.py --transport v1
```

## See also

- [OCI Responses API concept page](../concepts/oci-responses.md)
- [Hooks](../concepts/hooks.md) — works identically on all transports
- [MCP](../concepts/mcp.md) — host-side side effects via the same hook surface

### Oracle reference docs

- [OCI Generative AI — Chat](https://docs.oracle.com/iaas/Content/generative-ai/use-playground-chat.htm) — SDK transport
- [OCI OpenAI-compatible endpoints](https://docs.oracle.com/iaas/Content/generative-ai/openai-compatible-api.htm) — V1 transport
- [OCI Responses API](https://docs.oracle.com/iaas/Content/generative-ai/responses-api.htm) — Responses transport

## Source

```python
--8<-- "examples/notebook_01_oci_transports.py"
```
