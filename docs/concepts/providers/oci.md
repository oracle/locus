# OCI Generative AI

OCI Generative AI is the primary provider in Locus, the Oracle
Generative AI Agentic Harness. It exposes OpenAI commercial families,
Meta Llama, Anthropic Claude, Google Gemini, xAI Grok, Mistral, and
Cohere through Oracle's hosted inference service. New OCI model ids work
without an SDK update: pass the new id as the model string.

The headline value over the direct providers:

- **One auth surface.** Same `OCI_PROFILE` mechanism on a laptop, in
  CI, or running on OCI Compute / OKE / Functions.
- **New models as OCI lists them.** New OpenAI, Anthropic and Llama
  model ids are usable as soon as OCI publishes them; no SDK update is
  needed.
- **No per-provider API keys.** GPT, Claude, Llama all bill through
  your OCI tenancy.
- **Dedicated AI Cluster (DAC) endpoints** for predictable latency
  and isolation when on-demand isn't enough.

## When to pick OCI

| You want… | This is the right provider |
|---|---|
| GPT, Claude, Llama, Cohere, Gemini, Grok, Mistral all in one place | ✓ |
| Production inference on Oracle infrastructure (OKE / Compute / Functions) | ✓ |
| One auth surface across laptop, CI, OCI workloads | ✓ |
| Provisioned-capacity inference via [DAC](../../how-to/oci-dac.md) | ✓ |
| To avoid managing per-provider API keys | ✓ |
| Latest OpenAI features on release day | use [OpenAI](openai.md) direct; OCI may lag the OpenAI release |
| Local development without auth setup | use [Ollama or another OpenAI-compatible endpoint](openai-compatible.md) instead |

## Three transports under one prefix

OCI Generative AI exposes its inference service in three ways: the
OpenAI-compatible chat completions API, the OpenAI-compatible Responses
API, and the native OCI Generative AI Python SDK. Locus speaks all
three. The `oci:` prefix **picks the transport from the model id** —
Dedicated AI Cluster endpoints and Cohere models go through the SDK,
every other id through the OpenAI-compatible chat API — so you don't
have to know which transport a model uses to call it. The Responses
transport is opt-in: the prefix never selects it. You can also pick any
transport explicitly by instantiating its model class yourself.

{{ locus_diagram oci-transports }}

### V1 transport — `/openai/v1` (OpenAI-compatible)

`OCIChatCompletionsModel` calls the OpenAI-compatible chat completions
endpoint:

```text
https://inference.generativeai.<region>.oci.oraclecloud.com/openai/v1/chat/completions
```

This is the **default path for the majority of OCI models**:
OpenAI commercial, Meta Llama, xAI Grok, Mistral, Google Gemini, and
Claude on OCI. The wire format is identical to OpenAI's, so anything
you know about prompting OpenAI carries over: real SSE streaming,
OpenAI-style function calling, structured output, vision input.

```python
agent = Agent(model="oci:openai.gpt-5.5")           # OpenAI commercial
agent = Agent(model="oci:meta.llama-3.3-70b-instruct")  # Meta Llama
agent = Agent(model="oci:anthropic.claude-sonnet")  # Claude, no Anthropic key
```

### Responses transport — `/openai/v1/responses` (opt-in)

`OCIResponsesModel` calls the OpenAI-compatible Responses endpoint:

```text
https://inference.generativeai.<region>.oci.oraclecloud.com/openai/v1/responses
```

This is the **opt-in path** for Responses-only models
(`openai.gpt-5.5-pro` today) and for runs where you want OCI to hold
the conversation thread between turns. The runtime sends only the
latest-turn slice and threads `previous_response_id` via
`AgentState.provider_state`.

```python
from locus.models.providers.oci import OCIResponsesModel

agent = Agent(model=OCIResponsesModel(
    model="openai.gpt-5.5-pro",
    profile="MY_PROFILE",
    region="us-chicago-1",
    compartment_id="ocid1.compartment.oc1..…",
    # store=False for Zero-Data-Retention tenancies (full-history mode)
))
```

`store=True` (the default) has OCI persist each response so the next
turn continues it through `previous_response_id`; with `store=False`
nothing is retained server-side and the runtime sends the full history
on every request.

The only SDK primitive that bypasses on this path is
`ConversationManager`. Memory, Reflexion, GSAR, grounding, tool
hooks, idempotency, checkpointing, output schema, streaming, and
termination conditions all work identically. See
[OCI Responses concept page](../oci-responses.md) for the full
trade-off matrix.

### SDK transport — OCI native API

`OCIModel` calls the OCI Generative AI Python SDK directly. It's used
for Cohere (`cohere.command-r-*` and `cohere.command-a-*`), which OCI
exposes through the native API rather than the OpenAI-compatible
gateway, and for Dedicated AI Cluster endpoints (next section). Cohere
has its own request shape (separate `message` + `chat_history` instead
of a flat `messages` array).

```python
agent = Agent(model="oci:cohere.command-r-plus-08-2024")  # SDK transport
```

### DAC endpoints — dedicated capacity

When you've provisioned a Dedicated AI Cluster (DAC), OCI gives you
a **generative AI endpoint OCID**. Pass it as the model id and the SDK
auto-routes through the SDK transport with `DedicatedServingMode`:

```python
agent = Agent(
    model=get_model(
        "oci:ocid1.generativeaiendpoint.oc1.<region>....",
        compartment_id="ocid1.compartment.oc1...",
        profile_name="DEFAULT",
    ),
)
```

The [DAC how-to](../../how-to/oci-dac.md) covers Qwen on a DAC,
streaming, and per-model tool-call differences.

Note the spelling: `OCIModel` takes `profile_name=` (with `profile=`
accepted as an alias), while `OCIChatCompletionsModel` and
`OCIResponsesModel` take `profile=` only. Through the `oci:` factory use
`profile=` — it reaches every transport; `profile_name=` is honoured only
when the id routes to `OCIModel` (Cohere ids and DAC OCIDs) and is
silently ignored on the V1 path.

## Transport selection — by the `oci:` prefix

When you use the `oci:<model-id>` string factory, the SDK looks at the
model id and chooses for you:

| Model id pattern | Transport |
|---|---|
| `ocid1.generativeaiendpoint....` | SDK, `DedicatedServingMode` (DAC) |
| `cohere.*` | SDK, `OnDemandServingMode` |
| `openai.*` / `meta.*` / `xai.*` / `mistral.*` / `google.*` / `anthropic.*` | V1 (OpenAI-compatible) |

Need to override? Instantiate the transport class directly —
`OCIChatCompletionsModel` for V1, `OCIModel` for the SDK — instead of the
`oci:` prefix. The notebook harness (`examples/config.py`) also honours
`LOCUS_OCI_TRANSPORT=v1|sdk`.

For the **Responses transport**, instantiate `OCIResponsesModel`
explicitly — it's opt-in, not selected by prefix. See the
[Responses concept page](../oci-responses.md) and
[notebook 34](../../notebooks/notebook_34_oci_responses.md).

## One auth surface — laptop, CI, OCI workloads

`OCI_PROFILE` is read by the `oci:` factory for the V1 transport (every
id that is not `cohere.*` or a DAC OCID), and a profile with
`security_token_file` is signed with a session token automatically, so
the API-key and session-token paths need no code. The SDK transport
(`OCIModel`) takes `profile_name=` instead (default `DEFAULT`) and is the
only one that reads `OCI_AUTH_TYPE`; on the V1 and Responses transports
pass the principal on the model —
`get_model("oci:openai.gpt-5.5", auth_type="instance_principal", compartment_id=...)`
or `Agent(model="oci:...", model_kwargs={"auth_type": "instance_principal", "compartment_id": ...})`:

| `OCI_AUTH_TYPE` | Where it works | Notes |
|---|---|---|
| `api_key` | Laptop with a `~/.oci/config` profile | Also set `OCI_PROFILE` to the profile name |
| `session_token` | Federated SSO laptop | Run `oci session authenticate` first |
| `instance_principal` | OCI Compute, OKE pods | No key file needed |
| `resource_principal` | OCI Functions, serverless | Provider-injected |

```bash
export OCI_PROFILE=DEFAULT
export OCI_AUTH_TYPE=api_key
```

**The same `Agent` code and model id run on a laptop, on OKE, and on
Compute; only the auth arguments differ** — `OCI_PROFILE` on a laptop or
in CI, `auth_type=` plus `compartment_id=` on the model under a
principal.

## Region

OCI Generative AI is offered in a growing list of regions, including:

- `us-chicago-1`
- `eu-frankfurt-1`
- `uk-london-1`
- `sa-saopaulo-1`

On the `oci:` factory the V1 transport (every id that is not `cohere.*`
or a Dedicated AI Cluster OCID) defaults to `us-chicago-1`;
`LOCUS_OCI_REGION`, then `OCI_REGION`, overrides that, as does `region=`
on the model. The SDK transport behind Cohere ids and DAC endpoints takes
the region from the profile unless `region=` or `service_endpoint=` is
given:

```bash
export OCI_REGION=us-chicago-1
```

## Practical wiring: laptop to OKE

```python
# Same code on your laptop and on OKE:
from locus.agent import Agent
agent = Agent(
    model="oci:openai.gpt-5.5",
    system_prompt="You are a helpful assistant.",
)
```

```bash
# Laptop:
export OCI_PROFILE=DEFAULT
export OCI_AUTH_TYPE=api_key

# OKE pod — the V1 transport takes the principal as model kwargs:
# agent = Agent(
#     model="oci:openai.gpt-5.5",
#     model_kwargs={"auth_type": "instance_principal", "compartment_id": "ocid1.compartment.oc1..…"},
# )
# (no profile / key file — OKE injects the principal at runtime)
```

The model id and the agent code are unchanged; only the auth arguments
differ.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `404 Not Authorized` | OCI returns 404 rather than 403 for authorization failures. The principal lacks the `inspect generative-ai-endpoints` policy in the compartment. |
| `model_id not found` | Model id doesn't exist in your tenancy's region. List the models available in the region with the OCI CLI (below). |
| `compartment_id is required when auth_type= is set` | Principal auth on the V1/Responses transports cannot derive a compartment from a profile. Pass `compartment_id=` on the model (the SDK transport falls back to the profile's tenancy). |
| Streaming yields one big chunk | DAC endpoint rejected `is_stream`. The fall-back path in `OCIModel.stream()` swallows the failure and emits the full response as one chunk. |
| Cohere model fails on V1 | The `oci:` prefix already routes `cohere.*` ids to the SDK transport; if you built `OCIChatCompletionsModel` yourself, use `OCIModel` (in the notebooks, `LOCUS_OCI_TRANSPORT=sdk`). |

To list the models available in a region:

```bash
oci generative-ai model-collection list-models \
  --compartment-id <compartment-ocid> --region <region>
```

## Source

| Component | File |
|---|---|
| `OCIChatCompletionsModel` (V1) | [`oci/openai_compat.py`](https://github.com/oracle/locus/blob/main/src/locus/models/providers/oci/openai_compat.py) |
| `OCIModel` (SDK + DAC) | [`oci/__init__.py`](https://github.com/oracle/locus/blob/main/src/locus/models/providers/oci/__init__.py) |
| Per-family request builders | [`oci/models/`](https://github.com/oracle/locus/tree/main/src/locus/models/providers/oci/models) |
| Routing | [`registry.py`](https://github.com/oracle/locus/blob/main/src/locus/models/registry.py), `_make_oci()` |

## See also

- [Models overview](../models.md) — the full provider tree.
- [OCI Responses transport](../oci-responses.md) — when to opt in, ZDR mode, server-stateful continuation.
- [Notebook 01 — three OCI transports side-by-side](../../notebooks/notebook_01_oci_transports.md).
- [Notebook 33 — `OCIChatCompletionsModel` deep dive](../../notebooks/notebook_33_oci_openai_chat.md).
- [Notebook 34 — `OCIResponsesModel` deep dive](../../notebooks/notebook_34_oci_responses.md).
- [OCI GenAI models how-to](../../how-to/oci-models.md) — auth setup, region selection, debugging.
- [OCI Dedicated AI Cluster (DAC)](../../how-to/oci-dac.md) — provisioned-capacity endpoints.
- [OpenAI](openai.md) — direct OpenAI when OCI lags.
- [Anthropic](anthropic.md) — Claude direct when OCI lags.
- [OpenAI-compatible endpoints](openai-compatible.md) — Ollama, vLLM and other OpenAI-compatible servers for local development before moving to OCI.

### Oracle reference docs

- [OCI Generative AI — documentation hub](https://docs.oracle.com/iaas/Content/generative-ai/home.htm)
  — service overview, model catalog, regions.
- [OCI Generative AI — concepts](https://docs.oracle.com/iaas/Content/generative-ai/concepts.htm)
  — endpoints, serving modes, Dedicated AI Clusters.
- [OCI Generative AI — Chat (V1 / SDK transport)](https://docs.oracle.com/iaas/Content/generative-ai/use-playground-chat.htm)
  — the `/20231130/actions/chat` endpoint `OCIModel` calls.
- [OCI OpenAI-compatible endpoints](https://docs.oracle.com/iaas/Content/generative-ai/openai-compatible-api.htm)
  — the `/openai/v1/*` surface `OCIChatCompletionsModel` and `OCIResponsesModel` call.
- [OCI Responses API](https://docs.oracle.com/iaas/Content/generative-ai/responses-api.htm)
  — server-stateful Responses transport reference.
