# OCI GenAI models

Locus, the Oracle Generative AI Agentic Harness, connects to OCI Generative
AI through **three transports**.
The `oci:` string factory picks V1 or SDK by model family; the
Responses transport is opt-in.

!!! tip "Multi-tenant / cross-provider deployments"
    The direct OCI providers documented on this page are the right
    default for single-tenant production, dev / CI, and on-OKE
    workload identity. **If you need centralised virtual keys,
    per-team budgets, fallback chains across regions or providers,
    centralised observability (Langfuse / OTel / Datadog), or
    cross-provider routing, see the
    [LiteLLM AI Gateway guide](litellm-gateway.md) instead** — Locus
    connects to that via its existing `OpenAIModel(base_url=...)`
    client, no new Locus class needed.

| Model family | Transport | Class | Endpoint |
|---|---|---|---|
| OpenAI (`openai.gpt-*`, `openai.o*`) | V1 | `OCIChatCompletionsModel` | `/openai/v1/chat/completions` |
| Meta Llama (`meta.llama-*`) | V1 | `OCIChatCompletionsModel` | `/openai/v1/chat/completions` |
| xAI Grok (`xai.grok-*`) | V1 | `OCIChatCompletionsModel` | `/openai/v1/chat/completions` |
| Mistral (`mistral.*`) | V1 | `OCIChatCompletionsModel` | `/openai/v1/chat/completions` |
| Google Gemini (`google.gemini-*`) | V1 | `OCIChatCompletionsModel` | `/openai/v1/chat/completions` |
| **Cohere** (`cohere.command-r*`, `cohere.command-a*`) | SDK | `OCIModel` | `/20231130/actions/chat` |
| **Responses (opt-in)** — gpt-5.5-pro et al. | Responses | `OCIResponsesModel` | `/openai/v1/responses` |

V1 is the default. It uses the standard `openai` SDK against
OCI's OpenAI-compatible endpoint, gives you **real SSE streaming**
(token-by-token), picks up new OCI models on day-0 with no SDK update,
and uses the same OpenAI request shape for tool calls and system
prompts. The OCI SDK transport remains the only option for Cohere
models (Command R and Command A), which OCI does not expose on
`/openai/v1` (it returns `400 Unsupported OpenAI operation`). The Responses transport is the
only path that reaches Responses-only OCI models (`openai.gpt-5.5-pro`
today) and the only one that lets OCI hold conversation state between
turns via `previous_response_id` — both modes covered below.

## V1 transport — `OCIChatCompletionsModel`

### From a laptop or CI

```python
from locus.agent import Agent
from locus.models import OCIChatCompletionsModel

model = OCIChatCompletionsModel(
    model="openai.gpt-5.5",
    profile="DEFAULT",        # any [profile] in ~/.oci/config
)

agent = Agent(model=model, system_prompt="You are concise.")
print(agent.run_sync("hi").message)
```

`profile=` covers both API-key (`~/.oci/config` with `key_file`) and
session-token (`security_token_file`) authentication — V1 picks the
right signer based on the profile shape. The compartment
header that OCI requires for IAM auth is auto-derived from the
profile's `tenancy` field, so you don't pass it.

### On OCI compute / OKE / Functions (workload identity)

```python
import os
from locus.models import OCIChatCompletionsModel

model = OCIChatCompletionsModel(
    model="openai.gpt-5.5",
    auth_type="instance_principal",     # or "resource_principal"
    compartment_id=os.environ["OCI_COMPARTMENT_ID"],
)
```

For workload identity (no `~/.oci/config`), the compartment cannot be
auto-derived — pass it explicitly via `compartment_id=`. Use
`instance_principal` on OCI VMs and OKE node-identity pods;
`resource_principal` inside OCI Functions / Data Flow / OKE workload
identity.

### What you get

- **Real SSE streaming** — `agent.run(...)` yields events as the model
  produces tokens, not after the full response arrives.
- **Day-0 model coverage** — when OCI publishes a new model id (e.g.
  `openai.gpt-5.5` on launch day), it works immediately. No `oci`
  package release needed.
- **Standard OpenAI request shape** — tool calls, system messages,
  multimodal content, and seed/penalty/top_p knobs work the same way as
  with native OpenAI.

## Responses transport — `OCIResponsesModel` (opt-in)

Use this when you need a Responses-only OCI model
(`openai.gpt-5.5-pro` today) or when you want OCI to hold the
conversation thread across turns. The runtime sends only the
latest-turn slice each call and threads `previous_response_id` via
`AgentState.provider_state`.

```python
from locus.models.providers.oci import OCIResponsesModel

model = OCIResponsesModel(
    model="openai.gpt-5.5-pro",
    profile="DEFAULT",
    region="us-chicago-1",
    compartment_id="ocid1.compartment.oc1..…",
)
```

### Zero Data Retention (ZDR) tenancies

Enterprise OCI tenancies with ZDR reject `previous_response_id`
(server doesn't persist responses). For those, pass `store=False` —
the model sends `store: false` on every request, drops the
continuation token, and reports `server_stateful=False` so the agent
sends the full message history each turn. Responses-only models
remain reachable.

```python
model = OCIResponsesModel(
    model="openai.gpt-5.5-pro",
    profile="ENTERPRISE_PROFILE",
    region="us-chicago-1",
    compartment_id="ocid1.compartment.oc1..…",
    store=False,  # ZDR-safe
)
```

The only SDK primitive that bypasses on the Responses path is
`ConversationManager` (window/summarize have nothing to operate on
when the server owns the history). Memory, Reflexion, GSAR,
grounding, tool hooks, idempotency, checkpointing, output schema,
streaming, and termination conditions all work identically. See the
[OCI Responses concept page](../concepts/oci-responses.md) and
[notebook 34](../notebooks/notebook_34_oci_responses.md).

## Cohere (Command R and Command A) — `OCIModel`

```python
from locus.models import OCIModel

model = OCIModel(
    model_id="cohere.command-r-plus-08-2024",
    profile_name="DEFAULT",
    auth_type="api_key",
)
```

`OCIModel` uses the OCI Python SDK (`oci.generative_ai_inference`) and
sends OCI's proprietary `CohereChatRequest` payload for every
`cohere.command*` model — Command A included; OCI rejects the generic
request shape for it with `400 Chat request type does not match serving
model`. It supports the
same five auth modes (`api_key`, `security_token`, `session_token`,
`instance_principal`, `resource_principal`) but its argument names are
different from `OCIChatCompletionsModel` — `model_id=`, `profile_name=`,
`compartment_id=`, `auth_type=` as an enum / string. `stream()` on this
transport requests SSE (`is_stream=True` on the chat request) and
iterates the events; if the endpoint rejects the streaming request it
falls back to `complete()` and yields the full response as one chunk.

## String factory (`get_model("oci:...")`) auto-routes

```python
from locus.models import get_model

# Uses OCIChatCompletionsModel
m1 = get_model("oci:openai.gpt-5.5", profile="DEFAULT")

# Uses OCIModel (Cohere)
m2 = get_model("oci:cohere.command-r-plus", profile_name="DEFAULT", auth_type="api_key")
```

The string-form factory in `locus.models.registry` picks `OCIModel` for
any id starting with `cohere.` or `ocid1.generativeaiendpoint.` (a
Dedicated AI Cluster endpoint) and `OCIChatCompletionsModel` otherwise.
You pass kwargs in the shape the picked class expects.

## Notebooks and `examples/config.py`

The shared notebook harness (`examples/config.py`) reads
`LOCUS_MODEL_ID` and routes accordingly. To force a transport (rare —
useful only for debugging), set:

```bash
export LOCUS_OCI_TRANSPORT=v1     # or "sdk"
```

Notebooks that use `from examples.config import get_model` inherit
this routing. Notebooks that instantiate a class directly
(e.g. `notebook_11_model_providers.py`) show both transports
side-by-side.

## What's not supported today

- **Responses-only models depend on the region.** `openai.gpt-5.x-pro`,
  `openai.gpt-5.3-codex`, and tool calls on the `gpt-5.6` / `gpt-6`
  families are served only by `/openai/v1/responses`. `OCIResponsesModel`
  speaks that endpoint, but OCI does not expose it in every region yet
  (some regions return 404). Where it is missing
  those models complete plain prompts on V1 but cannot call tools.
- **Cohere on V1.** OCI's `/openai/v1` returns `400 Unsupported OpenAI
  operation` for every `cohere.*` id, Command R and Command A alike. Use
  `OCIModel`.
- **GenAI API key auth (Bearer token).** A "create an API key in the
  Console, hand it as `api_key=` and you're done" path is not yet
  reliable on OCI without a Project OCID. When it is, the SDK will add
  `api_key=` to `OCIChatCompletionsModel` as a third auth mode (additive,
  non-breaking).

## Model support on OCI

What the OCI Generative AI catalog can do through the same `Agent`, grouped
by capability. Catalogs differ per tenancy and region; the
[integration README](https://github.com/oracle/locus/tree/main/tests/integration)
has the scripts to check yours.

### Completion and tool calls

| Vendor | Models |
|---|---|
| OpenAI | `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.2`, `gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini`, `gpt-oss-120b`, `gpt-oss-20b`, `o3` (and their dated aliases) |
| Google | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` |
| Meta | `llama-3.3-70b-instruct` |
| Cohere | `command-a-03-2025` (SDK transport) |

### Completion only

`gpt-5.6-luna` / `-sol` / `-terra`, `gpt-6-astra` /
`-luna` / `-sol`, `gpt-6.1-sol`: tool calls need `/v1/responses`, which not
every region serves.

### Not usable

Retired by the service (`o1`, `o3-mini`, `o4-mini`,
`gpt-4.1-nano`, the `gpt-4o-search-preview` pair); not servable on-demand
(`command-r*`, `command-a-vision`, `llama-3-70b`, `llama-3.1-*`,
`gpt-5-codex`); Responses-only (`gpt-5.2-pro`, `gpt-5.4-pro`, `gpt-5.5-pro`,
`gpt-5.3-codex`); audio-only (`gpt-audio`).

### Embeddings and rerank

`cohere.embed-v4.0` (1536), `openai.text-embedding-3-small`
(1536), `openai.text-embedding-3-large` (3072) pass; the Cohere v3 family is
retired. `cohere.rerank-v4.0-fast` passes; `rerank-v3.5` and
`rerank-v4.0-pro` are retired.

## Testing

```bash
# V1 path (any non-Cohere model)
OCI_PROFILE=DEFAULT \
OCI_REGION=us-chicago-1 \
OCI_MODEL_ID=openai.gpt-5.5 \
pytest tests/integration/test_oci_openai_compat_integration.py

# SDK path (Cohere)
OCI_PROFILE=DEFAULT \
OCI_ENDPOINT=https://inference.generativeai.us-chicago-1.oci.oraclecloud.com \
OCI_COMPARTMENT=ocid1.compartment.oc1... \
OCI_MODEL_ID=cohere.command-a-03-2025 \
pytest tests/integration/test_oci_integration.py
```

`test_oci_openai_compat_integration.py` skips cleanly when
`OCI_MODEL_ID` is a Cohere model — V1 doesn't run there.

## Reference

- `src/locus/models/providers/oci/openai_compat.py` — `OCIChatCompletionsModel`.
- `src/locus/models/providers/oci/__init__.py` — `OCIModel`.
- `src/locus/models/providers/oci/_signing.py` — internal httpx OCI
  request signer used by V1's IAM path.
- `src/locus/models/registry.py` — the `oci:` string-factory routing.

### Oracle reference docs

- [OCI Generative AI — documentation hub](https://docs.oracle.com/iaas/Content/generative-ai/home.htm)
  — service overview, supported models, regional availability.
- [OCI Generative AI — Chat](https://docs.oracle.com/iaas/Content/generative-ai/use-playground-chat.htm)
  — the SDK transport's underlying API for Cohere models.
- [OCI OpenAI-compatible endpoints](https://docs.oracle.com/iaas/Content/generative-ai/openai-compatible-api.htm)
  — endpoint shape, auth modes, and the migration path from raw OpenAI.
- [OCI Responses API](https://docs.oracle.com/iaas/Content/generative-ai/responses-api.htm)
  — the opt-in Responses transport this how-to references.
