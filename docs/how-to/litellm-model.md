# LiteLLMModel — one Locus class for every LiteLLM-supported provider

`LiteLLMModel` is the unified Locus entry point for **every Oracle
Generative AI Infrastructure inference surface** — plus the rest of
LiteLLM's 100+ providers — without spinning up the LiteLLM proxy as a
separate process and without dressing requests up through the OpenAI SDK.

It is a thin, in-process wrapper around `litellm.acompletion()`. The
`litellm` package is a **lazy import** inside `complete()` / `stream()`,
so importing the class costs nothing for users who only call the other
Locus model providers.

| Transport | Endpoint | Backend |
|---|---|---|
| `oci_native` *(default)* | `POST /20231130/actions/chat` | `litellm.acompletion(model="oci/…")` with LiteLLM's vendor adapters (Cohere v1 transport, GENERIC apiFormat for Grok/Llama/Gemini/gpt-5) |
| `oci_openai_v1` | `POST /openai/v1/chat/completions` | delegates to [`OCIChatCompletionsModel`](oci-models.md#v1-transport-ocichatcompletionsmodel) |
| `oci_responses` | `POST /openai/v1/responses` | delegates to [`OCIResponsesModel`](oci-models.md#responses-transport-ociresponsesmodel-opt-in) (server-stateful, supports `previous_response_id`) |
| *(any LiteLLM provider)* | provider-specific | `litellm.acompletion(model="<provider>/<model>")` — OpenAI direct, Anthropic, Bedrock, Vertex, Cohere SaaS, Mistral, Groq, … |

All three OCI transports converge on the **same Locus interface**
(`complete` / `stream`), so callers pick the transport via config and
the rest of the agent loop doesn't care.

---

## Why `LiteLLMModel` and not the proxy?

There were two existing ways to point Locus at LiteLLM:

1. Spin up `litellm --config ...` as a subprocess and use
   `OpenAIModel(base_url=...)` to talk to it. Works but: extra process,
   extra hop, splits cost-tracking across two systems, and Locus
   pretends to be OpenAI when it isn't.
2. Use `OCIChatCompletionsModel` / `OCIResponsesModel` directly. Works
   but gives up LiteLLM's vendor adapters and cross-provider
   abstraction.

`LiteLLMModel` is the third path: **native, in-process, no subprocess,
no detour through OpenAI** — and the same class targets every OCI
transport plus every LiteLLM-supported third-party provider.

---

## Quickstart — OCI native

Default transport. All six `OCI_*` environment variables resolve
automatically through LiteLLM's `resolve_oci_credentials()`:

```bash
export OCI_REGION="us-chicago-1"
export OCI_USER="ocid1.user.oc1..xxx"
export OCI_FINGERPRINT="aa:bb:cc:..."
export OCI_TENANCY="ocid1.tenancy.oc1..xxx"
export OCI_KEY_FILE="$HOME/.oci/keys/your_api_key.pem"
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..xxx"
```

```python
from locus.agent import Agent
from locus.models.native.litellm import LiteLLMModel

model = LiteLLMModel(model="oci/xai.grok-4.20")          # vendor-adapter path
# or
model = LiteLLMModel(model="oci/cohere.command-latest")  # Cohere v1 preamble + chatHistory
# or
model = LiteLLMModel(model="oci/openai.gpt-5-mini",      # reasoning model
                     reasoning_effort="high")

agent = Agent(model=model, system_prompt="You are concise.")
print(agent.run_sync("hi").message)
```

What that gives you:

- **One client, every OCI model family** — Llama 4, Grok 4.20, Cohere
  Command (chat + reasoning + vision), Cohere Embed v4, Gemini 2.5,
  gpt-5 / gpt-5-mini / gpt-5-nano — all callable as `oci/<model>` with
  the same OpenAI-compatible request and response shape.
- **OpenAI-shape tool calling across vendors** — Cohere's `preamble` /
  `chatHistory` and the GENERIC apiFormat used by Grok / Llama / Gemini
  are normalized internally.
- **`reasoning_effort` + `reasoning_tokens`** plumbed through the
  generic-vendor reasoning models (Grok reasoning variants, Gemini 2.5,
  gpt-5 family).
- **`drop_params=True` by default** — OpenAI kwargs the target vendor
  doesn't accept get silently stripped instead of 400-ing your loop.

---

## OCI OpenAI Chat-Completions shim (V1)

Routes the same Locus interface through OCI's `/openai/v1/chat/completions`
surface via the existing `OCIChatCompletionsModel`. Same calling code,
different wire format.

```python
model = LiteLLMModel(
    model="openai.gpt-5-mini",          # bare id (no "oci/" prefix)
    transport="oci_openai_v1",
    oci_profile="DEFAULT",              # any [profile] in ~/.oci/config
    oci_region="us-chicago-1",
)
```

When `transport="oci_openai_v1"`, `LiteLLMModel` constructs an
`OCIChatCompletionsModel` internally and forwards `complete()` /
`stream()` calls to it. You inherit `OCIChatCompletionsModel`'s OCI
Signature v1 RSA-SHA256 signing, profile-based auth, and Instance
Principal / Workload Identity support — see
[OCI GenAI models](oci-models.md#v1-transport-ocichatcompletionsmodel)
for the full feature set.

---

## OCI OpenAI Responses API (server-stateful)

Routes through OCI's `/openai/v1/responses` (the newer Responses-shape
endpoint) via the existing `OCIResponsesModel`. Server-stateful by
default — OCI holds the thread between turns; Locus sends only the
latest turn and threads the continuation id via
`AgentState.provider_state`.

```python
model = LiteLLMModel(
    model="openai.gpt-5-mini",
    transport="oci_responses",
    oci_profile="DEFAULT",
    oci_region="us-chicago-1",
    oci_responses_store=True,           # default; set False for ZDR tenancies
)
```

For tenancies with Zero Data Retention enabled, pass
`oci_responses_store=False` — the server rejects `previous_response_id`
otherwise. Locus then sends the full message history each turn
(like chat/completions) but still targets the Responses endpoint.

!!! note "Tenancy entitlement"
    Not all OCI tenancies have the Responses API enabled yet. The
    endpoint returns a generic 404 *"Authorization failed or requested
    resource not found"* when entitlement is missing — that's an OCI
    access-control response, not a `LiteLLMModel` bug.

---

## Non-OCI providers through the same class

Set the LiteLLM model id directly. No transport kwarg needed — LiteLLM
routes by the provider prefix:

```python
# OpenAI direct
m = LiteLLMModel(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

# Anthropic direct
m = LiteLLMModel(model="anthropic/claude-sonnet-4-20250514",
                 api_key=os.environ["ANTHROPIC_API_KEY"])

# AWS Bedrock
m = LiteLLMModel(model="bedrock/anthropic.claude-sonnet-4-v1:0")

# Google Vertex AI
m = LiteLLMModel(model="vertex_ai/gemini-2.5-pro")

# Local Ollama
m = LiteLLMModel(model="ollama/llama3.3")
```

The full list of LiteLLM provider prefixes is at
[docs.litellm.ai/docs/providers](https://docs.litellm.ai/docs/providers).
Every one of them works behind this single Locus class.

---

## Streaming

`stream()` yields `ModelChunkEvent`s. LiteLLM's OpenAI-shape progressive
deltas (text + partial tool calls) are reassembled internally; the
terminal chunk (`done=True`) carries the fully-assembled `ToolCall`
list when the model issued any.

```python
async for ev in model.stream(messages=[Message.user("count to five")]):
    if ev.content:
        print(ev.content, end="", flush=True)
    if ev.done and ev.tool_calls:
        print("\ntools:", [tc.name for tc in ev.tool_calls])
```

---

## Tool calling

Pass OpenAI-shape tool definitions. Both LiteLLM (native path) and
`OCIChatCompletionsModel` / `OCIResponsesModel` (shim paths) accept the
same schema. For OCI native + Cohere models, LiteLLM adapts the
OpenAI schema to OCI's Cohere `tools` shape internally.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}]

resp = await model.complete(
    messages=[Message.user("Weather in Tokyo?")],
    tools=tools,
    tool_choice="auto",
)
print(resp.message.tool_calls[0].name, resp.message.tool_calls[0].arguments)
```

LiteLLM returns tool-call `arguments` as a JSON string; `LiteLLMModel`
parses it to a `dict` so Locus's `ToolCall.arguments: dict` invariant
holds. Malformed JSON falls back to `{}` rather than crashing the agent
loop on the next turn.

---

## Reasoning models

OpenAI's `reasoning_effort` parameter is mapped to OCI's native
reasoning controls (`reasoningEffort`), and the resulting reasoning-
token count is surfaced on `usage.completion_tokens_details`:

```python
model = LiteLLMModel(model="oci/openai.gpt-5-mini", reasoning_effort="high")
resp = await model.complete(messages=[Message.user("If 3x + 7 = 22, what is x?")])

print(resp.message.content)
print("reasoning tokens:", resp.usage.get("reasoning_tokens"))
```

`reasoning_effort` accepts `"low"`, `"medium"`, `"high"`, or
`"disable"`. It is honored on Generic-vendor reasoning models (xAI Grok
reasoning variants, Google Gemini, OpenAI gpt-5 family) and silently
ignored on OCI Cohere models, which are not reasoning models.

!!! note "Fixed-temperature reasoning models"
    OCI's hosted `openai.gpt-5*`, `openai.o1`, `openai.o3`, and
    `openai.o4` deployments reject any non-default `temperature` value
    with HTTP 400 (`Unsupported value: 'temperature' does not support
    0.7 with this model`). `LiteLLMModel` detects these prefixes and
    skips the `temperature` kwarg automatically.

---

## Authentication paths

| Path | Set on | Native (`oci_native`) | V1 shim (`oci_openai_v1`) | Responses (`oci_responses`) |
|---|---|---|---|---|
| `OCI_*` env vars | shell | ✅ resolved by `litellm.resolve_oci_credentials()` | – | – |
| `oci_*` kwargs (`oci_user`, `oci_key_file`, …) | constructor | ✅ | – | – |
| OCI SDK `Signer` | `oci_signer=` | ✅ (Instance Principal, OKE Workload Identity, custom) | – | – |
| OCI config profile (`~/.oci/config`) | `oci_profile=` | – | ✅ | ✅ |
| `auth_type="instance_principal" \| "resource_principal"` + `oci_compartment_id=` | constructor | – | ✅ | ✅ |

For the native path, signing happens inside LiteLLM. For the two shim
paths, signing happens inside `OCIChatCompletionsModel` /
`OCIResponsesModel` via their existing httpx-based OCI Signature v1
implementation. **In all three paths**, the signing keys never have to
leave the OCI tenancy boundary when using Instance Principal or OKE
Workload Identity.

---

## Configuration reference

```python
LiteLLMModel(
    model: str,
    *,
    transport: Literal["oci_native", "oci_openai_v1", "oci_responses"] = "oci_native",

    # generic LiteLLM passthrough (any transport)
    api_key: str | None = None,            # non-OCI providers
    api_base: str | None = None,           # custom endpoint
    max_tokens: int = 4096,
    temperature: float = 0.7,
    reasoning_effort: str | None = None,   # "low" | "medium" | "high" | "disable"
    request_timeout: float | None = None,
    max_retries: int = 2,                  # litellm.num_retries
    drop_params: bool = True,              # strip unsupported provider kwargs

    # oci_native — also auto-read from OCI_* env vars
    oci_region: str | None = None,
    oci_user: str | None = None,
    oci_fingerprint: str | None = None,
    oci_tenancy: str | None = None,
    oci_key: str | None = None,
    oci_key_file: str | None = None,
    oci_compartment_id: str | None = None,
    oci_serving_mode: str | None = None,   # "ON_DEMAND" | "DEDICATED"
    oci_endpoint_id: str | None = None,    # DEDICATED endpoint OCID
    oci_signer: Any = None,                # OCI SDK Signer

    # oci_openai_v1 / oci_responses — passed through to the backing class
    oci_profile: str | None = None,
    oci_auth_type: str | None = None,
    oci_config_file: str = "~/.oci/config",
    oci_base_url: str | None = None,

    # oci_responses-only
    oci_project_ocid: str | None = None,
    oci_responses_store: bool = True,
    oci_responses_request_timeout: float = 600.0,
)
```

---

## Notebooks

- [Notebook 56 — Model providers](../notebooks/notebook_56_model_providers.md) — sweeps the registry, including the LiteLLM path.
- [Notebook 01 — OCI transports](../notebooks/notebook_01_oci_transports.md) — the direct (non-LiteLLM) OCI transports for reference; `LiteLLMModel` offers an alternative path to the same endpoints.

The example notebooks under `examples/` already route their model
construction through `examples/config.py:get_model()`, so swapping the
default backend to `LiteLLMModel` is one config flip away:

```bash
export LOCUS_MODEL_PROVIDER=litellm           # falls into the LiteLLM factory
export LOCUS_LITELLM_TRANSPORT=oci_native     # or oci_openai_v1, oci_responses
export LOCUS_MODEL_ID="oci/openai.gpt-5-mini"
python examples/notebook_06_basic_agent.py
```

---

## Limitations and open follow-ups

- **No LiteLLM Router yet.** Today a `LiteLLMModel` instance is one
  model. A sibling `LiteLLMRouterModel` exposing fallback chains,
  weighted load-balancing, and shared retry/budget policies is the
  next iteration.
- **No `metadata` / `user` passthrough.** LiteLLM's observability
  callbacks (Langfuse, Helicone, OTel, Datadog) key off the `metadata`
  kwarg for trace IDs, user IDs, session IDs; per-user spend
  attribution uses the `user=` kwarg. These are not yet plumbed.
- **No automatic cost extraction.** LiteLLM populates
  `response._hidden_params["response_cost"]` per call — surfacing that
  on `ModelResponse.usage["cost_usd"]` is on the list.
- **No caching kwarg passthrough.** `litellm.acompletion(cache={...})`
  is not exposed yet.

See [the issue tracker](https://github.com/oracle-samples/locus/issues)
for current status.
