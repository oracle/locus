---
table_filter: true
---

# Environment variables

The SDK configures itself from environment variables. The notebook harness in
`examples/config.py` reads them with a consistent fallback chain; the SDK
itself reads provider API keys directly.

This page is the answer when an example prints *"skipped — missing env
vars"*.

## Minimum set for a notebook run

The harness takes the provider and the model id as two variables:

=== "OCI"

    ```bash
    export LOCUS_MODEL_PROVIDER=oci
    export LOCUS_MODEL_ID=openai.gpt-5.5
    export OCI_PROFILE=DEFAULT

    python examples/notebook_02_basic_agent.py
    ```

=== "OpenAI"

    ```bash
    export LOCUS_MODEL_PROVIDER=openai
    export LOCUS_MODEL_ID=gpt-4o
    export OPENAI_API_KEY=sk-...

    python examples/notebook_02_basic_agent.py
    ```

=== "Anthropic"

    ```bash
    export LOCUS_MODEL_PROVIDER=anthropic
    export LOCUS_MODEL_ID=claude-sonnet-5-5
    export ANTHROPIC_API_KEY=sk-ant-...

    python examples/notebook_02_basic_agent.py
    ```

=== "Self-hosted"

    ```bash
    # No key needed; default endpoint localhost:11434, LOCUS_OLLAMA_BASE_URL overrides
    export LOCUS_MODEL_PROVIDER=ollama
    export LOCUS_MODEL_ID=qwen3

    python examples/notebook_02_basic_agent.py
    ```

`LOCUS_MODEL_PROVIDER` is the switch. **Unset, it auto-detects**: `oci` when
`~/.oci/config` has a usable profile, otherwise `mock`, and the notebooks run
offline against a bundled model. Setting only `LOCUS_MODEL_ID` does not change
that choice — without an OCI profile it still gives the mock.

## Model selection

| Variable | Used by |
|---|---|
| `LOCUS_MODEL_PROVIDER` | `examples/config.py:get_model` — `oci` (auto-detected when `~/.oci/config` has a usable profile), `mock` (default otherwise), `openai`, `anthropic`, or any prefix from the [compatible-provider table](../concepts/providers/openai-compatible.md), e.g. `ollama` |
| `LOCUS_MODEL_ID` | Model id for the selected provider (e.g. `openai.gpt-5.5` on OCI, `gpt-4o`, `claude-sonnet-5-5`) |
| `LOCUS_MODEL_ID_B` | Secondary "model B" slot — comparison notebooks |
| `LOCUS_MODEL_ID_C` | Tertiary "model C" slot |
| `LOCUS_RESEARCH_MODEL` | Overrides the model for the research/eval scripts |
| `LOCUS_NOTEBOOK_BOOT` | Set to `1` to make notebook 65 start a live `AgentServer` on port 8000 instead of its TestClient smoke test |

### OCI Generative AI

Every `LOCUS_OCI_*` variable falls back to the plain `OCI_*` name, so a shell
that already exports `OCI_PROFILE` works unchanged. See
[OCI GenAI models](oci-models.md).

| Variable | Used by |
|---|---|
| `LOCUS_OCI_PROFILE` / `OCI_PROFILE` | `~/.oci/config` profile; when neither is set the notebooks pick `[DEFAULT]` or the only named profile |
| `LOCUS_OCI_AUTH_TYPE` / `OCI_AUTH_TYPE` | `api_key` (default), `security_token`, `instance_principal`, `resource_principal` |
| `LOCUS_OCI_REGION` / `OCI_REGION` | GenAI region, e.g. `us-chicago-1` |
| `LOCUS_OCI_COMPARTMENT` / `OCI_COMPARTMENT` | Compartment OCID; the OCI model classes also read `OCI_COMPARTMENT_ID`, the name [Configure a model provider](configure-providers.md) uses. Derived from the profile's tenancy when none is set |
| `LOCUS_OCI_ENDPOINT` / `OCI_ENDPOINT` | Service endpoint override — SDK transport (`OCIModel`) only |
| `LOCUS_OCI_TRANSPORT` / `OCI_TRANSPORT` | Force `v1` or `sdk` instead of routing by model id |
| `OCI_DAC_ENDPOINT_OCID`, `OCI_DAC_COMPARTMENT_ID`, `OCI_DAC_REGION` | Dedicated AI Cluster — notebook 35 |

## Provider keys

Per-provider setup, with the model string for each, is in
[Configure a model provider](configure-providers.md).

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | `OpenAIModel`, `OpenAIEmbeddings` |
| `OPENAI_BASE_URL` | OpenAI-shaped endpoint override — points the notebooks at a proxy or the LiteLLM gateway |
| `ANTHROPIC_API_KEY` | `AnthropicModel` |
| `COHERE_API_KEY` | `CohereEmbeddings`, `CohereReranker` |

### OpenAI-compatible providers

Every prefix in the [compatible-provider table](../concepts/providers/openai-compatible.md)
reads its own key variable; explicit `api_key=` and `base_url=` arguments win
over the environment.

| Variable | Used by |
|---|---|
| `GROQ_API_KEY` | `groq:` models |
| `TOGETHER_API_KEY` | `together:` models |
| `OPENROUTER_API_KEY` | `openrouter:` models |
| `DEEPSEEK_API_KEY` | `deepseek:` models |
| `MISTRAL_API_KEY` | `mistral:` models |
| `XAI_API_KEY` | `xai:` models |
| `FIREWORKS_API_KEY` | `fireworks:` models |
| `CEREBRAS_API_KEY` | `cerebras:` models |
| `PERPLEXITY_API_KEY` | `perplexity:` models |
| `NVIDIA_API_KEY` | `nvidia:` models |
| `GEMINI_API_KEY` / `GEMINI_BASE_URL` | `gemini:` models — Google's OpenAI-compatible endpoint |
| `RUNPOD_API_KEY` / `RUNPOD_ENDPOINT_ID` / `RUNPOD_BASE_URL` | `runpod:` models — the endpoint id fills the per-deployment URL when no base URL is given |
| `LITELLM_API_KEY` / `LITELLM_GATEWAY_URL` | `litellm:` models — see [LiteLLM gateway](#litellm-gateway) |
| `OLLAMA_BASE_URL`, `VLLM_BASE_URL`, `LMSTUDIO_BASE_URL`, `LLAMACPP_BASE_URL` | Local servers (`ollama`, `vllm`, `lmstudio`, `llamacpp`) — no key needed; `OLLAMA_API_KEY` and the matching names are honoured when set |
| `OPENAI_COMPATIBLE_API_KEY` | `openai-compatible:` models — the endpoint must come from `base_url=` or `LOCUS_OPENAI_COMPATIBLE_BASE_URL` |
| `LOCUS_<PREFIX>_BASE_URL` | Endpoint override for any prefix (e.g. `LOCUS_OLLAMA_BASE_URL`); it wins over the vendor's own variable |

## Storage backends

| Variable | Used by |
|---|---|
| `REDIS_URL` | Redis checkpointer — notebook 65 |
| `S3_BUCKET` | S3 checkpointer — notebook 50 |
| `ORACLE_DSN` | Oracle Database 26ai — tnsnames alias or `host:port/service` — notebooks 37-45 |
| `ORACLE_USER` / `ORACLE_PASSWORD` | Oracle Database 26ai credentials |
| `ORACLE_WALLET` / `ORACLE_WALLET_PASSWORD` | Autonomous Database wallet directory and password (empty string for auto-login wallets) |
| `S3_ENDPOINT_URL` | S3-compatible endpoint (MinIO, R2) — notebook 50 |

## Serving

| Variable | Used by |
|---|---|
| `LOCUS_SERVER_API_KEY` | `AgentServer` bearer token — notebook 65 |
| `LOCUS_A2A_API_KEY` | `A2AServer.__init__` bearer token |

## LiteLLM gateway

Notebooks 66 and 67. See [the gateway how-to](litellm-gateway.md).

| Variable | Used by |
|---|---|
| `LITELLM_GATEWAY_URL` | Gateway base URL |
| `LITELLM_GATEWAY_KEY` | Virtual key the agent presents |
| `LITELLM_GATEWAY_MODEL` | Model id as the gateway names it |
| `LITELLM_MASTER_KEY` | Admin key — `/key/generate` and `/spend/*` (needs Postgres) |

## Cloud and security domain

| Variable | Used by |
|---|---|
| `LOCUS_OCI_PROFILE` / `OCI_PROFILE` | OCI cloud-posture agent — the read-only profile notebook 81 audits as (see the OCI table above) |
| `LOCUS_OCI_AUTH_TYPE` / `OCI_AUTH_TYPE` | OCI cloud-posture agent — `api_key` (default) or `security_token` |
| `LOCUS_USE_OCI_ALLOW_MUTATIONS` | `use_oci` write opt-in — never set for the posture agent |
| `SCANNER_API_KEY` | Vulnerability-scanner adapter — notebook 80 |
| `DATAMAP_URL` | Data-map adapter — notebook 80 |
| `DATAMAP_API_KEY` | Data-map adapter — notebook 80 |
| `DATAMAP_TOKEN` | Data-map adapter — notebook 80 |

## Model fingerprinting

[Notebook 88](../notebooks/notebook_88_model_fingerprint.md), which ships with
examples 81–89 in the `locus-sdk-security` package
(`packages/locus-sdk-security/examples/`).

| Variable | Used by |
|---|---|
| `FINGERPRINT_MODEL` | Model id under test |
| `FINGERPRINT_ASSET` | Asset label recorded on the finding |
| `FINGERPRINT_SAMPLES` | Number of timing samples to collect |

## Other

| Variable | Used by |
|---|---|
| `OPENCLAW_DIR` | OpenClaw workspace root — the red-team example |
