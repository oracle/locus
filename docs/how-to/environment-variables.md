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
| `LOCUS_CONTEXT_WINDOW` | Input context window, in tokens, for every agent in the process whose `context_window` and model metadata do not name one — a self-hosted model the seed table cannot know |
| `LOCUS_MODEL_PROFILES` | Path to a JSON file of `ModelProfile` overrides keyed by model id or `fnmatch` glob; `profile_for()` reads it before the built-in metadata and family heuristics |

### OCI Generative AI

Every `LOCUS_OCI_*` variable falls back to the plain `OCI_*` name, so a shell
that already exports `OCI_PROFILE` works unchanged. See
[OCI GenAI models](oci-models.md).

| Variable | Used by |
|---|---|
| `LOCUS_OCI_PROFILE` / `OCI_PROFILE` | `~/.oci/config` profile; when neither is set the notebooks pick `[DEFAULT]` or the only named profile |
| `LOCUS_OCI_AUTH_TYPE` / `OCI_AUTH_TYPE` | `api_key` (default), `security_token`, `instance_principal`, `resource_principal`, `oke_workload_identity` (`resource_principal` detects an OKE pod on its own) |
| `LOCUS_OCI_REGION` / `OCI_REGION` | GenAI region, e.g. `us-chicago-1` |
| `LOCUS_OCI_COMPARTMENT` / `OCI_COMPARTMENT` | Compartment OCID; the OCI model classes also read `OCI_COMPARTMENT_ID`, the name [Configure a model provider](configure-providers.md) uses. Derived from the profile's tenancy when none is set |
| `LOCUS_OCI_ENDPOINT` / `OCI_ENDPOINT` | Service endpoint override — SDK transport (`OCIModel`) only |
| `LOCUS_OCI_TRANSPORT` / `OCI_TRANSPORT` | Force `v1` or `sdk` instead of routing by model id |
| `OCI_DAC_ENDPOINT_OCID`, `OCI_DAC_COMPARTMENT_ID`, `OCI_DAC_REGION` | Dedicated AI Cluster — notebook 35 |
| `OCI_SESSION_PROFILE` / `OCI_CLI_PROFILE` | `OracleAgentMemoryManager` signing an `oci/...` extraction model: the profile is `OCI_SESSION_PROFILE`, then `OCI_PROFILE`, then `OCI_CLI_PROFILE`, then `DEFAULT` |
| `OCI_USER` / `OCI_KEY_FILE` / `OCI_KEY_CONTENT` / `OCI_FINGERPRINT` | When any is set, `OracleAgentMemoryManager` leaves the `oci/...` model's auth to those variables instead of building a signer from the profile |
| `OCI_RESOURCE_PRINCIPAL_VERSION` | Set by the OCI runtime, not by you; the `locus deploy` entrypoint reads it to pick `resource_principal` over `instance_principal` when `LOCUS_OCI_AUTH_TYPE` is unset |

## Provider keys

Per-provider setup, with the model string for each, is in
[Configure a model provider](configure-providers.md).

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | `OpenAIModel`, `OpenAIEmbeddings` |
| `OPENAI_BASE_URL` | OpenAI-shaped endpoint override — points the notebooks at a proxy or the LiteLLM gateway |
| `ANTHROPIC_API_KEY` | `AnthropicModel` |
| `ANTHROPIC_BASE_URL` | `AnthropicModel` endpoint when `base_url=` is not given — any endpoint that speaks the Messages API; unset, `api.anthropic.com` |
| `ANTHROPIC_AUTH_TOKEN` | `AnthropicModel` bearer token sent as `Authorization` instead of `x-api-key` when `auth_token=` is not given — what most gateways take; with a token, `ANTHROPIC_API_KEY` is not sent |
| `COHERE_API_KEY` | `CohereEmbeddings`, `CohereReranker` |
| `JINA_API_KEY` | `JinaReranker` |
| `VOYAGE_API_KEY` | `VoyageReranker` |
| `PINECONE_API_KEY` | `PineconeVectorStore` |
| `MONGODB_URI` | `MongoDBAtlasVectorStore` connection string when `connection_string=` is not given |

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
| `REDIS_URL` | Redis checkpointer — notebook 65; a bare `redis://` approval-store spec |
| `S3_BUCKET` | S3 checkpointer — notebook 50 |
| `POSTGRES_HOST` / `POSTGRES_PORT` / `POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD` | A bare `postgresql://` approval-store spec (`approval_store("postgresql://")`, `locus approvals --store postgresql://`) |
| `ORACLE_DSN` | Oracle Database 26ai — tnsnames alias or `host:port/service` — notebooks 37-45; a bare `oracle://` approval-store spec |
| `ORACLE_USER` / `ORACLE_PASSWORD` | Oracle Database 26ai credentials |
| `ORACLE_CONFIG_DIR` | Directory holding `tnsnames.ora` when `ORACLE_DSN` is an alias on a TLS-only database (no wallet); a bare `oracle://` spec reads it, else `TNS_ADMIN` — notebooks 37-45 |
| `ORACLE_WALLET` / `ORACLE_WALLET_PASSWORD` | Autonomous Database wallet directory and password for mutual TLS only (empty string for auto-login wallets); the wallet directory resolves aliases too |
| `S3_ENDPOINT_URL` | S3-compatible endpoint (MinIO, R2) — notebook 50 |

## Approvals

| Variable | Used by |
|---|---|
| `LOCUS_APPROVALS_STORE` | The store `locus approvals` reads when `--store` is not given: a `FileApprovals` path, or `memory://`, `file://`, `postgresql://`, `redis://`, `oracle://`; `approval_store()` reads it when called without a spec |
| `LOCUS_APPROVER` | The principal `locus approvals approve` / `reject` record when `--by` is not given |

## Evaluation

| Variable | Used by |
|---|---|
| `LOCUS_EVAL_STORE` | The report store `locus eval` and `report_store()` open when no spec is given — see [Evaluate in CI](evaluate-in-ci.md) |

## Serving

| Variable | Used by |
|---|---|
| `LOCUS_SERVER_API_KEY` | `AgentServer` bearer token — notebook 65 |
| `LOCUS_A2A_API_KEY` | `A2AServer.__init__` bearer token |

## Deployed images

The image `locus deploy` builds starts with `python -m locus.deploy.entrypoint`
and serves the scaffolded `server.py`; see [Deploy to OCI](deploy-oci.md).

| Variable | Used by |
|---|---|
| `LOCUS_MODEL` | Model string the scaffolded server builds its agent from (default `oci:openai.gpt-4.1`); `mock` for a first deployment that proves the path before a model is wired |
| `LOCUS_SERVER_TITLE` | Title of the scaffolded `AgentServer` (default `Locus agent`) |
| `LOCUS_SERVER_ALLOW_UNAUTHENTICATED` | `1` lets the scaffolded server start without `LOCUS_SERVER_API_KEY`; otherwise it refuses to start |
| `<VAR>_SECRET_OCID` | The entrypoint reads the OCI Vault secret and sets `<VAR>` from it; a `<VAR>` already set wins |
| `LOCUS_APM_DATA_KEY` / `OTEL_EXPORTER_OTLP_HEADERS` | With the key resolved and no OTLP headers set, the entrypoint sets the Application Performance Monitoring authorization header from it |

## LiteLLM gateway

Notebooks 66 and 67. See [the gateway how-to](litellm-gateway.md).

| Variable | Used by |
|---|---|
| `LITELLM_GATEWAY_URL` | Gateway base URL |
| `LITELLM_GATEWAY_KEY` | Virtual key the agent presents |
| `LITELLM_GATEWAY_MODEL` | Model id as the gateway names it |
| `LITELLM_MASTER_KEY` | Admin key — `/key/generate` and `/spend/*` (needs Postgres) |

## Tools and MCP servers

| Variable | Used by |
|---|---|
| `LOCUS_SANDBOX` | Sandbox provider for `@tool(sandbox=True)` or a `SandboxSpec` that names none: `subprocess` (alias `local`) unless set; `docker`, or a name from the `locus.sandbox_providers` entry-point group |
| `LOCUS_ALLOW_PRIVATE_URLS` | `is_safe_url()` / `validate_url()` in `locus.tools.url_safety` when called without `allow_private=`: `true` / `1` / `yes` admits private-network targets (cloud metadata endpoints stay blocked). The MCP client passes its own `allow_private_url` field, so it does not read this |
| `LOCUS_MCP_SKIP_OSV` | `1` disables the OSV malware check run before an MCP server is spawned with `npx`, `uvx`, `pipx`, `bunx` or `pnpm dlx` — for hosts with no network egress |
| `OSV_ENDPOINT` | The OSV query endpoint that check calls (default `https://api.osv.dev/v1/query`) |
| `XDG_CONFIG_HOME` | Where `FileTokenStore` keeps MCP OAuth tokens when given no path: `$XDG_CONFIG_HOME/locus/mcp-tokens`, else `~/.config/locus/mcp-tokens` |

## OCI tools and vendor adapters

| Variable | Used by |
|---|---|
| `LOCUS_USE_OCI_ALLOW_MUTATIONS` | `use_oci` write opt-in — notebook 36; unset, the tool refuses mutating calls |
| `SCANNER_API_KEY` | Vulnerability-scanner adapter — notebook 80 |
| `DATAMAP_URL` | Data-map adapter — notebook 80 |
| `DATAMAP_API_KEY` | Data-map adapter — notebook 80 |
| `DATAMAP_TOKEN` | Data-map adapter — notebook 80 |

## Observability

| Variable | Used by |
|---|---|
| `OCI_APM_DATA_UPLOAD_ENDPOINT` / `OCI_APM_PRIVATE_DATA_KEY` | `oci_apm()` when not given `data_upload_endpoint=` / `private_data_key=` — see [Observability](../concepts/observability.md) |

## Rogue-agent challenge

`python -m locus.rogue` picks its model from these; with none set and no
`ANTHROPIC_API_KEY` it runs offline on a model that arrives compromised.

| Variable | Used by |
|---|---|
| `LOCUS_MODEL_URL` | Base URL of a self-hosted OpenAI-compatible server (`/v1` is appended); `LOCUS_ADVISORY_URL` is the older name and is still read |
| `LOCUS_MODEL_NAME` | Model id on that server; unset, the first id its `/v1/models` lists |
| `LOCUS_MODEL_API_KEY` | Key sent to that server (default `unused`; self-hosted servers ignore it) |
