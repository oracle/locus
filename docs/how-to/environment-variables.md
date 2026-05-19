# Environment variables — `OCI_*` vs `LOCUS_OCI_*`

Locus reads two families of environment variables to configure the OCI
GenAI transport. Both are valid, and the notebook harness in
`examples/config.py` reads them with a consistent fallback chain.

## The rule

| Look for first      | …then fall back to | …then use |
|---------------------|--------------------|-----------|
| `LOCUS_OCI_PROFILE` | `OCI_PROFILE`       | `DEFAULT` |
| `LOCUS_OCI_REGION`  | `OCI_REGION`        | `us-chicago-1` |
| `LOCUS_OCI_COMPARTMENT` | `OCI_COMPARTMENT` | *(profile-derived)* |
| `LOCUS_OCI_AUTH_TYPE`   | `OCI_AUTH_TYPE`   | `api_key` |
| `LOCUS_OCI_ENDPOINT`    | `OCI_ENDPOINT`    | *(region-derived)* |
| `LOCUS_OCI_TRANSPORT`   | `OCI_TRANSPORT`   | *(auto from model id)* |

The same rule applies to every `LOCUS_OCI_<name>` lookup the notebook
harness performs.

## Why two names?

- **`OCI_*`** is the OCI CLI / SDK standard. If you already typed
  `oci session authenticate --profile-name DEFAULT` and exported the
  resulting variables, the Locus notebooks pick them up unchanged.
- **`LOCUS_OCI_*`** is the namespaced form the notebook harness reads
  first. Useful when you want a Locus notebook to use a *different*
  profile or region from your shell-default OCI configuration —
  point `LOCUS_OCI_PROFILE` somewhere else without touching
  `OCI_PROFILE`.

## Minimum set for a notebook run

```bash
# Pick a model + provider.
export LOCUS_MODEL_PROVIDER=oci
export LOCUS_MODEL_ID=openai.gpt-4o-mini

# OCI auth. Set EITHER LOCUS_OCI_PROFILE OR OCI_PROFILE — the harness
# tries the LOCUS_ one first, then the bare one.
export OCI_PROFILE=DEFAULT
export OCI_REGION=us-chicago-1

# Run any notebook:
python examples/notebook_14_basic_agent.py
```

When the SDK itself is constructed directly (not via the notebook
harness), pass profile / region as constructor arguments — the
`LOCUS_OCI_*` fallback chain is a notebook-harness convenience, not
a runtime contract.

## Other env vars

| Variable             | Used by                       |
|----------------------|-------------------------------|
| `LOCUS_MODEL_PROVIDER` | `examples/config.py:get_model` |
| `LOCUS_MODEL_ID`       | model factory dispatch         |
| `LOCUS_MODEL_ID_B`     | secondary "model B" slot       |
| `LOCUS_MODEL_ID_C`     | tertiary "model C" slot        |
| `LOCUS_A2A_API_KEY`    | `A2AServer.__init__` bearer    |
| `OPENAI_API_KEY`       | `OpenAIModel`                  |
| `ANTHROPIC_API_KEY`    | `AnthropicModel`               |
