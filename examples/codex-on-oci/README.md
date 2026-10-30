# Codex on OCI Generative AI

OpenAI Codex can run with OCI Generative AI as its own model, so that Codex
and the Locus agent it builds both stay inside Oracle Cloud. Codex speaks the
Responses API and authenticates with a bearer token; OCI's OpenAI-compatible
endpoint serves chat completions and authenticates with an OCI request
signature. This directory is the piece between the two: a LiteLLM gateway on
loopback that translates Responses to chat completions and signs every call
with an OCI CLI profile. Nothing from `~/.oci` leaves the machine, and Codex
never holds an OCI credential.

| File | What it is |
|---|---|
| [`config.yaml`](config.yaml) | the gateway: one wildcard route, so any model id OCI serves in the region works (`openai.gpt-5.5` becomes `oci/openai.gpt-5.5`), and the signer callback |
| [`oci_signer.py`](oci_signer.py) | the callback: builds an OCI signer from the profile named by `OCI_PROFILE` (API key or session token, re-read once a minute), fills `OCI_COMPARTMENT_ID` from the profile's tenancy when unset, and drops the `reasoning_effort` Codex sends with tool calls, which OCI's chat API refuses for the GPT-5 family |
| [`oci.config.toml`](oci.config.toml) | the Codex profile: the provider entry for the gateway and the default model; copy it to `~/.codex/` and select it with `codex -p oci` |

## Run it

```bash
cd examples/codex-on-oci
python3 -m venv .venv && .venv/bin/pip install "litellm[proxy]" oci

export LITELLM_MASTER_KEY="sk-$(openssl rand -hex 32)"     # the gateway's own key
OCI_PROFILE=DEFAULT OCI_REGION=us-chicago-1 \
  .venv/bin/litellm --config config.yaml --host 127.0.0.1 --port 4000
```

`OCI_PROFILE` is a `[section]` of `~/.oci/config` (`OCI_CONFIG_FILE` names
another file); `OCI_REGION` is a region that serves Generative AI;
`OCI_COMPARTMENT_ID` is optional and defaults to the profile's tenancy. The
gateway refuses to start without `LITELLM_MASTER_KEY`; Codex sends that value
as its bearer token, so export it in the shell that starts Codex too.

Then, in a second shell with the same `LITELLM_MASTER_KEY`:

```bash
cp oci.config.toml ~/.codex/
codex -p oci "Build a support agent with an order lookup and a gated refund tool, with tests"
```

or, with the Locus skill installed and the sandbox settings it needs:

```bash
locus codex run --codex-profile oci "Build a support agent with an order lookup and a gated refund tool, with tests"
```

Check the gateway by hand with the Responses API Codex uses:

```bash
curl -sS http://127.0.0.1:4000/v1/responses \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"model":"openai.gpt-5.5","input":"Reply with the single word OK."}'
```

`-m <id>` on the Codex command line picks another model the region serves;
the `model` line in `oci.config.toml` sets the default.

## How it fits

```text
codex ──Responses API, bearer LITELLM_MASTER_KEY──▶ 127.0.0.1:4000 (LiteLLM)
                                                      │ chat completions, OCI request signature
                                                      │ from the profile in ~/.oci/config
                                                      ▼
                             inference.generativeai.<region>.oci.oraclecloud.com
```

The Locus agent that Codex writes does not go through the gateway: it calls
OCI Generative AI directly with `model="oci:<id>"` and the same profile, as
every Locus program does.

## Scope

Loopback only, one process, no persistence: a developer machine or a CI box,
not a shared host. For a shared gateway, start from
[`examples/litellm-gateway/`](../litellm-gateway/) (Docker Compose and Helm,
virtual keys, budgets) and add this directory's `oci_signer.py` and wildcard
route to it; the LiteLLM image needs the `oci` package installed beside it.
