---
title: Build it with Codex
description: OpenAI Codex builds on Locus. A skill teaches it the SDK, `locus codex` starts Codex with that skill and the sandbox settings it needs, a specification walks it through a first agent in a clone, and a gateway lets Codex itself run on OCI Generative AI.
---

# Build it with Codex

[OpenAI Codex](https://developers.openai.com/codex) builds on Locus. Three
pieces make that work, and this page covers each:

- **The Locus skill.** [`.agents/skills/locus/`](https://github.com/oracle/locus/tree/HEAD/.agents/skills/locus)
  is what Codex reads to use the SDK in any project: a `SKILL.md` with the
  shape of Locus and the rules, and four reference files with every option.
- **`locus codex`.** The command puts the skill where Codex looks and starts
  Codex with the sandbox settings the skill's recipes need.
- **Codex on OCI Generative AI.** [`examples/codex-on-oci/`](https://github.com/oracle/locus/tree/HEAD/examples/codex-on-oci)
  runs Codex's own model on OCI, so neither Codex nor the agent it builds
  needs an account outside Oracle Cloud.

In a clone of the repository there is a fourth: [`codex/`](https://github.com/oracle/locus/tree/HEAD/codex)
holds a specification and the prompt that drives Codex through a first agent,
a controlled action and the rogue-agent challenge, each step with a
checkpoint a person can follow by hand.

## Requirements

<dl class="locus-facts" markdown="block">
<div markdown="block"><dt>Codex</dt><dd>The Codex CLI (<code>npm install -g @openai/codex</code>), signed in or pointed at OCI below</dd></div>
<div markdown="block"><dt>Python</dt><dd>3.11–3.14</dd></div>
<div markdown="block"><dt>Locus</dt><dd><code>pip install "locus-sdk[oci]"</code>, or the extra for your provider</dd></div>
<div markdown="block"><dt>Provider</dt><dd>OCI Generative AI, OpenAI, Anthropic, or none</dd></div>
<div markdown="block"><dt>Credential</dt><dd>An OCI profile, a key in the environment, or none</dd></div>
</dl>

## In your project

```bash
pip install "locus-sdk[oci]"
locus codex run "Build a support agent with an order lookup and a gated refund tool, with tests"
```

`locus codex run` installs the skill into `.agents/skills/locus` of the
current directory, unless `~/.agents/skills/locus` already holds it, adds a
Locus section to `AGENTS.md`, and starts Codex with `-s workspace-write`
and `-c sandbox_workspace_write.network_access=true`: writes inside the
project, network for `pip` and the model. Without a prompt it opens an
interactive session. `--exec` runs non-interactively, `--codex-profile NAME`
selects a Codex profile such as the OCI one below, and anything after `--`
goes to Codex.

To place the skill without starting Codex:

| Command | Effect |
|---|---|
| `locus codex init` | `.agents/skills/locus` in the current project and a Locus section in its `AGENTS.md` |
| `locus codex init DIR` | the same, in `DIR` |
| `locus codex init --home` | `~/.agents/skills/locus`, for every project on the machine |
| `locus codex path` | where the bundled copy is, for a copy or a symlink by hand |

Codex picks the skill up when a task matches its description, or when you
name it: `$locus` in the prompt. A second `init` writes nothing a file does
not need; files of yours in the skill directory are kept.

## What the skill teaches

`SKILL.md` is the spine: what Locus is, the working procedure (confirm the
install, find the provider from the environment, read the reference, write
the agent and an offline test, gate every side effect, never show a
credential), the install matrix, a first agent, the gate with
`gate_tool()` and `admit()`, holding for a person and resuming, typed
answers and termination, how to pick a multi-agent shape, memory and RAG,
the coding harness, offline tests with the model doubles, the command line,
known behaviour, and the rules. The references carry the depth:

| Reference | Covers |
|---|---|
| `agent-and-tools.md` | `Agent` and `AgentConfig` options, `@tool` and `ToolContext`, structured output, streaming, hooks, termination, reflexion and grounding, decisions, skills, playbooks, evaluation, MCP, the testing doubles |
| `control-layer.md` | `ControlPolicy`, `Action`, `admit`, `gate_tool`, approvals and authorities, interrupt and resume, `AuditTrail` and signed exports, permission rules, shell parsing, spend ledgers, the governed wrapper |
| `multi-agent-harness-deploy.md` | each shape with code, `StateGraph` primitives, A2A, DeepAgent, the harness and its backends, the box runner, `AgentServer`, the command line, observability |
| `providers-memory-rag.md` | every provider prefix and its environment, OCI transports and errors, checkpointers and backends, memory managers, Oracle Database 26ai, embeddings, vector stores, rerankers |

Every Python snippet in the skill is checked against the installed SDK by
the repository's tests, the same check the documentation pages get, so the
skill cannot drift from the API it describes.

## A first agent in a clone

```bash
git clone https://github.com/oracle/locus.git
cd locus
codex -s workspace-write -c sandbox_workspace_write.network_access=true "$(cat codex/PROMPT.md)"
```

Codex then:

1. runs the preflight of the specification and shows you the table: Python,
   the clone, write access, the network, and the provider's prerequisite;
2. confirms the provider: OCI Generative AI when `~/.oci/config` exists,
   otherwise OpenAI or Anthropic from the key in the environment, otherwise
   offline; and asks, in one message, for anything it could not detect, such
   as which OCI profile to use;
3. creates `.venv`, installs Locus from the clone with the provider's extra,
   and writes `first_agent.py` and `controlled_refund.py` under
   `out/first-agent/`, the files from the [Quickstart](quickstart.md) and the
   [first controlled action](first-controlled-action.md);
4. runs the first agent (one live call), the controlled action and the
   rogue-agent challenge, checking each against the output the specification
   expects;
5. ends with a report: provider, versions, the three results, and where the
   files are.

If a check fails that only you can fix, such as an expired OCI session, Codex
tells you what to run in your own terminal and waits for `done`.

| File | Role |
|---|---|
| `AGENTS.md` | Loaded automatically from the repository root: the credential rule, what the agent may write, where things are |
| `.agents/skills/locus/` | The skill, discovered from the repository root like in any project |
| `codex/SPEC.md` | The specification: inputs, preflight, every step with its checkpoint, acceptance, known behaviour, the report |
| `codex/PROMPT.md` | The prompt: read the files above, run the steps, report |

Without a conversation, put your answers in an operator-context block and
pipe both into a non-interactive session:

```bash
cat > operator-context.md <<'EOT'
Operator context. PROVIDER=oci, OCI_PROFILE=DEFAULT, OCI_REGION=us-chicago-1.
Use the default model. If a check fails, report it and stop.
EOT
{ cat operator-context.md; echo; cat codex/PROMPT.md; } | codex exec \
  -s workspace-write -c sandbox_workspace_write.network_access=true -
```

`PROVIDER=offline` runs the controlled action and the rogue-agent challenge
with no credential and skips the live call. A non-interactive session cannot
answer approval requests, so the run has to fit inside the sandbox; the flags
above are enough for every command the specification issues.

## Credentials

On OCI no secret enters the environment. Locus reads the profile you name
from `~/.oci/config`, the OCI SDK signs each request with the key file that
profile points to, and the skill tells the agent to list the file's section
names and read nothing else from it. For OpenAI or Anthropic, export the key
in the shell that starts Codex: Codex passes its environment to the commands
it runs, which is how a Locus program finds `OPENAI_API_KEY`, and the skill
forbids any command whose output would show it.

## The Codex sandbox

`-s workspace-write` lets Codex write inside the project, which is where
`.venv` and the files it creates go, and nowhere else.
`-c sandbox_workspace_write.network_access=true` lets `pip` reach PyPI and
the agent reach the model; without it the sandbox has no network.
`locus codex run` passes both. To make the network setting permanent:

```toml
# ~/.codex/config.toml
[sandbox_workspace_write]
network_access = true
```

`codex exec` starts read-only, so the same two flags apply there too.

## Codex on OCI Generative AI

Codex speaks the Responses API and authenticates with a bearer token. OCI's
OpenAI-compatible endpoint serves chat completions and authenticates with an
OCI request signature. `examples/codex-on-oci/` is the piece between the
two: a LiteLLM gateway on loopback that translates Responses to chat
completions and signs every call with an OCI CLI profile. Nothing from
`~/.oci` leaves the machine, and Codex never holds an OCI credential.

```bash
cd examples/codex-on-oci
python3 -m venv .venv && .venv/bin/pip install "litellm[proxy]" oci

export LITELLM_MASTER_KEY="sk-$(openssl rand -hex 32)"     # the gateway's own key
OCI_PROFILE=DEFAULT OCI_REGION=us-chicago-1 \
  .venv/bin/litellm --config config.yaml --host 127.0.0.1 --port 4000
```

In a second shell with the same `LITELLM_MASTER_KEY` exported, install the
Codex profile and use it:

```bash
cp examples/codex-on-oci/oci.config.toml ~/.codex/
locus codex run --codex-profile oci "Build a support agent with a gated refund tool, with tests"
```

`codex -p oci` is the same without the Locus command. The profile sets
`openai.gpt-5.5` as the model; `-m` picks any other chat model id the region
serves, because the gateway's one route is a wildcard. The agent Codex writes
does not go through the gateway: it calls OCI Generative AI directly with
`model="oci:<id>"` and the same profile, as every Locus program does. The
example's README has the files, a `curl` to check the gateway, and what to
change for a shared deployment.

## What the agent never does

It never sees or shows a credential, installs nothing outside the project's
virtual environment, writes nothing outside the project, and makes live
model calls only where the task needs the answer. In a clone it never
modifies anything git tracks and fetches nothing from the web: the
specification is self-contained; `pip` and the model call are the only
network traffic.

## Next steps

- [Configure a model provider](configure-providers.md): the ids and the
  environment for every provider.
- [First controlled action](first-controlled-action.md): what the gate
  proves and what it does not.
- [Control-layer architecture](../concepts/control-layer.md): the
  enforcement boundary.
- [LiteLLM gateway](litellm-gateway.md): the same gateway in front of other
  providers, with virtual keys and budgets.
