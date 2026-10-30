# Codex on Locus

[OpenAI Codex](https://developers.openai.com/codex) builds applications with
Locus. This directory and three others in the repository make that work, and
this page is the map. The full guide is
[Build it with Codex](https://locusagents.ai/how-to/codex/) on the site.

| You want to | Use | Where |
|---|---|---|
| Codex to build or change something with Locus in your own project | `locus codex run "…"` (the skill is placed for you) | [`../.agents/skills/locus/`](../.agents/skills/locus/) |
| Codex itself to run on OCI Generative AI, no other account | the gateway, then `locus codex run --codex-profile oci "…"` | [`../examples/codex-on-oci/`](../examples/codex-on-oci/) |
| proof that Codex builds correct Locus applications from the skill | `python codex/scenarios/run.py` | [`scenarios/`](scenarios/) |
| to watch Codex do the Quickstart and the first controlled action in a clone | `codex "$(cat codex/PROMPT.md)"` | [`SPEC.md`](SPEC.md), [`PROMPT.md`](PROMPT.md) |

## The skill

`.agents/skills/locus/` is what Codex reads to use the SDK: `SKILL.md` gives
it the shape of Locus (install, the `Agent` and its tools, the admission gate,
holding for a person and resuming, typed answers, the multi-agent shapes,
memory and RAG, the coding harness, offline tests, the command line, known
behaviour, the rules) and five reference files carry every option with
runnable code. Codex discovers a skill at `.agents/skills/<name>/SKILL.md` in
the project it works in and in `~/.agents/skills`, picks it up when a task
matches its description, and loads it when you write `$locus`. The wheel
carries the skill, and `locus codex` places it:

```shell
pip install "locus-sdk[oci]"
locus codex init                 # .agents/skills/locus here, and a Locus section in AGENTS.md
locus codex init --home          # once for every project, in ~/.agents/skills/locus
locus codex run "Build a support agent with a gated refund tool, with tests"
locus codex path                 # where the bundled copy is
```

`locus codex run` installs the skill where it is missing and starts Codex
with the two sandbox settings below; `--exec` is non-interactive,
`--codex-profile NAME` selects a Codex profile, and anything after `--` goes
to Codex. `tests/docs/test_codex_skill.py` checks every Python snippet in the
skill against the installed SDK and measures the skill against each package's
public names, so it cannot drift from the API it describes.

## Codex on OCI Generative AI

Codex speaks the Responses API and sends a bearer token; OCI's
OpenAI-compatible endpoint serves chat completions and needs an OCI request
signature. `examples/codex-on-oci/` sits between the two on loopback:

```shell
cd examples/codex-on-oci
python3 -m venv .venv && .venv/bin/pip install "litellm[proxy]" oci
export LITELLM_MASTER_KEY="sk-$(openssl rand -hex 32)"
OCI_PROFILE=DEFAULT OCI_REGION=us-chicago-1 \
  .venv/bin/litellm --config config.yaml --host 127.0.0.1 --port 4000
```

Then, in a shell with the same `LITELLM_MASTER_KEY`, copy `oci.config.toml`
to `~/.codex/` and start Codex with `locus codex run --codex-profile oci "…"`
(or `codex -p oci`). Nothing from `~/.oci` leaves the machine; the agent Codex
writes calls OCI Generative AI directly with `model="oci:<id>"`, as every
Locus program does. The example's README has the details.

## The scenarios

`scenarios/` is a series of twelve tasks Codex builds from the skill alone,
each a `PROMPT.md` and a `check.sh` that runs in the project Codex built: a
gated refund with its audit trail, a hold a person approves and a resumed
run, a `StateGraph` fan-out, a retrieval tool, the coding harness behind the
gate, a checkpointed conversation, an agent served over HTTP, an orchestrator
with specialists, a typed answer with a termination rule and an evaluation,
an agent on OCI Generative AI, the permission rules of a coding agent, and a
playbook with a custom hook.

```shell
python codex/scenarios/run.py --skip-live                       # the offline series
python codex/scenarios/run.py --codex-profile oci               # Codex itself on OCI
python codex/scenarios/run.py --locus-install "$PWD[oci]"       # Locus from this checkout
```

The runner gives each scenario an empty directory, starts
`locus codex run --exec` with the task, runs the checks when Codex returns,
and prints a summary table; transcripts and reports land under
`out/scenarios/`. This is how a change to the SDK or to the skill is tried
the way an application would use it. [`scenarios/README.md`](scenarios/README.md)
describes each scenario and how to add one.

## A first agent in a clone

`SPEC.md` is a specification, written for a person to follow by hand as
well, of a first Locus agent with a typed tool, a controlled action that the
admission gate allows, holds and denies, and the rogue-agent challenge, each
step with a checkpoint. `PROMPT.md` drives Codex through it. From a clone:

```shell
git clone https://github.com/oracle/locus.git
cd locus
codex -s workspace-write -c sandbox_workspace_write.network_access=true "$(cat codex/PROMPT.md)"
```

Codex checks the machine, confirms the provider (OCI Generative AI when
`~/.oci/config` exists, otherwise OpenAI or Anthropic from the key in the
environment, otherwise offline), creates `.venv`, installs Locus from the
clone, writes and runs the first agent and the controlled action under
`out/first-agent/`, runs the rogue-agent challenge, and ends with a report.
If a check fails that only you can fix, such as an expired OCI session, it
tells you what to run in your own terminal and waits for `done`.

Without a conversation, put your answers in an operator-context block and
pipe both into a non-interactive session:

```shell
cat > operator-context.md <<'EOT'
Operator context. PROVIDER=oci, OCI_PROFILE=DEFAULT, OCI_REGION=us-chicago-1.
Use the default model. If a check fails, report it and stop.
EOT
{ cat operator-context.md; echo; cat codex/PROMPT.md; } | codex exec \
  -s workspace-write -c sandbox_workspace_write.network_access=true -
```

`PROVIDER=offline` runs the controlled action and the rogue-agent challenge
with no credential and skips the live call. `AGENTS.md` at the repository
root is loaded automatically and carries the rules the agent follows here.

## The Codex sandbox and credentials

`-s workspace-write` lets Codex write inside the project (`.venv`, its
files, `out/` in a clone) and nowhere else.
`-c sandbox_workspace_write.network_access=true` lets `pip` reach PyPI and
the agent reach the model; without it the sandbox has no network.
`locus codex run` passes both; `[sandbox_workspace_write] network_access = true`
in `~/.codex/config.toml` makes the second permanent. `codex exec` starts
read-only, so both apply there too.

On OCI no secret enters the environment: Locus reads the profile you name
from `~/.oci/config`, and the skill tells the agent to list that file's
section names and read nothing else from it. For OpenAI or Anthropic, export
the key in the shell that starts Codex; Codex passes its environment to the
commands it runs, and the skill forbids any command whose output would show
it.

## What the agent never does

It never sees or shows a credential, installs nothing outside the project's
virtual environment, writes nothing outside the project, and makes live
model calls only where the task needs the answer. In a clone it never
modifies anything git tracks and fetches nothing from the web: the
specification is self-contained; `pip` and the model call are the only
network traffic.
