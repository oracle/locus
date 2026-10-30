# Build it with Codex

This directory lets an AI coding agent build and verify a first Locus agent
in a clone of this repository, so that the agent, not you, learns the setup.
The specification is written for a person to follow by hand as well.

| File | What it is |
|---|---|
| [`SPEC.md`](SPEC.md) | the specification: inputs, preflight, every step with its checkpoint, acceptance, the rules the agent follows, known behaviour, the report |
| [`PROMPT.md`](PROMPT.md) | the prompt that drives OpenAI Codex through the specification; plain text, pass it as the argument or pipe it |
| [`../AGENTS.md`](../AGENTS.md) | what Codex loads automatically in this repository: the credential rule, what the agent may write, where things are |
| [`../.agents/skills/locus/`](../.agents/skills/locus/) | the Codex skill for building with Locus in any project: `SKILL.md` and its references; `locus codex init` places it |
| [`../examples/codex-on-oci/`](../examples/codex-on-oci/) | a loopback gateway so that Codex itself runs on OCI Generative AI |
| [`scenarios/`](scenarios/) | a series of tasks Codex builds from the skill, each with an automatic check, and the runner (`python codex/scenarios/run.py`) |

## Use it

Install [Codex](https://developers.openai.com/codex)
(`npm install -g @openai/codex`) and sign in. Then, from a clone:

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

On OCI no secret enters the environment: Locus reads the profile you name
from `~/.oci/config`. For OpenAI or Anthropic, export the key in the shell
that starts Codex; Codex passes its environment to the commands it runs, and
the specification forbids any command whose output would show it.

## In your own project

Outside a clone, Locus is a package and Codex learns it from the skill:

```shell
pip install "locus-sdk[oci]"
locus codex init          # .agents/skills/locus here, and a Locus section in AGENTS.md
locus codex run "Build a support agent with an order lookup and a gated refund tool, with tests"
```

`locus codex init --home` installs the skill once for every project, in
`~/.agents/skills/locus`. `locus codex run` adds the two sandbox settings
below and passes anything after `--` to Codex.

## The Codex sandbox

`-s workspace-write` lets Codex write inside the clone (`.venv`, `out/`) and
nowhere else. `-c sandbox_workspace_write.network_access=true` lets `pip`
reach PyPI and the first agent reach the model; without it the sandbox has no
network. `[sandbox_workspace_write] network_access = true` in
`~/.codex/config.toml` makes the second permanent. `codex exec` starts
read-only, so both flags apply there too.

## Without a conversation

Put your answers in an operator-context block and pipe both into a
non-interactive session:

```shell
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

## What the agent never does

It never sees or shows a credential, never modifies anything git tracks in
the clone, installs nothing outside `.venv`, fetches nothing from the web
(the specification is self-contained; `pip` and the model call are the only
network traffic), and makes one live model call, the first agent's, with at
most one retry.
