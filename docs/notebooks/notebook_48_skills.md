# Skills

A Skill (the AgentSkills.io shape) bundles a name, a description and a
block of instructions — a vetted operations procedure such as dispute
triage or incident triage. The `SkillsPlugin` exposes a catalog of
skills to the agent and delivers the full instructions only once the
agent activates a specific one. Progressive disclosure keeps the system
prompt small and the agent focused on the procedure the request needs.

Part 1 builds a payments procedure in code; Part 2 loads the bundled
operations library (code review, incident triage, indicator-of-compromise
enrichment, log-search queries), so one catalog can serve several teams;
Part 3 writes a `SKILL.md` to disk and reads it back through the same
loader.

What this notebook covers:

- `Skill` — built in code or loaded from a `SKILL.md` file with YAML
  front matter.
- `Skill.from_directory(path)` — load every skill under a directory.
- `Skill.from_file(path)` — load one skill directory or `SKILL.md`.
- `Agent(skills=[...])` — installs the SkillsPlugin and registers the
  `skills` tool that the agent calls to activate one.
- The `SKILL.md` format itself — front matter for `name`,
  `description`, `allowed-tools`, `metadata`, plus the instruction
  body and optional `scripts/`, `references/`, `assets/` directories.

## Prerequisites

- An OCI profile in `~/.oci/config` (picked up automatically), or
  `LOCUS_MODEL_PROVIDER` set to `openai` / `anthropic` / `mock` with the
  matching key.
- The bundled `examples/skills/` directory (`code-review`,
  `incident-triage`, `ioc-enrichment`, `siem-query`) for Part 2 to load.

The drawing follows a skill folder such as Part 3's `bin-enrichment` from disk through the three disclosure levels this notebook exercises:

- the catalog of names, descriptions and, for a skill loaded from disk, locations that every request carries (the four procedures Part 2 loads);
- the `SKILL.md` body that enters the conversation only when the agent calls the `skills` tool (the activation Parts 1 and 2 count);
- and the listing of `scripts/`, `references/` and `assets/` files that is named but not read.

Of the `SKILL.md` keys the drawing lists, `compatibility` is an optional one that the plugin appends to the activation footer when set; `metadata`, which Part 3 prints, is parsed onto the `Skill` object but never sent to the model, so it has no row:

{{ locus_diagram skills-disclosure }}

## Key lines

A `Skill` built in code from a name, a one-line description and the
triage checklist as its instructions; `AgentConfig(skills=[...])`
installs the plugin and the `skills` tool, and the run counts how many
times the agent activated one:

{{ locus_source examples/notebook_48_skills.py::example_programmatic }}

`Skill.from_directory` loads each subdirectory of `examples/skills/`
that holds a `SKILL.md`; the on-call operations agent receives the whole
catalog up front but the full instructions of only the skill it
activates:

{{ locus_source examples/notebook_48_skills.py::example_filesystem }}

Part 3 writes this `SKILL.md` — the `SKILL_MD` constant in the full
source below — into a temporary `bin-enrichment/` folder (BIN and PAN
are a card's bank identification number and primary account number):
YAML front matter between `---` lines, then the Markdown body that
becomes the instructions:

```markdown
---
name: bin-enrichment
description: Use when the user provides a card indicator (BIN, PAN prefix, issuer) to enrich.
allowed-tools: lookup_bin fetch_issuer
metadata:
  author: payments-team
  version: "1.0"
---

# Instructions for the Agent

1. First, identify the indicator type
2. Then, enrich it with the allowed tools
3. Finally, summarize the verdict with evidence

## Resource Files
Place additional files in:
- scripts/   — executable code
- references/ — documentation
- assets/    — templates, data
```

`Skill.from_file` reads the folder back, with a file under `references/`
beside the `SKILL.md`: the front matter becomes typed fields
(`allowed-tools` is split on whitespace into `allowed_tools`), the body
becomes the instructions returned on activation, and the resources are
listed by relative path:

{{ locus_source examples/notebook_48_skills.py::example_format }}

## Run it

```bash
python examples/notebook_48_skills.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

Offline, with the bundled mock model:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_48_skills.py
```

## See also

- [Notebook 47 — Plugins](notebook_47_plugins.md) — the `Plugin` base class that `SkillsPlugin` extends
- [Concepts: Skills](../concepts/skills.md) — the three disclosure levels and when to reach for a skill
- [API: Skills](../api/skills.md)

## Source

??? note "Full source — `examples/notebook_48_skills.py`"

    ````python
    --8<-- "examples/notebook_48_skills.py"
    ````
