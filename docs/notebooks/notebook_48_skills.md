# Skills

A Skill (the AgentSkills.io shape) bundles a name, a description, and
a block of instructions — a vetted payments-ops procedure such as
dispute triage or refund-policy review. The `SkillsPlugin` exposes a
catalog of skills to the agent and only injects the full instructions
once the agent activates a specific one. Progressive disclosure keeps
the system prompt small and the payments agent focused.

- `Skill` — built in code or loaded from a `SKILL.md` file with YAML
  front-matter.
- `Skill.from_directory(path)` — load every skill under a directory.
- `Agent(skills=[...])` — wires up the SkillsPlugin and the `skills`
  tool that the agent calls to activate one.
- The `SKILL.md` format itself — front-matter for `name`,
  `description`, `allowed-tools`, `metadata`, plus the instruction
  body and optional `scripts/`, `references/`, `assets/` directories.

## Run it

OCI Generative AI with an OCI profile, otherwise the bundled mock model; set `LOCUS_MODEL_PROVIDER` to force a provider:

```bash
LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_48_skills.py
```

Offline:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_48_skills.py
```

## Prerequisites

- An OCI profile in `~/.oci/config` (picked up automatically), or
  `LOCUS_MODEL_PROVIDER` set to `openai` / `anthropic` / `mock` with the
  matching key.
- The bundled `examples/skills/` directory (`code-review`,
  `incident-triage`, `ioc-enrichment`, `siem-query`) for Part 2 to load.

## Source

````python
--8<-- "examples/notebook_48_skills.py"
````
