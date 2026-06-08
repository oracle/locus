# Skills

A Skill (the AgentSkills.io shape) bundles a name, a description, and
a block of instructions. The `SkillsPlugin` exposes a catalog of
skills to the agent and only injects the full instructions once the
agent activates a specific one. Progressive disclosure keeps the
system prompt small and the agent focused.

- `Skill` — built in code or loaded from a `SKILL.md` file with YAML
  front-matter.
- `Skill.from_directory(path)` — load every skill under a directory.
- `Agent(skills=[...])` — wires up the SkillsPlugin and the `skills`
  tool that the agent calls to activate one.
- The `SKILL.md` format itself — front-matter for `name`,
  `description`, `allowed-tools`, `metadata`, plus the instruction
  body and optional `scripts/`, `references/`, `assets/` directories.

## Run it

The bundled mock model is the default; set `TULIP_MODEL_PROVIDER` for a live provider:

```bash
TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_48_skills.py
```

Offline:

```bash
TULIP_MODEL_PROVIDER=mock python examples/notebook_48_skills.py
```

## Prerequisites

- An OpenAI or Anthropic API key, or `TULIP_MODEL_PROVIDER` set to
  `openai` / `anthropic` / `mock`.
- Optional: an `examples/skills/` directory with one or more
  `SKILL.md` files for Part 2 to find.

## Source

```python
--8<-- "examples/notebook_48_skills.py"
```
