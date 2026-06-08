# Skills

AgentSkills.io-compatible packaged instruction bundles (`SKILL.md`
files) that agents load on demand via progressive disclosure:

- **L1:** the skill catalog (names + descriptions) appears in the
  system prompt.
- **L2:** the agent activates a skill — full instructions are loaded.
- **L3:** the agent reads supporting resource files (`scripts/`,
  `references/`, `assets/`).

Attach via `AgentConfig.skills=[skill_or_path, ...]`.

::: tulip.skills.models.Skill
::: tulip.skills.plugin.SkillsPlugin
