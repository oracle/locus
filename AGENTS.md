# Agent instructions

Locus (`locus-sdk`) is Oracle's Python agent harness: one `Agent` class,
eight multi-agent shapes, OCI Generative AI and Oracle Database 26ai first,
and an admission gate in front of every consequential action. These
instructions are for a coding agent working in this checkout. Two kinds of
task arrive here: building an application with Locus, driven by
`codex/PROMPT.md` and specified in `codex/SPEC.md`; and changing Locus
itself, governed by `CONTRIBUTING.md`. Read the one that applies before
running anything. The `locus` skill in `.agents/skills/locus/` is what a
coding agent reads to use the SDK in any project; in this checkout it is the
source of that skill, and a change to the SDK's public surface is reflected
there.

## Hard rules

- Credentials. Never ask for, print, log or store an API key, a token, a key
  file or a wallet. Read `~/.oci/config` only to list its section names.
  Anything that looks like a credential in an output is a stop condition,
  not something to repeat.
- Live model calls cost money and need a credential the operator chose. Make
  only the calls the task requires; never probe a provider with extra
  prompts; never use a profile or a key the operator did not name.
- Scope. For `codex/PROMPT.md`, write only under `out/` and `.venv/`; `src/`,
  `tests/`, `examples/`, `docs/` and `pyproject.toml` are read-only, and a
  problem found there is reported, not fixed. For a change to Locus, follow
  `CONTRIBUTING.md`.
- Nothing outside the checkout: no `sudo`, no edits to shell start-up files,
  `~/.oci` or `~/.codex`, no package installed outside the checkout's `.venv`.
- Nothing from the web that the task did not name. For `codex/PROMPT.md`
  the specification is self-contained; `pip` and the one model call are the
  only network traffic.
- Commands run from the repository root, in the foreground, with
  `.venv/bin/python`.

## Where things are

| Path | Purpose |
|---|---|
| `src/locus/` | the SDK; the module map is in `README.md` under "Repo layout" |
| `packages/locus-sdk-security/` | `locus-sdk-security`, the security findings package, released in lockstep |
| `examples/` | 89 numbered notebooks; each runs against OCI Generative AI with a profile, or against the bundled mock model (`examples/config.py`) without one |
| `docs/`, `mkdocs.yml`, `hooks/`, `overrides/` | the documentation site, published at [locusagents.ai](https://locusagents.ai/) |
| `tests/unit/`, `tests/integration/`, `tests/docs/` | unit tests (no credentials), integration tests (gated on credentials and services), site build tests |
| `codex/` | the specification and prompt for building a first agent with Codex |
| `.agents/skills/locus/` | the Codex skill for building with Locus (`SKILL.md` and `references/`); shipped in the wheel, placed by `locus codex init` |
| `examples/codex-on-oci/` | a loopback gateway that runs Codex itself on OCI Generative AI |
| `out/` | run output, git-ignored |

## Changing Locus

- Environment: `pip install -e ".[dev,sdk]"`, or `hatch env create`.
- Before a commit, `hatch run check` (ruff format check, ruff, mypy strict,
  the unit suite under its coverage gate), or the same by hand:
  `ruff format --check src tests examples packages`,
  `ruff check src tests examples packages`,
  `mypy src/locus packages/locus-sdk-security/src/locus_security`,
  `pytest tests/unit packages/locus-sdk-security/tests/unit`.
- Documentation: `mkdocs build --strict` must pass; every code block on a
  page must run as written; the prose carries no dates, account or tenancy
  names, test counts or claims of having been verified; identifiers in prose
  are the ones in the code.
- Notebooks under `examples/` must run offline against the mock model and
  follow the authoring rules in `CONTRIBUTING.md`.
- Commits follow Conventional Commits and carry a DCO sign-off
  (`git commit -s`); a pull request needs the Oracle Contributor Agreement.
