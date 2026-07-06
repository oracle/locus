# Contributing

Tulip is open to everyone. The bar everywhere: *a result ships on evidence or
it abstains; a side effect runs only gated and audited.* New probes prove their
success with an observable effect, and new actions go through `admit()`.

This document covers how to set up a development environment, the
review and sign-off process, the coding standards we hold the codebase
to, and how to verify your change against the workbench end-to-end
before opening a pull request.

Quick links for the most common drops:

- **Adding a notebook?** Jump to [Notebook authoring](#notebook-authoring).
- **Touching the README?** Jump to [README updates](#readme-updates).
- **Adding a model provider?** Jump to [Notebook authoring](#notebook-authoring) for the multi-model demo conventions, then [Coding Standards](#coding-standards) for the `ModelProtocol` interface.
- **Verifying against real OpenAI / Anthropic?** Jump to [Workbench end-to-end sweeps](#workbench-end-to-end-sweeps).

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Notebook authoring](#notebook-authoring)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Release checklist](#release-checklist)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold this code.

## Getting Started

### Developer Certificate of Origin (DCO)

Contributions are accepted under the [Developer Certificate of Origin](https://developercertificate.org/). You certify the DCO by signing off every commit — no separate agreement to file.

All commits must include a sign-off line:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Use `git commit --signoff` or `git commit -s` to add this automatically.

### Types of Contributions

We welcome:

- **Bug fixes** - Fix issues and improve stability
- **Features** - New capabilities aligned with the roadmap
- **Documentation** - Notebooks, examples, API docs
- **Tests** - Unit tests, integration tests, benchmarks
- **Performance** - Optimizations and efficiency improvements

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [Hatch](https://hatch.pypa.io/) for project management
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/tuliplabs-ai/sdk-python.git
cd tulip-agents

# Install Hatch if needed
pip install hatch

# Create development environment
hatch env create

# Install pre-commit hooks
hatch run pre-commit install

# Verify setup
hatch run test
```

### Environment Variables

For integration tests, configure these environment variables:

```bash
# OpenAI (required for most integration tests)
export OPENAI_API_KEY="sk-..."

# Cohere (optional, for the Cohere reranker / embeddings tests)
export COHERE_API_KEY="..."

# Anthropic (optional, for Anthropic provider tests)
export ANTHROPIC_API_KEY="sk-ant-..."

# Docker services (for checkpoint/vector store tests)
export REDIS_URL="redis://localhost:6379"
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="postgres"
export POSTGRES_DB="tulip"
```

## Notebook authoring

`examples/notebook_NN_*.py` is the primary teaching surface — every
notebook is also a workbench-runnable demo, a docs page, an
integration-test target, and a regression target for the matrix CI.
Treat them as production code, not throwaways.

### Where it goes

The numbering is contiguous and category-bound. Find the right track,
then take the next free number at the end of that range:

| Track | Range |
|---|---|
| Agent Foundations | 13–20 |
| Graphs & composition | 21–28 |
| Multi-agent | 29–39 |
| Reasoning & structured output | 40–42 |
| RAG | 43–45 |
| Skills, playbooks, plugins | 46–50 |
| Production | 51–56 |
| Cognitive router & observability | 57–61 |
| Real-world workflows | 62–66 |
| Server & full pipelines | 67–69 |

`NOTEBOOK_CATEGORIES` in the workbench's `backend/runner.py`
([tuliplabs-ai/workbench](https://github.com/tuliplabs-ai/workbench))
is the source of truth — open a companion PR there when you add a
notebook so the workbench sidebar groups the new entry correctly.

### File requirements

1. **Naming**: `examples/notebook_NN_short_topic.py` — two-digit
   number, snake_case slug.
2. **Header docstring**: first line is the title the workbench
   sidebar will show. Strip the leading `Notebook NN:` prefix if you
   include one — the runner cleans it. Follow with one short
   paragraph explaining what the reader gets.
3. **Multi-model demos**: if the notebook fits as a multi-model
   demonstration (orchestrator, handoff, debate, supervisor-critic),
   use `get_model_b()` and `get_model_c()` for the lighter roles —
   each falls back to slot A when the workbench's "Model B / C"
   dropdowns are empty, so plain CLI runs stay correct.
4. **Graceful skip**: if the notebook needs credentials the harness
   may not have (OPENAI_API_KEY, COHERE_API_KEY, etc.), check the env
   vars at the top of `main()` and print a skip banner with a one-line
   wiring snippet instead of crashing. The CI runs every notebook —
   nothing is allowed to traceback on a missing optional env.
5. **`tulip.core.interrupt()`**: notebooks that call `interrupt()`
   for human approval (Notebooks 24, 38, 62, 63, 64 today) get a
   `needs_stdin: true` badge in the workbench. Add the notebook
   number to `NOTEBOOK_NEEDS_STDIN` in the workbench repo's
   `backend/runner.py` when you add another one.
6. **Docs page**: the docs repo
   ([tuliplabs-ai/docs](https://github.com/tuliplabs-ai/docs)) carries
   one page per `examples/notebook_*.py` — run its
   `scripts/gen_notebook_pages.py` against your SDK checkout to
   scaffold the new page, edit the prose, and open a companion PR.
7. **Real-provider check**: run it through the workbench against the
   provider your audience cares about before opening the PR — the
   Playwright sweeps catch regressions, but a manual click-through
   catches the unloved corner cases (prompt phrasing, output
   readability).

## Making Changes

### Branch Naming

Use descriptive branch names:

```
feat/add-opensearch-store
fix/memory-leak-in-checkpointer
docs/rag-notebook-improvements
test/add-swarm-integration-tests
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
Signed-off-by: Your Name <email>
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `ci`: CI/CD changes
- `chore`: Maintenance

Examples:

```bash
git commit -s -m "feat(rag): add OpenSearch vector store support"
git commit -s -m "fix(memory): resolve checkpoint corruption on concurrent writes"
git commit -s -m "docs(notebooks): add RAG with Qdrant example"
```

### Code Changes

1. **Create a branch** from `main`:

   ```bash
   git checkout -b feat/my-feature
   ```

2. **Make changes** following coding standards

3. **Run checks** before committing:

   ```bash
   hatch run check          # ruff format-check + ruff lint + mypy strict (all-in-one)
   hatch run format         # auto-format with ruff format
   hatch run test           # unit tests on the active Python
   hatch run test-all       # unit tests across the 3.11 / 3.12 / 3.13 / 3.14 matrix
   hatch run test-cov       # unit tests with coverage report
   hatch run test-integration  # integration tests (gated on credentials)
   ```

   When the `check` group is clean and `test` passes, your commit will sail
   through every required CI job.

4. **Commit** with sign-off:

   ```bash
   git commit -s -m "feat: add my feature"
   ```

5. **Push** to your fork:

   ```bash
   git push origin feat/my-feature
   ```

## Pull Request Process

1. **Create an issue** first to discuss the change

2. **Open a PR** with:
   - Clear title following Conventional Commits
   - Description of changes
   - Link to related issue
   - Test results

3. **PR Template**:

   ```markdown
   ## Summary
   Brief description of changes.

   ## Related Issue
   Fixes #123

   ## Changes
   - Added X
   - Fixed Y
   - Updated Z

   ## Testing
   - [ ] Unit tests pass
   - [ ] Integration tests pass (if applicable)
   - [ ] Manual testing performed

   ## Checklist
   - [ ] Code follows project style
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] Commits are signed off
   ```

4. **Review process**:
   - Maintainers will review within 1 week
   - Address feedback promptly
   - Squash commits if requested

## Coding Standards

### Python Style

- **Formatter**: Ruff (line length: 100)
- **Linter**: Ruff + mypy (strict mode)
- **Type hints**: Required for all public APIs
- **Docstrings**: Google style

```python
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from tulip.core.state import AgentState


class MyConfig(BaseModel):
    """Configuration for MyComponent.

    Attributes:
        name: The component name.
        max_retries: Maximum retry attempts.
    """

    name: str = Field(description="Component name")
    max_retries: int = Field(default=3, ge=1, le=10)


async def process_state(state: "AgentState", config: MyConfig) -> "AgentState":
    """Process the agent state.

    Args:
        state: Current agent state.
        config: Processing configuration.

    Returns:
        Updated agent state.

    Raises:
        ValueError: If state is invalid.
    """
    if not state.messages:
        raise ValueError("State must have messages")

    # Processing logic here
    return state.with_metadata({"processed": True})
```

### Architecture Guidelines

1. **Pydantic-first**: Use Pydantic models for all data structures
2. **Immutable state**: Use frozen models, return new instances
3. **Async-native**: Prefer async functions
4. **Protocol-based**: Define interfaces with `typing.Protocol`
5. **No magic**: Explicit over implicit

### File Organization

```
src/tulip/
├── __init__.py          # Public exports
├── module/
│   ├── __init__.py      # Module exports
│   ├── base.py          # Base classes/protocols
│   ├── impl.py          # Implementations
│   └── utils.py         # Utilities
```

## Testing

### Running Tests

```bash
# All unit tests
hatch run test

# Specific test file
hatch run pytest tests/unit/test_agent.py -v

# With coverage
hatch run test-cov

# Integration tests (requires services)
hatch run pytest tests/integration -v

# Specific marker
hatch run pytest -m "requires_oci" -v
```

### Writing Tests

```python
import pytest
from tulip import Agent
from tulip.core.messages import Message


class TestAgent:
    """Tests for Agent class."""

    @pytest.fixture
    def mock_model(self, mocker):
        """Create a mock model."""
        model = mocker.MagicMock()
        model.complete = mocker.AsyncMock(return_value=...)
        return model

    @pytest.mark.asyncio
    async def test_agent_runs_successfully(self, mock_model):
        """Agent should complete a simple task."""
        agent = Agent(model=mock_model)

        result = await agent.run("Hello")

        assert result.success
        assert result.message is not None

    @pytest.mark.asyncio
    async def test_agent_handles_tool_error(self, mock_model):
        """Agent should handle tool execution errors gracefully."""
        # Test implementation
        pass
```

### Test Categories

- `tests/unit/` - Unit tests (no external dependencies)
- `tests/integration/` - Integration tests (require services)
- Markers: `@pytest.mark.requires_oci`, `@pytest.mark.requires_redis`, etc.

### Workbench end-to-end sweeps

The workbench
([tuliplabs-ai/workbench](https://github.com/tuliplabs-ai/workbench))
ships Playwright specs that drive every non-stdin notebook through the
UI against a single provider. Check it out next to this repo, point its
backend at your SDK checkout, and sweep:

```bash
git clone https://github.com/tuliplabs-ai/workbench.git ../workbench
cd ../workbench

# Bring up the three tiers (terminal 1–3); the backend runs your local
# SDK and cookbook instead of the published package. The backend env is
# hatch-managed — `sdk-local` installs ../../tulip-agents editable.
( cd backend && hatch run sdk-local && \
  TULIP_WORKBENCH_NOTEBOOKS=../../tulip-agents/examples hatch run serve )
( cd bff && npm install && npm run dev )    # terminal 2 — :3101
( cd web && npm install && npm run dev )    # terminal 3 — :5173
( cd e2e && npm install && npx playwright install chromium )

# Run a sweep against your provider of choice (terminal 4).
ANTHROPIC_API_KEY=sk-ant-... \
  npx --prefix e2e playwright test tests/all-anthropic.spec.ts --workers=3

OPENAI_API_KEY=sk-... \
  npx --prefix e2e playwright test tests/all-openai.spec.ts --workers=3
```

The sweeps honour the per-slot model env vars (slot A / B / C) so you
can validate multi-model notebooks end-to-end. Headless by default;
pass `--headed` to watch the run.

## Documentation

### Code Documentation

- All public functions/classes need docstrings
- Use Google-style docstrings
- Include type hints
- Add examples for complex APIs

### Notebooks

See [Notebook authoring](#notebook-authoring) above for the full
authoring guide — numbering, naming, multi-model demo conventions,
graceful-skip checklist, and how to wire a new notebook into the
workbench sidebar.

### README updates

For significant features, update:

- **Capability matrix** in the docs repo's `docs/FEATURES.md` and
  `docs/capabilities.md` (these are the source of truth — `README.md`
  mirrors them).
- **`README.md` hero** — the *"OpenAI · Anthropic"* provider
  strip and the *"Talk to any provider"* table only need updates when
  you add a new provider.
- **`README.md` "Vendor-neutral backends" block** — keep the backend
  table in sync with `tulip.rag.stores`, `tulip.rag.reranker`, and
  `tulip.memory.backends`. Bump when you add a new backend.
- **`README.md` notebook track table** — the ranges must match
  `NOTEBOOK_CATEGORIES` in the workbench repo's `backend/runner.py`
  and the docs repo's `docs/notebooks/index.md`.
- **`README.md` Quick Start examples** — only when the feature changes
  the *five-things-that-make-tulip-different* shape.
- **`README.md` Repo layout** — only when a new top-level module lands.

## Release checklist

Every release follows this checklist. Do not skip steps.

1. **Update `CHANGELOG.md`.** Move the relevant entries from
   `[Unreleased]` into a new version heading. Write the date in
   `YYYY-MM-DD`. Add `### Removed` / `### Changed` notes for anything
   breaking, with a migration snippet.
2. **Deprecation sweep.** For every item in `[Unreleased] > Removed`,
   confirm a `TulipDeprecationWarning` has been in place for at least
   one prior minor release, or document the single-release migration
   path in the CHANGELOG entry.
3. **Version bump.** Update `__version__` in `src/tulip/__init__.py`
   and the `version` field in `pyproject.toml`.
4. **Run the full matrix.** `hatch run all` locally plus
   `pytest tests/integration/ -v` with live services available.
5. **Tag.** `git tag -a v<version> -m 'Release v<version>'` and push
   the tag.
6. **Publish.** Build the wheel (`hatch build`), verify the wheel
   contents do not include compliance artifacts or test data, then
   upload.
7. **Announce.** CHANGELOG entry ships with the release notes; any
   deprecations are called out again in the release announcement.

See [`DEPRECATION.md`](DEPRECATION.md) for the full deprecation policy
and how `TulipDeprecationWarning` works.

## Questions?

- Open a [GitHub Discussion](https://github.com/tuliplabs-ai/sdk-python/discussions)
- Check existing [Issues](https://github.com/tuliplabs-ai/sdk-python/issues)

Thank you for contributing to the SDK!
