# Integration Test Services

Integration tests are opt-in and are skipped unless the required service
configuration is present. Unit tests remain mocked or in-memory and run by
default with `hatch run test`.

## Models

Most integration tests take the shared `model` fixture and never name a
provider. Point them at any provider the model registry knows
(`locus.models.registry.list_providers()`: openrouter, groq, gemini,
mistral, ollama, vllm, …) with a `provider:model` string; the
provider reads its own credentials from its usual environment variable:

```bash
LOCUS_TEST_MODEL=gemini:gemini-2.5-flash uv run hatch run test-integration
LOCUS_TEST_MODEL=anthropic:claude-haiku-4-5 uv run hatch run test-integration
```

`LOCUS_TEST_MAX_TOKENS` (default 4096) caps each response. Without
`LOCUS_TEST_MODEL` the fixture uses OpenAI, then Anthropic, then OCI, by which
credentials are set. Tests named for a provider (`skip_without_openai`, …) run
whenever that provider's key is present, whatever `LOCUS_TEST_MODEL` says.

The notebooks take the same idea through `LOCUS_MODEL_PROVIDER` and
`LOCUS_MODEL_ID`, and the runner reports a notebook whose prerequisites are
missing (exit status 77, e.g. the OCI-only ones without an OCI profile) as
SKIP rather than FAIL:

```bash
python scripts/run_notebooks.py --provider gemini --model gemini-2.5-flash
```

## PostgreSQL

PostgreSQL-backed tests use the same environment variables as the backend
fixtures:

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=locus_test
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
```

The development compose file includes a PostgreSQL service:

```bash
docker compose -f examples/docker-compose.yaml up -d postgres
uv run hatch run test:test tests/integration/test_checkpoint_backends.py::TestPostgreSQLBackend
```

## MySQL

MySQL integration tests require the optional `mysql` extra and an explicit
opt-in flag so local or CI runs do not connect to an arbitrary MySQL server.
The Hatch `test` environment includes the `mysql` extra.

```bash
export LOCUS_MYSQL_INTEGRATION=1
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_DB=locus_test
export MYSQL_USER=locus
export MYSQL_PASSWORD=locus
```

Start the local MySQL service with the dedicated compose file:

```bash
docker compose -f tests/integration/docker-compose.mysql.yml up -d
```

If port `3306` is already in use, choose another host port and pass the same
value to the tests:

```bash
MYSQL_PORT=13306 docker compose -f tests/integration/docker-compose.mysql.yml up -d

LOCUS_MYSQL_INTEGRATION=1 MYSQL_PORT=13306 \
uv run hatch run test:test \
  tests/integration/test_checkpoint_backends.py::TestMySQLBackend \
  tests/integration/test_checkpointer_adapters.py::TestMySQLAdapter
```

Stop the service when finished:

```bash
docker compose -f tests/integration/docker-compose.mysql.yml down
```
