# Integration Test Services

Integration tests are opt-in and are skipped unless the required service
configuration is present. Unit tests remain mocked or in-memory and run by
default with `hatch run test`.

## PostgreSQL

PostgreSQL-backed tests use the same environment variables as the backend
fixtures:

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=tulip_test
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
export TULIP_MYSQL_INTEGRATION=1
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_DB=tulip_test
export MYSQL_USER=tulip
export MYSQL_PASSWORD=tulip
```

Start the local MySQL service with the dedicated compose file:

```bash
docker compose -f tests/integration/docker-compose.mysql.yml up -d
```

If port `3306` is already in use, choose another host port and pass the same
value to the tests:

```bash
MYSQL_PORT=13306 docker compose -f tests/integration/docker-compose.mysql.yml up -d

TULIP_MYSQL_INTEGRATION=1 MYSQL_PORT=13306 \
uv run hatch run test:test \
  tests/integration/test_checkpoint_backends.py::TestMySQLBackend \
  tests/integration/test_checkpointer_adapters.py::TestMySQLAdapter
```

Stop the service when finished:

```bash
docker compose -f tests/integration/docker-compose.mysql.yml down
```
