#!/usr/bin/env bash
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

# Spin up Postgres (with pgvector) and Redis for tulip integration tests.
#
# Usage:
#   bash scripts/start_local_test_services.sh        # start
#   bash scripts/start_local_test_services.sh stop   # stop and remove
#
# Requires: Docker (Rancher Desktop's Docker daemon works fine).

set -euo pipefail

cmd=${1:-start}

case "$cmd" in
  start)
    if ! docker ps --filter "name=tulip-test-pg" --format '{{.Names}}' | grep -q .; then
      echo "starting postgres (pgvector)..."
      docker run -d --rm --name tulip-test-pg -p 5432:5432 \
        -e POSTGRES_PASSWORD=tulip \
        -e POSTGRES_USER=tulip \
        -e POSTGRES_DB=tulip_test \
        pgvector/pgvector:pg16 >/dev/null
      until docker exec tulip-test-pg pg_isready -U tulip >/dev/null 2>&1; do sleep 1; done
      docker exec tulip-test-pg psql -U tulip -d tulip_test \
        -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null
      echo "  postgres ready (5432) with vector extension"
    else
      echo "  postgres already running"
    fi

    if ! docker ps --filter "name=tulip-test-redis" --format '{{.Names}}' | grep -q .; then
      echo "starting redis..."
      docker run -d --rm --name tulip-test-redis -p 6379:6379 redis:7-alpine >/dev/null
      echo "  redis ready (6379)"
    else
      echo "  redis already running"
    fi

    cat <<'EOF'

Export these before running integration tests:

  export POSTGRES_HOST=localhost POSTGRES_PORT=5432
  export POSTGRES_DB=tulip_test POSTGRES_USER=tulip POSTGRES_PASSWORD=tulip
  export REDIS_URL=redis://localhost:6379

EOF
    ;;

  stop)
    docker stop tulip-test-pg tulip-test-redis 2>/dev/null || true
    echo "stopped"
    ;;

  *)
    echo "usage: $0 [start|stop]" >&2
    exit 2
    ;;
esac
