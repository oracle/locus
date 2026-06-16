#!/usr/bin/env bash
# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

# Spin up Postgres (with pgvector) and Redis for locus integration tests.
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
    if ! docker ps --filter "name=locus-test-pg" --format '{{.Names}}' | grep -q .; then
      echo "starting postgres (pgvector)..."
      docker run -d --rm --name locus-test-pg -p 5432:5432 \
        -e POSTGRES_PASSWORD=locus \
        -e POSTGRES_USER=locus \
        -e POSTGRES_DB=locus_test \
        pgvector/pgvector:pg16 >/dev/null
      until docker exec locus-test-pg pg_isready -U locus >/dev/null 2>&1; do sleep 1; done
      docker exec locus-test-pg psql -U locus -d locus_test \
        -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null
      echo "  postgres ready (5432) with vector extension"
    else
      echo "  postgres already running"
    fi

    if ! docker ps --filter "name=locus-test-redis" --format '{{.Names}}' | grep -q .; then
      echo "starting redis..."
      docker run -d --rm --name locus-test-redis -p 6379:6379 redis:7-alpine >/dev/null
      echo "  redis ready (6379)"
    else
      echo "  redis already running"
    fi

    cat <<'EOF'

Export these before running integration tests:

  export POSTGRES_HOST=localhost POSTGRES_PORT=5432
  export POSTGRES_DB=locus_test POSTGRES_USER=locus POSTGRES_PASSWORD=locus
  export REDIS_URL=redis://localhost:6379

EOF
    ;;

  stop)
    docker stop locus-test-pg locus-test-redis 2>/dev/null || true
    echo "stopped"
    ;;

  *)
    echo "usage: $0 [start|stop]" >&2
    exit 2
    ;;
esac
