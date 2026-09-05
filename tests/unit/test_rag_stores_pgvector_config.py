# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""#171 — PgVectorStore must never silently discard its configuration.

The failure this guards against: a caller built a ``PgVectorConfig`` with a
real DSN, handed it over under a plausible keyword, and the store accepted it,
threw it away, and connected to ``postgres@localhost:5432/postgres`` — every
declared DSN, table name and pool size gone, discovered only when a real
ingest connected to the wrong database.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tulip.rag.stores.pgvector import PgVectorConfig, PgVectorStore


DSN = "postgresql://u:p@db:5432/x"


class TestExplicitConfig:
    def test_config_object_is_used(self) -> None:
        store = PgVectorStore(config=PgVectorConfig(dsn=DSN, dimension=256))
        assert store.pgvector_config.dsn == DSN
        assert store.pgvector_config.dimension == 256

    def test_config_under_a_wrong_keyword_raises_with_a_pointer(self) -> None:
        """The exact call that used to silently do nothing."""
        with pytest.raises(TypeError, match="config="):
            PgVectorStore(pg_config=PgVectorConfig(dsn=DSN))

    def test_config_plus_individual_settings_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            PgVectorStore(config=PgVectorConfig(dsn=DSN), dimension=64)


class TestUnknownKeywords:
    def test_typoed_keyword_raises_instead_of_vanishing(self) -> None:
        with pytest.raises(ValidationError):
            PgVectorStore(dsn=DSN, dimensions=256)  # note the plural typo

    def test_config_model_forbids_extras(self) -> None:
        with pytest.raises(ValidationError):
            PgVectorConfig(dsn=DSN, tablename="docs")


class TestNoSilentLocalhostFallback:
    def test_no_connection_settings_raises(self) -> None:
        """Nothing configured must not mean postgres@localhost."""
        with pytest.raises(ValueError, match="no connection settings"):
            PgVectorStore(table_name="docs", dimension=4)

    def test_explicit_localhost_still_works(self) -> None:
        store = PgVectorStore(host="localhost", table_name="docs", dimension=4)
        assert store.pgvector_config.host == "localhost"
        assert store.pgvector_config.database == "postgres"

    def test_dsn_alone_is_enough(self) -> None:
        store = PgVectorStore(dsn=DSN)
        assert store.pgvector_config.dsn == DSN


class TestEventTextAliases:
    """#165 — assistant text is reachable under the first-guess name."""

    def test_think_event_content_aliases_reasoning(self) -> None:
        from tulip.core.events import ThinkEvent

        ev = ThinkEvent(agent_name="a", iteration=1, reasoning="hello")
        assert ev.content == "hello"

    def test_terminate_event_content_aliases_final_message(self) -> None:
        from tulip.core.events import TerminateEvent

        ev = TerminateEvent(
            agent_name="a",
            reason="complete",
            iterations_used=1,
            final_confidence=1.0,
            total_tool_calls=0,
            final_message="the answer",
        )
        assert ev.content == "the answer"
