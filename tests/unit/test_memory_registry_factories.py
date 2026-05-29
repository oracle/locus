# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Coverage tests that exercise the *real* factory closures in
``locus.memory.registry``.

The existing ``test_memory_registry.py`` swaps each closure for a stub
inside ``_CHECKPOINTERS``. That measures the public API but bypasses the
factory bodies, leaving them at 60% line coverage. This file leaves the
closures in place and only patches the underlying ``adapters.*`` factory
functions so the closure body executes end-to-end.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from locus.memory.registry import get_checkpointer, list_checkpointers


def _patch_adapter(monkeypatch: pytest.MonkeyPatch, name: str) -> dict[str, Any]:
    """Replace ``locus.memory.backends.adapters.<name>`` with a captor."""
    captured: dict[str, Any] = {}

    def _captor(**kwargs: Any) -> MagicMock:
        captured["kwargs"] = kwargs
        return MagicMock()

    from locus.memory.backends import adapters

    monkeypatch.setattr(adapters, name, _captor, raising=True)
    return captured


class TestRealFactoryClosures:
    def test_redis_factory_normalises_short_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if "redis" not in list_checkpointers():
            pytest.skip("redis backend not registered")
        captured = _patch_adapter(monkeypatch, "redis_checkpointer")
        get_checkpointer("redis:host.example:6379")
        assert captured["kwargs"]["url"] == "redis://host.example:6379"

    def test_redis_factory_keeps_full_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if "redis" not in list_checkpointers():
            pytest.skip("redis backend not registered")
        captured = _patch_adapter(monkeypatch, "redis_checkpointer")
        get_checkpointer("redis:redis://prod.example:6379")
        assert captured["kwargs"]["url"] == "redis://prod.example:6379"

    def test_redis_factory_no_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if "redis" not in list_checkpointers():
            pytest.skip("redis backend not registered")
        captured = _patch_adapter(monkeypatch, "redis_checkpointer")
        get_checkpointer("redis")
        assert "url" not in captured["kwargs"]

    def test_postgresql_factory_passes_database(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if "postgresql" not in list_checkpointers():
            pytest.skip("postgresql backend not registered")
        captured = _patch_adapter(monkeypatch, "postgresql_checkpointer")
        get_checkpointer("postgresql:mydb")
        assert captured["kwargs"]["database"] == "mydb"

    def test_postgresql_factory_no_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if "postgresql" not in list_checkpointers():
            pytest.skip("postgresql backend not registered")
        captured = _patch_adapter(monkeypatch, "postgresql_checkpointer")
        get_checkpointer("postgresql")
        assert "database" not in captured["kwargs"]

    def test_mysql_factory_passes_database(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if "mysql" not in list_checkpointers():
            pytest.skip("mysql backend not registered")
        captured = _patch_adapter(monkeypatch, "mysql_checkpointer")
        get_checkpointer("mysql:mydb")
        assert captured["kwargs"]["database"] == "mydb"

    def test_mysql_factory_no_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if "mysql" not in list_checkpointers():
            pytest.skip("mysql backend not registered")
        captured = _patch_adapter(monkeypatch, "mysql_checkpointer")
        get_checkpointer("mysql")
        assert "database" not in captured["kwargs"]

    def test_opensearch_factory_single_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if "opensearch" not in list_checkpointers():
            pytest.skip("opensearch backend not registered")
        captured = _patch_adapter(monkeypatch, "opensearch_checkpointer")
        get_checkpointer("opensearch:host1:9200")
        assert captured["kwargs"]["hosts"] == ["host1:9200"]

    def test_opensearch_factory_multi_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if "opensearch" not in list_checkpointers():
            pytest.skip("opensearch backend not registered")
        captured = _patch_adapter(monkeypatch, "opensearch_checkpointer")
        get_checkpointer("opensearch:h1:9200,h2:9200")
        assert captured["kwargs"]["hosts"] == ["h1:9200", "h2:9200"]

    def test_opensearch_factory_no_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if "opensearch" not in list_checkpointers():
            pytest.skip("opensearch backend not registered")
        captured = _patch_adapter(monkeypatch, "opensearch_checkpointer")
        get_checkpointer("opensearch")
        assert "hosts" not in captured["kwargs"]

    def test_oci_factory_splits_bucket_namespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if "oci" not in list_checkpointers():
            pytest.skip("oci backend not registered")
        captured = _patch_adapter(monkeypatch, "oci_bucket_checkpointer")
        get_checkpointer("oci:my-bucket/my-namespace")
        assert captured["kwargs"]["bucket_name"] == "my-bucket"
        assert captured["kwargs"]["namespace"] == "my-namespace"

    def test_oci_factory_skips_when_no_slash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if "oci" not in list_checkpointers():
            pytest.skip("oci backend not registered")
        captured = _patch_adapter(monkeypatch, "oci_bucket_checkpointer")
        get_checkpointer("oci:bucket-only")
        # No slash → factory's ``"/" in config_hint`` is False → kwargs stays empty
        assert "bucket_name" not in captured["kwargs"]

    def test_oracle_factory_passes_database(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if "oracle" not in list_checkpointers():
            pytest.skip("oracle backend not registered")
        captured = _patch_adapter(monkeypatch, "oracle_checkpointer")
        get_checkpointer("oracle:mydb_high")
        assert captured["kwargs"]["database"] == "mydb_high"

    def test_oracle_factory_no_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if "oracle" not in list_checkpointers():
            pytest.skip("oracle backend not registered")
        captured = _patch_adapter(monkeypatch, "oracle_checkpointer")
        get_checkpointer("oracle")
        assert "database" not in captured["kwargs"]
