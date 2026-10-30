# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""The Oracle Cloud and Oracle Database entry points are wired in.

The modules themselves have their own suites; these tests pin the plumbing
that the restore touched — the ``oci:`` provider routing, the ``oci`` /
``oracle`` checkpointer factories, the adapter shortcuts, and every lazy
``Oracle*`` / ``OCI*`` export — so a refactor cannot silently drop one.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

import locus
from locus.memory.backends import adapters
from locus.memory.backends.adapters import StorageBackendAdapter
from locus.memory.registry import get_checkpointer, list_checkpointers
from locus.models.registry import get_model, list_providers


# --------------------------------------------------------------------- root --


def test_the_package_root_resolves_subpackages_on_demand(monkeypatch: pytest.MonkeyPatch) -> None:
    assert locus.agent.__name__ == "locus.agent"
    # A subpackage nothing has imported yet is still reachable as an attribute:
    # forget ``locus.voice`` and the root must import it on demand.
    monkeypatch.delitem(sys.modules, "locus.voice", raising=False)
    monkeypatch.delattr(locus, "voice", raising=False)
    assert locus.voice.__name__ == "locus.voice"
    with pytest.raises(AttributeError, match="definitely_missing"):
        _ = locus.definitely_missing


def test_the_package_root_exports_the_oracle_rag_classes() -> None:
    from locus.rag.embeddings.oci import OCIEmbeddings
    from locus.rag.stores.oracle import OracleVectorStore

    assert locus.OCIEmbeddings is OCIEmbeddings
    assert locus.OracleVectorStore is OracleVectorStore
    assert "OCIEmbeddings" in locus.__all__
    assert "OracleVectorStore" in locus.__all__


# -------------------------------------------------------------- providers --


def test_oci_is_a_registered_provider() -> None:
    assert "oci" in list_providers()


def test_oci_routes_cohere_r_series_and_dac_endpoints_to_the_sdk_transport() -> None:
    from locus.models.providers.oci import OCIModel

    assert isinstance(get_model("oci:cohere.command-r-plus"), OCIModel)
    assert isinstance(get_model("oci:Cohere.Command-R-08-2024"), OCIModel)
    assert isinstance(get_model("oci:cohere.command-a-03-2025"), OCIModel)
    assert isinstance(get_model("oci:ocid1.generativeaiendpoint.oc1.phx.abc"), OCIModel)


def test_oci_routes_everything_else_to_the_v1_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    from locus.models.providers.oci import OCIChatCompletionsModel

    monkeypatch.setenv("OCI_PROFILE", "FROM_ENV")
    model = get_model("oci:openai.gpt-5")
    assert isinstance(model, OCIChatCompletionsModel)
    assert model.config.profile == "FROM_ENV"

    # Explicit kwargs always beat the environment fallback.
    explicit = get_model("oci:meta.llama-3.3-70b-instruct", profile="EXPLICIT")
    assert isinstance(explicit, OCIChatCompletionsModel)
    assert explicit.config.profile == "EXPLICIT"


def test_oci_v1_region_comes_from_the_environment_when_not_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from locus.models.providers.oci import OCIChatCompletionsModel

    monkeypatch.setenv("OCI_PROFILE", "FROM_ENV")
    monkeypatch.delenv("LOCUS_OCI_REGION", raising=False)
    monkeypatch.delenv("OCI_REGION", raising=False)
    default = get_model("oci:openai.gpt-5.5")
    assert isinstance(default, OCIChatCompletionsModel)
    assert default.config.region == "us-chicago-1"

    monkeypatch.setenv("OCI_REGION", "eu-frankfurt-1")
    model = get_model("oci:openai.gpt-5.5")
    assert isinstance(model, OCIChatCompletionsModel)
    assert model.config.region == "eu-frankfurt-1"
    assert "eu-frankfurt-1" in str(model.config.base_url)

    # LOCUS_OCI_REGION wins over OCI_REGION, and an explicit region= wins over both.
    monkeypatch.setenv("LOCUS_OCI_REGION", "uk-london-1")
    assert get_model("oci:openai.gpt-5.5").config.region == "uk-london-1"
    assert get_model("oci:openai.gpt-5.5", region="sa-saopaulo-1").config.region == "sa-saopaulo-1"


def test_the_provider_table_lists_oci() -> None:
    from locus.models.providers import provider_table

    assert "| `oci` | OCI Generative AI |" in provider_table()


def test_models_packages_export_the_oci_classes_lazily() -> None:
    import locus.models
    import locus.models.providers
    from locus.models.providers import oci

    for name in (
        "OCIModel",
        "OCIConfig",
        "OCIAuthType",
        "OCIChatCompletionsModel",
        "OCIChatCompletionsConfig",
        "OCIResponsesModel",
        "OCIResponsesConfig",
    ):
        assert getattr(locus.models, name) is getattr(oci, name)
    for name in (
        "OCIModel",
        "OCIConfig",
        "OCIAuthType",
        "OCIChatCompletionsModel",
        "OCIResponsesModel",
    ):
        assert getattr(locus.models.providers, name) is getattr(oci, name)
    with pytest.raises(AttributeError):
        _ = locus.models.providers.Nope


# ----------------------------------------------------------- checkpointers --


class _Captured:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def test_oci_and_oracle_checkpointers_are_registered() -> None:
    assert {"oci", "oracle"} <= set(list_checkpointers())


def test_the_oci_factory_splits_bucket_and_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return "checkpointer"

    monkeypatch.setattr(adapters, "oci_bucket_checkpointer", fake)
    assert get_checkpointer("oci:my-bucket/my-namespace") == "checkpointer"
    assert seen == {"bucket_name": "my-bucket", "namespace": "my-namespace"}


def test_the_factories_work_without_a_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(adapters, "oci_bucket_checkpointer", lambda **kw: calls.append(("oci", kw)))
    monkeypatch.setattr(adapters, "oracle_checkpointer", lambda **kw: calls.append(("oracle", kw)))
    monkeypatch.setattr(adapters, "s3_checkpointer", lambda **kw: calls.append(("s3", kw)))
    get_checkpointer("oci", bucket_name="b", namespace="ns")
    get_checkpointer("oracle", dsn="mydb_high")
    get_checkpointer("s3:my-bucket")
    assert calls == [
        ("oci", {"bucket_name": "b", "namespace": "ns"}),
        ("oracle", {"dsn": "mydb_high"}),
        ("s3", {"bucket": "my-bucket"}),
    ]


def test_the_oracle_factory_passes_the_hint_as_database(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return "checkpointer"

    monkeypatch.setattr(adapters, "oracle_checkpointer", fake)
    assert get_checkpointer("oracle:mydb_high", user="app") == "checkpointer"
    assert seen == {"database": "mydb_high", "user": "app"}


def test_the_oci_bucket_shortcut_returns_the_native_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from locus.memory.backends import oci_bucket

    monkeypatch.setattr(oci_bucket, "OCIBucketBackend", _Captured)
    backend = adapters.oci_bucket_checkpointer(
        bucket_name="b", namespace="ns", region="us-ashburn-1"
    )
    assert isinstance(backend, _Captured)
    assert backend.kwargs == {
        "bucket_name": "b",
        "namespace": "ns",
        "prefix": "locus/checkpoints/",
        "region": "us-ashburn-1",
    }


def test_the_oracle_shortcut_wraps_the_backend_in_the_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from locus.memory.backends import oracle

    monkeypatch.setattr(oracle, "OracleBackend", _Captured)
    checkpointer = adapters.oracle_checkpointer(dsn="mydb_high", user="admin", password="pw")  # noqa: S106
    assert isinstance(checkpointer, StorageBackendAdapter)
    assert checkpointer._backend.kwargs == {"dsn": "mydb_high", "user": "admin", "password": "pw"}


# ------------------------------------------------------------ lazy exports --


@pytest.mark.parametrize(
    ("package", "name", "module"),
    [
        ("locus.rag.embeddings", "OCIEmbeddings", "locus.rag.embeddings.oci"),
        ("locus.rag.embeddings", "OracleInDBEmbeddings", "locus.rag.embeddings.oracle_indb"),
        ("locus.rag.embeddings", "OracleSyncInDBEmbeddings", "locus.rag.embeddings.oracle_sync"),
        ("locus.rag.reranker", "OCICohereReranker", "locus.rag.reranker.cohere_oci"),
        ("locus.rag.stores", "OracleVectorStore", "locus.rag.stores.oracle"),
        ("locus.rag.stores", "OracleSyncVectorStore", "locus.rag.stores.oracle_sync"),
        ("locus.rag.chunkers", "OracleInDBChunker", "locus.rag.chunkers.oracle_indb"),
        ("locus.rag.chunkers", "OracleSyncInDBChunker", "locus.rag.chunkers.oracle_sync"),
        ("locus.rag.loaders", "OracleADBLoader", "locus.rag.loaders.oracle"),
        ("locus.rag.loaders", "OracleSyncADBLoader", "locus.rag.loaders.oracle_sync"),
        ("locus.rag", "OCIEmbeddings", "locus.rag.embeddings.oci"),
        ("locus.rag", "OracleInDBEmbeddings", "locus.rag.embeddings.oracle_indb"),
        ("locus.rag", "OracleVectorStore", "locus.rag.stores.oracle"),
        ("locus.rag", "OracleSyncVectorStore", "locus.rag.stores.oracle_sync"),
        ("locus.rag", "OracleADBLoader", "locus.rag.loaders.oracle"),
        ("locus.rag", "OracleInDBChunker", "locus.rag.chunkers.oracle_indb"),
        ("locus.rag", "OCICohereReranker", "locus.rag.reranker.cohere_oci"),
        ("locus.memory.backends", "OCIBucketBackend", "locus.memory.backends.oci_bucket"),
        ("locus.memory.backends", "OracleBackend", "locus.memory.backends.oracle"),
        (
            "locus.memory.backends",
            "OracleCheckpointSaver",
            "locus.memory.backends.oracle_versioned",
        ),
        ("locus.memory.backends", "OracleSyncBackend", "locus.memory.backends.oracle_sync"),
        ("locus.memory.backends", "OracleSyncCheckpointSaver", "locus.memory.backends.oracle_sync"),
        ("locus.memory.store_backends", "OracleStore", "locus.memory.store_backends.oracle"),
        ("locus.memory.store_backends", "OracleStoreConfig", "locus.memory.store_backends.oracle"),
        (
            "locus.memory.store_backends",
            "OracleSyncStore",
            "locus.memory.store_backends.oracle_sync",
        ),
        (
            "locus.memory.managers",
            "OracleAgentMemoryManager",
            "locus.memory.managers.oracle_agent_memory",
        ),
    ],
)
def test_lazy_export_resolves_to_the_real_class(package: str, name: str, module: str) -> None:
    pkg = importlib.import_module(package)
    assert getattr(pkg, name) is getattr(importlib.import_module(module), name)
    assert name in pkg.__all__


@pytest.mark.parametrize(
    "package",
    [
        "locus.rag.embeddings",
        "locus.rag.reranker",
        "locus.rag.stores",
        "locus.rag.chunkers",
        "locus.rag.loaders",
        "locus.rag",
        "locus.memory.backends",
        "locus.memory.store_backends",
        "locus.memory.managers",
    ],
)
def test_unknown_names_still_raise_attribute_error(package: str) -> None:
    pkg = importlib.import_module(package)
    with pytest.raises(AttributeError, match="NoSuchThing"):
        _ = pkg.NoSuchThing
