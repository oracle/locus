# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the Weaviate store that need no running instance.

The store itself is exercised live by ``tests/integration/rag/
test_store_conformance_live.py`` (embedded Weaviate behind
``WEAVIATE_EMBEDDED=1``); these tests cover the parts that are pure
Python: constructor validation, collection naming, property naming and
score mapping.
"""

from __future__ import annotations

import pytest

from locus.rag.stores.weaviate import WeaviateVectorStore, meta_property


def test_collection_name_is_capitalised() -> None:
    store = WeaviateVectorStore(collection_name="runbooks", _client=object())
    assert store._collection_name == "Runbooks"
    assert "Runbooks" in repr(store)


def test_metric_validation_and_config() -> None:
    with pytest.raises(ValueError, match="distance_metric"):
        WeaviateVectorStore(distance_metric="hamming", _client=object())
    store = WeaviateVectorStore(dimension=8, distance_metric="l2", _client=object())
    assert store.config.dimension == 8
    assert store.config.distance_metric == "l2"


def test_meta_property_names() -> None:
    assert meta_property("topic") == "meta_topic"
    assert meta_property("_private") == "meta__private"
    assert meta_property("bad key") is None
    assert meta_property("1abc") is None


def test_score_mapping() -> None:
    cosine = WeaviateVectorStore(_client=object())
    assert cosine._score(0.25) == pytest.approx(0.75)
    dot = WeaviateVectorStore(distance_metric="dot", _client=object())
    assert dot._score(-0.4) == pytest.approx(0.4)
    l2 = WeaviateVectorStore(distance_metric="euclidean", _client=object())
    assert l2._score(3.0) == pytest.approx(0.25)


def test_repr_for_each_target() -> None:
    assert "embedded" in repr(WeaviateVectorStore(embedded=True, _client=object()))
    assert "cluster" in repr(WeaviateVectorStore(cluster_url="https://cluster", _client=object()))
    assert "myhost" in repr(WeaviateVectorStore(host="myhost", _client=object()))
