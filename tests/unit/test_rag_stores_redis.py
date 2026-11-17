# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the Redis vector store — FT.* command shaping on a fake client."""

from __future__ import annotations

import json
from typing import Any

import pytest

from locus.rag.stores.base import Document
from locus.rag.stores.redis import (
    RedisVectorStore,
    escape_tag,
    pack_vector,
    unpack_vector,
)


class _Pipeline:
    def __init__(self, parent: _FakeRedis) -> None:
        self._parent = parent
        self._ops: list[tuple[str, Any]] = []

    def delete(self, key: str) -> None:
        self._ops.append(("delete", key))

    def hset(self, key: str, mapping: dict[str, Any]) -> None:
        self._ops.append(("hset", (key, mapping)))

    async def execute(self) -> None:
        for op, payload in self._ops:
            if op == "delete":
                self._parent.hashes.pop(payload, None)
            else:
                key, mapping = payload
                self._parent.hashes[key] = {
                    k.encode(): (v if isinstance(v, bytes) else str(v).encode())
                    for k, v in mapping.items()
                }
        self._parent.pipelines.append(self._ops)


class _FakeRedis:
    """Stores hashes in a dict and records every FT.* command."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[bytes, bytes]] = {}
        self.commands: list[tuple[Any, ...]] = []
        self.pipelines: list[list[tuple[str, Any]]] = []
        self.index_exists = False
        self.search_reply: list[Any] = [0]
        self.closed = False

    async def execute_command(self, *args: Any) -> Any:
        self.commands.append(args)
        if args[0] == "FT.INFO":
            if not self.index_exists:
                raise Exception("Unknown index name")  # noqa: TRY002 — mirrors redis.ResponseError text
            return []
        if args[0] == "FT.CREATE":
            self.index_exists = True
            return b"OK"
        if args[0] == "FT.SEARCH":
            if args[3:5] == ("LIMIT", "0") and args[5] == "0":
                return [len(self.hashes)]
            return self.search_reply
        raise AssertionError(f"unexpected command {args[0]}")

    def pipeline(self, transaction: bool = False) -> _Pipeline:
        return _Pipeline(self)

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        return dict(self.hashes.get(key, {}))

    async def delete(self, *keys: str | bytes) -> int:
        removed = 0
        for key in keys:
            name = key.decode() if isinstance(key, bytes) else key
            if name in self.hashes:
                del self.hashes[name]
                removed += 1
        return removed

    async def scan(
        self, cursor: int = 0, match: str = "*", count: int = 10
    ) -> tuple[int, list[bytes]]:
        prefix = match.rstrip("*")
        return 0, [k.encode() for k in self.hashes if k.startswith(prefix)]

    async def aclose(self) -> None:
        self.closed = True


def _store(**kwargs: Any) -> tuple[RedisVectorStore, _FakeRedis]:
    fake = _FakeRedis()
    store = RedisVectorStore(dimension=3, index_name="t", _client=fake, **kwargs)
    return store, fake


def _reply(*rows: tuple[str, float, dict[str, Any]]) -> list[Any]:
    reply: list[Any] = [len(rows)]
    for doc_id, distance, meta in rows:
        reply.append(f"t:doc:{doc_id}".encode())
        reply.append(
            [
                b"doc_id",
                doc_id.encode(),
                b"content",
                f"content {doc_id}".encode(),
                b"content_type",
                b"text",
                b"metadata_json",
                json.dumps(meta).encode(),
                b"created_at",
                b"2026-01-01T00:00:00+00:00",
                b"embedding",
                pack_vector([1.0, 0.0, 0.0]),
                b"vector_distance",
                str(distance).encode(),
            ]
        )
    return reply


def test_vector_packing_round_trips() -> None:
    values = [1.0, -0.5, 0.25]
    assert unpack_vector(pack_vector(values)) == pytest.approx(values)
    assert len(pack_vector(values)) == 12


def test_escape_tag_escapes_punctuation_and_bools() -> None:
    assert escape_tag("a-b c") == "a\\-b\\ c"
    assert escape_tag(True) == "true"
    assert escape_tag(2026) == "2026"


async def test_index_created_with_vector_and_declared_tag_fields() -> None:
    store, fake = _store(metadata_fields=["topic"])
    await store.add(
        Document(id="a", content="x", embedding=[1.0, 0.0, 0.0], metadata={"topic": "x"})
    )
    create = next(c for c in fake.commands if c[0] == "FT.CREATE")
    assert create[1] == "t"
    assert create[create.index("PREFIX") + 2] == "t:doc:"
    assert "meta_topic" in create
    assert create[create.index("meta_topic") + 1 : create.index("meta_topic") + 3] == (
        "TAG",
        "CASESENSITIVE",
    )
    vector = create[create.index("embedding") :]
    assert vector[1:3] == ("VECTOR", "HNSW")
    assert vector[vector.index("DIM") + 1] == "3"
    assert vector[vector.index("DISTANCE_METRIC") + 1] == "COSINE"
    # Index exists afterwards: a second add does not recreate it.
    await store.add(Document(id="b", content="y", embedding=[0.0, 1.0, 0.0]))
    assert len([c for c in fake.commands if c[0] == "FT.CREATE"]) == 1


async def test_add_writes_hash_with_packed_vector_and_tags() -> None:
    store, fake = _store(metadata_fields=["topic", "flag"])
    await store.add(
        Document(
            id="a",
            content="alpha",
            embedding=[1.0, 0.0, 0.0],
            metadata={"topic": "x", "flag": True, "other": 1},
        )
    )
    ops = fake.pipelines[-1]
    assert ops[0] == ("delete", "t:doc:a")
    key, mapping = ops[1][1]
    assert key == "t:doc:a"
    assert mapping["doc_id"] == "a"
    assert mapping["embedding"] == pack_vector([1.0, 0.0, 0.0])
    assert mapping["meta_topic"] == "x"
    assert mapping["meta_flag"] == "true"
    assert "meta_other" not in mapping
    assert json.loads(mapping["metadata_json"]) == {"topic": "x", "flag": True, "other": 1}

    got = await store.get("a")
    assert got is not None
    assert got.content == "alpha"
    assert got.embedding == pytest.approx([1.0, 0.0, 0.0])
    assert got.metadata == {"topic": "x", "flag": True, "other": 1}


async def test_add_requires_embedding() -> None:
    store, _ = _store()
    with pytest.raises(ValueError, match="embedding"):
        await store.add(Document(id="a", content="x"))


async def test_search_uses_declared_tag_prefilter() -> None:
    store, fake = _store(metadata_fields=["topic"])
    fake.index_exists = True
    fake.search_reply = _reply(("a", 0.0, {"topic": "x"}), ("c", 0.3, {"topic": "x"}))

    results = await store.search([1.0, 0.0, 0.0], limit=2, metadata_filter={"topic": "x-y"})

    search = fake.commands[-1]
    assert search[0:2] == ("FT.SEARCH", "t")
    assert search[2] == "(@meta_topic:{x\\-y})=>[KNN 2 @embedding $vec AS vector_distance]"
    assert search[search.index("PARAMS") + 3] == pack_vector([1.0, 0.0, 0.0])
    assert search[search.index("SORTBY") + 1 : search.index("SORTBY") + 3] == (
        "vector_distance",
        "ASC",
    )
    assert search[search.index("DIALECT") + 1] == "2"
    assert [r.document.id for r in results] == ["a", "c"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].distance == pytest.approx(0.0)
    assert results[1].score == pytest.approx(0.7)
    assert results[0].document.embedding == pytest.approx([1.0, 0.0, 0.0])


async def test_search_rejects_empty_value_on_declared_tag_field() -> None:
    store, fake = _store(metadata_fields=["topic"])
    fake.index_exists = True
    with pytest.raises(ValueError, match="metadata_filter\\['topic'\\] is empty"):
        await store.search([1.0, 0.0, 0.0], limit=2, metadata_filter={"topic": ""})
    assert not [c for c in fake.commands if c[0] == "FT.SEARCH"]
    # An undeclared key is filtered client-side, where an empty value is fine.
    fake.search_reply = _reply(("a", 0.0, {"other": ""}), ("b", 0.2, {"other": "x"}))
    results = await store.search([1.0, 0.0, 0.0], limit=2, metadata_filter={"other": ""})
    assert [r.document.id for r in results] == ["a"]


async def test_search_undeclared_filter_overfetches_and_filters_client_side() -> None:
    store, fake = _store()
    fake.index_exists = True
    fake.search_reply = _reply(
        ("a", 0.0, {"topic": "x"}), ("b", 0.2, {"topic": "y"}), ("c", 0.3, {"topic": "x"})
    )

    results = await store.search([1.0, 0.0, 0.0], limit=1, metadata_filter={"topic": "y"})

    search = fake.commands[-1]
    assert search[2] == "(*)=>[KNN 4 @embedding $vec AS vector_distance]"
    assert [r.document.id for r in results] == ["b"]


async def test_search_threshold_and_limit() -> None:
    store, fake = _store()
    fake.index_exists = True
    fake.search_reply = _reply(("a", 0.0, {}), ("b", 0.5, {}))
    results = await store.search([1.0, 0.0, 0.0], limit=2, threshold=0.9)
    assert [r.document.id for r in results] == ["a"]


async def test_delete_count_clear_close() -> None:
    store, fake = _store()
    await store.add_batch(
        [
            Document(id="a", content="x", embedding=[1.0, 0.0, 0.0]),
            Document(id="b", content="y", embedding=[0.0, 1.0, 0.0]),
        ]
    )
    assert await store.count() == 2
    assert await store.delete("a") is True
    assert await store.delete("a") is False
    assert await store.clear() == 1
    assert fake.hashes == {}
    await store.close()
    assert fake.closed is True


def test_constructor_validation_and_config() -> None:
    with pytest.raises(ValueError, match="distance_metric"):
        RedisVectorStore(distance_metric="hamming", _client=object())
    with pytest.raises(ValueError, match="algorithm"):
        RedisVectorStore(algorithm="IVF", _client=object())
    store = RedisVectorStore(
        dimension=8, distance_metric="l2", algorithm="flat", key_prefix="kb:", _client=object()
    )
    assert store.config.dimension == 8
    assert store.config.index_type == "flat"
    assert store.schema_args()[store.schema_args().index("PREFIX") + 2] == "kb:"
    assert store._score(3.0) == (pytest.approx(0.25), 3.0)


async def test_parsed_dict_reply_shape_is_supported() -> None:
    """redis-py 5+ parses FT.SEARCH into a dict; older versions return the raw array."""
    store, fake = _store()
    fake.index_exists = True
    row = _reply(("a", 0.1, {"topic": "x"}))
    fields = row[2]
    fake.search_reply = {
        b"total_results": 1,
        b"results": [
            {
                b"id": b"t:doc:a",
                b"extra_attributes": {fields[j]: fields[j + 1] for j in range(0, len(fields), 2)},
                b"values": [],
            }
        ],
    }
    results = await store.search([1.0, 0.0, 0.0], limit=1)
    assert [r.document.id for r in results] == ["a"]
    assert results[0].score == pytest.approx(0.9)
    assert results[0].document.metadata == {"topic": "x"}
    assert store._total({b"total_results": 7}) == 7
    assert store._total([3]) == 3
