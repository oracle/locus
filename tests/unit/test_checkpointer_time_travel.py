# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Time travel on every checkpointer the unit suite can run without a service.

The shared cases live in ``checkpointer_time_travel_conformance.py``; this
module runs them over the in-memory, file, adapter-wrapped (a dict store),
S3 (``moto``) and OCI bucket (an in-memory client) checkpointers, over a
retained wrapper, and over a minimal ``BaseCheckpointer`` subclass that has
nothing but the base defaults — the path a custom backend gets for free.
Below them: the pieces on their own — the writer recording a run's origin,
the lineage helpers, and the history drawn as a Mermaid branch.
"""

from __future__ import annotations

import copy
import fnmatch
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from locus.agent.durability import CheckpointWriter
from locus.core.messages import Message
from locus.core.state import AgentState
from locus.memory import CheckpointInfo
from locus.memory.backends.adapters import StorageBackendAdapter
from locus.memory.backends.file import FileCheckpointer
from locus.memory.backends.memory import MemoryCheckpointer
from locus.memory.checkpointer import BaseCheckpointer
from locus.memory.retention import RetainedCheckpointer
from locus.memory.time_travel import (
    checkpoint_size,
    fork_metadata,
    info_from,
    link_parents,
    parse_time,
)
from locus.multiagent.graph import END, START, StateGraph
from locus.multiagent.visualize import draw_mermaid_history
from tests.unit.checkpointer_time_travel_conformance import (
    BoundedHistoryConformance,
    CheckpointerTimeTravelConformance,
    _state,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _DictStore:
    """A storage backend of the adapter's shape: one JSON document per key."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    async def save(self, key: str, data: dict[str, Any]) -> None:
        self.rows[key] = copy.deepcopy(data)

    async def load(self, key: str) -> dict[str, Any] | None:
        row = self.rows.get(key)
        return copy.deepcopy(row) if row is not None else None

    async def delete(self, key: str) -> bool:
        return self.rows.pop(key, None) is not None

    async def exists(self, key: str) -> bool:
        return key in self.rows

    async def list_threads(self, limit: int = 100, pattern: str = "*") -> list[str]:
        return [k for k in self.rows if fnmatch.fnmatch(k, pattern)][:limit]


class _FakeObjectStorage:
    """The slice of the OCI Object Storage client the bucket checkpointer uses."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, datetime]] = {}

    def get_bucket(self, **_: Any) -> Any:
        return SimpleNamespace(data=SimpleNamespace(name="fake"))

    def put_object(self, *, object_name: str, put_object_body: bytes, **_: Any) -> None:
        self.objects[object_name] = (bytes(put_object_body), datetime.now(UTC))

    def get_object(self, *, object_name: str, **_: Any) -> Any:
        if object_name not in self.objects:
            raise Exception(f"ObjectNotFound: {object_name}")  # noqa: TRY002
        return SimpleNamespace(data=SimpleNamespace(content=self.objects[object_name][0]))

    def delete_object(self, *, object_name: str, **_: Any) -> None:
        if object_name not in self.objects:
            raise Exception(f"ObjectNotFound: {object_name}")  # noqa: TRY002
        del self.objects[object_name]

    def list_objects(
        self, *, prefix: str, limit: int = 1000, delimiter: str | None = None, **_: Any
    ) -> Any:
        found: list[Any] = []
        prefixes: list[str] = []
        for name, (body, modified) in sorted(self.objects.items()):
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix) :]
            if delimiter and delimiter in rest:
                common = prefix + rest.split(delimiter, 1)[0] + delimiter
                if common not in prefixes:
                    prefixes.append(common)
                continue
            found.append(SimpleNamespace(name=name, time_modified=modified, size=len(body)))
        return SimpleNamespace(data=SimpleNamespace(objects=found[:limit], prefixes=prefixes))


class _MinimalCheckpointer(BaseCheckpointer):
    """Only what the base class requires, plus metadata: a custom backend as written."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, tuple[dict[str, Any], dict[str, Any], datetime]]] = {}

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        from uuid import uuid4

        checkpoint_id = checkpoint_id or uuid4().hex
        self.rows.setdefault(thread_id, {})[checkpoint_id] = (
            state.to_checkpoint(),
            dict(metadata or {}),
            datetime.now(UTC),
        )
        return checkpoint_id

    async def load(self, thread_id: str, checkpoint_id: str | None = None) -> AgentState | None:
        ids = await self.list_checkpoints(thread_id, limit=1)
        checkpoint_id = checkpoint_id or (ids[0] if ids else None)
        row = self.rows.get(thread_id, {}).get(checkpoint_id or "")
        return AgentState.from_checkpoint(row[0]) if row else None

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
        rows = self.rows.get(thread_id, {})
        return sorted(rows, key=lambda k: rows[k][2], reverse=True)[:limit]

    async def delete(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        if checkpoint_id is None:
            return self.rows.pop(thread_id, None) is not None
        return self.rows.get(thread_id, {}).pop(checkpoint_id, None) is not None

    @property
    def capabilities(self) -> Any:
        from locus.core.protocols import CheckpointerCapabilities

        return CheckpointerCapabilities(metadata_query=True)

    async def get_metadata(
        self, thread_id: str, checkpoint_id: str | None = None
    ) -> dict[str, Any] | None:
        row = self.rows.get(thread_id, {}).get(checkpoint_id or "")
        return {"metadata": row[1], "created_at": row[2].isoformat()} if row else None


# ---------------------------------------------------------------------------
# The conformance suite, per backend
# ---------------------------------------------------------------------------


class TestMemoryTimeTravel(CheckpointerTimeTravelConformance, BoundedHistoryConformance):
    @pytest.fixture
    def make_checkpointer(self) -> Any:
        return MemoryCheckpointer

    @pytest.fixture
    def make_bounded(self) -> Any:
        return lambda keep: MemoryCheckpointer(max_checkpoints_per_thread=keep)


class TestFileTimeTravel(CheckpointerTimeTravelConformance, BoundedHistoryConformance):
    @pytest.fixture
    def make_checkpointer(self, tmp_path: Path) -> Any:
        return lambda: FileCheckpointer(tmp_path / "threads")

    @pytest.fixture
    def make_bounded(self, tmp_path: Path) -> Any:
        return lambda keep: FileCheckpointer(tmp_path / "bounded", max_checkpoints_per_thread=keep)


class TestAdapterTimeTravel(CheckpointerTimeTravelConformance, BoundedHistoryConformance):
    @pytest.fixture
    def make_checkpointer(self) -> Any:
        return lambda: StorageBackendAdapter(_DictStore())

    @pytest.fixture
    def make_bounded(self) -> Any:
        return lambda keep: StorageBackendAdapter(_DictStore(), max_checkpoints_per_thread=keep)


class TestRetainedTimeTravel(CheckpointerTimeTravelConformance):
    @pytest.fixture
    def make_checkpointer(self) -> Any:
        return lambda: RetainedCheckpointer(MemoryCheckpointer())


class TestBaseDefaultsTimeTravel(CheckpointerTimeTravelConformance):
    """A subclass with only the abstract methods gets ``list`` and ``fork`` from the base."""

    @pytest.fixture
    def make_checkpointer(self) -> Any:
        return _MinimalCheckpointer


class TestOCIBucketTimeTravel(CheckpointerTimeTravelConformance, BoundedHistoryConformance):
    @pytest.fixture
    def make_checkpointer(self) -> Any:
        from locus.memory.backends.oci_bucket import OCIBucketBackend

        def make(keep: int | None = None) -> OCIBucketBackend:
            backend = OCIBucketBackend(
                bucket_name="b",
                namespace="n",
                retry_strategy=object(),
                max_checkpoints_per_thread=keep,
            )
            backend._client = _FakeObjectStorage()  # type: ignore[assignment]
            return backend

        return make

    @pytest.fixture
    def make_bounded(self, make_checkpointer: Any) -> Any:
        return make_checkpointer


moto = pytest.importorskip("moto", reason="S3 time travel runs on moto")


class TestS3TimeTravel(CheckpointerTimeTravelConformance, BoundedHistoryConformance):
    # moto reports object modification times to the second.
    settle_seconds = 1.05

    @pytest.fixture(autouse=True)
    def _mock_aws(self) -> Iterator[None]:
        pytest.importorskip("boto3")
        with moto.mock_aws():
            yield

    @pytest.fixture
    def make_checkpointer(self) -> Any:
        from locus.memory.backends.s3 import S3Backend

        return lambda keep=None: S3Backend(
            bucket="locus-tt", region_name="us-east-1", max_checkpoints_per_thread=keep
        )

    @pytest.fixture
    def make_bounded(self, make_checkpointer: Any) -> Any:
        return make_checkpointer


# ---------------------------------------------------------------------------
# The pieces on their own
# ---------------------------------------------------------------------------


class _Recording(BaseCheckpointer):
    def __init__(self) -> None:
        self.saves: list[tuple[str, str | None, dict[str, Any] | None]] = []

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.saves.append((thread_id, checkpoint_id, metadata))
        return checkpoint_id or f"cp-{len(self.saves)}"

    async def load(self, thread_id: str, checkpoint_id: str | None = None) -> AgentState | None:
        return None

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
        return []


async def test_the_writer_records_the_origin_of_a_run_started_from_a_checkpoint() -> None:
    cp = _Recording()
    writer = CheckpointWriter(cp, "sync")
    await writer.save_iteration(AgentState(), "t", "i1")
    writer.parent_checkpoint_id = "origin"

    await writer.save_iteration(AgentState(), "t", "i2", {"source": "step"})
    await writer.save_now(AgentState(), "t")

    assert cp.saves == [
        ("t", "i1", None),
        ("t", "i2", {"source": "step", "parent_checkpoint_id": "origin"}),
        ("t", None, {"parent_checkpoint_id": "origin"}),
    ]


def test_link_parents_follows_the_line_unless_a_parent_is_recorded() -> None:
    newest_first = [
        info_from("c", "t", metadata={"parent_checkpoint_id": "a"}),
        info_from("b", "t"),
        info_from("f", "t", metadata=fork_metadata("other", "x")),
        info_from("a", "t"),
    ]

    linked = link_parents(newest_first)

    assert [i.parent_checkpoint_id for i in linked] == ["a", "f", "x", None]
    assert linked[2].forked_from == "other"
    assert linked[2].source == "fork"


def test_info_from_normalises_what_a_backend_stored() -> None:
    info = info_from(
        "c1",
        "t",
        created_at="2026-01-02T03:04:05+00:00",
        iteration="3",
        size=12.0,
        metadata={"parent_checkpoint_id": 7},
    )

    assert info.created_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert (info.iteration, info.size, info.parent_checkpoint_id) == (3, 12, "7")
    assert info.to_dict()["created_at"] == "2026-01-02T03:04:05+00:00"
    assert parse_time("not a time") is None
    assert parse_time(None) is None
    assert checkpoint_size({"a": 1}) == len(b'{"a":1}')


def test_checkpoint_info_is_in_the_memory_namespace() -> None:
    assert CheckpointInfo("c", "t").to_dict()["checkpoint_id"] == "c"


async def test_history_mermaid_draws_a_fork_as_a_branch() -> None:
    cp = MemoryCheckpointer()
    for step in range(3):
        await cp.save(_state(step), "main", f"c{step}")
    await cp.fork("main", "c1", new_thread_id="main-alt")
    await cp.save(_state(2), "main-alt", "alt2")

    diagram = await draw_mermaid_history(cp, "main", "main-alt")

    assert diagram.startswith("graph LR")
    assert 'subgraph thread_main["main"]' in diagram
    assert 'subgraph thread_main_alt["main-alt"]' in diagram
    assert "cp_main__c0 --> cp_main__c1" in diagram
    assert "cp_main__c1 --> cp_main__c2" in diagram
    [fork_id] = [i.checkpoint_id for i in await cp.list("main-alt") if i.forked_from]
    fork_node = f"cp_main_alt__{fork_id}"
    assert f"cp_main__c1 -.->|fork| {fork_node}" in diagram
    assert f"{fork_node} --> cp_main_alt__alt2" in diagram
    assert "iteration 1<br/>fork" in diagram
    # The fork alone: its parent is not drawn, so no dangling arrow.
    alone = await draw_mermaid_history(cp, "main-alt")
    assert "-.->" not in alone
    assert "cp_main__" not in alone


async def test_history_mermaid_shows_a_run_from_a_checkpoint_as_a_branch_in_its_thread() -> None:
    cp = MemoryCheckpointer()
    for step in range(2):
        await cp.save(_state(step), "t", f"c{step}")
    await cp.save(_state(1), "t", "c1b", metadata={"parent_checkpoint_id": "c0"})

    diagram = await draw_mermaid_history(cp, "t")

    assert "cp_t__c0 --> cp_t__c1" in diagram
    assert "cp_t__c0 --> cp_t__c1b" in diagram
    assert "cp_t__c1 --> cp_t__c1b" not in diagram


async def test_a_graph_draws_its_thread_and_its_fork() -> None:
    cp = MemoryCheckpointer()
    graph = StateGraph()

    async def step(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"count": inputs.get("count", 0) + 1}

    graph.add_node("step", step)
    graph.add_edge(START, "step")
    graph.add_edge("step", END)
    graph.compile(checkpointer=cp)
    graph.config.thread_id = "g"
    await graph.execute({"count": 0})
    forked = await graph.afork("g", new_thread_id="g-alt")

    diagram = await graph.adraw_mermaid_history("g", forked)

    assert 'subgraph thread_g["g"]' in diagram
    assert "-.->|fork|" in diagram
    [first] = await cp.list("g-alt")
    assert first.forked_from == "g"
    with pytest.raises(ValueError, match="thread id"):
        await StateGraph().compile(checkpointer=cp).adraw_mermaid_history()


async def test_an_old_file_layout_is_still_described() -> None:
    """A checkpoint written before the step fields existed is read in full."""
    import json

    cp = FileCheckpointer(Path(__file__).parent / "_unused")
    cp.base_dir = Path(__file__).parent / "_tt_old_layout"
    thread_dir = cp._get_thread_dir("t")
    thread_dir.mkdir(parents=True, exist_ok=True)
    try:
        state = _state(2).with_message(Message.user("old"))
        (thread_dir / "old.json").write_text(
            json.dumps(
                {
                    "checkpoint_id": "old",
                    "thread_id": "t",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "state": state.to_checkpoint(),
                    "metadata": {"legacy": True},
                }
            )
        )

        [info] = await cp.list("t")

        assert (info.checkpoint_id, info.iteration) == ("old", 2)
        assert info.size == checkpoint_size(state.to_checkpoint())
        assert info.metadata == {"legacy": True}
    finally:
        import shutil

        shutil.rmtree(cp.base_dir, ignore_errors=True)


def test_bounds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        MemoryCheckpointer(max_checkpoints_per_thread=0)
    with pytest.raises(ValueError, match="at least 1"):
        StorageBackendAdapter(_DictStore(), max_checkpoints_per_thread=0)
