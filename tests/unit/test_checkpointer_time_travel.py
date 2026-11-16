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

import asyncio
import copy
import fnmatch
import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from locus.agent import Agent
from locus.agent.durability import CheckpointWriter
from locus.agent.runtime_loop import ITERATION_CHECKPOINTS_KEY
from locus.core.messages import Message, Role, ToolCall
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
    order_by_write_time,
    parse_time,
    split_stored_metadata,
)
from locus.multiagent.graph import END, START, StateGraph
from locus.multiagent.visualize import draw_mermaid_history
from locus.testing import FunctionModel, text, tool_call
from locus.tools.decorator import tool
from tests.unit.checkpointer_time_travel_conformance import (
    BoundedHistoryConformance,
    CheckpointerTimeTravelConformance,
    _agent,
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
        # Stamped to the second — coarser than the service's milliseconds —
        # so same-second writes are told apart by the metadata, as on S3.
        self.objects[object_name] = (
            bytes(put_object_body),
            datetime.now(UTC).replace(microsecond=0),
        )

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
    """moto reports object modification times to the second, as S3 does."""

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


async def test_the_writer_records_the_origin_on_the_first_lasting_save_only() -> None:
    cp = _Recording()
    writer = CheckpointWriter(cp, "sync")
    await writer.save_iteration(AgentState(), "t", "i1")
    writer.parent_checkpoint_id = "origin"

    # Saves the final one will delete name the origin but do not use it up.
    await writer.save_iteration(AgentState(), "t", "i2", {"source": "step"}, kept=False)
    await writer.save_iteration(AgentState(), "t", "i3", kept=False)
    # The first save that stays does; the ones after it follow by position.
    await writer.save_iteration(AgentState(), "t", "i4")
    await writer.save_iteration(AgentState(), "t", "i5")
    await writer.save_now(AgentState(), "t")

    assert cp.saves == [
        ("t", "i1", None),
        ("t", "i2", {"source": "step", "parent_checkpoint_id": "origin"}),
        ("t", "i3", {"parent_checkpoint_id": "origin"}),
        ("t", "i4", {"parent_checkpoint_id": "origin"}),
        ("t", "i5", None),
        ("t", None, None),
    ]


async def test_the_final_save_records_the_origin_when_no_iteration_save_stays() -> None:
    cp = _Recording()
    writer = CheckpointWriter(cp, "sync")
    writer.parent_checkpoint_id = "origin"

    await writer.save_iteration(AgentState(), "t", "i1", kept=False)
    await writer.save_now(AgentState(), "t", metadata={"note": 1})

    assert cp.saves == [
        ("t", "i1", {"parent_checkpoint_id": "origin"}),
        ("t", None, {"note": 1, "parent_checkpoint_id": "origin"}),
    ]


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return f"found {q}"


def _looking_up(checkpointer: Any, **kwargs: Any) -> Agent:
    """An agent that calls ``lookup`` until two results are in the thread, then answers."""

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        answered = [m for m in messages if m.role == Role.TOOL]
        if len(answered) < 2:
            return tool_call("lookup", q=f"q{len(answered)}")
        return text("done")

    return Agent(
        model=FunctionModel(handler),
        tools=[lookup],
        checkpointer=checkpointer,
        max_iterations=6,
        reflexion=False,
        grounding=False,
        **kwargs,
    )


async def test_a_run_from_a_checkpoint_with_kept_iteration_saves_is_a_chain() -> None:
    cp = MemoryCheckpointer()
    agent = _looking_up(cp, keep_iteration_checkpoints=True)
    await agent.arun("first", thread_id="t")
    first_turn = await cp.list("t")
    assert [i.iteration for i in first_turn] == [3, 2, 1], "two iteration saves and the final"
    origin = first_turn[-1]
    await asyncio.sleep(0.002)

    await agent.arun("again", thread_id="t", from_checkpoint=origin.checkpoint_id)

    history = await cp.list("t", limit=50)
    old = {i.checkpoint_id for i in first_turn}
    final, step = [i for i in history if i.checkpoint_id not in old]
    # origin <- step <- final: a chain, not a star around the origin.
    assert step.parent_checkpoint_id == origin.checkpoint_id
    assert final.parent_checkpoint_id == step.checkpoint_id
    assert [i.metadata.get("parent_checkpoint_id") for i in (step, final)] == [
        origin.checkpoint_id,
        None,
    ], "only the first lasting save records the origin; the rest follow by position"
    assert [i.parent_checkpoint_id for i in first_turn] == [
        first_turn[1].checkpoint_id,
        origin.checkpoint_id,
        None,
    ], "the first turn's history is untouched"


async def test_a_run_from_an_iteration_checkpoint_never_deletes_it() -> None:
    """A crashed turn's iteration save names itself for deletion; a run from it must not oblige."""
    cp = MemoryCheckpointer()
    await cp.save(_state(0).with_metadata(ITERATION_CHECKPOINTS_KEY, ["i0"]), "t", "i0")
    await asyncio.sleep(0.002)
    await cp.save(_state(1).with_metadata(ITERATION_CHECKPOINTS_KEY, ["i0", "i1"]), "t", "i1")
    await asyncio.sleep(0.002)

    await _agent(cp).arun("go", thread_id="t", from_checkpoint="i1")

    assert await cp.load("t", "i1") is not None, "the origin stays"
    assert await cp.load("t", "i0") is not None, "and so do the crashed turn's other saves"
    newest, *_ = await cp.list("t")
    assert newest.parent_checkpoint_id == "i1"
    assert newest.checkpoint_id not in {"i0", "i1"}
    latest = await cp.load("t")
    assert latest is not None
    assert ITERATION_CHECKPOINTS_KEY not in latest.metadata


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
    # A window of the history: its oldest descends from the checkpoint before it.
    window = link_parents(newest_first[:2], "f")
    assert [i.parent_checkpoint_id for i in window] == ["a", "f"]


def test_split_stored_metadata_reads_each_shape_a_backend_returns() -> None:
    assert split_stored_metadata(None) == ({}, None)
    assert split_stored_metadata({"note": 1}) == ({"note": 1}, None)
    metadata, when = split_stored_metadata(
        {"metadata": {"note": 1}, "timestamp": "2026-01-02T03:04:05+00:00"}
    )
    assert (metadata, when) == ({"note": 1}, datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))


async def test_the_base_list_skips_what_does_not_load_and_a_backend_without_metadata() -> None:
    class _Partial(_MinimalCheckpointer):
        async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
            return ["ghost", *await super().list_checkpoints(thread_id, limit)]

        async def get_metadata(
            self, thread_id: str, checkpoint_id: str | None = None
        ) -> dict[str, Any] | None:
            raise NotImplementedError

    cp = _Partial()
    await cp.save(_state(0), "t", "cp-0", metadata={"note": 0})

    [info] = await cp.list("t")

    assert (info.checkpoint_id, info.iteration, info.metadata) == ("cp-0", 0, {})


async def test_trimming_counts_only_what_the_backend_deleted() -> None:
    class _Sticky(_MinimalCheckpointer):
        async def delete(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
            return False

    cp = _Sticky()
    for step in range(3):
        await cp.save(_state(step), "t", f"cp-{step}")
        await asyncio.sleep(0.002)

    assert await cp._trim_history("t", 1) == 0
    assert len(await cp.list_checkpoints("t")) == 3
    assert await cp._trim_history("t~ledger", 1) == 0, "a ledger sidecar is never trimmed"


async def test_order_by_write_time_reads_the_metadata_for_same_second_writes_only() -> None:
    second = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    metas = {
        cid: {"created_at": (second + timedelta(microseconds=n)).isoformat()}
        for n, cid in enumerate(["a", "b", "c", "d"], start=1)
    }
    read: list[str] = []

    async def read_meta(checkpoint_id: str) -> dict[str, Any] | None:
        read.append(checkpoint_id)
        return metas.get(checkpoint_id)

    objects = [
        ("b", second),
        ("d", second + timedelta(seconds=1)),  # alone in its second: the listing is enough
        ("a", second.replace(tzinfo=None)),  # a naive stamp is read as UTC
        ("c", second),
        ("legacy", None),  # no time and no metadata object: last, by itself
    ]

    ordered = await order_by_write_time(objects, read_meta, "created_at")

    assert [cid for cid, _, _ in ordered] == ["d", "c", "b", "a", "legacy"]
    assert sorted(read) == ["a", "b", "c"]
    assert ordered[0][2] is None
    assert ordered[1][2] == metas["c"]


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
    # A naive value is UTC wall-clock time and comes back aware, so infos from
    # a backend whose driver drops the zone compare with every other backend's.
    naive = datetime(2026, 1, 2, 3)  # noqa: DTZ001 — what a zone-dropping driver returns
    assert parse_time(naive) == datetime(2026, 1, 2, 3, tzinfo=UTC)
    assert parse_time("2026-01-02T03:00:00") == datetime(2026, 1, 2, 3, tzinfo=UTC)
    assert parse_time(datetime(2026, 1, 2, 3, tzinfo=UTC)) == datetime(2026, 1, 2, 3, tzinfo=UTC)
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
    # A thread id with a replaced character carries a hash of the original.
    assert 'subgraph thread_main_alt_59e97666["main-alt"]' in diagram
    assert "cp_main__c0 --> cp_main__c1" in diagram
    assert "cp_main__c1 --> cp_main__c2" in diagram
    [fork_id] = [i.checkpoint_id for i in await cp.list("main-alt") if i.forked_from]
    fork_node = f"cp_main_alt_59e97666__{fork_id}"
    assert f"cp_main__c1 -.->|fork| {fork_node}" in diagram
    assert f"{fork_node} --> cp_main_alt_59e97666__alt2" in diagram
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


async def test_history_mermaid_keeps_threads_apart_and_escapes_labels() -> None:
    cp = MemoryCheckpointer()
    state = AgentState()
    for thread_id, second in (("a-b", "c2"), ("a_b", 'c"<#>2')):
        await cp.save(state, thread_id, "c1")
        await asyncio.sleep(0.002)
        await cp.save(state, thread_id, second)

    diagram = await draw_mermaid_history(cp, "a-b", "a_b")

    lanes = re.findall(r"subgraph thread_(\S+)\[", diagram)
    assert len(lanes) == len(set(lanes)) == 2, "a-b and a_b are not the same lane"
    nodes = re.findall(r"^\s+(\S+)\[\"", diagram, flags=re.MULTILINE)
    assert len(nodes) == 4
    assert all(re.fullmatch(r"[A-Za-z0-9_]+", n) for n in nodes), "ids hold nothing Mermaid minds"
    edges = re.findall(r"^\s+(\S+) --> (\S+)$", diagram, flags=re.MULTILINE)
    assert len(edges) == 2
    for src, dst in edges:
        [lane] = [lane for lane in lanes if src.startswith(f"cp_{lane}__")]
        assert dst.startswith(f"cp_{lane}__"), "no edge crosses from one thread to the other"
    assert '["c#quot;#lt;#35;#gt;2<br/>iteration 0"]' in diagram
    assert 'c"<#>2' not in diagram


async def test_history_mermaid_labels_a_checkpoint_without_an_iteration_by_its_id() -> None:
    class _Listing:
        async def list(self, thread_id: str, limit: int = 10) -> list[CheckpointInfo]:
            return [CheckpointInfo("c", thread_id)]

    diagram = await draw_mermaid_history(_Listing(), "t")

    assert 'cp_t__c["c"]' in diagram


async def test_a_graph_forks_through_a_checkpointer_without_the_time_travel_surface() -> None:
    """A duck-typed checkpointer with save/load alone still gets a plain copy."""

    class _Duck:
        def __init__(self) -> None:
            self.inner = MemoryCheckpointer()

        async def save(
            self,
            state: AgentState,
            thread_id: str,
            checkpoint_id: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> str:
            return await self.inner.save(state, thread_id, checkpoint_id, metadata)

        async def load(self, thread_id: str, checkpoint_id: str | None = None) -> Any:
            return await self.inner.load(thread_id, checkpoint_id)

    duck = _Duck()
    graph = StateGraph().compile(checkpointer=duck)
    await duck.save(AgentState(metadata={"count": 1}), "g")

    forked = await graph.afork("g", new_thread_id="g-alt")

    assert forked == "g-alt"
    copied = await duck.load("g-alt")
    assert copied is not None
    assert copied.metadata["count"] == 1
    [first] = await duck.inner.list("g-alt")
    assert first.forked_from == "g"
    unnamed = await graph.afork("g")
    assert unnamed.startswith("g-fork-")
    with pytest.raises(LookupError):
        await graph.afork("no-such-thread")
    with pytest.raises(ValueError, match="different"):
        await graph.afork("g", new_thread_id="g")


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
    cp = FileCheckpointer(Path(__file__).parent / "_unused")
    cp.base_dir = Path(__file__).parent / "_tt_old_layout"
    thread_dir = cp._get_thread_dir("t")
    thread_dir.mkdir(parents=True, exist_ok=True)
    try:
        # The payload holds the header's own key names — a ``size`` in a
        # tool's arguments and in the state's metadata, a message's
        # ``metadata`` — which must not be read as the header.
        state = (
            _state(2)
            .with_metadata("size", 7)
            .with_message(
                Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="resize", arguments={"size": 3})]
                )
            )
            .with_message(Message(role=Role.USER, content="old", metadata={"iteration": 9}))
        )
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


async def test_a_header_whose_metadata_names_state_is_parsed_in_full(tmp_path: Path) -> None:
    cp = FileCheckpointer(tmp_path)
    await cp.save(_state(1), "t", "c1", metadata={"state": "CA", "note": 1})

    [info] = await cp.list("t")

    assert info.metadata == {"state": "CA", "note": 1}
    assert (info.iteration, info.size) == (1, checkpoint_size(_state(1).to_checkpoint()))


async def test_file_list_skips_what_is_not_a_checkpoint(tmp_path: Path) -> None:
    cp = FileCheckpointer(tmp_path)
    await cp.save(_state(0), "t", "c0")
    thread_dir = cp._get_thread_dir("t")
    (thread_dir / "notes.json").write_text('{"hello": "world"}')
    (thread_dir / "cut.json").write_text('{"checkpoint_id": "cut", "state": ')
    (tmp_path / "stray.txt").write_text("not a thread")

    assert [i.checkpoint_id for i in await cp.list("t")] == ["c0"]
    assert await cp.get_disk_usage() == sum(f.stat().st_size for f in thread_dir.glob("*.json"))


async def test_adapter_list_reads_the_step_fields_off_the_checkpoint_for_an_old_index() -> None:
    store = _DictStore()
    cp = StorageBackendAdapter(store)
    await cp.save(_state(2), "t", "c2")
    index = store.rows["t:_checkpoints"]
    for entry in index["checkpoints"]:
        del entry["iteration"], entry["size"]
    index["checkpoints"].append({"checkpoint_id": "gone", "timestamp": "2026-01-01T00:00:00"})
    index["checkpoints"].append({"timestamp": "2025-01-01T00:00:00"})

    described, gone = await cp.list("t")

    assert (described.checkpoint_id, described.iteration) == ("c2", 2)
    assert described.size == checkpoint_size(store.rows["t:c2"]), "the document as stored"
    assert (gone.checkpoint_id, gone.iteration, gone.size) == ("gone", None, None)


async def test_s3_deleting_an_older_checkpoint_leaves_the_latest_pointer_alone() -> None:
    from locus.memory.backends.adapters import s3_checkpointer
    from locus.memory.backends.s3 import S3Backend

    with moto.mock_aws():
        cp = s3_checkpointer("locus-tt", region_name="us-east-1")
        assert isinstance(cp, S3Backend)
        for step in range(3):
            await cp.save(_state(step), "t", f"cp-{step}")

        assert await cp.delete("t", "cp-0") is True
        latest = await cp.load("t")
        assert latest is not None
        assert latest.iteration == 2
        assert await cp.delete("t", "cp-2") is True, "the newest: the pointer moves back"
        latest = await cp.load("t")
        assert latest is not None
        assert latest.iteration == 1
        assert await cp.delete("t", "cp-1") is True
        assert await cp.load("t") is None


def test_bounds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        MemoryCheckpointer(max_checkpoints_per_thread=0)
    with pytest.raises(ValueError, match="at least 1"):
        StorageBackendAdapter(_DictStore(), max_checkpoints_per_thread=0)
