# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""ObservationPack: large old tool outputs go as placeholders, and come back exactly.

The transform works on the request alone, so these check that the history and
the checkpoint keep every byte, that ``obs_recall`` returns the original bytes
page by page, that nothing is swapped before its ``full_sends`` requests, that
the cost model holds swaps back until a batch pays, that a failure sends the
full output, and that compaction clears into recallable stubs whose ids
survive a summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tulip.agent import Agent, AgentConfig, ObservationPackConfig
from tulip.core.events import CustomEvent, TerminateEvent
from tulip.core.media import encode_image
from tulip.core.messages import Message, Role, ToolCall
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.memory.compaction import (
    CLEARED_OUTPUT_KEY,
    RECALLABLE_OUTPUTS_KEY,
    CompactionTracker,
    ContextCompactor,
    is_summary_message,
)
from tulip.memory.observation_pack import (
    OBSERVATION_ID_KEY,
    Observation,
    ObservationArchive,
    ObservationPack,
    SwapCostModel,
    is_observation_id,
    placeholder_for,
    session_directory_name,
)
from tulip.models.base import ModelResponse
from tulip.testing import FunctionModel
from tulip.tools.decorator import tool


#: Swaps as soon as an output is due: no batching, no pricing.
EAGER = SwapCostModel(min_batch_bytes=0, horizon_requests=1_000_000)


def _big(tag: str, lines: int = 400) -> str:
    # Multibyte characters on purpose: pages must never split one.
    return "\n".join(f"{tag} line {n}: é ✓ {'x' * 30}" for n in range(lines))


def _history(outputs: list[str], *, said: str = "") -> list[Message]:
    """System, task, then one call and result per output, each answered by the model."""
    messages = [Message.system("instructions"), Message.user("task")]
    for n, text in enumerate(outputs):
        call = ToolCall(id=f"c{n}", name="read", arguments={"path": f"f{n}"})
        messages.append(Message.assistant(f"step {n}{said}", tool_calls=[call]))
        messages.append(Message(role=Role.TOOL, tool_call_id=f"c{n}", name="read", content=text))
    return messages


def _pack(tmp_path: Path, **kwargs: Any) -> ObservationPack:
    kwargs.setdefault("cost_model", EAGER)
    return ObservationPack(directory=tmp_path, **kwargs)


def _placeholders(messages: list[Message]) -> list[str]:
    return [
        str(m.metadata[OBSERVATION_ID_KEY]) for m in messages if OBSERVATION_ID_KEY in m.metadata
    ]


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------


def test_the_transform_never_changes_the_history(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    history = [*_history([_big("a")]), Message.assistant("a"), Message.assistant("b")]
    before = [m.model_dump() for m in history]
    sent = pack.project(history, session="s")

    assert [m.model_dump() for m in history] == before
    assert sent is not history
    assert _placeholders(sent)
    assert not _placeholders(history)
    assert history[3].content == _big("a")


def test_an_output_goes_whole_for_its_first_requests(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    history = _history([_big("a")])
    shown = []
    for _ in range(4):
        sent = pack.project(history, session="s")
        shown.append(bool(_placeholders(sent)))
        history = [*history, Message.assistant("thinking")]
    # Sent whole in requests one and two, a placeholder from the third on.
    assert shown == [False, False, True, True]


def test_full_sends_is_configurable(tmp_path: Path) -> None:
    pack = _pack(tmp_path, full_sends=0)
    assert _placeholders(pack.project(_history([_big("a")]), session="s"))


def test_small_outputs_errors_with_images_and_stubs_are_left_alone(tmp_path: Path) -> None:
    pack = _pack(tmp_path, full_sends=0)
    image = f"{_big('i')}\n{encode_image(b'png-bytes')}"
    history = _history(["short", image])
    cleared = Message(
        role=Role.TOOL,
        tool_call_id="c9",
        name="read",
        content=_big("c"),
        metadata={CLEARED_OUTPUT_KEY: True},
    )
    history += [Message.assistant("x", tool_calls=[ToolCall(id="c9", name="read")]), cleared]
    assert pack.project(history, session="s") == history


def test_the_placeholder_names_the_id_size_and_both_ends(tmp_path: Path) -> None:
    text = _big("a")
    pack = _pack(tmp_path, full_sends=0)
    sent = pack.project(_history([text]), session="s")
    placeholder = sent[3].content or ""
    obs_id = _placeholders(sent)[0]

    assert is_observation_id(obs_id)
    assert f"id: {obs_id}" in placeholder
    assert f"original_bytes: {len(text.encode())}" in placeholder
    assert '"offset": 0' in placeholder
    assert "obs_recall" in placeholder
    assert "a line 0:" in placeholder
    assert "a line 399:" in placeholder
    assert len(placeholder.encode()) < 2_000
    # Deterministic: the next request carries the same bytes.
    assert pack.project(_history([text]), session="s")[3].content == placeholder


def test_a_single_long_line_still_gets_an_excerpt() -> None:
    minified = json.dumps({"k": ["v" * 50] * 400})
    message = Message(role=Role.TOOL, tool_call_id="c", name="t", content=minified)
    placeholder = placeholder_for(Observation.of(message), 1024)
    assert '{"k": ["vvvv' in placeholder
    assert 'vvvv"]}' in placeholder


def test_view_shows_committed_swaps_without_deciding_new_ones(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    history = [*_history([_big("a")]), Message.assistant("1"), Message.assistant("2")]
    assert not _placeholders(pack.view(history, session="s"))
    pack.project(history, session="s")
    assert _placeholders(pack.view(history, session="s"))


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


def _recall_all(pack: ObservationPack, obs_id: str, session: str = "s") -> str:
    """Every page of ``obs_id``, joined."""
    offset, chunks = 0, []
    while True:
        page, details = pack.recall(session, obs_id, offset=offset)
        chunks.append(_page(page)[1])
        if details["eof"]:
            return "".join(chunks)
        offset = details["next_offset"]


def _swapped_id(pack: ObservationPack, text: str, session: str = "s") -> str:
    sent = pack.project(_history([text]), session=session)
    return _placeholders(sent)[0]


def _page(text: str) -> tuple[dict[str, str], str]:
    first, second, body = text.split("\n", 2)
    fields = dict(part.split("=", 1) for part in first.strip("[]").split()[1:])
    assert second.startswith("[chunk_bytes=")
    return fields, body


def test_recall_returns_the_exact_bytes_page_by_page(tmp_path: Path) -> None:
    text = _big("a", lines=1_500)
    pack = _pack(tmp_path, full_sends=0)
    obs_id = _swapped_id(pack, text)

    offset, pages, chunks = 0, 0, []
    while True:
        page, details = pack.recall("s", obs_id, offset=offset)
        assert len(page.encode()) <= 16 * 1024
        assert page.count("\n") + 1 <= 400
        fields, body = _page(page)
        chunks.append(body.encode())
        assert details["next_offset"] == offset + len(body.encode())
        pages += 1
        if details["eof"]:
            assert fields["eof"] == "true"
            break
        offset = details["next_offset"]
    assert b"".join(chunks) == text.encode()
    assert pages > 1
    stats = pack.stats("s")
    assert stats.recalls == pages
    assert stats.recalled_bytes == len(text.encode())


def test_recall_by_line(tmp_path: Path) -> None:
    text = _big("a")
    pack = _pack(tmp_path, full_sends=0)
    obs_id = _swapped_id(pack, text)
    page, details = pack.recall("s", obs_id, line=101)
    _, body = _page(page)
    assert body.startswith("a line 100:")
    assert "first_line=101" in page
    assert details["offset"] == len("\n".join(text.split("\n")[:100]).encode()) + 1


def test_a_page_never_splits_a_character(tmp_path: Path) -> None:
    text = "é" * 20_000
    pack = _pack(tmp_path, full_sends=0, recall_max_bytes=1_025)
    obs_id = _swapped_id(pack, text)
    page, details = pack.recall("s", obs_id, offset=0)
    assert details["bytes"] % 2 == 0
    assert _page(page)[1] == "é" * (details["bytes"] // 2)


@pytest.mark.parametrize(
    ("obs_id", "kwargs", "match"),
    [
        ("obs_0123456789abcdef01234567", {}, "unknown observation"),
        ("../../etc/passwd", {}, "unknown observation"),
        (None, {"offset": 10**9}, "outside"),
        (None, {"line": 10**6}, "past the end"),
        (None, {"line": 0}, "at least 1"),
    ],
)
def test_bad_recalls_are_refused(
    tmp_path: Path, obs_id: str | None, kwargs: dict[str, Any], match: str
) -> None:
    pack = _pack(tmp_path, full_sends=0)
    real = _swapped_id(pack, _big("a"))
    with pytest.raises(ValueError, match=match):
        pack.recall("s", obs_id or real, **kwargs)


def test_an_archived_object_that_is_a_symlink_is_not_followed(tmp_path: Path) -> None:
    pack = _pack(tmp_path, full_sends=0)
    obs_id = _swapped_id(pack, _big("a"))
    archive = pack.archive_for("s")._state.archive
    target = tmp_path / "secret"
    target.write_text("secret")
    path = archive.path_of(obs_id)
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(OSError):
        pack.recall("s", obs_id)


def test_a_tampered_object_is_not_reused(tmp_path: Path) -> None:
    message = Message(role=Role.TOOL, tool_call_id="c", name="t", content=_big("a"))
    observation = Observation.of(message)
    archive = ObservationArchive(tmp_path / "archive")
    archive.store(observation)
    archive.store(observation)  # the same bytes again: reused
    archive.path_of(observation.id).write_text("tampered")
    with pytest.raises(OSError, match="does not match"):
        archive.store(observation)


# ---------------------------------------------------------------------------
# Batching and the cost model
# ---------------------------------------------------------------------------


def test_the_cost_model_prices_a_swap_against_the_rewrite() -> None:
    model = SwapCostModel(min_batch_bytes=1_000)
    free = model.decide(saved_bytes=10, saved_tokens=3, rewrite_tokens=0, requests_so_far=1)
    assert (free.swap, free.reason) == (True, "free")
    small = model.decide(saved_bytes=999, saved_tokens=250, rewrite_tokens=10, requests_so_far=50)
    assert (small.swap, small.reason) == (False, "batching")
    # 10k tokens saved for 20 requests at 0.1 beats rewriting 10k at 0.9...
    pays = model.decide(
        saved_bytes=40_000, saved_tokens=10_000, rewrite_tokens=10_000, requests_so_far=20
    )
    assert (pays.swap, pays.reason) == (True, "pays")
    assert pays.benefit == pytest.approx(20_000)
    assert pays.cost == pytest.approx(9_000)
    # ...but not when the run is young and the horizon short,
    young = model.decide(
        saved_bytes=40_000, saved_tokens=10_000, rewrite_tokens=10_000, requests_so_far=2
    )
    assert (young.swap, young.reason) == (False, "costly")
    # nor when compaction is due in two requests and will clear them anyway.
    soon = model.decide(
        saved_bytes=40_000,
        saved_tokens=10_000,
        rewrite_tokens=10_000,
        requests_so_far=50,
        requests_left=2,
    )
    assert not soon.swap
    nothing = model.decide(saved_bytes=0, saved_tokens=0, rewrite_tokens=0, requests_so_far=9)
    assert not nothing.swap


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cache_read_cost": 2.0, "cache_write_cost": 1.0},
        {"horizon_requests": 0},
        {"min_horizon_requests": 0},
        {"min_batch_bytes": -1},
    ],
)
def test_invalid_cost_models_are_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SwapCostModel(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold_bytes": -1},
        {"full_sends": -1},
        {"excerpt_bytes": -1},
        {"recall_max_bytes": 100},
        {"recall_max_lines": 2},
    ],
)
def test_invalid_packs_are_refused(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ObservationPack(directory=tmp_path, **kwargs)


def test_due_outputs_wait_for_a_batch_then_swap_together(tmp_path: Path) -> None:
    one = len(_big("a").encode())
    # More than one output frees, less than two.
    pack = _pack(tmp_path, cost_model=SwapCostModel(min_batch_bytes=one))
    history = _history([_big("a")])
    for _ in range(3):
        history.append(Message.assistant("thinking"))
        assert not _placeholders(pack.project(history, session="s"))
    history = [*history, *_history([_big("b")])[2:]]
    for _ in range(3):
        history.append(Message.assistant("thinking"))
        sent = pack.project(history, session="s")
    assert len(_placeholders(sent)) == 2
    assert pack.stats("s").batches == 1


def test_a_swap_behind_a_break_that_happens_anyway_is_free(tmp_path: Path) -> None:
    # Too costly to break the cache for on its own...
    pack = _pack(tmp_path, cost_model=SwapCostModel(min_batch_bytes=10**9))
    history = [*_history([_big("a")]), Message.assistant("1"), Message.assistant("2")]
    pack.project(history, session="s")
    assert not _placeholders(pack.project(history, session="s"))
    # ...but once something before it changes (a compaction rewrote the
    # history), the request is rewritten from there whatever happens.
    rewritten = [history[0], Message.user("task, summarised"), *history[2:]]
    assert _placeholders(pack.project(rewritten, session="s"))
    stats = pack.stats("s")
    assert (stats.batches, stats.prefix_breaks) == (1, 0)


def test_swaps_survive_a_restart(tmp_path: Path) -> None:
    history = [*_history([_big("a")]), Message.assistant("1"), Message.assistant("2")]
    _pack(tmp_path).project(history, session="thread/1")
    fresh = ObservationPack(directory=tmp_path, cost_model=SwapCostModel(min_batch_bytes=10**9))
    # Already swapped before the restart: still a placeholder, no new decision.
    assert _placeholders(fresh.view(history, session="thread/1"))
    assert (tmp_path / session_directory_name("thread/1") / "observation-pack").is_dir()


def test_swaps_and_recalls_are_in_the_ledger(tmp_path: Path) -> None:
    pack = _pack(tmp_path, full_sends=0)
    obs_id = _swapped_id(pack, _big("a"))
    pack.recall("s", obs_id)
    ledger = tmp_path / "s" / "observation-pack" / "ledger.jsonl"
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [e["event"] for e in events] == ["swap", "recall"]
    assert events[0]["ids"] == [obs_id]
    assert {"saved_tokens", "rewrite_tokens", "benefit", "cost"} <= set(events[0])


# ---------------------------------------------------------------------------
# Failing open
# ---------------------------------------------------------------------------


def test_an_archive_that_cannot_be_written_sends_the_full_output(tmp_path: Path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where the archive directory should be")
    pack = ObservationPack(directory=blocker, full_sends=0, cost_model=EAGER)
    history = _history([_big("a")])
    assert pack.project(history, session="s") == history
    assert pack.stats("s").fail_open == 1


def test_any_failure_sends_the_request_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = _pack(tmp_path, full_sends=0)

    def broken(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(pack, "_project", broken)
    monkeypatch.setattr(pack, "_apply", broken)
    history = _history([_big("a")])
    assert pack.project(history, session="s") == history
    assert pack.view(history, session="s") == history
    assert pack.stats("s").fail_open == 1


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


def test_compaction_clears_into_recallable_stubs(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    archive = pack.archive_for("s")
    compactor = ContextCompactor(context_length=40_000, tool_output_keep_tokens=2_000)
    history = _history([_big(f"o{n}") for n in range(8)])
    cleared = compactor._clear_tool_outputs(history, archive)

    stubs = [m for m in cleared if m.metadata.get(CLEARED_OUTPUT_KEY)]
    assert stubs
    for stub in stubs:
        obs_id = str(stub.metadata[OBSERVATION_ID_KEY])
        assert obs_id in (stub.content or "")
        assert "obs_recall" in (stub.content or "")
        assert "read(path='f" in (stub.content or "")
        original = next(m for m in history if m.tool_call_id == stub.tool_call_id)
        assert _recall_all(pack, obs_id) == original.content
    # Recallable clearing frees about as much as the lossy stub.
    assert all(len(s.content or "") < 400 for s in stubs)
    assert pack.stats("s").cleared_recallable == len(stubs)


def test_clearing_falls_back_to_the_lossy_stub_when_archiving_fails(tmp_path: Path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("x")
    archive = ObservationPack(directory=blocker).archive_for("s")
    compactor = ContextCompactor(context_length=40_000, tool_output_keep_tokens=2_000)
    cleared = compactor._clear_tool_outputs(_history([_big(f"o{n}") for n in range(8)]), archive)
    stubs = [m for m in cleared if m.metadata.get(CLEARED_OUTPUT_KEY)]
    assert stubs
    assert all(OBSERVATION_ID_KEY not in m.metadata for m in stubs)
    assert all("Call the tool again" in (m.content or "") for m in stubs)


class _Summariser:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, messages: list[Message], **_: Any) -> ModelResponse:
        self.prompts.append(messages[-1].content or "")
        return ModelResponse(message=Message.assistant(f"summary {len(self.prompts)}"))


async def test_a_summary_keeps_the_ids_of_the_outputs_it_folded(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    summariser = _Summariser()
    compactor = ContextCompactor(
        context_length=30_000,
        summary_model=summariser,
        tail_turns=1,
        tool_output_keep_tokens=0,
        min_iterations_between_summaries=0,
    )
    # Long replies, so clearing tool output alone cannot make room.
    said = " " + "y" * 10_000
    history = _history([_big(f"o{n}", lines=100) for n in range(12)], said=said)
    outcome = await compactor.compact(
        history, iteration=1, tracker=CompactionTracker(), archive=pack.archive_for("s")
    )
    assert outcome is not None
    assert outcome.stage == "summarize"
    summary = next(m for m in outcome.messages if is_summary_message(m))
    listed = summary.metadata[RECALLABLE_OUTPUTS_KEY]
    assert len(listed) >= 10
    for obs_id, label, _ in listed:
        assert obs_id in (summary.content or "")
        assert label.startswith("read(")
        pack.recall("s", obs_id)  # every listed id is recallable
    # The summariser was told which archive each output is in.
    assert listed[0][0] in summariser.prompts[0]

    # The next summary carries the earlier ids forward and does not summarise
    # the list itself.
    later = _history([_big(f"p{n}", lines=100) for n in range(12)], said=said)[2:]
    more = [
        *outcome.messages,
        *(
            m.model_copy(
                update={
                    "tool_call_id": f"p{m.tool_call_id}",
                    "tool_calls": [c.model_copy(update={"id": f"p{c.id}"}) for c in m.tool_calls],
                }
            )
            for m in later
        ),
    ]
    again = await compactor.compact(
        more, iteration=9, tracker=CompactionTracker(), archive=pack.archive_for("s")
    )
    assert again is not None
    second = next(m for m in again.messages if is_summary_message(m))
    ids = {entry[0] for entry in second.metadata[RECALLABLE_OUTPUTS_KEY]}
    assert {entry[0] for entry in listed} <= ids
    assert "Recallable tool outputs" not in summariser.prompts[-1]
    assert (second.content or "").count("## Recallable tool outputs") == 1


# ---------------------------------------------------------------------------
# Through the agent
# ---------------------------------------------------------------------------

BIG_FILE = _big("big", lines=600)


@tool(idempotent=False)
def read(path: str) -> str:
    """Read a file."""
    return BIG_FILE if path == "big.txt" else f"contents of {path}"


def _scripted_recall(
    requests: list[list[Message]],
) -> FunctionModel:
    """Reads big.txt, does three small steps, recalls the big file's start, answers."""
    recalled: dict[str, Any] = {}

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        turn = len(requests)
        requests.append(list(messages))
        usage = {"prompt_tokens": 100, "completion_tokens": 5}
        if turn == 0:
            call = ToolCall(id="big", name="read", arguments={"path": "big.txt"})
            return ModelResponse(
                message=Message.assistant("Reading.", tool_calls=[call]), usage=usage
            )
        if turn < 4:
            call = ToolCall(id=f"s{turn}", name="read", arguments={"path": f"s{turn}"})
            return ModelResponse(message=Message.assistant("More.", tool_calls=[call]), usage=usage)
        if turn == 4:
            placeholder = next(m for m in messages if m.tool_call_id == "big")
            obs_id = next(
                line.split(": ", 1)[1]
                for line in (placeholder.content or "").splitlines()
                if line.startswith("id: ")
            )
            recalled["id"] = obs_id
            call = ToolCall(id="r", name="obs_recall", arguments={"id": obs_id, "offset": 0})
            return ModelResponse(
                message=Message.assistant("Recall.", tool_calls=[call]), usage=usage
            )
        return ModelResponse(message=Message.assistant("Done."), usage=usage)

    model = FunctionModel(handler)
    model.config = SimpleNamespace(model="tulip-test-observation-pack")  # type: ignore[attr-defined]
    return model


async def test_an_agent_swaps_a_large_output_and_recalls_it_exactly(tmp_path: Path) -> None:
    requests: list[list[Message]] = []
    checkpointer = MemoryCheckpointer()
    agent = Agent(
        model=_scripted_recall(requests),
        tools=[read],
        context_window=1_000_000,
        checkpointer=checkpointer,
        observation_pack=ObservationPackConfig(
            enabled=True, directory=tmp_path, min_batch_bytes=0, horizon_requests=1_000
        ),
        reflexion=False,
        grounding=False,
    )
    events = [e async for e in agent.run("Look at big.txt", thread_id="t1")]
    assert isinstance(events[-1], TerminateEvent)
    assert "obs_recall" in [t["function"]["name"] for t in agent._tool_registry.to_openai_schemas()]

    def big_in(request: list[Message]) -> str:
        return next(m.content or "" for m in request if m.tool_call_id == "big")

    # Requests two and three carry it whole; from the fourth, the placeholder.
    assert [big_in(r) == BIG_FILE for r in requests[1:]] == [True, True, False, False, False]
    recall = next(m.content or "" for m in requests[-1] if m.tool_call_id == "r")
    assert BIG_FILE.startswith(_page(recall)[1])

    # The run's history and its checkpoint keep the full output.
    saved = await checkpointer.load("t1")
    assert saved is not None
    stored = next(m for m in saved.messages if m.tool_call_id == "big")
    assert stored.content == BIG_FILE

    stats = agent.observation_pack.stats("t1")
    assert (stats.swaps, stats.recalls) == (1, 1)
    assert stats.bytes_saved > 3 * len(BIG_FILE.encode()) // 2
    announced = [
        e.data["event"]
        for e in events
        if isinstance(e, CustomEvent) and e.name == "observation_pack"
    ]
    assert announced == ["swap", "recall"]


def test_obs_recall_is_only_registered_when_the_pack_is_on(tmp_path: Path) -> None:
    def names(agent: Agent) -> list[str]:
        agent._initialize()
        return list(agent._tool_registry.tools)

    model = FunctionModel(lambda m, t: ModelResponse(message=Message.assistant("ok")))
    assert "obs_recall" not in names(Agent(model=model, tools=[read]))
    assert Agent(model=model, tools=[read]).observation_pack is None
    on = Agent(model=model, tools=[read], observation_pack=True)
    assert "obs_recall" in names(on)
    assert isinstance(on.observation_pack, ObservationPack)
    assert (
        AgentConfig(model="openai:gpt-4o", observation_pack=None).observation_pack.enabled is False
    )


async def test_obs_recall_without_a_pack_is_an_error(tmp_path: Path) -> None:
    model = FunctionModel(lambda m, t: ModelResponse(message=Message.assistant("ok")))
    agent = Agent(model=model, tools=[read], observation_pack=ObservationPackConfig(enabled=True))
    agent._initialize()
    agent._observation_pack = None
    with pytest.raises(ValueError, match="not enabled"):
        await agent._tool_registry.tools["obs_recall"].execute(id="obs_x")


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def test_a_page_stops_at_the_line_limit(tmp_path: Path) -> None:
    text = "\n".join(f"{n}" for n in range(5_000))
    pack = _pack(tmp_path, full_sends=0, threshold_bytes=1_000)
    obs_id = _swapped_id(pack, text)
    page, details = pack.recall("s", obs_id)
    assert details["lines"] == 398
    assert len(page.splitlines()) <= 400
    assert _recall_all(pack, obs_id) == text


def test_a_long_multibyte_line_gets_whole_characters_at_both_ends() -> None:
    message = Message(role=Role.TOOL, tool_call_id="c", name="t", content="é✓" * 10_000)
    placeholder = placeholder_for(Observation.of(message), 101)
    assert "�" not in placeholder
    assert "é✓é" in placeholder


def test_an_archive_directory_that_is_a_symlink_is_refused(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    root = tmp_path / "s" / "observation-pack"
    root.mkdir(parents=True)
    (root / "objects").symlink_to(elsewhere)
    pack = ObservationPack(directory=tmp_path, full_sends=0, cost_model=EAGER)
    history = _history([_big("a")])
    assert pack.project(history, session="s") == history
    assert pack.stats("s").fail_open == 1
    assert not list(elsewhere.iterdir())


def test_an_archived_object_that_is_not_a_file_is_refused(tmp_path: Path) -> None:
    pack = _pack(tmp_path, full_sends=0)
    obs_id = _swapped_id(pack, _big("a"))
    path = pack.archive_for("s")._state.archive.path_of(obs_id)
    path.unlink()
    path.mkdir()
    with pytest.raises(OSError, match="not a regular file"):
        pack.recall("s", obs_id)


def test_the_caches_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tulip.memory.observation_pack as module

    monkeypatch.setattr(module, "_OBSERVED_CACHE_LIMIT", 1)
    monkeypatch.setattr(module, "_PLACEHOLDER_CACHE_LIMIT", 1)
    pack = _pack(tmp_path, full_sends=0)
    sent = pack.project(_history([_big(f"o{n}") for n in range(5)]), session="s")
    assert len(_placeholders(sent)) == 5
    assert len(pack._observed) <= 2
    assert len(pack._placeholders) <= 2


def test_the_archive_skips_what_is_not_a_text_output(tmp_path: Path) -> None:
    archive = _pack(tmp_path).archive_for("s")
    assert archive.recall_id(Message.user("hello")) is None
    image = Message(role=Role.TOOL, tool_call_id="c", content=encode_image(b"png"))
    assert archive.clear(image, "shot()") is None
    already = Message(
        role=Role.TOOL,
        tool_call_id="c",
        content="stub",
        metadata={OBSERVATION_ID_KEY: "obs_0123456789abcdef01234567"},
    )
    assert archive.recall_id(already) == "obs_0123456789abcdef01234567"
    assert archive.clear(already, "read()") is None


def test_a_session_is_its_thread_or_its_run() -> None:
    assert ObservationPack.session_key("t", "r") == "t"
    assert ObservationPack.session_key(None, "r") == "r"
    assert ObservationPack.session_key(None, None) == "default"
    assert session_directory_name("a/b#2") == "a_b_2"
    assert session_directory_name("") == "_"


def test_the_default_directory_is_under_tmp() -> None:
    import tempfile

    assert ObservationPack().directory == Path(tempfile.gettempdir()) / "tulip-observation-pack"


async def test_a_summary_without_a_working_archive_lists_nothing(tmp_path: Path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("x")
    compactor = ContextCompactor(
        context_length=30_000,
        summary_model=_Summariser(),
        tail_turns=1,
        tool_output_keep_tokens=0,
    )
    history = _history([_big(f"o{n}", lines=100) for n in range(12)], said=" " + "y" * 10_000)
    outcome = await compactor.compact(
        history,
        iteration=1,
        tracker=CompactionTracker(),
        archive=ObservationPack(directory=blocker).archive_for("s"),
    )
    assert outcome is not None
    summary = next(m for m in outcome.messages if is_summary_message(m))
    assert RECALLABLE_OUTPUTS_KEY not in summary.metadata
    assert "Recallable" not in (summary.content or "")
