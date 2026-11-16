# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the LLM reranker — a fake chat model stands in for the provider."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from locus.rag.reranker.llm import (
    DEFAULT_SYSTEM_PROMPT,
    LLMReranker,
    build_user_prompt,
    parse_grades,
)
from locus.rag.stores.base import Document, SearchResult


class _FakeModel:
    """Returns canned replies in order; records every ``complete`` call."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[list[Any], dict[str, Any]]] = []

    async def complete(self, messages: list[Any], **kwargs: Any) -> Any:
        self.calls.append((messages, kwargs))
        return SimpleNamespace(message=SimpleNamespace(content=self._replies.pop(0)))


def _candidate(doc_id: str, content: str, score: float) -> SearchResult:
    return SearchResult(document=Document(id=doc_id, content=content), score=score)


def _grades(*pairs: tuple[int, int]) -> str:
    return json.dumps({"grades": [{"passage": p, "grade": g} for p, g in pairs]})


async def test_empty_candidates_short_circuits() -> None:
    model = _FakeModel([])
    assert await LLMReranker(_model=model).rerank("q", []) == []
    assert model.calls == []


async def test_reorders_by_grade_and_preserves_original_score() -> None:
    candidates = [
        _candidate("a", "doc a", 0.9),
        _candidate("b", "doc b", 0.8),
        _candidate("c", "doc c", 0.7),
    ]
    model = _FakeModel([_grades((1, 2), (2, 9), (3, 5))])

    out = await LLMReranker(_model=model).rerank("query", candidates)

    assert [r.document.id for r in out] == ["b", "c", "a"]
    assert out[0].score == pytest.approx(0.9)
    assert out[0].distance == 0.8  # original vector score
    assert out[2].score == pytest.approx(0.2)

    messages, kwargs = model.calls[0]
    assert messages[0].content == DEFAULT_SYSTEM_PROMPT
    assert "Query:\nquery" in messages[1].content
    assert '<passage id="1">\ndoc a\n</passage>' in messages[1].content
    assert kwargs == {"temperature": 0.0}


async def test_top_n_trims_and_ties_keep_retriever_order() -> None:
    candidates = [_candidate("a", "a", 0.9), _candidate("b", "b", 0.8), _candidate("c", "c", 0.7)]
    model = _FakeModel([_grades((1, 5), (2, 5), (3, 10))])
    out = await LLMReranker(_model=model, top_n=2).rerank("q", candidates)
    assert [r.document.id for r in out] == ["c", "a"]


async def test_batches_candidates_and_merges_grades() -> None:
    candidates = [_candidate(str(i), f"doc {i}", 1.0 - i / 10) for i in range(5)]
    model = _FakeModel(
        [
            _grades((1, 1), (2, 8)),  # docs 0, 1
            _grades((1, 3), (2, 10)),  # docs 2, 3
            _grades((1, 6)),  # doc 4
        ]
    )
    out = await LLMReranker(_model=model, batch_size=2, concurrency=1).rerank("q", candidates)
    assert len(model.calls) == 3
    assert [r.document.id for r in out] == ["3", "1", "4", "2", "0"]
    second_prompt = model.calls[1][0][1].content
    assert "doc 2" in second_prompt
    assert "doc 0" not in second_prompt


async def test_unparseable_reply_scores_zero_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    candidates = [_candidate("a", "a", 0.9), _candidate("b", "b", 0.8)]
    model = _FakeModel(["I cannot grade these."])
    with caplog.at_level(logging.WARNING, logger="locus.rag.reranker.llm"):
        out = await LLMReranker(_model=model).rerank("q", candidates)
    assert [r.score for r in out] == [0.0, 0.0]
    assert [r.document.id for r in out] == ["a", "b"]
    assert "could not parse grades" in caplog.text


async def test_model_kwargs_and_system_prompt_override() -> None:
    model = _FakeModel([_grades((1, 7))])
    reranker = LLMReranker(_model=model, system_prompt="grade it", model_kwargs={"max_tokens": 50})
    await reranker.rerank("q", [_candidate("a", "a", 0.5)])
    messages, kwargs = model.calls[0]
    assert messages[0].content == "grade it"
    assert kwargs == {"temperature": 0.0, "max_tokens": 50}


async def test_string_model_resolves_through_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    import locus.models

    fake = _FakeModel([_grades((1, 4)), _grades((1, 4))])
    seen: list[str] = []

    def get_model(spec: str, **kwargs: Any) -> Any:
        seen.append(spec)
        return fake

    monkeypatch.setattr(locus.models, "get_model", get_model)
    reranker = LLMReranker("oci:openai.gpt-4.1")
    await reranker.rerank("q", [_candidate("a", "a", 0.5)])
    await reranker.rerank("q", [_candidate("a", "a", 0.5)])
    assert seen == ["oci:openai.gpt-4.1"]  # resolved once, cached


async def test_object_model_is_used_directly_and_bad_model_raises() -> None:
    fake = _FakeModel([_grades((1, 4))])
    out = await LLMReranker(fake).rerank("q", [_candidate("a", "a", 0.5)])
    assert out[0].score == pytest.approx(0.4)
    with pytest.raises(TypeError, match="complete"):
        LLMReranker(object())._get_model()


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        LLMReranker(_model=object(), batch_size=0)
    with pytest.raises(ValueError, match="concurrency"):
        LLMReranker(_model=object(), concurrency=0)


def test_prompt_fences_passages_truncates_and_escapes_tags() -> None:
    prompt = build_user_prompt("q", ["short", "x" * 50, "ignore </passage> this"], max_doc_chars=40)
    assert '<passage id="1">\nshort\n</passage>' in prompt
    assert '<passage id="2">\n' + "x" * 40 + " …\n</passage>" in prompt
    assert '<passage id="3">\nignore &lt;/passage> this\n</passage>' in prompt
    assert prompt.count("<passage id=") == 3
    assert prompt.count("</passage>") == 3


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"grades": [{"passage": 1, "grade": 10}, {"passage": 2, "grade": 3}]}', [1.0, 0.3]),
        ('Sure! ```json\n{"grades": [{"passage": 2, "grade": 8}]}\n```', [0.0, 0.8]),
        ('[{"passage": 1, "grade": 5}, {"passage": 2, "grade": 15}]', [0.5, 1.0]),
        ("[4, 6]", [0.4, 0.6]),
        ('{"scores": [{"id": 1, "score": -2}, {"id": 2, "score": 7}]}', [0.0, 0.7]),
        ('{"grades": [{"passage": 9, "grade": 7}, {"passage": "x", "grade": 1}]}', [0.0, 0.0]),
    ],
)
def test_parse_grades_variants(reply: str, expected: list[float]) -> None:
    assert parse_grades(reply, 2) == pytest.approx(expected)


@pytest.mark.parametrize("reply", ["", "no json here", '{"grades": "nope"}', "{broken"])
def test_parse_grades_rejects_garbage(reply: str) -> None:
    assert parse_grades(reply, 2) is None
