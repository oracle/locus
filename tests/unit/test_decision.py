# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`tulip.decision` — offline, against a fake OpenAI-compatible server."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from tulip.control import Action, ApprovalOutcome, AuditTrail, ControlPolicy, approve
from tulip.decision import (
    ADMIT_FIELD,
    LETTERS,
    SYSTEM_PROMPT,
    Answer,
    Choice,
    Decision,
    DecisionAdvisor,
    DecisionError,
    DecisionProvider,
    Field,
    LogprobDecider,
    Score,
    TenantDecisionRouter,
    UnknownTenantError,
    YesNo,
    answer_from_logprobs,
    decide_sync,
    describe_action,
    per_tenant_trails,
    render,
    verification_from_decision,
)


# ---------------------------------------------------------------------------
# A fake OpenAI-compatible server
# ---------------------------------------------------------------------------

Probs = Mapping[str, float]


def _response(probs: Probs, *, sampled: str | None = None) -> dict[str, Any]:
    """A chat-completions body whose first token's top logprobs are ``probs``."""
    top = [{"token": tok, "logprob": math.log(p)} for tok, p in probs.items()]
    best = sampled or max(probs, key=lambda k: probs[k])
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": best},
                "logprobs": {
                    "content": [
                        {"token": best, "logprob": math.log(probs[best]), "top_logprobs": top}
                    ]
                },
            }
        ]
    }


class FakeServer:
    """Answers by the question in the rendered user message; records every request."""

    def __init__(self, answers: Mapping[str, Probs], *, delay: float = 0.0) -> None:
        self.answers = answers
        self.delay = delay
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []
        self.in_flight = 0
        self.max_in_flight = 0

    def _question(self, body: dict[str, Any]) -> str:
        user = body["messages"][1]["content"]
        return str(user.split("[question]\n", 1)[1].split("\n", 1)[0])

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        self.headers.append(request.headers)
        return httpx.Response(200, json=_response(self.answers[self._question(body)]))

    async def handle_async(self, request: httpx.Request) -> httpx.Response:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            return self.handle(request)
        finally:
            self.in_flight -= 1


def _decider(server: FakeServer, **kw: Any) -> LogprobDecider:
    return LogprobDecider(
        "http://heads.local/v1/",
        "companion-head",
        transport=httpx.MockTransport(server.handle_async),
        sync_transport=httpx.MockTransport(server.handle),
        **kw,
    )


INTENT = Choice("intent", "What does the speaker want?", ("come", "build", "chat"))
PERSONAL = YesNo("personal", "Does it ask for personal details?")
MEETING = YesNo("meeting", "Does it suggest meeting outside the game?")
MOOD = Score("mood", "How upset is the speaker?", 5)


# ---------------------------------------------------------------------------
# The wire format — pinned verbatim. Trained heads depend on every character.
# ---------------------------------------------------------------------------


def test_render_is_the_contract() -> None:
    assert render("Is it for the companion?", ("yes", "no"), "  hi there \n") == (
        "[input]\n"
        "hi there\n"
        "\n"
        "[question]\n"
        "Is it for the companion?\n"
        "\n"
        "[answers]\n"
        "A) yes\n"
        "B) no\n"
        "\n"
        "Answer with one letter."
    )
    assert SYSTEM_PROMPT == (
        "You answer one question about the input. "
        "Reply with the letter of one listed answer and nothing else."
    )
    assert LETTERS == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def test_fields_render_and_labels() -> None:
    assert PERSONAL.labels == ("yes", "no")
    assert MOOD.labels == ("1", "2", "3", "4", "5")
    assert Score("s", "q?", ("low", "high")).labels == ("low", "high")
    assert INTENT.render("x") == render(INTENT.question, INTENT.labels, "x")
    assert PERSONAL.render("x").endswith("A) yes\nB) no\n\nAnswer with one letter.")
    assert MOOD.render("x").count(")") == 5
    assert Choice("c", "q?", ["a", "b"]).options == ("a", "b")


@pytest.mark.parametrize(
    "build",
    [
        lambda: Choice("", "q?", ("a", "b")),
        lambda: Choice("c", " ", ("a", "b")),
        lambda: Choice("c", "q?", ("a",)),
        lambda: Choice("c", "q?", ("a", "a")),
        lambda: Choice("c", "q?", ("a", "")),
        lambda: Choice("c", "q?", ("a", "b\nc")),
        lambda: Choice("c", "q?", tuple(f"o{i}" for i in range(27))),
        lambda: Score("s", "q?", 1),
    ],
)
def test_bad_fields_are_refused(build: Callable[[], Field]) -> None:
    with pytest.raises(ValueError):
        build()


# ---------------------------------------------------------------------------
# Reading an answer from logprobs
# ---------------------------------------------------------------------------


def test_answer_strips_whitespace_and_sums_variants() -> None:
    lp = [("A", math.log(0.5)), (" A", math.log(0.2)), ("B", math.log(0.1)), ("\n", math.log(0.2))]
    answer = answer_from_logprobs(PERSONAL, lp)
    assert answer.label == "yes"
    assert answer.p_yes == pytest.approx(0.7 / 0.8)
    assert answer.p("no") == pytest.approx(0.1 / 0.8)
    assert answer.coverage == pytest.approx(0.8)
    assert answer.margin == pytest.approx(0.6 / 0.8)
    assert answer.probability == pytest.approx(0.875)


def test_answer_ignores_letters_off_the_list_and_missing_letters_are_zero() -> None:
    answer = answer_from_logprobs(INTENT, {"B": math.log(0.6), "D": math.log(0.3)})
    assert answer.label == "build"
    assert answer.distribution == {"come": 0.0, "build": 1.0, "chat": 0.0}
    assert answer.coverage == pytest.approx(0.6)


def test_answer_with_no_listed_letter_raises() -> None:
    with pytest.raises(DecisionError, match="none of the letters"):
        answer_from_logprobs(PERSONAL, {"Sure": math.log(0.9), "C": math.log(0.1)})
    with pytest.raises(DecisionError):
        answer_from_logprobs(PERSONAL, {"A": float("-inf")})


def test_answer_helpers() -> None:
    answer = answer_from_logprobs(MOOD, {"A": math.log(0.5), "E": math.log(0.5)})
    assert answer.expected == pytest.approx(0.5)
    with pytest.raises(ValueError):
        _ = answer.p_yes
    with pytest.raises(ValueError):
        answer.flagged(0.5)
    assert answer.flagged(0.5, "5")
    with pytest.raises(KeyError):
        answer.p("6")
    yes = answer_from_logprobs(PERSONAL, {"A": math.log(0.3), "B": math.log(0.7)})
    assert yes.flagged(0.3)
    assert not yes.flagged(0.31)
    with pytest.raises(ValueError):
        _ = yes.expected
    tie = answer_from_logprobs(INTENT, {"A": math.log(0.5), "B": math.log(0.5)})
    assert tie.label == "come"  # first listed wins a tie
    record = yes.as_record()
    assert record["label"] == "no"
    assert set(record["distribution"]) == {"yes", "no"}


def test_decision_access() -> None:
    a = answer_from_logprobs(PERSONAL, {"A": 0.0})
    decision = Decision(answers={"personal": a}, model="m", provider="p", latency_ms=1.5)
    assert decision["personal"] is a
    assert "personal" in decision
    assert len(decision) == 1
    assert list(decision) == ["personal"]
    with pytest.raises(KeyError, match="no field"):
        decision["intent"]
    with pytest.raises(TypeError):
        decision.answers["x"] = a  # type: ignore[index]
    assert "text" not in decision.as_record()


# ---------------------------------------------------------------------------
# LogprobDecider against the fake server
# ---------------------------------------------------------------------------


async def test_decider_sends_the_contract_request() -> None:
    server = FakeServer({INTENT.question: {"B": 0.9, "C": 0.1}})
    decider = _decider(
        server,
        api_key="secret-key",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 99},
    )
    decision = await decider.decide("build me a castle", [INTENT])
    assert decision["intent"].label == "build"
    assert decision.model == "companion-head"
    assert decision.provider == "logprob"
    body = server.requests[0]
    assert body == {
        "chat_template_kwargs": {"enable_thinking": False},
        "model": "companion-head",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": INTENT.render("build me a castle")},
        ],
        "max_tokens": 1,  # extra_body cannot override the contract
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 20,
    }
    assert server.headers[0]["authorization"] == "Bearer secret-key"
    assert "secret-key" not in repr(decider)
    await decider.aclose()


async def test_decider_asks_fields_concurrently() -> None:
    server = FakeServer(
        {
            INTENT.question: {"C": 0.8, "A": 0.2},
            PERSONAL.question: {"B": 0.99, "A": 0.01},
            MEETING.question: {"A": 0.6, "B": 0.4},
        },
        delay=0.05,
    )
    decider = _decider(server, concurrency=3)
    decision = await decider.decide("hi", [INTENT, PERSONAL, MEETING])
    assert server.max_in_flight == 3
    assert decision["intent"].label == "chat"
    assert decision["personal"].p_yes == pytest.approx(0.01)
    assert decision["meeting"].flagged(0.5)
    assert decision.latency_ms > 0
    # the input comes first, so all three share the prefix up to the question
    prefixes = {b["messages"][1]["content"].split("[question]")[0] for b in server.requests}
    assert prefixes == {"[input]\nhi\n\n"}


async def test_decider_concurrency_limit() -> None:
    server = FakeServer(
        {INTENT.question: {"A": 1.0}, PERSONAL.question: {"A": 1.0}, MEETING.question: {"A": 1.0}},
        delay=0.02,
    )
    await _decider(server, concurrency=1).decide("x", [INTENT, PERSONAL, MEETING])
    assert server.max_in_flight == 1


def test_decider_sync_and_sampled_token_counted_once() -> None:
    server = FakeServer({PERSONAL.question: {"A": 0.25, "B": 0.75}})
    decider = _decider(server)
    decision = decider.decide_sync("x", [PERSONAL])
    assert decision["personal"].p_yes == pytest.approx(0.25)
    assert decide_sync(decider, "x", [PERSONAL])["personal"].label == "no"
    decider.close()
    decider.close()


def test_decider_uses_sampled_token_when_alternatives_are_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = {"choices": [{"logprobs": {"content": [{"token": "B", "logprob": -0.1}]}}]}
        return httpx.Response(200, json=body)

    decider = LogprobDecider("http://x/v1", "m", sync_transport=httpx.MockTransport(handler))
    assert decider.decide_sync("x", [PERSONAL])["personal"].label == "no"


@pytest.mark.parametrize(
    "respond",
    [
        lambda: httpx.Response(500, json={"error": "boom"}),
        lambda: httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]}),
        lambda: httpx.Response(200, content=b"not json"),
        lambda: httpx.Response(200, json=_response({"Hello": 0.9})),
    ],
)
async def test_decider_raises_rather_than_guesses(respond: Callable[[], httpx.Response]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return respond()

    decider = LogprobDecider(
        "http://x/v1",
        "m",
        transport=httpx.MockTransport(handler),
        sync_transport=httpx.MockTransport(handler),
    )
    with pytest.raises(DecisionError):
        await decider.decide("x", [PERSONAL])
    with pytest.raises(DecisionError):
        decider.decide_sync("x", [PERSONAL])


async def test_decider_transport_failure_is_a_decision_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    decider = LogprobDecider(
        "http://x/v1",
        "m",
        transport=httpx.MockTransport(handler),
        sync_transport=httpx.MockTransport(handler),
    )
    with pytest.raises(DecisionError, match="ConnectError"):
        await decider.decide("x", [PERSONAL])
    with pytest.raises(DecisionError, match="ConnectError"):
        decider.decide_sync("x", [PERSONAL])


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("", "m"), {}),
        (("http://x", ""), {}),
        (("http://x", "m"), {"top_logprobs": 0}),
        (("http://x", "m"), {"concurrency": 0}),
    ],
)
def test_decider_arguments(args: tuple[str, str], kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        LogprobDecider(*args, **kwargs)  # type: ignore[arg-type]


async def test_decide_refuses_empty_and_duplicate_fields() -> None:
    decider = _decider(FakeServer({}))
    with pytest.raises(ValueError):
        await decider.decide("x", [])
    with pytest.raises(ValueError):
        await decider.decide("x", [PERSONAL, YesNo("personal", "again?")])


def test_decider_satisfies_the_protocol() -> None:
    assert isinstance(_decider(FakeServer({})), DecisionProvider)


# ---------------------------------------------------------------------------
# A provider with no sync method, called from sync code inside and outside a loop
# ---------------------------------------------------------------------------


class AsyncOnly:
    """A provider that only implements the protocol's coroutine."""

    def __init__(self, probs: Mapping[str, Probs]) -> None:
        self.probs = probs
        self.calls: list[str] = []

    async def decide(self, text: str, fields: Sequence[Field]) -> Decision:
        self.calls.append(text)
        answers = {
            f.name: answer_from_logprobs(f, {k: math.log(v) for k, v in self.probs[f.name].items()})
            for f in fields
        }
        return Decision(answers=answers, model="async-only", provider="test", latency_ms=0.1)


class Broken:
    async def decide(self, text: str, fields: Sequence[Field]) -> Decision:
        raise DecisionError("down")


def test_decide_sync_without_a_loop() -> None:
    provider = AsyncOnly({"personal": {"A": 0.9, "B": 0.1}})
    assert decide_sync(provider, "x", [PERSONAL])["personal"].label == "yes"


async def test_decide_sync_inside_a_running_loop() -> None:
    provider = AsyncOnly({"personal": {"A": 0.9, "B": 0.1}})
    assert decide_sync(provider, "x", [PERSONAL])["personal"].label == "yes"
    with pytest.raises(DecisionError, match="down"):
        decide_sync(Broken(), "x", [PERSONAL])


def test_decide_sync_checks_what_a_sync_method_returns() -> None:
    class Liar:
        async def decide(self, text: str, fields: Sequence[Field]) -> Decision:
            raise AssertionError

        def decide_sync(self, text: str, fields: Sequence[Field]) -> object:
            return "allow"

    with pytest.raises(DecisionError):
        decide_sync(Liar(), "x", [PERSONAL])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The admit head as a ControlAdvisor: escalate-only
# ---------------------------------------------------------------------------

OPEN = ControlPolicy(require_verification_score=0.0, require_human_for=frozenset())
REFUND = Action(name="refund", asset="cust:1", environment="staging", kind="payment")


def _admit(probs: Probs) -> AsyncOnly:
    return AsyncOnly({"admit": probs})


@pytest.mark.parametrize(
    ("probs", "expected"),
    [
        ({"A": 0.9, "B": 0.05, "C": 0.05}, ApprovalOutcome.ALLOW),
        ({"A": 0.1, "B": 0.8, "C": 0.1}, ApprovalOutcome.REQUIRE_HUMAN),
        ({"A": 0.1, "B": 0.1, "C": 0.8}, ApprovalOutcome.DENY),
    ],
)
def test_advisor_can_raise_an_allow(probs: Probs, expected: str) -> None:
    decision = approve(REFUND, policy=OPEN, advisor=DecisionAdvisor(_admit(probs)))
    assert decision.policy_outcome == ApprovalOutcome.ALLOW
    assert decision.outcome == expected


@pytest.mark.parametrize(
    ("policy", "held"),
    [
        (ControlPolicy(require_verification_score=0.0), ApprovalOutcome.REQUIRE_HUMAN),
        (
            ControlPolicy(require_verification_score=0.0, deny_for=frozenset({"payment"})),
            ApprovalOutcome.DENY,
        ),
    ],
)
def test_advisor_never_lowers_a_hold(policy: ControlPolicy, held: str) -> None:
    production = Action(name="refund", environment="production", kind="payment")
    advisor = DecisionAdvisor(_admit({"A": 0.99, "B": 0.005, "C": 0.005}))
    decision = approve(production, policy=policy, advisor=advisor)
    assert decision.outcome == held


async def test_advisor_works_inside_admit_style_async_code() -> None:
    advisor = DecisionAdvisor(_admit({"A": 0.1, "B": 0.1, "C": 0.8}))
    assert approve(REFUND, policy=OPEN, advisor=advisor).outcome == ApprovalOutcome.DENY


def test_advisor_down_is_absent() -> None:
    decision = approve(REFUND, policy=OPEN, advisor=DecisionAdvisor(Broken()))
    assert decision.outcome == ApprovalOutcome.ALLOW
    assert decision.model_outcome is None


def test_advisor_min_probability_and_certified_threshold() -> None:
    unsure = _admit({"A": 0.3, "B": 0.4, "C": 0.3})
    assert DecisionAdvisor(unsure, min_probability=0.5).advise(REFUND) is None
    assert DecisionAdvisor(unsure).advise(REFUND) == ApprovalOutcome.REQUIRE_HUMAN
    # hold_at reads the risk 1 - P(allow): 0.7 here
    assert DecisionAdvisor(unsure, hold_at=0.7).advise(REFUND) == ApprovalOutcome.REQUIRE_HUMAN
    assert DecisionAdvisor(unsure, hold_at=0.71).advise(REFUND) == ApprovalOutcome.ALLOW
    denyish = _admit({"A": 0.3, "B": 0.2, "C": 0.5})
    assert DecisionAdvisor(denyish, hold_at=0.5).advise(REFUND) == ApprovalOutcome.DENY


def test_advisor_reads_the_described_action() -> None:
    provider = _admit({"A": 1.0})
    DecisionAdvisor(provider).advise(REFUND)
    assert provider.calls == [describe_action(REFUND)]
    rich = Action(name="pay", asset="a", tags=frozenset({"x", "y"}), cost_usd=12.5, kind="k")
    text = describe_action(rich)
    assert "tags: x, y" in text
    assert "cost: $12.50" in text
    assert "asset: a" in text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"field": Choice("admit", "q?", ("allow", "deny"))},
        {"min_probability": 1.5},
        {"hold_at": 0.0},
    ],
)
def test_advisor_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DecisionAdvisor(_admit({}), **kwargs)


def test_admit_field() -> None:
    assert ADMIT_FIELD.labels == ("allow", "require_human", "deny")


# ---------------------------------------------------------------------------
# Safety heads as ControlPolicy verification
# ---------------------------------------------------------------------------


def _safety(personal: float, meeting: float) -> Decision:
    answers = {
        "personal": answer_from_logprobs(
            PERSONAL, {"A": math.log(personal), "B": math.log(1 - personal)}
        ),
        "meeting": answer_from_logprobs(
            MEETING, {"A": math.log(meeting), "B": math.log(1 - meeting)}
        ),
    }
    return Decision(answers=answers, model="safety-head", provider="test", latency_ms=1.0)


SAY = Action(name="say", environment="world", kind="speech")
STRICT = ControlPolicy(require_verification_score=0.8, require_human_for=frozenset())


def test_verification_clear_reply_is_allowed() -> None:
    verdict = verification_from_decision(_safety(0.01, 0.02), {"personal": 0.4, "meeting": 0.5})
    assert verdict.survives
    assert verdict.confidence == pytest.approx(1 - 0.04)
    assert verdict.evidence_quality == pytest.approx(1.0)
    assert "safety-head" in verdict.notes
    assert approve(SAY, policy=STRICT, verdict=verdict).outcome == ApprovalOutcome.ALLOW


def test_verification_flagged_reply_is_denied() -> None:
    verdict = verification_from_decision(_safety(0.45, 0.02), {"personal": 0.4, "meeting": 0.5})
    assert not verdict.survives
    assert verdict.confidence == 0.0
    assert [r.weight for r in verdict.refutations] == ["fatal"]
    assert "personal" in verdict.refutations[0].reason
    assert approve(SAY, policy=STRICT, verdict=verdict).outcome == ApprovalOutcome.DENY


def test_verification_near_miss_wants_a_person() -> None:
    verdict = verification_from_decision(_safety(0.2, 0.02), {"personal": 0.4, "meeting": 0.5})
    assert verdict.survives
    assert verdict.confidence == pytest.approx(0.5)
    assert approve(SAY, policy=STRICT, verdict=verdict).outcome == ApprovalOutcome.REQUIRE_HUMAN


@pytest.mark.parametrize(
    "thresholds",
    [{}, {"personal": 0.0}, {"personal": 1.5}],
)
def test_verification_arguments(thresholds: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        verification_from_decision(_safety(0.1, 0.1), thresholds)


def test_verification_needs_yes_no_fields() -> None:
    decision = Decision(
        answers={"intent": answer_from_logprobs(INTENT, {"A": 0.0})},
        model="m",
        provider="p",
        latency_ms=0,
    )
    with pytest.raises(ValueError):
        verification_from_decision(decision, {"intent": 0.5})
    with pytest.raises(KeyError):
        verification_from_decision(decision, {"personal": 0.5})


# ---------------------------------------------------------------------------
# Tenants: each served by its own head, each on its own chain
# ---------------------------------------------------------------------------


class Head:
    """A tenant's head that remembers every input it saw."""

    def __init__(self, name: str, p_yes: float) -> None:
        self.name = name
        self.p_yes = p_yes
        self.seen: list[str] = []

    async def decide(self, text: str, fields: Sequence[Field]) -> Decision:
        self.seen.append(text)
        answers = {
            f.name: answer_from_logprobs(
                f, {"A": math.log(self.p_yes), "B": math.log(1 - self.p_yes)}
            )
            for f in fields
        }
        return Decision(answers=answers, model=self.name, provider="head", latency_ms=1.0)


def _router(tmp_path: Path, **kw: Any) -> tuple[TenantDecisionRouter, Head, Head, Any]:
    acme, globex = Head("acme-head", 0.9), Head("globex-head", 0.1)
    audit_for = per_tenant_trails(tmp_path)
    router = TenantDecisionRouter({"acme": acme, "globex": globex}, audit_for=audit_for, **kw)
    return router, acme, globex, audit_for


async def test_tenant_is_served_only_by_its_own_head(tmp_path: Path) -> None:
    router, acme, globex, audit_for = _router(tmp_path)
    decision = await router.decide("acme's secret words", [PERSONAL], tenant="acme")
    assert decision.model == "acme-head"
    assert decision.meta == {"tenant": "acme", "served_by": "tenant"}
    assert acme.seen == ["acme's secret words"]
    assert globex.seen == []

    acme_trail, globex_trail = audit_for("acme"), audit_for("globex")
    assert len(acme_trail) == 1
    assert len(globex_trail) == 0
    record = acme_trail.records()[0]
    assert record.event_type == "decision"
    assert record.payload["tenant"] == "acme"
    assert record.payload["answers"]["personal"]["label"] == "yes"
    assert "text" not in record.payload  # the input is not recorded by default
    assert "secret" not in json.dumps(record.payload)
    assert acme_trail.verify()
    assert (tmp_path / "acme" / "decisions.jsonl").exists()
    assert not (tmp_path / "globex" / "decisions.jsonl").exists()


async def test_two_tenants_never_touch(tmp_path: Path) -> None:
    router, acme, globex, audit_for = _router(tmp_path)
    await asyncio.gather(
        router.decide("from acme", [PERSONAL], tenant="acme"),
        router.decide("from globex", [PERSONAL], tenant="globex"),
    )
    assert acme.seen == ["from acme"]
    assert globex.seen == ["from globex"]
    assert [r.payload["tenant"] for r in audit_for("acme").records()] == ["acme"]
    assert [r.payload["tenant"] for r in audit_for("globex").records()] == ["globex"]
    assert audit_for("acme").records()[0].payload["model"] == "acme-head"
    assert audit_for("globex").records()[0].payload["model"] == "globex-head"


async def test_unknown_tenant_is_refused_never_borrowed(tmp_path: Path) -> None:
    router, acme, globex, audit_for = _router(tmp_path)
    for tenant in ("initech", "", "../acme", "acme/../globex", ".."):
        with pytest.raises(UnknownTenantError):
            await router.decide("x", [PERSONAL], tenant=tenant)
    assert acme.seen == []
    assert globex.seen == []
    with pytest.raises(UnknownTenantError):
        audit_for("../acme")
    with pytest.raises(UnknownTenantError):
        TenantDecisionRouter({"../evil": acme})


async def test_public_head_is_opt_in_and_recorded(tmp_path: Path) -> None:
    base = Head("public-base", 0.2)
    router, acme, _, audit_for = _router(tmp_path, public=base)
    decision = await router.decide("x", [PERSONAL], tenant="initech")
    assert decision.model == "public-base"
    assert decision.meta["served_by"] == "public"
    assert audit_for("initech").records()[0].payload["served_by"] == "public"
    assert acme.seen == []


async def test_record_text_is_opt_in(tmp_path: Path) -> None:
    router, _, _, audit_for = _router(tmp_path, record_text=True)
    await router.decide("hello", [PERSONAL], tenant="acme")
    assert audit_for("acme").records()[0].payload["text"] == "hello"


async def test_a_decision_the_record_cannot_hold_is_not_returned() -> None:
    def audit_for(tenant: str) -> AuditTrail:
        raise OSError("disk full")

    router = TenantDecisionRouter({"acme": Head("a", 0.5)}, audit_for=audit_for)
    with pytest.raises(DecisionError, match="could not be recorded"):
        await router.decide("x", [PERSONAL], tenant="acme")


def test_router_resolver_sync_and_bound_tenant(tmp_path: Path) -> None:
    acme = Head("acme-head", 0.7)
    sync_server = FakeServer({PERSONAL.question: {"A": 0.6, "B": 0.4}})
    sync_head = _decider(sync_server)
    heads: dict[str, DecisionProvider] = {"acme": acme, "globex": sync_head}
    audit_for = per_tenant_trails(tmp_path)
    router = TenantDecisionRouter(heads.get, audit_for=audit_for)

    assert router.decide_sync("x", [PERSONAL], tenant="acme").model == "acme-head"
    assert router.decide_sync("y", [PERSONAL], tenant="globex").model == "companion-head"
    assert len(sync_server.requests) == 1
    with pytest.raises(UnknownTenantError):
        router.decide_sync("x", [PERSONAL], tenant="initech")

    bound = router.for_tenant("acme")
    assert isinstance(bound, DecisionProvider)
    advisor = DecisionAdvisor(
        bound,
        field=Choice("personal", "q?", ("allow", "require_human", "deny")),
    )
    # the bound head answers yes/no letters A/B, read as allow/require_human
    assert advisor.advise(REFUND) == ApprovalOutcome.ALLOW
    assert len(audit_for("acme")) == 2
    assert len(audit_for("globex")) == 1


async def test_bound_tenant_async(tmp_path: Path) -> None:
    router, acme, globex, audit_for = _router(tmp_path)
    decision = await router.for_tenant("globex").decide("z", [PERSONAL])
    assert decision.model == "globex-head"
    assert globex.seen == ["z"]
    assert acme.seen == []


def test_router_rejects_non_decisions() -> None:
    class Liar:
        async def decide(self, text: str, fields: Sequence[Field]) -> Decision:
            raise AssertionError

        def decide_sync(self, text: str, fields: Sequence[Field]) -> object:
            return {"personal": "yes"}

    router = TenantDecisionRouter({"acme": Liar()})  # type: ignore[dict-item]
    with pytest.raises(DecisionError):
        router.decide_sync("x", [PERSONAL], tenant="acme")


def test_router_async_only_head_from_sync(tmp_path: Path) -> None:
    router = TenantDecisionRouter({"acme": Head("a", 0.5)}, audit_for=per_tenant_trails(tmp_path))
    assert router.decide_sync("x", [PERSONAL], tenant="acme").model == "a"


def test_answer_type_is_frozen() -> None:
    answer = answer_from_logprobs(PERSONAL, {"A": 0.0})
    assert isinstance(answer, Answer)
    with pytest.raises(AttributeError):
        answer.field = "x"  # type: ignore[misc]
