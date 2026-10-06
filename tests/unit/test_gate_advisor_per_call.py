# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``gate_tool`` with an advisor, and with a verdict worked out for each call.

A gated tool is wrapped once and serves every call a shared agent makes. A
verification that depends on the call (the safety of the text this call would
send) cannot be fixed when the tool is wrapped, and a trained control model has
to see every call. These tests pin both, and the one property neither may
break: a model or a verification that fails never lets a call through.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tulip.control import (
    Action,
    AdmissionError,
    AuditTrail,
    ControlPolicy,
    InMemoryApprovals,
    admit,
    admit_sync,
    gate_tool,
)
from tulip.control.verification import VerificationResult
from tulip.tools.decorator import Tool, tool


def _sender(sent: list[str]) -> Tool:
    @tool
    def send(text: str) -> str:
        """Send a message."""
        sent.append(text)
        return "sent"

    return send


def _open_policy(**overrides: Any) -> ControlPolicy:
    return ControlPolicy(
        require_verification_score=overrides.pop("require_verification_score", 0.0),
        require_human_for=overrides.pop("require_human_for", frozenset()),
        **overrides,
    )


class _Advisor:
    def __init__(self, advice: Any = None, *, fail: bool = False) -> None:
        self.advice = advice
        self.fail = fail
        self.seen: list[Action] = []

    def advise(self, action: Action) -> str | None:
        self.seen.append(action)
        if self.fail:
            raise RuntimeError("model down")
        return self.advice  # type: ignore[no-any-return]


def _verified(confidence: float) -> VerificationResult:
    return VerificationResult(survives=True, confidence=confidence, evidence_quality=1.0)


# --------------------------------------------------------------------------
# advisor=
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_advisor_denies_what_the_policy_allows() -> None:
    sent: list[str] = []
    trail = AuditTrail()
    advisor = _Advisor("deny")
    gated = gate_tool(_sender(sent), policy=_open_policy(), advisor=advisor, trail=trail)

    payload = json.loads(await gated.execute(text="hi"))

    assert sent == []
    assert payload["status"] == "denied"
    assert "control model escalated allow -> deny" in payload["reason"]
    assert [a.name for a in advisor.seen] == ["send"]
    entry = trail.records()[-1].payload
    assert entry["policy_outcome"] == "allow"
    assert entry["model_outcome"] == "deny"


@pytest.mark.asyncio
@pytest.mark.parametrize("advisor", [_Advisor(None), _Advisor(fail=True), _Advisor("allow")])
async def test_a_silent_or_broken_advisor_changes_nothing(advisor: _Advisor) -> None:
    sent: list[str] = []
    gated = gate_tool(_sender(sent), policy=_open_policy(), advisor=advisor)

    assert await gated.execute(text="hi") == "sent"
    assert sent == ["hi"]


@pytest.mark.asyncio
async def test_an_advisor_never_releases_what_the_policy_holds() -> None:
    sent: list[str] = []
    advisor = _Advisor("allow")
    gated = gate_tool(
        _sender(sent),
        policy=_open_policy(deny_for=frozenset({"send"})),
        action=lambda name, kw: Action(name=name, tags=frozenset({"send"})),
        advisor=advisor,
    )

    payload = json.loads(await gated.execute(text="hi"))

    assert payload["status"] == "denied"
    assert sent == []
    assert advisor.seen == []  # consulted only on the allow branch


@pytest.mark.asyncio
async def test_without_an_advisor_the_trail_entry_keeps_its_shape() -> None:
    trail = AuditTrail()
    gated = gate_tool(_sender([]), policy=_open_policy(), trail=trail)

    await gated.execute(text="hi")

    assert set(trail.records()[-1].payload) == {"action", "asset", "outcome", "reason"}


@pytest.mark.asyncio
async def test_admit_and_admit_sync_take_an_advisor() -> None:
    action = Action(name="send")

    async def perform() -> str:
        return "ran"

    with pytest.raises(AdmissionError) as held:
        await admit(action, perform, policy=_open_policy(), advisor=_Advisor("require_human"))
    assert held.value.decision.outcome == "require_human"
    assert held.value.decision.escalated_by_model

    with pytest.raises(AdmissionError):
        admit_sync(action, lambda: "ran", policy=_open_policy(), advisor=_Advisor("deny"))
    assert admit_sync(action, lambda: "ran", policy=_open_policy(), advisor=_Advisor()) == "ran"


@pytest.mark.asyncio
async def test_an_advisor_hold_pauses_and_an_approval_runs_it_once() -> None:
    sent: list[str] = []
    store = InMemoryApprovals()
    advisor = _Advisor("require_human")
    gated = gate_tool(
        _sender(sent),
        policy=_open_policy(),
        advisor=advisor,
        approval=store,
        on_refusal="interrupt",
    )

    held = json.loads(await gated.execute(text="hi"))
    assert held["__interrupt__"] is True
    store.decide(held["metadata"]["approval_id"], "approved", by="alice")

    assert await gated.execute(text="hi") == "sent"
    assert sent == ["hi"]
    # Asked about the first call, about the second before its approval is
    # found, and once more when the approved call is admitted.
    assert len(advisor.seen) == 3


@pytest.mark.asyncio
async def test_an_approved_call_is_still_refused_when_the_advisor_now_denies() -> None:
    sent: list[str] = []
    store = InMemoryApprovals()
    advisor = _Advisor("require_human")
    gated = gate_tool(
        _sender(sent),
        policy=_open_policy(),
        advisor=advisor,
        approval=store,
        on_refusal="interrupt",
    )

    held = json.loads(await gated.execute(text="hi"))
    store.decide(held["metadata"]["approval_id"], "approved", by="alice")
    advisor.advice = "deny"

    payload = json.loads(await gated.execute(text="hi"))

    assert payload["status"] == "denied"
    assert sent == []


# --------------------------------------------------------------------------
# verdict= / finding= per call
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_verdict_callable_is_asked_about_every_call() -> None:
    sent: list[str] = []
    asked: list[tuple[str, dict[str, Any]]] = []

    def safety(name: str, arguments: dict[str, Any]) -> VerificationResult:
        asked.append((name, arguments))
        return _verified(0.1 if "bad" in arguments["text"] else 0.99)

    gated = gate_tool(
        _sender(sent),
        policy=_open_policy(require_verification_score=0.8),
        verdict=safety,
    )

    assert await gated.execute(text="hello") == "sent"
    held = json.loads(await gated.execute(text="bad words"))

    assert sent == ["hello"]
    assert held["status"] == "held_for_approval"
    assert "verification confidence 0.10" in held["reason"]
    assert asked == [("send", {"text": "hello"}), ("send", {"text": "bad words"})]


@pytest.mark.asyncio
async def test_an_async_verdict_callable_is_awaited() -> None:
    sent: list[str] = []

    async def safety(name: str, arguments: dict[str, Any]) -> VerificationResult:
        return VerificationResult(
            survives=arguments["text"] != "refuted", confidence=0.99, evidence_quality=1.0
        )

    gated = gate_tool(
        _sender(sent), policy=_open_policy(require_verification_score=0.5), verdict=safety
    )

    assert await gated.execute(text="ok") == "sent"
    payload = json.loads(await gated.execute(text="refuted"))
    assert payload["status"] == "denied"
    assert sent == ["ok"]


@pytest.mark.asyncio
async def test_a_verdict_callable_returning_none_is_no_verification() -> None:
    sent: list[str] = []
    strict = gate_tool(
        _sender(sent),
        policy=_open_policy(require_verification_score=0.5),
        verdict=lambda name, arguments: None,
    )
    lenient = gate_tool(_sender(sent), policy=_open_policy(), verdict=lambda name, arguments: None)

    held = json.loads(await strict.execute(text="a"))
    assert held["outcome"] == "require_human"
    assert "no verification provided" in held["reason"]
    assert await lenient.execute(text="b") == "sent"
    assert sent == ["b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["verdict", "finding"])
async def test_a_failing_callable_fails_closed_even_when_the_policy_needs_none(
    kind: str,
) -> None:
    sent: list[str] = []
    trail = AuditTrail()

    def broken(name: str, arguments: dict[str, Any]) -> Any:
        raise TimeoutError("safety head too slow")

    gated = gate_tool(
        _sender(sent),
        # Requires no verification: reading the failure as `None` would allow.
        policy=_open_policy(),
        trail=trail,
        on_refusal="interrupt",
        approval=InMemoryApprovals(),
        **{kind: broken},
    )

    payload = json.loads(await gated.execute(text="hi"))

    assert sent == []
    assert payload["status"] == "denied"  # a deny never pauses, even in interrupt mode
    assert "TimeoutError" in payload["reason"]
    assert trail.records()[-1].payload["outcome"] == "deny"


@pytest.mark.asyncio
async def test_a_failing_callable_raises_admission_error_when_asked_to() -> None:
    def broken(name: str, arguments: dict[str, Any]) -> Any:
        raise RuntimeError("no")

    gated = gate_tool(_sender([]), policy=_open_policy(), verdict=broken, on_refusal="raise")

    with pytest.raises(AdmissionError) as refused:
        await gated.execute(text="hi")
    assert refused.value.decision.outcome == "deny"


@pytest.mark.asyncio
async def test_an_edited_approval_is_verified_again_with_the_edited_arguments() -> None:
    sent: list[str] = []
    store = InMemoryApprovals()
    asked: list[str] = []

    def safety(name: str, arguments: dict[str, Any]) -> VerificationResult:
        asked.append(arguments["text"])
        return VerificationResult(
            survives="secret" not in arguments["text"], confidence=0.99, evidence_quality=1.0
        )

    gated = gate_tool(
        _sender(sent),
        policy=_open_policy(require_verification_score=0.5, require_human_for=frozenset({"x"})),
        action=lambda name, kw: Action(name=name, kind="x"),
        verdict=safety,
        approval=store,
        on_refusal="interrupt",
    )

    held = json.loads(await gated.execute(text="hello"))
    store.decide(
        held["metadata"]["approval_id"],
        "approved",
        by="alice",
        arguments={"text": "the secret"},
    )

    payload = json.loads(await gated.execute(text="hello"))

    assert payload["status"] == "denied"
    assert sent == []
    assert asked[-1] == "the secret"


@pytest.mark.asyncio
async def test_plain_values_still_work() -> None:
    sent: list[str] = []
    gated = gate_tool(
        _sender(sent),
        policy=_open_policy(require_verification_score=0.5),
        verdict=_verified(0.9),
    )

    assert await gated.execute(text="hi") == "sent"
