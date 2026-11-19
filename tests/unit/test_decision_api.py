# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The Decisions API providers and the ``provider:model`` registry, over mocked HTTP."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from locus.decision import (
    MAX_QUESTIONS,
    Choice,
    DecisionError,
    DecisionProvider,
    OCIDecider,
    OpenAIDecider,
    Score,
    YesNo,
    decide_sync,
    get_decider,
    list_deciders,
    register_decider,
)


FIELDS = [
    YesNo("asks_refund", "Does the customer ask for a refund?"),
    Choice("department", "Which department?", ("shipping", "billing", "technical")),
    Score("urgency", "How urgent?", ("low", "medium", "high")),
]

TEXT = "My laptop is late; refund me if it cannot arrive by Monday."

# The shape OCI returned for these three questions, with probabilities that
# are not all 0 and 1 so the reading is exercised.
GOOD = {
    "model": "openai.gpt-6-luna",
    "answers": [
        {"type": "predicate", "name": "asks_refund", "probability": 0.9},
        {
            "type": "choice",
            "name": "department",
            "choice": "shipping",
            "probabilities": [
                {"value": "shipping", "probability": 0.7},
                {"value": "billing", "probability": 0.2},
                {"value": "technical", "probability": 0.1},
            ],
            "confidence": 0.8,
        },
        {
            "type": "score",
            "name": "urgency",
            "score": 1.5,
            "probabilities": [
                {"value": 0, "label": "low", "probability": 0.0},
                {"value": 1, "label": "medium", "probability": 0.5},
                {"value": 2, "label": "high", "probability": 0.5},
            ],
            "confidence": 0.6,
        },
    ],
    "usage": {"input_tokens": 461, "output_tokens": 0, "total_tokens": 461},
}


class _Recorder:
    """An httpx handler that records requests and returns a canned reply."""

    def __init__(self, status: int = 200, payload: Any = GOOD) -> None:
        self.status = status
        self.payload = payload
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.payload, bytes):
            return httpx.Response(self.status, content=self.payload)
        return httpx.Response(self.status, json=self.payload)

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


def _openai(rec: _Recorder, **kw: Any) -> OpenAIDecider:
    return OpenAIDecider(
        "gpt-6-luna",
        api_key="test-key",
        transport=httpx.MockTransport(rec),
        sync_transport=httpx.MockTransport(rec),
        **kw,
    )


def _oci(rec: _Recorder, **kw: Any) -> OCIDecider:
    kw.setdefault("api_key", "oci-test-key")
    return OCIDecider(
        "openai.gpt-6-luna",
        region="eu-frankfurt-1",
        compartment_id="ocid1.compartment.oc1..fake",
        transport=httpx.MockTransport(rec),
        sync_transport=httpx.MockTransport(rec),
        **kw,
    )


# -- the request --------------------------------------------------------------


async def test_one_request_carries_every_field() -> None:
    rec = _Recorder()
    await _openai(rec).decide(TEXT, FIELDS)
    assert len(rec.requests) == 1
    request = rec.requests[0]
    assert str(request.url) == "https://api.openai.com/v1/decisions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert rec.body == {
        "model": "gpt-6-luna",
        "input": TEXT,
        "questions": [
            {
                "type": "predicate",
                "name": "asks_refund",
                "instructions": "Does the customer ask for a refund?",
            },
            {
                "type": "choice",
                "name": "department",
                "instructions": "Which department?",
                "choices": [{"value": "shipping"}, {"value": "billing"}, {"value": "technical"}],
            },
            {
                "type": "score",
                "name": "urgency",
                "instructions": "How urgent?",
                "levels": [
                    {"label": "low", "description": "low"},
                    {"label": "medium", "description": "medium"},
                    {"label": "high", "description": "high"},
                ],
            },
        ],
    }


async def test_extra_body_cannot_change_the_fixed_keys() -> None:
    rec = _Recorder()
    await _openai(rec, extra_body={"prompt_cache_key": "k", "model": "other"}).decide(TEXT, FIELDS)
    assert rec.body["prompt_cache_key"] == "k"
    assert rec.body["model"] == "gpt-6-luna"


def test_request_limits_are_checked_before_sending() -> None:
    decider = _openai(_Recorder())
    with pytest.raises(ValueError, match="at most 10"):
        decider.request_body(TEXT, [Score("s", "How much?", 11)])
    too_many = [YesNo(f"q{i}", "Is it?") for i in range(MAX_QUESTIONS + 1)]
    with pytest.raises(ValueError, match="at most 400"):
        decider.request_body(TEXT, too_many)
    with pytest.raises(ValueError, match="empty"):
        decider.request_body("  ", FIELDS)
    with pytest.raises(ValueError, match="at least one field"):
        decider.request_body(TEXT, [])


# -- the answer ---------------------------------------------------------------


async def test_answers_become_a_decision() -> None:
    decision = await _openai(_Recorder()).decide(TEXT, FIELDS)
    assert decision.provider == "openai"
    assert decision.model == "openai.gpt-6-luna"
    assert decision["asks_refund"].p_yes == pytest.approx(0.9)
    assert decision["asks_refund"].p("no") == pytest.approx(0.1)
    assert decision["department"].label == "shipping"
    assert decision["department"].p("billing") == pytest.approx(0.2)
    assert decision["urgency"].expected == pytest.approx(0.75)  # score 1.5 of 0..2
    assert decision["department"].coverage == pytest.approx(1.0)
    assert decision.meta["confidence"] == {"department": 0.8, "urgency": 0.6}
    assert decision.meta["usage"]["input_tokens"] == 461


@pytest.mark.parametrize(
    ("answers", "match"),
    [
        ([], "no answer came back"),
        ([{"type": "refusal", "name": "asks_refund"}], "refused"),
        ([{"type": "choice", "name": "asks_refund"}], "expected a predicate"),
        ([{"type": "predicate", "name": "asks_refund"}], "no probability"),
        ([{"type": "predicate", "name": "asks_refund", "probability": 1.5}], "no probability"),
    ],
)
async def test_an_unreadable_predicate_raises(answers: list[dict[str, Any]], match: str) -> None:
    rec = _Recorder(payload={"answers": answers})
    with pytest.raises(DecisionError, match=match):
        await _openai(rec).decide(TEXT, FIELDS[:1])


async def test_a_label_off_the_list_raises() -> None:
    payload = {
        "answers": [
            {
                "type": "choice",
                "name": "department",
                "probabilities": [{"value": "legal", "probability": 1.0}],
            }
        ]
    }
    with pytest.raises(DecisionError, match="not one of"):
        await _openai(_Recorder(payload=payload)).decide(TEXT, FIELDS[1:2])


async def test_no_probability_mass_raises() -> None:
    payload = {"answers": [{"type": "choice", "name": "department", "probabilities": []}]}
    with pytest.raises(DecisionError, match="no probabilities"):
        await _openai(_Recorder(payload=payload)).decide(TEXT, FIELDS[1:2])


async def test_http_errors_raise_with_the_server_message() -> None:
    rec = _Recorder(status=403, payload={"error": {"message": "tenancy not allowlisted"}})
    with pytest.raises(DecisionError, match="HTTP 403: tenancy not allowlisted"):
        await _openai(rec).decide(TEXT, FIELDS)


async def test_a_body_without_answers_raises() -> None:
    with pytest.raises(DecisionError, match="no answers"):
        await _openai(_Recorder(payload=b"not json")).decide(TEXT, FIELDS)


async def test_a_transport_failure_raises() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    decider = OpenAIDecider("gpt-6-luna", api_key="k", transport=httpx.MockTransport(boom))
    with pytest.raises(DecisionError, match="ConnectError"):
        await decider.decide(TEXT, FIELDS)
    await decider.aclose()


def test_decide_sync_reads_the_same_answer() -> None:
    rec = _Recorder()
    decider = _openai(rec)
    decision = decide_sync(decider, TEXT, FIELDS)
    assert decision["department"].label == "shipping"
    assert len(rec.requests) == 1
    decider.close()


# -- OpenAI authentication ----------------------------------------------------


def test_openai_key_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    decider = OpenAIDecider("gpt-6-luna")
    assert decider._headers()["Authorization"] == "Bearer from-env"
    assert "from-env" not in repr(decider)


def test_openai_without_a_key_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIDecider("gpt-6-luna")


# -- OCI ----------------------------------------------------------------------


async def test_oci_posts_to_the_regional_openai_endpoint() -> None:
    rec = _Recorder()
    decision = await _oci(rec, project_ocid="ocid1.generativeaiproject.oc1..p").decide(TEXT, FIELDS)
    request = rec.requests[0]
    assert str(request.url) == (
        "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com/openai/v1/decisions"
    )
    assert request.headers["Authorization"] == "Bearer oci-test-key"
    assert request.headers["opc-compartment-id"] == "ocid1.compartment.oc1..fake"
    assert request.headers["opc-project-id"] == "ocid1.generativeaiproject.oc1..p"
    assert rec.body["model"] == "openai.gpt-6-luna"
    assert decision.provider == "oci"


def test_oci_never_reads_the_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    decider = OCIDecider(
        "openai.gpt-6-luna", auth_type="instance_principal", compartment_id="ocid1.c"
    )
    assert "Authorization" not in decider._headers()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "exactly one"),
        ({"profile": "P", "api_key": "k"}, "exactly one"),
        ({"auth_type": "magic", "compartment_id": "c"}, "auth_type must be"),
        ({"auth_type": "instance_principal"}, "compartment_id is required"),
    ],
)
def test_oci_auth_modes_are_checked(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        OCIDecider("openai.gpt-6-luna", **kwargs)


async def test_oci_signs_with_the_profile_signer(monkeypatch: pytest.MonkeyPatch) -> None:
    import locus.models.providers.oci.openai_compat as compat

    signed: list[str] = []

    class _Signer:
        def do_request_sign(self, prepared: Any) -> None:
            signed.append(prepared.url)
            prepared.headers["authorization"] = "Signature fake"

    monkeypatch.setattr(compat, "_build_signer_from_profile", lambda profile, cfg: _Signer())
    monkeypatch.setattr(compat, "_refresh_callable_for", lambda signer, **kw: None)
    rec = _Recorder()
    decider = OCIDecider(
        "openai.gpt-6-luna",
        profile="P",
        region="eu-frankfurt-1",
        compartment_id="ocid1.compartment.oc1..fake",
        transport=httpx.MockTransport(rec),
    )
    await decider.decide(TEXT, FIELDS)
    assert signed == [
        "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com/openai/v1/decisions"
    ]
    assert rec.requests[0].headers["authorization"] == "Signature fake"
    await decider.aclose()


def test_oci_compartment_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..env")
    decider = OCIDecider("openai.gpt-6-luna", api_key="k")
    assert decider.compartment_id == "ocid1.compartment.oc1..env"


# -- the registry -------------------------------------------------------------


def test_registry_resolves_both_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert {"oci", "openai"} <= set(list_deciders())
    oai = get_decider("openai:gpt-6-luna")
    oci = get_decider("oci:openai.gpt-6-luna", api_key="k", region="eu-frankfurt-1")
    assert isinstance(oai, OpenAIDecider)
    assert oai.model == "gpt-6-luna"
    assert isinstance(oci, OCIDecider)
    assert oci.model == "openai.gpt-6-luna"
    assert isinstance(oci, DecisionProvider)


@pytest.mark.parametrize("spec", ["gpt-6-luna", "oci:", ":model", "nope:model"])
def test_registry_refuses_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        get_decider(spec)


def test_registry_takes_a_custom_provider() -> None:
    made: list[tuple[str, dict[str, Any]]] = []

    def factory(model_id: str, **kwargs: Any) -> OpenAIDecider:
        made.append((model_id, kwargs))
        return OpenAIDecider(model_id, api_key="k")

    register_decider("mine", factory)
    try:
        get_decider("mine:small", timeout=5.0)
        assert made == [("small", {"timeout": 5.0})]
    finally:
        from locus.decision import registry

        registry._DECIDERS.pop("mine", None)
    with pytest.raises(ValueError):
        register_decider("bad:prefix", factory)
