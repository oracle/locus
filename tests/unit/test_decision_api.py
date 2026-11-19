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
        # json.dumps writes NaN / Infinity, so malformed numbers can be served.
        return httpx.Response(self.status, content=json.dumps(self.payload).encode())

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
        ([{"type": "predicate", "name": "asks_refund", "probability": 1.5}], r"not in \[0, 1\]"),
    ],
)
async def test_an_unreadable_predicate_raises(answers: list[dict[str, Any]], match: str) -> None:
    rec = _Recorder(payload={"answers": answers})
    with pytest.raises(DecisionError, match=match):
        await _openai(rec).decide(TEXT, FIELDS[:1])


_REFUSAL = {"type": "refusal", "name": "asks_refund"}
_YES = {"type": "predicate", "name": "asks_refund", "probability": 0.9}
_NO = {"type": "predicate", "name": "asks_refund", "probability": 0.1}


@pytest.mark.parametrize(
    "answers",
    [[_REFUSAL, _YES], [_YES, _REFUSAL], [_YES, _NO]],
    ids=["refusal-then-answer", "answer-then-refusal", "two-answers"],
)
async def test_a_duplicate_answer_raises(answers: list[dict[str, Any]]) -> None:
    """Neither answer is kept: a refusal must never vanish behind a valid one."""
    rec = _Recorder(payload={"answers": answers})
    with pytest.raises(DecisionError, match="'asks_refund' twice"):
        await _openai(rec).decide(TEXT, FIELDS[:1])


async def test_a_duplicate_for_a_field_not_asked_raises() -> None:
    extra = {"type": "predicate", "name": "other", "probability": 0.5}
    rec = _Recorder(payload={"answers": [_YES, extra, extra]})
    with pytest.raises(DecisionError, match="'other' twice"):
        await _openai(rec).decide(TEXT, FIELDS[:1])


def _choice(probabilities: list[dict[str, Any]]) -> dict[str, Any]:
    return {"answers": [{"type": "choice", "name": "department", "probabilities": probabilities}]}


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (-0.1, r"not in \[0, 1\]"),
        (1.5, r"not in \[0, 1\]"),
        (float("inf"), r"not in \[0, 1\]"),
        (float("nan"), r"not in \[0, 1\]"),
        (True, "not a number"),
        (False, "not a number"),
        ("0.5", "not a number"),
    ],
)
async def test_a_malformed_predicate_probability_raises(value: Any, match: str) -> None:
    payload = {"answers": [{"type": "predicate", "name": "asks_refund", "probability": value}]}
    with pytest.raises(DecisionError, match=match):
        await _openai(_Recorder(payload=payload)).decide(TEXT, FIELDS[:1])


@pytest.mark.parametrize("field_index", [0, 1, 2], ids=["predicate", "choice", "score"])
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
@pytest.mark.parametrize("value", [10**400, -(10**400)], ids=["huge-positive", "huge-negative"])
async def test_huge_integer_probability_raises_decision_error(
    field_index: int, sync: bool, value: int
) -> None:
    question = FIELDS[field_index]
    raw: dict[str, Any] = {
        "type": ("predicate", "choice", "score")[field_index],
        "name": question.name,
    }
    if field_index == 0:
        raw["probability"] = value
    else:
        key = "label" if field_index == 2 else "value"
        raw["probabilities"] = [{key: question.labels[0], "probability": value}]
    decider = _openai(_Recorder(payload={"answers": [raw]}))
    try:
        if sync:
            with pytest.raises(DecisionError, match=r"not in \[0, 1\]"):
                decider.decide_sync(TEXT, [question])
        else:
            with pytest.raises(DecisionError, match=r"not in \[0, 1\]"):
                await decider.decide(TEXT, [question])
    finally:
        decider.close()
        await decider.aclose()


@pytest.mark.parametrize(
    ("probabilities", "match"),
    [
        # A negative mass is refused, not dropped: 0.2 / -0.8 must not read as a certain 0.2.
        (
            [{"value": "shipping", "probability": 0.2}, {"value": "billing", "probability": -0.8}],
            r"not in \[0, 1\]",
        ),
        (
            [{"value": "shipping", "probability": 5}, {"value": "billing", "probability": 5}],
            r"not in \[0, 1\]",
        ),
        ([{"value": "shipping", "probability": float("inf")}], r"not in \[0, 1\]"),
        ([{"value": "shipping", "probability": float("nan")}], r"not in \[0, 1\]"),
        ([{"value": "shipping", "probability": True}], "not a number"),
        ([{"value": "shipping"}], "not a number"),
        (
            [{"value": "shipping", "probability": 0.5}, {"value": "shipping", "probability": 0.5}],
            "'shipping' twice",
        ),
        (
            [{"value": "shipping", "probability": 0.9}, {"value": "billing", "probability": 0.8}],
            "sum to 1.7000",
        ),
        ([{"value": "shipping", "probability": 0.5}], "sum to 0.5000"),
    ],
)
async def test_a_malformed_distribution_raises(
    probabilities: list[dict[str, Any]], match: str
) -> None:
    with pytest.raises(DecisionError, match=match):
        await _openai(_Recorder(payload=_choice(probabilities))).decide(TEXT, FIELDS[1:2])


async def test_infinity_in_the_json_body_raises() -> None:
    """``json.loads`` reads a bare ``Infinity``; it must not reach the distribution."""
    body = (
        b'{"answers": [{"type": "choice", "name": "department", "probabilities": '
        b'[{"value": "shipping", "probability": Infinity}]}]}'
    )
    with pytest.raises(DecisionError, match=r"not in \[0, 1\]"):
        await _openai(_Recorder(payload=body)).decide(TEXT, FIELDS[1:2])


async def test_rounding_within_the_tolerance_is_accepted() -> None:
    probabilities = [
        {"value": "shipping", "probability": 0.333},
        {"value": "billing", "probability": 0.333},
        {"value": "technical", "probability": 0.333},
    ]
    decision = await _openai(_Recorder(payload=_choice(probabilities))).decide(TEXT, FIELDS[1:2])
    answer = decision["department"]
    assert sum(answer.distribution.values()) == pytest.approx(1.0)
    assert answer.p("shipping") == pytest.approx(1 / 3)
    assert answer.coverage == pytest.approx(0.999)


async def test_a_listed_label_left_out_reads_as_zero() -> None:
    probabilities = [
        {"value": "shipping", "probability": 0.75},
        {"value": "billing", "probability": 0.25},
    ]
    decision = await _openai(_Recorder(payload=_choice(probabilities))).decide(TEXT, FIELDS[1:2])
    assert decision["department"].p("technical") == 0.0
    assert decision["department"].label == "shipping"


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


@pytest.mark.parametrize("field_index", [1, 2])
@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize(
    ("probabilities", "match"),
    [
        (123, "probabilities must be an array"),
        ({}, "probabilities must be an array"),
        ("invalid", "probabilities must be an array"),
        (None, "probabilities must be an array"),
        ([None], "entry must be an object"),
        ([{"value": [], "label": [], "probability": 0.5}], "not one of"),
        ([{"value": {}, "label": {}, "probability": 0.5}], "not one of"),
    ],
)
async def test_malformed_probabilities_raise_decision_error(
    field_index: int, sync: bool, probabilities: Any, match: str
) -> None:
    question = FIELDS[field_index]
    payload = {
        "answers": [
            {
                "type": "choice" if field_index == 1 else "score",
                "name": question.name,
                "probabilities": probabilities,
            }
        ]
    }
    decider = _openai(_Recorder(payload=payload))
    try:
        if sync:
            with pytest.raises(DecisionError, match=match):
                decider.decide_sync(TEXT, [question])
        else:
            with pytest.raises(DecisionError, match=match):
                await decider.decide(TEXT, [question])
    finally:
        decider.close()
        await decider.aclose()


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
def test_oci_auth_modes_are_checked(
    kwargs: dict[str, Any], match: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OCI_PROFILE", raising=False)
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


@pytest.mark.parametrize("via_registry", [False, True])
def test_oci_profile_falls_back_to_the_environment(
    monkeypatch: pytest.MonkeyPatch, via_registry: bool
) -> None:
    monkeypatch.setenv("OCI_PROFILE", "ENV_PROFILE")
    kwargs = {"compartment_id": "ocid1.compartment.oc1..fake"}
    decider = (
        get_decider("oci:openai.gpt-6-luna", **kwargs)
        if via_registry
        else OCIDecider("openai.gpt-6-luna", **kwargs)
    )
    assert isinstance(decider, OCIDecider)
    assert decider.profile == "ENV_PROFILE"


@pytest.mark.parametrize("via_registry", [False, True])
@pytest.mark.parametrize(
    "auth", [{"profile": "EXPLICIT"}, {"auth_type": "instance_principal"}, {"api_key": "k"}]
)
def test_oci_explicit_auth_overrides_the_environment(
    monkeypatch: pytest.MonkeyPatch, via_registry: bool, auth: dict[str, str]
) -> None:
    monkeypatch.setenv("OCI_PROFILE", "ENV_PROFILE")
    kwargs = {"compartment_id": "ocid1.compartment.oc1..fake", **auth}
    decider = (
        get_decider("oci:openai.gpt-6-luna", **kwargs)
        if via_registry
        else OCIDecider("openai.gpt-6-luna", **kwargs)
    )
    assert isinstance(decider, OCIDecider)
    assert decider.profile == auth.get("profile")
    assert decider.auth_type == auth.get("auth_type")


@pytest.mark.parametrize("via_registry", [False, True])
@pytest.mark.parametrize(
    ("locus_region", "oci_region", "explicit", "expected"),
    [
        (None, None, None, "us-chicago-1"),
        (None, "eu-frankfurt-1", None, "eu-frankfurt-1"),
        ("eu-paris-1", "eu-frankfurt-1", None, "eu-paris-1"),
        ("", "eu-frankfurt-1", None, "eu-frankfurt-1"),
        ("eu-paris-1", "eu-frankfurt-1", "uk-london-1", "uk-london-1"),
    ],
)
def test_oci_region_precedence(
    monkeypatch: pytest.MonkeyPatch,
    via_registry: bool,
    locus_region: str | None,
    oci_region: str | None,
    explicit: str | None,
    expected: str,
) -> None:
    for name, value in (("LOCUS_OCI_REGION", locus_region), ("OCI_REGION", oci_region)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    kwargs: dict[str, Any] = {"api_key": "k"}
    if explicit is not None:
        kwargs["region"] = explicit
    decider = (
        get_decider("oci:openai.gpt-6-luna", **kwargs)
        if via_registry
        else OCIDecider("openai.gpt-6-luna", **kwargs)
    )
    assert isinstance(decider, OCIDecider)
    assert decider.region == expected
    assert decider.url == (
        f"https://inference.generativeai.{expected}.oci.oraclecloud.com/openai/v1/decisions"
    )


@pytest.mark.parametrize("via_registry", [False, True])
def test_oci_explicit_endpoint_overrides_the_environment(
    monkeypatch: pytest.MonkeyPatch, via_registry: bool
) -> None:
    monkeypatch.setenv("LOCUS_OCI_REGION", "eu-paris-1")
    monkeypatch.setenv("OCI_REGION", "eu-frankfurt-1")
    kwargs = {"api_key": "k", "base_url": "https://example.test/openai/v1"}
    decider = (
        get_decider("oci:openai.gpt-6-luna", **kwargs)
        if via_registry
        else OCIDecider("openai.gpt-6-luna", **kwargs)
    )
    assert isinstance(decider, OCIDecider)
    assert decider.region == "us-chicago-1"
    assert decider.url == "https://example.test/openai/v1/decisions"


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
