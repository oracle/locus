# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A hosted provider: the OpenAI Decisions API, served by OCI Generative AI or by OpenAI.

``POST /decisions`` answers typed questions about one shared input, each with
a probability per answer, in one request of up to 400 questions. That is the
contract :mod:`locus.decision` draws, so the fields map onto its question
types one to one:

=========  ===========  ====================================================
Field      Question     What the :class:`~locus.decision.Answer` is read from
=========  ===========  ====================================================
YesNo      predicate    ``probability`` is P(yes); P(no) is the rest
Choice     choice       ``probabilities[].probability``, matched by ``value``
Score      score        ``probabilities[].probability``, matched by ``label``
=========  ===========  ====================================================

OCI Generative AI serves the endpoint on ``/openai/v1/decisions`` under IAM
signing, with the compartment in ``opc-compartment-id`` and the model
``openai.gpt-6-luna`` (:meth:`DecisionsAPIDecider.on_oci`). OpenAI serves it
on ``https://api.openai.com/v1/decisions`` behind a bearer key, model
``gpt-6-luna``. Either way one decision is one request, so the fields of an
input cost one round trip instead of one per field.

A refusal, an answer of the wrong type, a missing answer or an HTTP error is a
:class:`~locus.decision.DecisionError`: the provider never guesses.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeGuard

from locus.core.loop_bound import loop_bound
from locus.decision.fields import (
    YES_NO,
    Answer,
    Decision,
    DecisionError,
    Field,
    Score,
    YesNo,
    check_fields,
)


if TYPE_CHECKING:
    import httpx


__all__ = [
    "MAX_QUESTIONS_PER_REQUEST",
    "MAX_SCORE_LEVELS",
    "OCI_DECISIONS_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_DECISIONS_MODEL",
    "DecisionsAPIDecider",
]

#: The most questions one ``POST /decisions`` takes.
MAX_QUESTIONS_PER_REQUEST = 400
#: The most levels a ``score`` question takes (a :class:`Score` may list up to 26).
MAX_SCORE_LEVELS = 10
#: OpenAI's base URL; the endpoint is ``/decisions`` under it.
OPENAI_BASE_URL = "https://api.openai.com/v1"
#: The model that serves the Decisions API at OpenAI.
OPENAI_DECISIONS_MODEL = "gpt-6-luna"
#: The same model as OCI Generative AI names it.
OCI_DECISIONS_MODEL = "openai.gpt-6-luna"


def _is_probability(value: object) -> TypeGuard[float]:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _detail(payload: bytes) -> str:
    """The server's ``message`` (OCI) or ``error.message`` (OpenAI), for an error."""
    try:
        body = json.loads(payload)
    except ValueError:
        return ""
    if not isinstance(body, dict):
        return ""
    message = body.get("message")
    error = body.get("error")
    if message is None and isinstance(error, dict):
        message = error.get("message")
    return f": {message}" if isinstance(message, str) and message else ""


class DecisionsAPIDecider:
    """Decisions from the OpenAI Decisions API, at OCI Generative AI or at OpenAI.

    Every field of an input goes in one request (``batch_size`` at most, 400
    by default, the endpoint's limit); the answers come back in the same
    order and are read into :class:`~locus.decision.Answer` objects by field
    name. The endpoint's per-answer ``confidence`` for choice and score
    questions, and its token ``usage``, land in :attr:`Decision.meta`.

    Args:
        base_url: The OpenAI-style base URL; ``/decisions`` is appended.
            Defaults to OpenAI's. :meth:`on_oci` derives the OCI one.
        model: The model name as the endpoint knows it: ``gpt-6-luna`` at
            OpenAI, ``openai.gpt-6-luna`` on OCI.
        api_key: Sent as a bearer token when given (OpenAI).
        auth: An ``httpx.Auth`` that signs each request instead of a key: the
            OCI request signer :meth:`on_oci` builds, or your own.
        headers: Extra headers on every request.
        timeout: Seconds per request.
        batch_size: Most questions per request, 1 to 400.
        transport: An ``httpx`` async transport (tests pass
            ``httpx.MockTransport``).
        sync_transport: The same, for :meth:`decide_sync`.
    """

    provider_name = "decisions"

    def __init__(
        self,
        base_url: str = OPENAI_BASE_URL,
        model: str = OPENAI_DECISIONS_MODEL,
        *,
        api_key: str | None = None,
        auth: httpx.Auth | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        batch_size: int = MAX_QUESTIONS_PER_REQUEST,
        transport: httpx.AsyncBaseTransport | None = None,
        sync_transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        if api_key and auth is not None:
            raise ValueError("pass api_key or auth, not both")
        if not 1 <= batch_size <= MAX_QUESTIONS_PER_REQUEST:
            raise ValueError(f"batch_size must be 1 to {MAX_QUESTIONS_PER_REQUEST}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._auth = auth
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.batch_size = batch_size
        self._transport = transport
        self._sync_transport = sync_transport
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        self._sync_lock = threading.Lock()

    def __repr__(self) -> str:
        # Neither the key nor the signer appears in a repr, a log line or an exception.
        return f"DecisionsAPIDecider(base_url={self.base_url!r}, model={self.model!r})"

    # -- OCI -----------------------------------------------------------------

    @classmethod
    def on_oci(
        cls,
        *,
        model: str = OCI_DECISIONS_MODEL,
        profile: str | None = None,
        auth_type: str | None = None,
        compartment_id: str | None = None,
        region: str | None = None,
        config_file: str = "~/.oci/config",
        base_url: str | None = None,
        **kwargs: Any,
    ) -> DecisionsAPIDecider:
        """The endpoint as OCI Generative AI serves it, signed with OCI IAM.

        The auth switches are those of the ``oci:`` chat transport. Pass
        ``profile=`` (a section of ``config_file``; the default is
        ``LOCUS_OCI_PROFILE`` / ``OCI_PROFILE``, else ``DEFAULT``), or a
        principal ``auth_type=`` (``"instance_principal"``,
        ``"resource_principal"``, ``"oke_workload_identity"``; the default
        is ``LOCUS_OCI_AUTH_TYPE`` / ``OCI_AUTH_TYPE``). ``region`` defaults
        to ``LOCUS_OCI_REGION`` / ``OCI_REGION``, then the SDK's default
        GenAI region. ``compartment_id`` goes out as ``opc-compartment-id``;
        it defaults to ``LOCUS_OCI_COMPARTMENT`` / ``OCI_COMPARTMENT`` /
        ``OCI_COMPARTMENT_ID``, then to the profile's tenancy, and must be
        given under a principal. Session tokens and principals are
        refreshed the way the chat transport refreshes them. ``kwargs``
        go to the constructor (``timeout``, ``batch_size``, ``headers``,
        the transports).

        Raises:
            ValueError: Both or neither auth mode after the defaults, an
                unknown ``auth_type``, or no compartment to send.
        """
        from locus.models.providers.oci import openai_compat as oci

        if auth_type is None and profile is None:
            auth_type = os.environ.get("LOCUS_OCI_AUTH_TYPE") or os.environ.get("OCI_AUTH_TYPE")
        profile, auth_type = oci.resolve_profile_auth(profile, auth_type)
        if profile is None and auth_type is None:
            profile, _ = oci.resolve_profile_auth(None, "api_key")
        if profile is not None and auth_type is not None:
            raise ValueError(
                "DecisionsAPIDecider.on_oci: pass profile= or a principal auth_type=, "
                f"not both (got profile={profile!r}, auth_type={auth_type!r})"
            )
        if auth_type is not None and auth_type not in oci.PRINCIPAL_AUTH_TYPES:
            raise ValueError(
                f"auth_type must be one of {oci.PRINCIPAL_AUTH_TYPES}, got {auth_type!r}"
            )

        region = (
            region
            or os.environ.get("LOCUS_OCI_REGION")
            or os.environ.get("OCI_REGION")
            or oci.DEFAULT_OCI_GENAI_REGION
        )
        compartment_id = oci.resolve_oci_compartment(
            compartment_id, profile=profile, config_file=config_file
        )
        if not compartment_id:
            raise ValueError(
                "DecisionsAPIDecider.on_oci: compartment_id is required (the endpoint "
                "takes it as opc-compartment-id); pass it or set OCI_COMPARTMENT"
            )
        auth = oci.build_oci_request_auth(
            profile=profile,
            auth_type=auth_type,
            compartment_id=compartment_id,
            config_file=config_file,
        )
        return cls(base_url or oci.build_oci_openai_base_url(region), model, auth=auth, **kwargs)

    # -- the request ---------------------------------------------------------

    @staticmethod
    def question(field: Field) -> dict[str, Any]:
        """The Decisions API question for one field. Public so a request can be checked."""
        if isinstance(field, YesNo):
            return {"type": "predicate", "name": field.name, "instructions": field.question}
        if isinstance(field, Score):
            if len(field.labels) > MAX_SCORE_LEVELS:
                raise ValueError(
                    f"field {field.name!r} has {len(field.labels)} levels; "
                    f"a Decisions API score takes 2 to {MAX_SCORE_LEVELS}"
                )
            return {
                "type": "score",
                "name": field.name,
                "instructions": field.question,
                "levels": [{"label": label} for label in field.labels],
            }
        return {
            "type": "choice",
            "name": field.name,
            "instructions": field.question,
            "choices": [{"value": option} for option in field.options],
        }

    def request_body(self, text: str, fields: Sequence[Field]) -> dict[str, Any]:
        """The JSON body sent for ``fields`` about ``text`` (one batch)."""
        return {
            "model": self.model,
            "input": text,
            "questions": [self.question(field) for field in fields],
        }

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @property
    def _url(self) -> str:
        return f"{self.base_url}/decisions"

    def _batches(self, fields: Sequence[Field]) -> list[Sequence[Field]]:
        return [fields[i : i + self.batch_size] for i in range(0, len(fields), self.batch_size)]

    # -- the answers ---------------------------------------------------------

    @staticmethod
    def _answer(field: Field, item: Mapping[str, Any]) -> Answer:
        kind = item.get("type")
        if kind == "refusal":
            raise DecisionError(f"field {field.name!r}: the model declined to answer")
        if isinstance(field, YesNo):
            probability = item.get("probability")
            if kind != "predicate" or not _is_probability(probability):
                raise DecisionError(
                    f"field {field.name!r}: expected a predicate answer with a "
                    f"probability, got {kind!r}"
                )
            p_yes = float(probability)
            return Answer(
                field=field.name,
                kind="yesno",
                labels=YES_NO,
                distribution={"yes": p_yes, "no": 1.0 - p_yes},
                coverage=1.0,
            )
        if isinstance(field, Score):
            expected, key, noun = "score", "label", "levels"
        else:
            expected, key, noun = "choice", "value", "options"
        if kind != expected:
            raise DecisionError(f"field {field.name!r}: expected a {expected} answer, got {kind!r}")
        mass = dict.fromkeys(field.labels, 0.0)
        for entry in item.get("probabilities") or []:
            if not isinstance(entry, Mapping):
                continue
            label, probability = entry.get(key), entry.get("probability")
            if isinstance(label, str) and label in mass and _is_probability(probability):
                mass[label] += float(probability)
        coverage = sum(mass.values())
        if coverage <= 0.0:
            raise DecisionError(
                f"field {field.name!r}: the answer lists none of the field's {noun} {field.labels}"
            )
        return Answer(
            field=field.name,
            kind=expected,
            labels=field.labels,
            distribution={label: m / coverage for label, m in mass.items()},
            coverage=min(1.0, coverage),
        )

    def _read(
        self, fields: Sequence[Field], status: int, payload: bytes
    ) -> tuple[list[Answer], dict[str, Any]]:
        if status >= 400:
            raise DecisionError(f"the Decisions endpoint answered HTTP {status}{_detail(payload)}")
        try:
            body = json.loads(payload)
            raw = body["answers"]
        except (ValueError, KeyError, TypeError) as exc:
            raise DecisionError("the Decisions endpoint returned no answers") from exc
        if not isinstance(raw, list) or len(raw) != len(fields):
            got = len(raw) if isinstance(raw, list) else 0
            raise DecisionError(f"asked {len(fields)} question(s), the endpoint answered {got}")
        items: list[Mapping[str, Any]] = [item if isinstance(item, Mapping) else {} for item in raw]
        by_name = {item["name"]: item for item in items if isinstance(item.get("name"), str)}
        answers: list[Answer] = []
        confidence: dict[str, float] = {}
        for position, field in enumerate(fields):
            item = by_name.get(field.name, items[position])
            answers.append(self._answer(field, item))
            value = item.get("confidence")
            if _is_probability(value):
                confidence[field.name] = float(value)
        usage = body.get("usage")
        meta: dict[str, Any] = {
            "confidence": confidence,
            "usage": usage if isinstance(usage, dict) else {},
        }
        return answers, meta

    def _decision(
        self, results: Sequence[tuple[list[Answer], dict[str, Any]]], started: float
    ) -> Decision:
        answers: dict[str, Answer] = {}
        confidence: dict[str, float] = {}
        usage: dict[str, Any] = {}
        for batch_answers, meta in results:
            answers.update({a.field: a for a in batch_answers})
            confidence.update(meta["confidence"])
            for key, value in meta["usage"].items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    usage[key] = usage.get(key, 0) + value
                elif key not in usage:
                    usage[key] = value
        return Decision(
            answers=answers,
            model=self.model,
            provider=self.provider_name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            meta={"confidence": confidence, "usage": usage},
        )

    # -- async ---------------------------------------------------------------

    def _build_client(self) -> httpx.AsyncClient:
        import httpx  # noqa: PLC0415

        return httpx.AsyncClient(timeout=self.timeout, transport=self._transport, auth=self._auth)

    async def _ask(self, text: str, fields: Sequence[Field]) -> tuple[list[Answer], dict[str, Any]]:
        import httpx  # noqa: PLC0415

        client = loop_bound(self, "_client", self._build_client)
        try:
            response = await client.post(
                self._url, json=self.request_body(text, fields), headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise DecisionError(f"the Decisions request failed ({type(exc).__name__})") from exc
        return self._read(fields, response.status_code, response.content)

    async def decide(self, text: str, fields: Sequence[Field]) -> Decision:
        """Answer every field about ``text`` in one request (one per ``batch_size`` fields)."""
        check_fields(fields)
        started = time.perf_counter()
        results = await asyncio.gather(*(self._ask(text, batch) for batch in self._batches(fields)))
        return self._decision(results, started)

    async def aclose(self) -> None:
        """Close the async client, while its loop is still running."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- sync ----------------------------------------------------------------

    def decide_sync(self, text: str, fields: Sequence[Field]) -> Decision:
        """The same, from synchronous code."""
        import httpx  # noqa: PLC0415

        check_fields(fields)
        started = time.perf_counter()
        with self._sync_lock:
            if self._sync_client is None:
                self._sync_client = httpx.Client(
                    timeout=self.timeout, transport=self._sync_transport, auth=self._auth
                )
            client = self._sync_client
        results = []
        for batch in self._batches(fields):
            try:
                response = client.post(
                    self._url, json=self.request_body(text, batch), headers=self._headers()
                )
            except httpx.HTTPError as exc:
                raise DecisionError(f"the Decisions request failed ({type(exc).__name__})") from exc
            results.append(self._read(batch, response.status_code, response.content))
        return self._decision(results, started)

    def close(self) -> None:
        """Close the synchronous client."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
