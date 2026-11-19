# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Decisions from the OpenAI Decisions API, on OpenAI or on OCI Generative AI.

The Decisions API (``POST /decisions``) answers typed questions about one
input with a probability per answer, which is what a
:class:`~locus.decision.DecisionProvider` returns. Its question types map onto
the fields one to one:

- :class:`~locus.decision.YesNo` is a ``predicate`` (the probability the
  condition holds);
- :class:`~locus.decision.Choice` is a ``choice`` (one of the listed values);
- :class:`~locus.decision.Score` is a ``score`` on ordered ``levels``.

Every field of one decision goes in **one** request, which shares the input
across the questions. :class:`OpenAIDecider` speaks to OpenAI with a bearer
key; :class:`OCIDecider` speaks to OCI Generative AI's ``/openai/v1`` endpoint
with OCI request signing (or an OCI Generative AI API key). The wire format
and the reading of the answers are the same; only the endpoint and the
authentication differ.

    from locus.decision import Choice, YesNo, get_decider

    decider = get_decider("oci:openai.gpt-6-luna", profile="DEFAULT", region="eu-frankfurt-1")
    decision = await decider.decide(
        "My laptop is late; refund me if it cannot arrive by Monday.",
        [
            YesNo("asks_refund", "Does the customer ask for a refund?"),
            Choice("department", "Which department?", ("shipping", "billing", "technical")),
        ],
    )
    decision["department"].label     # "shipping"
    decision.meta["confidence"]       # {"department": 0.98}

The API's ``confidence`` (choice and score answers) is kept in
``Decision.meta["confidence"]``, not folded into a probability. Token usage is
in ``Decision.meta["usage"]``. A refusal, an answer missing for an asked field
or an answer off the listed labels raises :class:`~locus.decision.DecisionError`.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from locus.core.loop_bound import loop_bound
from locus.decision.fields import (
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


__all__ = ["MAX_QUESTIONS", "MAX_SCORE_LEVELS", "OCIDecider", "OpenAIDecider"]

#: Most questions one Decisions request takes.
MAX_QUESTIONS = 400
#: Most levels a ``score`` question takes (and at least two).
MAX_SCORE_LEVELS = 10

_OPENAI_BASE_URL = "https://api.openai.com/v1"
_OCI_AUTH_TYPES = frozenset({"instance_principal", "resource_principal"})


def question_body(question: Field) -> dict[str, Any]:
    """One field as a Decisions API question."""
    if isinstance(question, YesNo):
        return {"type": "predicate", "name": question.name, "instructions": question.question}
    if isinstance(question, Score):
        if len(question.labels) > MAX_SCORE_LEVELS:
            raise ValueError(
                f"field {question.name!r} has {len(question.labels)} levels; "
                f"the Decisions API takes at most {MAX_SCORE_LEVELS}"
            )
        # The API asks for a description per level; the level's own label is
        # the only description a Score carries.
        return {
            "type": "score",
            "name": question.name,
            "instructions": question.question,
            "levels": [{"label": lbl, "description": lbl} for lbl in question.labels],
        }
    return {
        "type": "choice",
        "name": question.name,
        "instructions": question.question,
        "choices": [{"value": opt} for opt in question.labels],
    }


def _distribution(question: Field, raw: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    """The listed labels' probabilities, renormalised, and the mass they had."""
    if isinstance(question, YesNo):
        p = raw.get("probability")
        if not isinstance(p, int | float) or not 0.0 <= float(p) <= 1.0:
            raise DecisionError(f"field {question.name!r}: the answer carries no probability")
        return {"yes": float(p), "no": 1.0 - float(p)}, 1.0

    key = "label" if isinstance(question, Score) else "value"
    mass = dict.fromkeys(question.labels, 0.0)
    probabilities = raw.get("probabilities")
    if not isinstance(probabilities, list):
        raise DecisionError(f"field {question.name!r}: probabilities must be an array")
    for item in probabilities:
        if not isinstance(item, Mapping):
            raise DecisionError(f"field {question.name!r}: a probability entry must be an object")
        label, p = item.get(key), item.get("probability")
        if not isinstance(label, str) or label not in mass:
            raise DecisionError(
                f"field {question.name!r}: the answer names {label!r}, "
                f"which is not one of {question.labels}"
            )
        if isinstance(p, int | float) and float(p) > 0.0:
            mass[label] += float(p)
    coverage = sum(mass.values())
    if coverage <= 0.0:
        raise DecisionError(f"field {question.name!r}: the answer carries no probabilities")
    return {lbl: m / coverage for lbl, m in mass.items()}, min(1.0, coverage)


def _kind(question: Field) -> str:
    if isinstance(question, YesNo):
        return "yesno"
    if isinstance(question, Score):
        return "score"
    return "choice"


_API_TYPE = {"yesno": "predicate", "choice": "choice", "score": "score"}


class OpenAIDecider:
    """Decisions from the OpenAI Decisions API, with a bearer key.

    Args:
        model: The decision model, e.g. ``gpt-6-luna``.
        api_key: The OpenAI key; read from ``OPENAI_API_KEY`` when omitted.
        base_url: The API base URL (default ``https://api.openai.com/v1``).
        timeout: Seconds per request.
        extra_body: Merged into every request body *under* the fixed keys
            (``model``, ``input``, ``questions``), e.g. ``prompt_cache_key``.
        transport: An ``httpx`` async transport (tests pass
            ``httpx.MockTransport``).
        sync_transport: The same, for :meth:`decide_sync`.
    """

    provider_name = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        extra_body: Mapping[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sync_transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        self.model = model
        self.base_url = (base_url or _OPENAI_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})
        self._api_key = api_key if api_key is not None else self._default_api_key()
        self._transport = transport
        self._sync_transport = sync_transport
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        self._sync_lock = threading.Lock()
        self._check_auth()

    def __repr__(self) -> str:
        # The key never appears in a repr, a log line or an exception.
        return f"{type(self).__name__}(model={self.model!r}, base_url={self.base_url!r})"

    # -- authentication (overridden by OCIDecider) ---------------------------

    def _default_api_key(self) -> str | None:
        return os.environ.get("OPENAI_API_KEY") or None

    def _check_auth(self) -> None:
        if not self._api_key:
            raise ValueError("an OpenAI key is required: pass api_key= or set OPENAI_API_KEY")

    def _auth(self) -> httpx.Auth | None:
        return None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    # -- the request ---------------------------------------------------------

    @property
    def url(self) -> str:
        """Where a decision is posted."""
        return f"{self.base_url}/decisions"

    def request_body(self, text: str, fields: Sequence[Field]) -> dict[str, Any]:
        """The JSON body for one decision. Public so a server can be checked against it."""
        check_fields(fields)
        if len(fields) > MAX_QUESTIONS:
            raise ValueError(
                f"{len(fields)} fields; the Decisions API takes at most {MAX_QUESTIONS} per request"
            )
        if not text or not text.strip():
            raise ValueError("the input is empty")
        return {
            **self.extra_body,
            "model": self.model,
            "input": text,
            "questions": [question_body(f) for f in fields],
        }

    # -- the answer ----------------------------------------------------------

    def _read(
        self, fields: Sequence[Field], status: int, payload: bytes, started: float
    ) -> Decision:
        if status >= 400:
            raise DecisionError(
                f"the Decisions API answered HTTP {status}: {_error_message(payload)}"
            )
        try:
            body = json.loads(payload)
            raw_answers = body["answers"]
        except (ValueError, KeyError, TypeError) as exc:
            raise DecisionError("the response carries no answers") from exc
        by_name: dict[str, Mapping[str, Any]] = {}
        for item in raw_answers if isinstance(raw_answers, list) else []:
            if isinstance(item, Mapping) and isinstance(item.get("name"), str):
                # Two answers for one name are ambiguous whatever their order
                # or type: keeping either would let one hide the other (a
                # refusal followed by a valid answer, say).
                if item["name"] in by_name:
                    raise DecisionError(f"the response answers {item['name']!r} twice")
                by_name[item["name"]] = item

        answers: dict[str, Answer] = {}
        confidence: dict[str, float] = {}
        for question in fields:
            raw = by_name.get(question.name)
            if raw is None:
                raise DecisionError(f"field {question.name!r}: no answer came back")
            kind = _kind(question)
            if raw.get("type") == "refusal":
                raise DecisionError(f"field {question.name!r}: the model refused to answer")
            if raw.get("type") != _API_TYPE[kind]:
                raise DecisionError(
                    f"field {question.name!r}: expected a {_API_TYPE[kind]} answer, "
                    f"got {raw.get('type')!r}"
                )
            distribution, coverage = _distribution(question, raw)
            answers[question.name] = Answer(
                field=question.name,
                kind=kind,
                labels=question.labels,
                distribution=distribution,
                coverage=coverage,
            )
            if isinstance(raw.get("confidence"), int | float):
                confidence[question.name] = float(raw["confidence"])

        meta: dict[str, Any] = {"confidence": confidence}
        if isinstance(body.get("usage"), Mapping):
            meta["usage"] = dict(body["usage"])
        return Decision(
            answers=answers,
            model=str(body.get("model") or self.model),
            provider=self.provider_name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            meta=meta,
        )

    # -- async ---------------------------------------------------------------

    def _build_client(self) -> httpx.AsyncClient:
        import httpx  # noqa: PLC0415

        return httpx.AsyncClient(timeout=self.timeout, transport=self._transport, auth=self._auth())

    async def decide(self, text: str, fields: Sequence[Field]) -> Decision:
        """Answer every field about ``text`` in one request."""
        import httpx  # noqa: PLC0415

        body = self.request_body(text, fields)
        started = time.perf_counter()
        client = loop_bound(self, "_client", self._build_client)
        try:
            response = await client.post(self.url, json=body, headers=self._headers())
        except httpx.HTTPError as exc:
            raise DecisionError(f"the request failed ({type(exc).__name__})") from exc
        return self._read(fields, response.status_code, response.content, started)

    async def aclose(self) -> None:
        """Close the async client, while its loop is still running."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- sync ----------------------------------------------------------------

    def decide_sync(self, text: str, fields: Sequence[Field]) -> Decision:
        """The same, from synchronous code."""
        import httpx  # noqa: PLC0415

        body = self.request_body(text, fields)
        started = time.perf_counter()
        with self._sync_lock:
            if self._sync_client is None:
                self._sync_client = httpx.Client(
                    timeout=self.timeout, transport=self._sync_transport, auth=self._auth()
                )
            client = self._sync_client
        try:
            response = client.post(self.url, json=body, headers=self._headers())
        except httpx.HTTPError as exc:
            raise DecisionError(f"the request failed ({type(exc).__name__})") from exc
        return self._read(fields, response.status_code, response.content, started)

    def close(self) -> None:
        """Close the synchronous client."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None


class OCIDecider(OpenAIDecider):
    """Decisions from the Decisions API on OCI Generative AI.

    Posts to ``https://inference.generativeai.<region>.oci.oraclecloud.com/openai/v1/decisions``.
    Authenticate with exactly one of:

    - ``profile=``: a profile in ``config_file`` (API key or session token);
    - ``auth_type="instance_principal"`` or ``"resource_principal"``, which
      need ``compartment_id``;
    - ``api_key=``: an OCI Generative AI API key, sent as a bearer token.

    Args:
        model: The OCI model id, e.g. ``openai.gpt-6-luna``.
        profile: OCI config profile name. Falls back to ``OCI_PROFILE`` when
            no authentication mode is given.
        auth_type: ``"instance_principal"`` or ``"resource_principal"``.
        api_key: An OCI Generative AI API key.
        compartment_id: Sent as ``opc-compartment-id``. Read from
            ``OCI_COMPARTMENT`` / ``OCI_COMPARTMENT_ID``, else the profile's
            tenancy, when omitted.
        project_ocid: Sent as ``opc-project-id`` when given.
        region: OCI region. Defaults to ``LOCUS_OCI_REGION``, then
            ``OCI_REGION``, then the SDK's OCI default region. An explicit
            ``base_url`` bypasses the region environment defaults.
        config_file: The OCI config file, for ``profile=``.
        base_url: Override the derived endpoint (ending in ``/openai/v1``).
        timeout, extra_body, transport, sync_transport: As for
            :class:`OpenAIDecider`.
    """

    provider_name = "oci"

    def __init__(
        self,
        model: str,
        *,
        profile: str | None = None,
        auth_type: str | None = None,
        api_key: str | None = None,
        compartment_id: str | None = None,
        project_ocid: str | None = None,
        region: str | None = None,
        config_file: str = "~/.oci/config",
        base_url: str | None = None,
        timeout: float = 30.0,
        extra_body: Mapping[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sync_transport: httpx.BaseTransport | None = None,
    ) -> None:
        from locus.models.providers.oci.openai_compat import (  # noqa: PLC0415
            DEFAULT_OCI_GENAI_REGION,
            build_oci_openai_base_url,
        )

        if profile is None and auth_type is None and api_key is None:
            profile = os.environ.get("OCI_PROFILE") or None
        if sum(x is not None for x in (profile, auth_type, api_key)) != 1:
            raise ValueError("specify exactly one of profile=, auth_type=, api_key=")
        if auth_type is not None and auth_type not in _OCI_AUTH_TYPES:
            raise ValueError(
                f"auth_type must be one of {sorted(_OCI_AUTH_TYPES)}, got {auth_type!r}"
            )
        if auth_type is not None and compartment_id is None:
            raise ValueError("compartment_id is required when auth_type= is set")
        self.profile = profile
        self.auth_type = auth_type
        self.config_file = config_file
        if region is None and base_url is None:
            region = os.environ.get("LOCUS_OCI_REGION") or os.environ.get("OCI_REGION")
        self.region = region or DEFAULT_OCI_GENAI_REGION
        self.project_ocid = project_ocid
        self.compartment_id = compartment_id or self._resolve_compartment(profile, config_file)
        super().__init__(
            model,
            api_key=api_key,
            base_url=base_url or build_oci_openai_base_url(self.region),
            timeout=timeout,
            extra_body=extra_body,
            transport=transport,
            sync_transport=sync_transport,
        )

    @staticmethod
    def _resolve_compartment(profile: str | None, config_file: str) -> str | None:
        found = os.getenv("OCI_COMPARTMENT") or os.getenv("OCI_COMPARTMENT_ID")
        if found or profile is None:
            return found
        from locus.models.providers.oci.openai_compat import _load_profile_config  # noqa: PLC0415

        try:
            tenancy = _load_profile_config(profile, config_file).get("tenancy")
        except Exception:  # noqa: BLE001 - best-effort, as OCIResponsesModel does
            return None
        return str(tenancy) if tenancy else None

    def _default_api_key(self) -> str | None:
        # An OCI decider never falls back to OPENAI_API_KEY.
        return None

    def _check_auth(self) -> None:
        # __init__ has already required exactly one auth mode.
        return

    def _auth(self) -> httpx.Auth | None:
        if self._api_key:
            return None
        from locus.models.providers.oci._signing import OCIRequestSigner  # noqa: PLC0415
        from locus.models.providers.oci.openai_compat import (  # noqa: PLC0415
            _build_instance_principal_signer,
            _build_resource_principal_signer,
            _build_signer_from_profile,
            _refresh_callable_for,
        )

        if self.auth_type == "instance_principal":
            signer = _build_instance_principal_signer()
        elif self.auth_type == "resource_principal":
            signer = _build_resource_principal_signer()
        else:
            assert self.profile is not None  # noqa: S101 - invariant from __init__
            signer = _build_signer_from_profile(self.profile, self.config_file)
        return OCIRequestSigner(
            signer,
            compartment_id=self.compartment_id,
            refresh_signer=_refresh_callable_for(
                signer,
                auth_type=self.auth_type,
                profile=self.profile,
                config_file=self.config_file,
            ),
            refresh_interval=600.0,
        )

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        if self._api_key and self.compartment_id:
            headers["opc-compartment-id"] = self.compartment_id
        if self.project_ocid:
            headers["opc-project-id"] = self.project_ocid
        return headers


def _error_message(payload: bytes) -> str:
    """The server's error message, short, for an exception."""
    try:
        body = json.loads(payload)
    except ValueError:
        return payload[:200].decode("utf-8", "replace")
    error = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "")[:300]
    return str(error)[:300]
