# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OCI GenAI access via the OpenAI-compatible ``/openai/v1`` endpoint.

This is the V1 OCI transport in locus for any model family that
speaks the OpenAI request shape on OCI (OpenAI / Meta / xAI / Mistral /
Gemini). Cohere R-series models are not supported by OCI on this endpoint
and will fail with a 400 from OCI — for those, use :class:`OCIModel` (the
OCI SDK transport).

Two mutually-exclusive auth modes are supported. Pass **exactly one**:

- ``profile=...`` — name of a profile in ``~/.oci/config`` (or
  ``config_file=``). Used for both API-key IAM signing and session-token
  signing — the profile shape determines which.
- ``auth_type="instance_principal"`` / ``"resource_principal"`` — for
  workloads running on OCI compute or in OCI Functions / OKE with
  workload identity.
- ``auth_type="api_key"`` (``"security_token"`` too) — the profile path by
  another name: sign with ``profile=``, else the ``OCI_PROFILE`` profile,
  else ``DEFAULT``. It is what ``LOCUS_OCI_AUTH_TYPE`` defaults to.

The endpoint is ``/openai/v1/chat/completions``. We do **not** use the
Responses API and do **not** require a GenAI Project OCID — those imply
server-side conversation state that locus already owns.

Example::

    import os
    from locus.core.messages import Message
    from locus.models.providers.oci import OCIChatCompletionsModel

    # IAM via OCI config profile (typical local dev / CI)
    model = OCIChatCompletionsModel(
        model="meta.llama-3.3-70b-instruct",
        profile="DEFAULT",
    )

    # Workload identity (OCI VM / OKE / Functions)
    model = OCIChatCompletionsModel(
        model="meta.llama-3.3-70b-instruct",
        auth_type="instance_principal",
        compartment_id=os.environ["OCI_COMPARTMENT_ID"],
    )

    response = await model.complete([Message.user("Hello!")])
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx
from pydantic import Field

from locus.core.loop_bound import loop_bound
from locus.models.native.openai import OpenAIConfig, OpenAIModel


if TYPE_CHECKING:
    import openai
    from oci.signer import AbstractBaseSigner

    from locus.core.events import ModelChunkEvent
    from locus.core.messages import Message
    from locus.models.base import ModelResponse
    from locus.models.providers.oci._signing import OCIRequestSigner


DEFAULT_OCI_GENAI_REGION = "us-chicago-1"

AuthType = Literal[
    "instance_principal",
    "resource_principal",
    "oke_workload_identity",
    "api_key",
    "security_token",
    "session_token",
]
#: ``auth_type`` values that name a principal rather than a config profile.
PRINCIPAL_AUTH_TYPES: tuple[str, ...] = (
    "instance_principal",
    "resource_principal",
    "oke_workload_identity",
)
_VALID_AUTH_TYPES = PRINCIPAL_AUTH_TYPES

#: ``auth_type`` values that name the profile path rather than a principal:
#: the API key of a ``~/.oci/config`` profile (the default of every other OCI
#: auth switch in the SDK and of ``LOCUS_OCI_AUTH_TYPE``) or the session
#: token the profile names. The profile's shape picks the signer either way.
PROFILE_AUTH_TYPES: frozenset[str] = frozenset({"api_key", "security_token", "session_token"})


def resolve_profile_auth(
    profile: str | None, auth_type: str | None
) -> tuple[str | None, str | None]:
    """Fold a profile-path ``auth_type`` into ``profile``.

    ``auth_type="api_key"`` (or ``"security_token"`` / ``"session_token"``)
    says "sign with a profile": the result is ``(profile, None)`` with the
    profile taken from ``profile=``, else ``LOCUS_OCI_PROFILE`` / ``OCI_PROFILE``,
    else ``DEFAULT`` — the section ``oci.config`` reads on its own. Any other
    pair is returned as given.
    """
    if auth_type in PROFILE_AUTH_TYPES:
        env_profile = os.environ.get("LOCUS_OCI_PROFILE") or os.environ.get("OCI_PROFILE")
        return profile or env_profile or "DEFAULT", None
    return profile, auth_type


def build_oci_openai_base_url(region: str) -> str:
    """Construct the OCI GenAI OpenAI-compatible base URL for a region."""
    return f"https://inference.generativeai.{region}.oci.oraclecloud.com/openai/v1"


#: A sync callable handed each outgoing :class:`httpx.Request` (``on_request=``).
RequestHook = Callable[[httpx.Request], None]
#: A sync callable handed each :class:`httpx.Response` as it arrives (``on_response=``).
ResponseHook = Callable[[httpx.Response], None]

_ON_REQUEST_DESCRIPTION = (
    "Called with each outgoing httpx.Request after it is signed and before it "
    "is sent: the seam for seeing what goes on the wire (``request.content`` is "
    "the JSON body, ``request.headers`` the signed headers) in a test or when "
    "debugging. Runs once per HTTP attempt, so a retry shows twice. None "
    "(default) installs no hook."
)
_ON_RESPONSE_DESCRIPTION = (
    "Called with each httpx.Response as its headers arrive and before the body "
    "is read: ``status_code``, ``headers`` and ``request`` are available, the "
    "body is not yet. Runs once per HTTP attempt. None (default) installs no "
    "hook."
)


def request_event_hooks(
    on_request: RequestHook | None,
    on_response: ResponseHook | None,
) -> dict[str, list[Callable[..., Awaitable[None]]]]:
    """The httpx ``event_hooks`` mapping for a pair of sync inspection callables.

    :class:`httpx.AsyncClient` awaits its hooks, so each sync callable is
    wrapped in a coroutine function; an absent callable installs nothing. The
    HTTP-backed OCI models (:class:`OCIChatCompletionsModel`,
    :class:`~locus.models.providers.oci.OCIResponsesModel`) build their
    client with this so ``on_request=`` / ``on_response=`` behave the same
    on both transports.
    """
    hooks: dict[str, list[Callable[..., Awaitable[None]]]] = {}
    if on_request is not None:

        async def _request(request: httpx.Request) -> None:
            on_request(request)

        hooks["request"] = [_request]
    if on_response is not None:

        async def _response(response: httpx.Response) -> None:
            on_response(response)

        hooks["response"] = [_response]
    return hooks


logger = logging.getLogger(__name__)


class OCIChatCompletionsConfig(OpenAIConfig):
    """Configuration for :class:`OCIChatCompletionsModel`.

    Inherits all OpenAI knobs from :class:`OpenAIConfig`. Adds region,
    profile, ``auth_type``, ``compartment_id``, and ``config_file`` for OCI
    auth selection, and the ``on_request`` / ``on_response`` inspection hooks.
    """

    region: str = Field(
        default=DEFAULT_OCI_GENAI_REGION,
        description="OCI region for the GenAI inference endpoint",
    )
    profile: str | None = Field(
        default=None,
        description="OCI config profile name (in config_file)",
    )
    auth_type: str | None = Field(
        default=None,
        description=(
            '"instance_principal", "resource_principal" or "oke_workload_identity"; '
            '"api_key" / "security_token" fold into profile= at construction'
        ),
    )
    config_file: str = Field(
        default="~/.oci/config",
        description="Path to the OCI config file (only used with profile=)",
    )
    compartment_id: str | None = Field(
        default=None,
        description=(
            "OCI compartment OCID (sent as opc-compartment-id). "
            "Auto-derived from the profile's tenancy when profile= is used. "
            "Required for instance_principal / resource_principal."
        ),
    )
    on_request: RequestHook | None = Field(default=None, description=_ON_REQUEST_DESCRIPTION)
    on_response: ResponseHook | None = Field(default=None, description=_ON_RESPONSE_DESCRIPTION)


def _load_profile_config(profile: str, config_file: str) -> dict[str, Any]:
    """Load an OCI config profile as a dict (tenancy, user, key_file, ...)."""
    from oci import config as oci_config_module

    cfg_path = str(Path(config_file).expanduser())
    cfg: dict[str, Any] = oci_config_module.from_file(cfg_path, profile_name=profile)
    return cfg


#: JSON Schema keywords OCI's Gemini translation rejects outright
#: (``400 Unsupported JSON Schema feature for Gemini: <keyword>``). They are
#: validation metadata; Locus validates structured output with Pydantic after
#: the fact, so dropping them for the request loses nothing.
_GEMINI_UNSUPPORTED_SCHEMA_KEYS: frozenset[str] = frozenset(
    {
        "additionalProperties",
        "title",
        "$schema",
        "$id",
        "default",
        "examples",
        "pattern",
        "format",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
    }
)


def fold_schema_enums(schema: Any) -> Any:
    """Return a copy of a JSON schema in the subset OCI's Gemini translation accepts.

    The translation rejects ``enum`` and ``const`` (``400 Unsupported JSON
    Schema feature for Gemini: enum``) and a set of validation keywords
    (``additionalProperties``, ``title``, ``pattern``, ``minimum``, …; see
    ``_GEMINI_UNSUPPORTED_SCHEMA_KEYS``). The allowed values are kept where
    the model still sees them — appended to the property's ``description``
    as ``Allowed values: "a", "b".`` — and the ``type`` stays, so structured
    output and tool schemas keep working; the validation keywords are
    dropped. Nested objects, arrays, ``anyOf`` and friends are walked;
    non-dict input is returned unchanged.
    """
    if isinstance(schema, list):
        return [fold_schema_enums(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    values: list[Any] | None = None
    for key, value in schema.items():
        if key == "enum" and isinstance(value, list):
            values = list(value)
            continue
        if key == "const":
            values = [value]
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: fold_schema_enums(sub) for name, sub in value.items()}
            continue
        if key in _GEMINI_UNSUPPORTED_SCHEMA_KEYS:
            continue
        out[key] = fold_schema_enums(value)
    if values is not None:
        allowed = "Allowed values: " + ", ".join(json.dumps(v) for v in values) + "."
        desc = str(out.get("description") or "").rstrip()
        out["description"] = f"{desc} {allowed}".strip() if desc else allowed
        out.setdefault("type", "string")
    return out


def wrap_tool_results_as_objects(messages: list[Message]) -> list[Message]:
    """Make every tool result a JSON object, as Gemini on OCI requires.

    OCI maps a ``tool`` message onto Gemini's ``function_response.response``,
    a protobuf ``Struct``; a bare string or number is rejected with ``400
    Invalid value at 'contents[n].parts[0].function_response.response'``.
    Results that already are JSON objects pass through unchanged; anything
    else becomes ``{"result": <value>}`` — the parsed value when the text is
    valid JSON, the raw text otherwise. Other roles are untouched.
    """
    from locus.core.messages import Role

    out: list[Message] = []
    for msg in messages:
        if msg.role != Role.TOOL or msg.content is None:
            out.append(msg)
            continue
        try:
            parsed: Any = json.loads(msg.content)
        except ValueError:
            parsed = msg.content
        if isinstance(parsed, dict):
            out.append(msg)
            continue
        out.append(msg.model_copy(update={"content": json.dumps({"result": parsed})}))
    return out


def blank_null_assistant_content(messages: list[Message]) -> list[Message]:
    """Return ``messages`` with every text-less assistant turn carrying ``""``.

    An assistant turn that only made tool calls has ``content=None``; OpenAI
    accepts ``"content": null`` for it, OCI's V1 endpoint does not: ``400
    Invalid type for 'messages[n].content': expected an object, but got null
    instead``. The history is replayed on every turn after a tool call, so one
    such message breaks the whole conversation. Other roles are untouched;
    the input is never mutated and is returned as the same object when
    nothing needed changing.
    """
    from locus.core.messages import Role

    if not any(m.role == Role.ASSISTANT and m.content is None for m in messages):
        return messages
    return [
        m.model_copy(update={"content": ""})
        if m.role == Role.ASSISTANT and m.content is None
        else m
        for m in messages
    ]


def fold_leading_system_into_first_user(messages: list[Message]) -> list[Message]:
    """Return ``messages`` with leading system messages merged into the first user turn.

    The system text is prepended to the first user message's content,
    separated by a blank line; the system messages are dropped. Lists
    that don't start with a system message, or that have no user
    message at all, are returned unchanged (same object). The input is
    never mutated.
    """
    from locus.core.messages import Role

    if not messages or messages[0].role != Role.SYSTEM:
        return messages
    i = 0
    system_parts: list[str] = []
    while i < len(messages) and messages[i].role == Role.SYSTEM:
        if messages[i].content:
            system_parts.append(messages[i].content or "")
        i += 1
    rest = messages[i:]
    first_user = next((k for k, m in enumerate(rest) if m.role == Role.USER), None)
    if first_user is None:
        return messages
    user = rest[first_user]
    merged = "\n\n".join([*system_parts, user.content or ""]).strip()
    folded = user.model_copy(update={"content": merged})
    return [*rest[:first_user], folded, *rest[first_user + 1 :]]


def _build_signer_from_profile(profile: str, config_file: str) -> AbstractBaseSigner:
    """Build an OCI signer from a config profile.

    Picks security-token signing if the profile has ``security_token_file``,
    otherwise user-principal API-key signing.
    """
    cfg = _load_profile_config(profile, config_file)
    if cfg.get("security_token_file"):
        return _build_session_signer(cfg)
    return _build_user_principal_signer(cfg)


def _build_user_principal_signer(cfg: dict[str, Any]) -> AbstractBaseSigner:
    from oci.signer import Signer

    return Signer(
        tenancy=cfg["tenancy"],
        user=cfg["user"],
        fingerprint=cfg["fingerprint"],
        private_key_file_location=cfg["key_file"],
        pass_phrase=cfg.get("pass_phrase"),
    )


def _build_session_signer(cfg: dict[str, Any]) -> AbstractBaseSigner:
    from oci.auth.signers import SecurityTokenSigner
    from oci.signer import load_private_key_from_file

    token_path = Path(cfg["security_token_file"]).expanduser()
    token = token_path.read_text().strip()
    private_key = load_private_key_from_file(cfg["key_file"], cfg.get("pass_phrase"))
    return SecurityTokenSigner(token=token, private_key=private_key)


def _build_instance_principal_signer() -> AbstractBaseSigner:
    from oci.auth.signers import InstancePrincipalsSecurityTokenSigner

    return InstancePrincipalsSecurityTokenSigner()


def _build_resource_principal_signer() -> AbstractBaseSigner:
    from locus._oci_auth import resource_principal_signer

    return cast("AbstractBaseSigner", resource_principal_signer("resource_principal"))


def _build_oke_workload_identity_signer() -> AbstractBaseSigner:
    from locus._oci_auth import resource_principal_signer

    return cast("AbstractBaseSigner", resource_principal_signer("oke_workload_identity"))


def resolve_oci_compartment(
    compartment_id: str | None, *, profile: str | None, config_file: str
) -> str | None:
    """The compartment OCID to send as ``opc-compartment-id``, or ``None``.

    Precedence: the explicit ``compartment_id``; then ``LOCUS_OCI_COMPARTMENT``,
    ``OCI_COMPARTMENT`` or ``OCI_COMPARTMENT_ID`` (one export drives every
    OCI switch in the SDK and its tests); then the ``profile``'s tenancy, a
    last-resort default that may lack GenAI policy when the profile's home
    tenancy is not the one hosting the compartment.
    """
    if compartment_id is None:
        compartment_id = (
            os.environ.get("LOCUS_OCI_COMPARTMENT")
            or os.environ.get("OCI_COMPARTMENT")
            or os.environ.get("OCI_COMPARTMENT_ID")
        )
    if compartment_id is None and profile is not None:
        try:
            compartment_id = _load_profile_config(profile, config_file).get("tenancy")
        except Exception:  # noqa: BLE001 — the profile may be unreadable; the caller decides
            compartment_id = None
    return compartment_id


def build_oci_signer(
    *, profile: str | None, auth_type: str | None, config_file: str
) -> AbstractBaseSigner:
    """The OCI signer for one resolved auth mode: a principal, or a config profile.

    ``auth_type`` is one of :data:`PRINCIPAL_AUTH_TYPES` or ``None``; with
    ``None``, ``profile`` names the section of ``config_file`` to sign with
    (API key, or the session token the section names).
    """
    if auth_type == "instance_principal":
        return _build_instance_principal_signer()
    if auth_type == "resource_principal":
        return _build_resource_principal_signer()
    if auth_type == "oke_workload_identity":
        return _build_oke_workload_identity_signer()
    if auth_type is not None:
        raise ValueError(f"auth_type must be one of {PRINCIPAL_AUTH_TYPES}, got {auth_type!r}")
    if profile is None:
        raise ValueError("an OCI profile or a principal auth_type is required")
    return _build_signer_from_profile(profile, config_file)


def build_oci_request_auth(
    *,
    profile: str | None,
    auth_type: str | None,
    compartment_id: str | None,
    config_file: str,
    refresh_interval: float = 600.0,
    signer: AbstractBaseSigner | None = None,
) -> OCIRequestSigner:
    """The ``httpx.Auth`` that signs requests to the ``/openai/v1`` endpoint.

    Builds the signer for the auth mode (:func:`build_oci_signer`) unless one
    is passed in (a test's stand-in, or a signer built once and shared), sends
    ``compartment_id`` as ``opc-compartment-id``, and refreshes expiring
    credentials: token-based signers (instance principal, resource principal,
    session token) expire within minutes to hours, and without a refresh the
    401 retry in :class:`OCIRequestSigner` is a no-op and every call fails
    after about half an hour. The refresh rebuilds a fresh signer rather than
    mutating one in place (see :func:`_refresh_callable_for`); a user-principal
    API key never expires and gets no refresh. ``refresh_interval`` is 10
    minutes, inside every principal credential's lifetime. Shared by
    :class:`OCIChatCompletionsModel` and ``locus.decision.DecisionsAPIDecider``.
    """
    from locus.models.providers.oci._signing import OCIRequestSigner

    if signer is None:
        signer = build_oci_signer(profile=profile, auth_type=auth_type, config_file=config_file)
    return OCIRequestSigner(
        signer,
        compartment_id=compartment_id,
        refresh_signer=_refresh_callable_for(
            signer, auth_type=auth_type, profile=profile, config_file=config_file
        ),
        refresh_interval=refresh_interval,
    )


def _refresh_callable_for(
    signer: AbstractBaseSigner,
    *,
    auth_type: str | None = None,
    profile: str | None = None,
    config_file: str | None = None,
) -> Callable[[], Any] | None:
    """Return a callable that yields a *fresh* signer, or ``None`` when
    there is no recoverable expiring credential.

    Every recoverable case rebuilds the signer from scratch rather than
    mutating one in place — this mirrors oracle-samples/oci-genai-auth-python
    (``_refresh_signer``) and is the fix for long-lived-process 401s: the
    in-place ``signer.refresh_security_token()`` path mutates a cached
    federation client, whereas a full rebuild re-reads the instance leaf
    certificate / token from the metadata service. :class:`OCIRequestSigner`
    swaps in whatever signer the callable returns.

    Three cases:

    1. **Instance / resource principal.** Rebuild a brand-new
       ``InstancePrincipalsSecurityTokenSigner`` /
       ``get_resource_principals_signer`` on each refresh. Selected by
       ``auth_type`` (the only place that knows which principal we built).
    2. **Session token (``SecurityTokenSigner`` from a profile with a
       ``security_token_file``).** The file on disk is the source of truth
       — ``oci session refresh`` (and any external token-rotation daemon)
       updates it out-of-band. When ``profile`` + ``config_file`` are
       supplied, capture them in a closure that re-reads the token file and
       returns a fresh ``SecurityTokenSigner``. Without those args we can't
       rebuild, so return None and the auth flow stays dormant.
    3. **User principal (API key).** No expiring credential, no refresh.
       Returns None.

    Args:
        signer: The OCI signer in use (unused for dispatch, kept for API
            symmetry and future signer-type introspection).
        auth_type: ``"instance_principal"`` / ``"resource_principal"`` when
            a principal signer is in use; ``None`` for profile-based signers.
        profile: OCI config profile name. Required to enable session-token
            refresh; ignored for instance/resource principal.
        config_file: Path to the OCI config file. Same constraint as
            ``profile``.

    Returns:
        A zero-arg callable returning a fresh signer, or ``None`` when the
        credential is non-expiring.
    """
    # Case 1: instance / resource principal — rebuild a fresh signer.
    if auth_type == "instance_principal":
        return _build_instance_principal_signer
    if auth_type == "resource_principal":
        return _build_resource_principal_signer
    if auth_type == "oke_workload_identity":
        return _build_oke_workload_identity_signer

    # Case 2: SecurityTokenSigner rebuilt from disk.
    if profile is not None and config_file is not None:
        try:
            cfg = _load_profile_config(profile, config_file)
        except Exception:  # noqa: BLE001 — refresh stays disabled if config is unreadable
            return None
        if cfg.get("security_token_file"):
            return _session_token_disk_refresh(cfg)

    return None


def _session_token_disk_refresh(cfg: dict[str, Any]) -> Callable[[], Any]:
    """Build a refresh callable for a session-token profile.

    Captures the ``security_token_file`` path, ``key_file`` path, and
    ``pass_phrase`` from ``cfg`` in a closure. Each invocation re-reads
    the token file from disk (picking up out-of-band refreshes from
    ``oci session refresh`` or equivalent) and returns a fresh
    ``SecurityTokenSigner`` that the caller can swap in.
    """
    from oci.auth.signers import SecurityTokenSigner
    from oci.signer import load_private_key_from_file

    token_path = Path(cfg["security_token_file"]).expanduser()
    key_path = cfg["key_file"]
    pass_phrase = cfg.get("pass_phrase")

    def _refresh() -> AbstractBaseSigner:
        token = token_path.read_text().strip()
        private_key = load_private_key_from_file(key_path, pass_phrase)
        return SecurityTokenSigner(token=token, private_key=private_key)

    return _refresh


class OCIChatCompletionsModel(OpenAIModel):
    """OCI GenAI model accessed through the ``/openai/v1`` endpoint.

    Reuses :class:`OpenAIModel` for message conversion, tool handling,
    response parsing, and streaming. The only thing this class adds is
    the OCI-specific auth wiring.

    Pass exactly one of ``profile``, ``auth_type``.
    """

    config: OCIChatCompletionsConfig

    def __init__(
        self,
        model: str,
        *,
        profile: str | None = None,
        auth_type: str | None = None,
        compartment_id: str | None = None,
        region: str = DEFAULT_OCI_GENAI_REGION,
        config_file: str = "~/.oci/config",
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        on_request: RequestHook | None = None,
        on_response: ResponseHook | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OCI OpenAI-compat model.

        Args:
            model: OCI model identifier (e.g. ``openai.gpt-5``,
                ``meta.llama-3.3-70b-instruct``).
            profile: OCI config profile name from ``config_file``. Mutually
                exclusive with ``auth_type``.
            auth_type: ``"instance_principal"``, ``"resource_principal"`` or
                ``"oke_workload_identity"``: a principal, mutually exclusive
                with ``profile`` and requiring ``compartment_id``. Or
                ``"api_key"`` (``"security_token"``), which is the profile
                path by another name: ``profile=`` is used, else the
                ``OCI_PROFILE`` profile, else ``DEFAULT``.
            compartment_id: OCI compartment OCID, sent as
                ``opc-compartment-id``. Auto-derived from the profile's
                tenancy under ``profile=``. Must be supplied explicitly
                under a principal ``auth_type=``.
            region: OCI region hosting the inference endpoint.
            config_file: Path to the OCI config file (used with ``profile``).
            base_url: Override the derived endpoint URL (e.g. for a custom
                realm). Defaults to the OpenAI-compat URL for ``region``.
            max_tokens: Default max tokens. For ``o1``/``o3``/``gpt-5*``
                this is automatically forwarded as
                ``max_completion_tokens`` — see :class:`OpenAIModel`.
            temperature: Default sampling temperature.
            reasoning_effort: Reasoning effort for reasoning models
                (``"none"``, ``"minimal"``, ``"low"``, ``"medium"``,
                ``"high"``; the server validates the set a model accepts).
                ``None`` (default) sends nothing. Folded into ``extra_body``
                as ``reasoning_effort``, so an explicit ``extra_body`` key of
                the same name wins. The gpt-6 family rejects function tools
                on the chat endpoint unless this is ``"none"``; a model that
                does not accept ``"none"`` (``openai.gpt-6.2-sol``) needs
                ``OCIResponsesModel`` for tools.
            on_request: Sync callable handed each outgoing :class:`httpx.Request`
                after signing, before it is sent — ``request.content`` is the
                JSON body. The seam for request-shape tests and for seeing what
                goes on the wire; once per HTTP attempt. ``None`` installs none.
            on_response: Sync callable handed each :class:`httpx.Response` as
                its headers arrive (``status_code``, ``headers``; the body is
                not read yet). ``None`` installs none.
            **kwargs: Forwarded to :class:`OCIChatCompletionsConfig` (top_p, seed,
                frequency_penalty, presence_penalty, ...).

        Raises:
            ValueError: If zero or both auth modes are set, if ``auth_type``
                is invalid, or if ``auth_type`` is set without
                ``compartment_id``.
        """
        profile, auth_type = resolve_profile_auth(profile, auth_type)
        modes_set = sum(x is not None for x in (profile, auth_type))
        if modes_set != 1:
            msg = (
                "OCIChatCompletionsModel: specify exactly one auth mode. "
                "Either profile='<section_from_~/.oci/config>' (api-key path; "
                "use profile='DEFAULT' for the default section, or "
                "auth_type='api_key' for the OCI_PROFILE / DEFAULT section), "
                "or auth_type='instance_principal'|'resource_principal'|"
                "'oke_workload_identity' together with compartment_id=. "
                f"Got profile={profile!r}, auth_type={auth_type!r}."
            )
            raise ValueError(msg)
        if auth_type is not None and auth_type not in _VALID_AUTH_TYPES:
            msg = f"auth_type must be one of {_VALID_AUTH_TYPES}, got {auth_type!r}"
            raise ValueError(msg)
        if auth_type is not None and compartment_id is None:
            msg = "compartment_id is required when auth_type= is set"
            raise ValueError(msg)

        # Pop fields we set explicitly to avoid duplicate-kwarg errors
        # when callers splat a config dict that includes the same keys.
        for explicit in (
            "model",
            "profile",
            "auth_type",
            "compartment_id",
            "region",
            "config_file",
            "base_url",
            "max_tokens",
            "temperature",
            "reasoning_effort",
            "on_request",
            "on_response",
        ):
            kwargs.pop(explicit, None)

        if reasoning_effort is not None:
            kwargs["extra_body"] = {
                "reasoning_effort": reasoning_effort,
                **(kwargs.get("extra_body") or {}),
            }

        compartment_id = resolve_oci_compartment(
            compartment_id, profile=profile, config_file=config_file
        )

        config = OCIChatCompletionsConfig(
            model=model,
            api_key=None,
            base_url=base_url or build_oci_openai_base_url(region),
            region=region,
            profile=profile,
            auth_type=auth_type,
            compartment_id=compartment_id,
            config_file=config_file,
            max_tokens=max_tokens,
            temperature=temperature,
            on_request=on_request,
            on_response=on_response,
            **kwargs,
        )
        # Skip OpenAIModel.__init__ — it would rebuild the config without
        # OCI fields. Go straight to the Pydantic BaseModel init.
        super(OpenAIModel, self).__init__(config=config)

    @property
    def _requires_inlined_schema_refs(self) -> bool:
        """Gemini rejects ``$ref`` in JSON schemas — inline before sending."""
        model = (self.config.model or "").lower()
        return model.startswith("google.")

    def _munge_response_format_for_gemini(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Inline ``$ref`` for Gemini's response_format schema.

        Gemini's Proto validator rejects ``$ref`` and ``$defs`` in
        ``response_format.json_schema.schema`` — Pydantic emits both
        whenever a model contains a nested ``BaseModel``. ``inline_schema_refs``
        walks the schema and substitutes definitions inline. Vendor-gated
        because OpenAI handles ``$ref`` natively.
        """
        if "response_format" not in kwargs:
            return kwargs
        if not self._requires_inlined_schema_refs:
            return kwargs
        rf = kwargs["response_format"]
        if not (isinstance(rf, dict) and rf.get("type") == "json_schema"):
            return kwargs
        from locus.core.structured import inline_schema_refs

        schema = fold_schema_enums(inline_schema_refs(rf["json_schema"].get("schema", {})))
        rf = {**rf, "json_schema": {**rf["json_schema"], "schema": schema}}
        return {**kwargs, "response_format": rf}

    def _sanitize_tools_for_oci(
        self, tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Sanitize ``function.parameters`` on every tool before sending.

        OCI GenAI's V1 (OpenAI-compat) endpoint forwards tool schemas
        to vendor-specific validators. Gemini's Proto validator is the
        strictest of those vendors — it rejects ``type: ["string", "null"]``
        (Pydantic's emission for ``Optional[str]``), ``type: "any"``
        (Pydantic's emission for ``Any``), metadata keys like ``title``,
        ``const``, ``x-*``, and ``default: None``. The OCI tool API itself
        also rejects these metadata keys for non-Gemini models, so the
        sanitization is safe to apply unconditionally — matching the
        pattern used by ``langchain-oci``'s ``OCIUtils.sanitize_schema``.

        Mutation lands on ``tool["function"]["parameters"]``; tools
        without a ``function`` block (or without ``parameters``) are
        passed through untouched. The input list is not mutated; a new
        list of new dicts is returned so callers can re-send the
        originals to a non-OCI provider in the same process.
        """
        if not tools:
            return tools
        from locus.core.structured import sanitize_oci_tool_schema

        gemini = self._requires_inlined_schema_refs
        out: list[dict[str, Any]] = []
        for tool in tools:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(fn, dict) or "parameters" not in fn:
                out.append(tool)
                continue
            parameters = sanitize_oci_tool_schema(fn["parameters"])
            if gemini:
                # Gemini's translation also rejects ``enum`` in tool parameters.
                parameters = fold_schema_enums(parameters)
            out.append(
                {
                    **tool,
                    "function": {
                        **fn,
                        "parameters": parameters,
                    },
                }
            )
        return out

    def _prepare_messages_for_oci(
        self, messages: list[Message], tools: list[dict[str, Any]] | None
    ) -> list[Message]:
        """Apply vendor quirks of OCI's V1 endpoint to the message list.

        All vendors: an assistant turn with ``content=None`` (tool calls only)
        is sent with ``""``; the endpoint rejects ``null`` where OpenAI accepts
        it.

        Meta Llama with tools: OCI rejects the request with ``400 Tool
        calling for this Llama model requires the first message to be a
        user message`` whenever a system message leads the conversation —
        which is every Locus agent with a ``system_prompt``. Fold the
        leading system text into the first user message instead. Other
        vendors accept a leading system message, and Llama without tools
        does too, so the rewrite is scoped to exactly that case.
        """
        model = (self.config.model or "").lower()
        # Every vendor on OCI: a tool-call-only assistant turn must not carry
        # a null content (see the helper).
        messages = blank_null_assistant_content(messages)
        if tools and model.startswith("meta.llama"):
            messages = fold_leading_system_into_first_user(messages)
        if model.startswith("google."):
            # Gemini: tool results must be JSON objects (see the helper).
            messages = wrap_tool_results_as_objects(messages)
        return messages

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        kwargs = self._munge_response_format_for_gemini(kwargs)
        tools = self._sanitize_tools_for_oci(tools)
        messages = self._prepare_messages_for_oci(messages, tools)
        return await super().complete(messages, tools=tools, **kwargs)

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        """Stream the turn; answer a structured request in one piece when the service streams nothing.

        Some models behind this endpoint (the Llama family, observed on
        ``meta.llama-3.3-70b-instruct``) answer a ``json_schema``
        ``response_format`` without streaming it: the stream carries an empty
        content delta and ends with ``finish_reason: "tool_calls"`` and no
        tool-call delta, while the same request without ``stream`` returns the
        JSON. A structured answer can never be legitimately empty, so when a
        ``response_format`` was sent and nothing came through, the turn is
        asked again without streaming and its content is yielded as one chunk,
        followed by the terminal event of that completion.
        """
        kwargs = self._munge_response_format_for_gemini(kwargs)
        tools = self._sanitize_tools_for_oci(tools)
        messages = self._prepare_messages_for_oci(messages, tools)

        structured = kwargs.get("response_format") is not None
        streamed_anything = False
        terminal: ModelChunkEvent | None = None
        async for event in super().stream(messages, tools=tools, **kwargs):
            if event.done:
                terminal = event
                continue
            if event.content or event.tool_calls or event.reasoning:
                streamed_anything = True
            yield event

        if structured and not streamed_anything:
            from locus.core.events import ModelChunkEvent as _ChunkEvent  # noqa: PLC0415

            served = terminal.model if terminal is not None else None
            logger.debug(
                "OCI model %s streamed nothing for a response_format (stop_reason=%r); "
                "answering the structured turn without streaming",
                served or self.config.model,
                terminal.stop_reason if terminal is not None else None,
            )
            response = await super().complete(messages, tools=tools, **kwargs)
            if response.message.content:
                yield _ChunkEvent(content=response.message.content, model=served)
            if response.message.tool_calls:
                yield _ChunkEvent(tool_calls=list(response.message.tool_calls), model=served)
            yield _ChunkEvent(
                done=True,
                usage=response.usage or None,
                stop_reason=response.stop_reason,
                model=served,
                cost_usd=terminal.cost_usd if terminal is not None else None,
            )
            return

        if terminal is not None:
            yield terminal

    @property
    def client(self) -> openai.AsyncOpenAI:
        """Build the AsyncOpenAI client wired with the OCI request signer."""

        def build() -> openai.AsyncOpenAI:
            import openai

            # Pin the same timeout/retry knobs the parent OpenAIModel
            # uses so a stuck OCI request can't hang asyncio.gather()
            # forever (see ParallelPipeline) — without an explicit
            # timeout, the httpx default + openai-SDK default stack
            # together to ~10 minutes per call.
            http_client = httpx.AsyncClient(
                auth=build_oci_request_auth(
                    profile=self.config.profile,
                    auth_type=self.config.auth_type,
                    compartment_id=self.config.compartment_id,
                    config_file=self.config.config_file,
                    signer=self._build_signer(),
                ),
                timeout=httpx.Timeout(self.config.request_timeout),
                event_hooks=request_event_hooks(self.config.on_request, self.config.on_response),
            )
            return openai.AsyncOpenAI(
                api_key="not-used",
                base_url=self.config.base_url,
                http_client=http_client,
                max_retries=self.config.max_retries,
                timeout=self.config.request_timeout,
            )

        # Bound to the loop that built it — see locus.core.loop_bound.
        return loop_bound(self, "_client", build)

    def _build_signer(self) -> AbstractBaseSigner:
        return build_oci_signer(
            profile=self.config.profile,
            auth_type=self.config.auth_type,
            config_file=self.config.config_file,
        )


__all__ = [
    "DEFAULT_OCI_GENAI_REGION",
    "PRINCIPAL_AUTH_TYPES",
    "PROFILE_AUTH_TYPES",
    "OCIChatCompletionsConfig",
    "OCIChatCompletionsModel",
    "build_oci_openai_base_url",
    "build_oci_request_auth",
    "build_oci_signer",
    "resolve_oci_compartment",
    "resolve_profile_auth",
]
