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
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from locus.core.loop_bound import loop_bound
from locus.models.native.openai import OpenAIConfig, OpenAIModel


if TYPE_CHECKING:
    import openai
    from oci.signer import AbstractBaseSigner

    from locus.core.events import ModelChunkEvent
    from locus.core.messages import Message
    from locus.models.base import ModelResponse


DEFAULT_OCI_GENAI_REGION = "us-chicago-1"

AuthType = Literal["instance_principal", "resource_principal"]
_VALID_AUTH_TYPES: tuple[str, ...] = ("instance_principal", "resource_principal")


def build_oci_openai_base_url(region: str) -> str:
    """Construct the OCI GenAI OpenAI-compatible base URL for a region."""
    return f"https://inference.generativeai.{region}.oci.oraclecloud.com/openai/v1"


class OCIChatCompletionsConfig(OpenAIConfig):
    """Configuration for :class:`OCIChatCompletionsModel`.

    Inherits all OpenAI knobs from :class:`OpenAIConfig`. Adds region,
    profile, ``auth_type``, ``compartment_id``, and ``config_file`` for OCI
    auth selection.
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
        description='"instance_principal" or "resource_principal"',
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
    from oci.auth.signers import get_resource_principals_signer

    return get_resource_principals_signer()


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
        **kwargs: Any,
    ) -> None:
        """Initialize the OCI OpenAI-compat model.

        Args:
            model: OCI model identifier (e.g. ``openai.gpt-5``,
                ``meta.llama-3.3-70b-instruct``).
            profile: OCI config profile name from ``config_file``. Mutually
                exclusive with ``auth_type``.
            auth_type: ``"instance_principal"`` or ``"resource_principal"``.
                Mutually exclusive with ``profile``. Requires
                ``compartment_id``.
            compartment_id: OCI compartment OCID, sent as
                ``opc-compartment-id``. Auto-derived from the profile's
                tenancy under ``profile=``. Must be supplied explicitly
                under ``auth_type=``.
            region: OCI region hosting the inference endpoint.
            config_file: Path to the OCI config file (used with ``profile``).
            base_url: Override the derived endpoint URL (e.g. for a custom
                realm). Defaults to the OpenAI-compat URL for ``region``.
            max_tokens: Default max tokens. For ``o1``/``o3``/``gpt-5*``
                this is automatically forwarded as
                ``max_completion_tokens`` — see :class:`OpenAIModel`.
            temperature: Default sampling temperature.
            **kwargs: Forwarded to :class:`OCIChatCompletionsConfig` (top_p, seed,
                frequency_penalty, presence_penalty, ...).

        Raises:
            ValueError: If zero or both auth modes are set, if ``auth_type``
                is invalid, or if ``auth_type`` is set without
                ``compartment_id``.
        """
        modes_set = sum(x is not None for x in (profile, auth_type))
        if modes_set != 1:
            msg = (
                "OCIChatCompletionsModel: specify exactly one auth mode. "
                "Either profile='<section_from_~/.oci/config>' (api-key path; "
                "use profile='DEFAULT' for the default section), "
                "or auth_type='instance_principal'|'resource_principal'|"
                "'security_token'|'delegation_token' together with compartment_id=. "
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
        ):
            kwargs.pop(explicit, None)

        # Resolve compartment_id with this precedence:
        #   1. explicit ``compartment_id=`` arg
        #   2. ``OCI_COMPARTMENT`` env var (matches OCIModel + the test
        #      conftest convention so a single export drives every test)
        #   3. profile's tenancy (last-resort default; may not have GenAI
        #      policy if the profile lives in a different home tenancy than
        #      the GenAI compartment, which is the MY_PROFILE / cross-tenancy case)
        if compartment_id is None:
            import os

            compartment_id = os.getenv("OCI_COMPARTMENT") or os.getenv("OCI_COMPARTMENT_ID")
        if compartment_id is None and profile is not None:
            try:
                profile_cfg = _load_profile_config(profile, config_file)
                compartment_id = profile_cfg.get("tenancy")
            except Exception:  # noqa: BLE001 — profile load may fail; keep None
                compartment_id = None

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
        kwargs = self._munge_response_format_for_gemini(kwargs)
        tools = self._sanitize_tools_for_oci(tools)
        messages = self._prepare_messages_for_oci(messages, tools)
        async for event in super().stream(messages, tools=tools, **kwargs):
            yield event

    @property
    def client(self) -> openai.AsyncOpenAI:
        """Build the AsyncOpenAI client wired with the OCI request signer."""

        def build() -> openai.AsyncOpenAI:
            import httpx
            import openai

            from locus.models.providers.oci._signing import OCIRequestSigner

            signer = self._build_signer()
            # Token-based signers (instance principal, resource principal,
            # session token) carry credentials that expire on the order of
            # minutes-to-hours; without a refresh callback the 401 retry
            # path in OCIRequestSigner is a no-op and the process dies a
            # slow death of "all GenAI calls 401 after ~15-30min". The
            # callback rebuilds a fresh signer on each refresh (rebuild, not
            # in-place mutation — see _refresh_callable_for), so auth_flow
            # can both (a) refresh-on-401 and (b) refresh proactively on a
            # timer. Static signers (user-principal API key) return None and
            # the refresh path stays dormant. refresh_interval is tightened
            # to 10 minutes (upstream default is 1h) to re-mint principal
            # credentials well inside their lifetime.
            #
            # Pin the same timeout/retry knobs the parent OpenAIModel
            # uses so a stuck OCI request can't hang asyncio.gather()
            # forever (see ParallelPipeline) — without an explicit
            # timeout, the httpx default + openai-SDK default stack
            # together to ~10 minutes per call.
            http_client = httpx.AsyncClient(
                auth=OCIRequestSigner(
                    signer,
                    compartment_id=self.config.compartment_id,
                    refresh_signer=_refresh_callable_for(
                        signer,
                        auth_type=self.config.auth_type,
                        profile=self.config.profile,
                        config_file=self.config.config_file,
                    ),
                    refresh_interval=600.0,
                ),
                timeout=httpx.Timeout(self.config.request_timeout),
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
        if self.config.auth_type == "instance_principal":
            return _build_instance_principal_signer()
        if self.config.auth_type == "resource_principal":
            return _build_resource_principal_signer()
        # Validation in __init__ guarantees profile is set if we got here.
        assert self.config.profile is not None  # noqa: S101 — invariant
        return _build_signer_from_profile(self.config.profile, self.config.config_file)


__all__ = [
    "DEFAULT_OCI_GENAI_REGION",
    "OCIChatCompletionsConfig",
    "OCIChatCompletionsModel",
    "build_oci_openai_base_url",
]
