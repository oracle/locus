# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OCI GenAI Responses API transport.

A server-stateful transport — the OCI Responses endpoint holds the
conversation thread between turns and is keyed by ``previous_response_id``.
Locus's runtime loop sends only the input added since the last turn and
threads the continuation token through :attr:`AgentState.provider_state`.

The only Locus primitive that doesn't apply on this path is the
``ConversationManager`` (window/summarize strategies have nothing to
operate on — the history is server-side). Everything else — memory,
reflexion, GSAR, grounding, tool hooks, idempotency, checkpointing,
streaming, structured output, termination conditions — works
identically to the chat/completions path.

Auth mirrors :class:`OCIChatCompletionsModel`: API key via ``profile=``, or
``auth_type="instance_principal"|"resource_principal"`` for workload
identity. Reuses the existing httpx signer (``OCIRequestSigner``).
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from locus.core.events import ModelChunkEvent
from locus.core.loop_bound import loop_bound
from locus.core.messages import Message, ToolCall
from locus.core.warnings import LocusWarning
from locus.models.base import ModelResponse
from locus.models.providers.oci._responses_parse import (
    OCI_RESPONSES_ITEMS_METADATA_KEY,
    build_request_body,
    parse_compaction_response,
    parse_response,
    parse_stream_event,
    reasoning_text,
)
from locus.models.providers.oci._signing import OCIRequestSigner
from locus.models.providers.oci.openai_compat import (
    DEFAULT_OCI_GENAI_REGION,
    _build_instance_principal_signer,
    _build_resource_principal_signer,
    _build_signer_from_profile,
    _load_profile_config,
    _refresh_callable_for,
    build_oci_openai_base_url,
)


if TYPE_CHECKING:
    from oci.signer import AbstractBaseSigner


_VALID_AUTH_TYPES = frozenset({"instance_principal", "resource_principal"})


class OCIProjectRequiredError(Exception):
    """Raised when an OCI Responses request requires a Project OCID.

    Locus keeps the Project OCID dependency optional. If a specific
    Responses feature demands one (some preview features do), the OCI
    endpoint returns a 403 / 404 with a project-related error body, and
    for non-OpenAI models it returns a 400 naming the ``OpenAI-Project``
    / ``opc-conversation-store-id`` header; this exception surfaces both
    with a pointer at the constructor kwarg so the caller can fix the
    call without grepping wire dumps.
    """


#: Lower-cased fragments of the 400 the Responses endpoint returns when a
#: model needs a project header (non-OpenAI models on ``/responses``).
_PROJECT_HEADER_MARKERS: tuple[str, ...] = (
    "openai-project",
    "opc-project-id",
    "opc-conversation-store-id",
)


class OCIResponsesStateLostError(Exception):
    """Raised when ``previous_response_id`` is unknown or expired.

    Responses threads expire (typically 30 days at the OCI side). When
    the continuation token in :attr:`AgentState.provider_state` is no
    longer valid, the agent should usually restart the run rather than
    silently dropping the conversation.
    """


class OCIResponsesZeroDataRetentionError(OCIResponsesStateLostError):
    """Raised when the tenancy enforces Zero Data Retention (ZDR).

    On a ZDR tenancy the Responses endpoint answers every request that
    carries ``previous_response_id`` with HTTP 400 ("Previous response
    cannot be used for this organization due to Zero Data Retention"),
    so the server-stateful flow can never continue a thread there. A
    subclass of :class:`OCIResponsesStateLostError`, so handlers written
    for the expired-thread case keep working.

    Under ``store="auto"`` (the default) the model switches itself to the
    stateless path when it first sees this error and the agent runtime
    replays the turn from the history it holds; the error still
    propagates out of :meth:`OCIResponsesModel.complete` so a direct
    caller knows to re-send the full conversation. Under ``store=True``
    nothing is switched and the error is the final word.
    """


#: Lower-cased fragment of the 400 a Zero Data Retention tenancy returns
#: for ``previous_response_id``.
_ZDR_MARKER = "zero data retention"


class OCIResponsesConfig(BaseModel):
    """Configuration for :class:`OCIResponsesModel`."""

    model: str
    region: str = Field(default=DEFAULT_OCI_GENAI_REGION)
    profile: str | None = None
    auth_type: str | None = None
    config_file: str = "~/.oci/config"
    compartment_id: str | None = None
    project_ocid: str | None = Field(
        default=None,
        description=(
            "OCI GenAI Project OCID. Optional by design — required only "
            "when a specific Responses feature demands it. When unset and "
            "such a feature is requested, the model raises "
            "OCIProjectRequiredError."
        ),
    )
    base_url: str | None = None
    max_output_tokens: int = 4096
    temperature: float = 0.7
    request_timeout: float = 600.0
    reasoning_effort: str | None = Field(
        default=None,
        description=(
            "Reasoning effort for reasoning models (e.g. ``low``, "
            "``medium``, ``high``), sent as ``reasoning={'effort': ...}``. "
            "The accepted values depend on the model and are validated by "
            "the server. None (default) sends no ``reasoning`` field, so "
            "the server's default applies. A ``reasoning_effort`` call "
            "kwarg overrides it for that call."
        ),
    )
    reasoning: dict[str, Any] | None = Field(
        default=None,
        description=(
            "The Responses-native ``reasoning`` object (``effort``, "
            "``summary``, ...), sent as given. When set together with "
            "``reasoning_effort`` the dict wins and ``reasoning_effort`` only "
            "fills ``effort`` if the dict lacks it. A ``reasoning`` call kwarg "
            "replaces it for that call. None (default) sends nothing."
        ),
    )
    store: bool | Literal["auto"] = Field(
        default="auto",
        description=(
            "Whether the OCI server should persist responses and accept "
            "``previous_response_id`` for multi-turn continuation. "
            "``'auto'`` (default) starts stateful and, if the endpoint "
            "reports Zero Data Retention (ZDR) for the tenancy, switches the "
            "model to the stateless path for the rest of its lifetime with "
            "one warning. ``True`` is always stateful and raises "
            "``OCIResponsesZeroDataRetentionError`` on a ZDR tenancy. "
            "``False`` is always stateless: ``store=false`` is sent on every "
            "request, the model advertises ``server_stateful=False`` so the "
            "agent sends the full message history each turn, and "
            "``provider_state`` stays empty. ZDR tenants still benefit from "
            "access to Responses-only models like ``openai.gpt-5.5-pro``."
        ),
    )


class OCIResponsesModel(BaseModel):
    """OCI GenAI model via the Responses API.

    Server-stateful transport — Locus's runtime loop sends only the
    input added since the last turn and threads
    ``previous_response_id`` through :attr:`AgentState.provider_state`
    so the OCI server can reattach to the existing thread.

    Example:
        >>> model = OCIResponsesModel(
        ...     model="openai.gpt-5.5-pro",
        ...     profile="LUIGI_FRA_API",
        ...     region="us-chicago-1",
        ... )
        >>> agent = Agent(model=model, tools=[my_tool])
        >>> result = agent.run_sync("Plan a trip to Tokyo.")

    Pass exactly one of ``profile``, ``auth_type``.
    """

    config: OCIResponsesConfig
    _client: httpx.AsyncClient | None = None
    _client_loop: Any = None  # event loop ``_client`` was built on
    _signer: Any = None  # AbstractBaseSigner — lazy import-free
    _zdr_detected: bool = False  # set once under store="auto", never reset

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model: str,
        *,
        profile: str | None = None,
        auth_type: str | None = None,
        compartment_id: str | None = None,
        project_ocid: str | None = None,
        region: str = DEFAULT_OCI_GENAI_REGION,
        config_file: str = "~/.oci/config",
        base_url: str | None = None,
        max_output_tokens: int = 4096,
        temperature: float = 0.7,
        request_timeout: float = 600.0,
        store: bool | Literal["auto"] = "auto",
        reasoning_effort: str | None = None,
        reasoning: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OCI Responses model.

        Args:
            model: OCI model identifier (e.g. ``openai.gpt-5.5-pro``).
            profile: OCI config profile name in ``config_file``.
                Mutually exclusive with ``auth_type``.
            auth_type: ``"instance_principal"`` or ``"resource_principal"``.
                Mutually exclusive with ``profile``. Requires
                ``compartment_id``.
            compartment_id: OCI compartment OCID, sent as
                ``opc-compartment-id``. Auto-derived from the profile's
                tenancy under ``profile=``. Required under ``auth_type=``.
            project_ocid: Optional GenAI Project OCID. Locus does **not**
                require this — leave unset for the standard path.
                ``OCIProjectRequiredError`` is raised at request time if
                the model's response indicates a project is needed.
            region: OCI region (default us-chicago-1).
            config_file: Path to the OCI config file (with ``profile=``).
            base_url: Override the derived endpoint URL.
            max_output_tokens: Default output token cap.
            temperature: Default sampling temperature.
            request_timeout: HTTP request timeout in seconds.
            store: ``"auto"`` (default) starts stateful: the OCI server
                persists responses and the agent uses
                ``previous_response_id`` for cheap multi-turn continuation.
                If the endpoint reports Zero Data Retention (ZDR) for the
                tenancy, the model switches to the stateless path for the
                rest of its lifetime (see :attr:`zdr_detected`), warns
                once, and the agent runtime replays the turn from its own
                history. ``True`` never switches: a ZDR tenancy raises
                ``OCIResponsesZeroDataRetentionError``. ``False`` is
                stateless from the start — the model reports
                ``server_stateful=False`` to the agent runtime, so the
                agent sends the full message history each turn (like
                chat/completions) but still uses the Responses endpoint —
                the right choice when the tenancy is known to be ZDR.
            reasoning_effort: Reasoning effort for reasoning models (e.g.
                ``"low"``, ``"medium"``, ``"high"``), sent as
                ``reasoning={"effort": ...}``. The accepted values depend on
                the model and are validated by the server. ``None`` (default)
                sends no ``reasoning`` field. A ``reasoning_effort`` kwarg on
                a call, such as an agent's ``model_kwargs``, overrides it.
            reasoning: The Responses-native ``reasoning`` object, e.g.
                ``{"effort": "high", "summary": "auto"}``, sent as given.
                With ``reasoning_effort`` set too, the dict wins and the
                effort only fills ``effort`` when the dict lacks it. A
                ``reasoning`` call kwarg replaces it for that call. Under
                ``store=False`` any reasoning control also requests
                ``include=["reasoning.encrypted_content"]``, and the reasoning
                items that come back are replayed on the next request so a
                tool-calling turn keeps its reasoning.

        Raises:
            ValueError: If zero or both auth modes are set, if
                ``auth_type`` is invalid, or if ``auth_type`` is set
                without ``compartment_id``.
        """
        modes_set = sum(x is not None for x in (profile, auth_type))
        if modes_set != 1:
            msg = "specify exactly one of profile=, auth_type="
            raise ValueError(msg)
        if auth_type is not None and auth_type not in _VALID_AUTH_TYPES:
            msg = f"auth_type must be one of {sorted(_VALID_AUTH_TYPES)}, got {auth_type!r}"
            raise ValueError(msg)
        if auth_type is not None and compartment_id is None:
            msg = "compartment_id is required when auth_type= is set"
            raise ValueError(msg)

        # Resolve compartment_id with same precedence as OCIChatCompletionsModel.
        if compartment_id is None:
            import os

            compartment_id = os.getenv("OCI_COMPARTMENT") or os.getenv("OCI_COMPARTMENT_ID")
        if compartment_id is None and profile is not None:
            try:
                profile_cfg = _load_profile_config(profile, config_file)
                compartment_id = profile_cfg.get("tenancy")
            except Exception:  # noqa: BLE001 — best-effort resolution
                compartment_id = None

        config = OCIResponsesConfig(
            model=model,
            region=region,
            profile=profile,
            auth_type=auth_type,
            config_file=config_file,
            compartment_id=compartment_id,
            project_ocid=project_ocid,
            store=store,
            base_url=base_url or build_oci_openai_base_url(region),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            request_timeout=request_timeout,
            reasoning_effort=reasoning_effort,
            reasoning=reasoning,
            **kwargs,
        )
        super().__init__(config=config)

    @property
    def server_stateful(self) -> bool:
        """Whether this model relies on OCI-side response persistence.

        Reflects :attr:`OCIResponsesConfig.store` and, under
        ``store="auto"``, whether Zero Data Retention has been detected
        (:attr:`zdr_detected`). The agent runtime reads this before every
        model call to decide between the latest-turn-slice +
        ``provider_state`` flow (True) and the stateless full-history
        flow (False), so the switch takes effect on the very turn that
        met the ZDR error.
        """
        if self.config.store is False:
            return False
        return not self._zdr_detected

    @property
    def zdr_detected(self) -> bool:
        """``True`` once the endpoint reported Zero Data Retention.

        Only ever set under ``store="auto"``, on the first
        :class:`OCIResponsesZeroDataRetentionError`; it stays set for the
        lifetime of the instance, so no later run on this model attempts
        the stateful path again. A plain flag written once from the
        coroutine that saw the error — safe for concurrent asyncio
        callers on one loop.
        """
        return self._zdr_detected

    def _note_zero_data_retention(self) -> None:
        """Switch to the stateless path under ``store="auto"``; no-op otherwise."""
        if self.config.store != "auto" or self._zdr_detected:
            return
        self._zdr_detected = True
        warnings.warn(
            "OCI Responses: the endpoint reported Zero Data Retention for this "
            "tenancy, so previous_response_id cannot be used. "
            f"OCIResponsesModel({self.config.model!r}) now sends store=false "
            "with the full history on every request for the rest of its "
            "lifetime. Pass store=False at construction to skip the first "
            "stateful attempt.",
            LocusWarning,
            stacklevel=3,
        )

    def _build_signer(self) -> AbstractBaseSigner:
        if self.config.auth_type == "instance_principal":
            return _build_instance_principal_signer()
        if self.config.auth_type == "resource_principal":
            return _build_resource_principal_signer()
        assert self.config.profile is not None  # noqa: S101 — invariant from __init__
        return _build_signer_from_profile(self.config.profile, self.config.config_file)

    def _http_client(self) -> httpx.AsyncClient:
        """Build (or reuse) the OCI-signed httpx client."""

        def build() -> httpx.AsyncClient:
            signer = self._build_signer()
            # Wire the signer's refresh hook so instance-principal /
            # resource-principal federation tokens auto-refresh both
            # on 401 and on a 10-minute periodic timer. Without this,
            # the captured signer's token expires after ~15-30 min and
            # every subsequent /v1/responses call 401s until the
            # process restarts. Same fix as openai_compat.py — kept
            # parallel via the shared _refresh_callable_for helper so
            # future signer types get refresh support in one place.
            # Static signers (user-principal API key) return None and
            # the refresh branches stay dormant.
            return httpx.AsyncClient(
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
                base_url=self.config.base_url or "",
            )

        # Bound to the loop that built it — see locus.core.loop_bound.
        return loop_bound(self, "_client", build)

    def _extra_headers(self) -> dict[str, str]:
        """Headers added on top of OCIRequestSigner's signature.

        ``opc-compartment-id`` is set by the signer for IAM-auth paths;
        the optional Project OCID is folded in here so it's part of the
        signed payload too. It goes out under both names: ``OpenAI-Project``
        is the header the Responses endpoint reads (non-OpenAI models
        require it there), ``opc-project-id`` is kept for the OCI-side
        features that look for it.
        """
        headers: dict[str, str] = {}
        if self.config.project_ocid is not None:
            headers["OpenAI-Project"] = self.config.project_ocid
            headers["opc-project-id"] = self.config.project_ocid
        return headers

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        provider_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Send a Responses request and return the parsed reply.

        ``provider_state`` is the continuation token from
        :attr:`AgentState.provider_state`. On turn 1 it's ``None``; on
        subsequent turns it carries ``{"previous_response_id": <id>}``.
        """
        previous_response_id = (
            provider_state.get("previous_response_id") if provider_state else None
        )

        # OpenAI reasoning families (gpt-5, gpt-5.5, o1, o3, o4) reject
        # ``temperature`` on the Responses endpoint. The Agent loop
        # always passes ``temperature=self.config.temperature`` as a
        # kwarg, so a "pop unless explicit" guard isn't enough — drop
        # it unconditionally on the Responses path.
        kwargs.pop("temperature", None)
        body = build_request_body(
            messages,
            model=self.config.model,
            tools=tools,
            previous_response_id=previous_response_id,
            temperature=None,
            max_output_tokens=kwargs.pop("max_tokens", None)
            or kwargs.pop("max_output_tokens", self.config.max_output_tokens),
            stream=False,
            response_format=kwargs.pop("response_format", None),
            store=self.server_stateful,
            reasoning=kwargs.pop("reasoning", None) or self.config.reasoning,
            reasoning_effort=kwargs.pop("reasoning_effort", None) or self.config.reasoning_effort,
        )

        client = self._http_client()
        try:
            resp = await client.post(
                "/responses",
                json=body,
                headers=self._extra_headers(),
            )
        except httpx.HTTPError as exc:
            msg = f"OCI Responses request failed: {exc}"
            raise RuntimeError(msg) from exc

        try:
            self._raise_for_error(resp)
        except OCIResponsesZeroDataRetentionError:
            self._note_zero_data_retention()
            raise

        try:
            payload = resp.json()
        except ValueError as exc:
            msg = f"OCI Responses returned non-JSON body: {resp.text[:300]}"
            raise RuntimeError(msg) from exc

        message, usage, stop_reason, parsed_provider_state = parse_response(payload)
        return ModelResponse(
            message=message,
            usage=usage,
            stop_reason=stop_reason,
            provider_state=parsed_provider_state or None,
            reasoning=reasoning_text(message),
        )

    async def compact_history(
        self,
        messages: list[Message],
        *,
        instructions: str | None = None,
    ) -> tuple[list[Message], dict[str, int]]:
        """Hand a conversation to the endpoint's own compaction.

        Posts the thread to ``/responses/compact``; the endpoint keeps the
        messages a continuation needs and folds the rest into an encrypted
        ``compaction`` item. The result is a shorter Locus history that this
        model replays verbatim on later requests (the compaction item rides
        on an assistant message under
        :data:`OCI_RESPONSES_ITEMS_METADATA_KEY`), so it is only meaningful
        for a thread that keeps going through this transport.

        :class:`~locus.memory.compaction.ContextCompactor` calls this when
        built with ``native_compaction=True``; it is the opt-in that hands
        the summary stage to the endpoint instead of writing one client-side.

        Args:
            messages: The history to compact. System messages become the
                request's ``instructions``; the rest is the ``input``.
            instructions: Extra guidance appended to the instructions (a
                before-compaction hook's steering).

        Returns:
            ``(messages, usage)``: the compacted history, without the system
            messages, and the compaction call's token usage.
        """
        body = build_request_body(messages, model=self.config.model, store=True)
        request: dict[str, Any] = {"model": body["model"], "input": body["input"]}
        guidance = [body.get("instructions"), instructions]
        joined = "\n\n".join(part for part in guidance if part)
        if joined:
            request["instructions"] = joined

        client = self._http_client()
        try:
            resp = await client.post(
                "/responses/compact",
                json=request,
                headers=self._extra_headers(),
            )
        except httpx.HTTPError as exc:
            msg = f"OCI Responses compaction request failed: {exc}"
            raise RuntimeError(msg) from exc
        self._raise_for_error(resp)
        try:
            payload = resp.json()
        except ValueError as exc:
            msg = f"OCI Responses compaction returned non-JSON body: {resp.text[:300]}"
            raise RuntimeError(msg) from exc
        return parse_compaction_response(payload)

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        provider_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        """Stream a Responses request as Locus ModelChunkEvents.

        Accumulates ``function_call_arguments.delta`` events per call
        id and emits a single :class:`ToolCall` per id once arguments
        finish. Text deltas emit one event per chunk. ``done=True``
        fires on ``response.completed`` carrying the continuation id
        in ``provider_state``.
        """
        previous_response_id = (
            provider_state.get("previous_response_id") if provider_state else None
        )
        # See complete() — temperature is always dropped on the
        # Responses path (reasoning models reject it).
        kwargs.pop("temperature", None)
        body = build_request_body(
            messages,
            model=self.config.model,
            tools=tools,
            previous_response_id=previous_response_id,
            temperature=None,
            max_output_tokens=kwargs.pop("max_tokens", None)
            or kwargs.pop("max_output_tokens", self.config.max_output_tokens),
            stream=True,
            response_format=kwargs.pop("response_format", None),
            store=self.server_stateful,
            reasoning=kwargs.pop("reasoning", None) or self.config.reasoning,
            reasoning_effort=kwargs.pop("reasoning_effort", None) or self.config.reasoning_effort,
        )

        client = self._http_client()
        tool_args_buf: dict[str, str] = {}
        tool_names: dict[str, str] = {}
        # Argument deltas are keyed by the stream item id; ``output_item.added``
        # announces the call id the server expects back. Accumulate under the
        # call id so the emitted ToolCall matches what ``complete()`` returns.
        item_to_call: dict[str, str] = {}

        async with client.stream(
            "POST", "/responses", json=body, headers=self._extra_headers()
        ) as resp:
            if not resp.is_success:
                # A streaming response has no body yet; read it so the
                # server's message reaches _raise_for_error instead of
                # surfacing as httpx.ResponseNotRead.
                await resp.aread()
            try:
                self._raise_for_error(resp)
            except OCIResponsesZeroDataRetentionError:
                self._note_zero_data_retention()
                raise
            async for raw_event in _iter_sse_events(resp):
                parsed = parse_stream_event(raw_event)
                if not parsed:
                    continue
                if "content" in parsed:
                    yield ModelChunkEvent(content=parsed["content"])
                if "tool_calls" in parsed:
                    for partial in parsed["tool_calls"]:
                        cid = partial.get("id", "")
                        item_id = partial.get("item_id", "")
                        if item_id and cid and cid != item_id:
                            item_to_call[item_id] = cid
                            # Deltas that arrived under the item id move over.
                            if item_id in tool_args_buf:
                                tool_args_buf[cid] = tool_args_buf.pop(item_id) + tool_args_buf.get(
                                    cid, ""
                                )
                            if item_id in tool_names and cid not in tool_names:
                                tool_names[cid] = tool_names.pop(item_id)
                        cid = item_to_call.get(cid, cid)
                        if partial.get("final"):
                            # ``output_item.done`` carries the whole argument string.
                            tool_args_buf[cid] = partial.get("arguments") or tool_args_buf.get(
                                cid, ""
                            )
                        else:
                            delta = partial.get("arguments_delta", "")
                            tool_args_buf[cid] = tool_args_buf.get(cid, "") + delta
                        if partial.get("name"):
                            tool_names[cid] = partial["name"]
                if parsed.get("error"):
                    msg = parsed["error"]
                    raise RuntimeError(f"OCI Responses stream error: {msg}")
                if parsed.get("done"):
                    # Emit accumulated tool calls before the done marker.
                    if tool_args_buf:
                        import json

                        completed_calls: list[ToolCall] = []
                        for cid, raw in tool_args_buf.items():
                            try:
                                args = json.loads(raw) if raw else {}
                            except (ValueError, TypeError):
                                args = {}
                            completed_calls.append(
                                ToolCall(id=cid, name=tool_names.get(cid, ""), arguments=args)
                            )
                        yield ModelChunkEvent(tool_calls=completed_calls)
                    yield ModelChunkEvent(done=True)

    def _raise_for_error(self, response: httpx.Response) -> None:
        """Translate OCI HTTP error responses into Locus-typed exceptions."""
        if response.is_success:
            return

        try:
            body = response.json()
        except ValueError:
            body = {}

        # Try to surface OCI's structured error.
        error_obj = body.get("error", body) if isinstance(body, dict) else {}
        message = ""
        param = ""
        if isinstance(error_obj, dict):
            message = str(error_obj.get("message") or error_obj.get("code") or "")
            # The unwrapped envelope names the offending field here rather
            # than in the message (``"param": "previous_response_id"``).
            param = str(error_obj.get("param") or "")

        text = f"{message} {param}".strip().lower() or response.text.lower()

        project_signalled = (response.status_code in (403, 404) and "project" in text) or (
            response.status_code == 400 and any(m in text for m in _PROJECT_HEADER_MARKERS)
        )
        if project_signalled:
            raise OCIProjectRequiredError(
                "OCI Responses request requires a Project OCID — pass "
                "project_ocid=<ocid> to OCIResponsesModel(...). "
                f"Server said: {message or response.text[:200]}"
            )

        if response.status_code == 400 and _ZDR_MARKER in text:
            raise OCIResponsesZeroDataRetentionError(
                "OCI Responses: this tenancy enforces Zero Data Retention, so "
                "previous_response_id cannot be used. Build the model with "
                "store=False (or leave store='auto' and let it switch). "
                f"Server said: {message or response.text[:200]}"
            )

        if response.status_code in (400, 404) and (
            "previous_response_id" in text or "response not found" in text or "thread" in text
        ):
            raise OCIResponsesStateLostError(
                "OCI Responses thread state is unknown or expired. "
                "The agent should restart the conversation. "
                f"Server said: {message or response.text[:200]}"
            )

        # Generic — let the caller see status + body.
        msg = f"OCI Responses returned {response.status_code}: {message or response.text[:300]}"
        raise RuntimeError(msg)

    async def aclose(self) -> None:
        """Close the underlying httpx client."""
        if self._client is not None:
            await self._client.aclose()
            object.__setattr__(self, "_client", None)


async def _iter_sse_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Iterate SSE events from a streaming httpx response.

    Each event is a JSON object on a ``data:`` line. Multi-line events
    are concatenated; ``[DONE]`` sentinels terminate the stream.
    """
    import json as _json

    buffer: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if buffer:
                joined = "\n".join(buffer).strip()
                buffer = []
                if joined == "[DONE]":
                    return
                try:
                    yield _json.loads(joined)
                except (ValueError, TypeError):
                    continue
            continue
        if line.startswith("data:"):
            buffer.append(line[5:].lstrip())


__all__ = [
    "OCI_RESPONSES_ITEMS_METADATA_KEY",
    "OCIProjectRequiredError",
    "OCIResponsesConfig",
    "OCIResponsesModel",
    "OCIResponsesStateLostError",
    "OCIResponsesZeroDataRetentionError",
]
