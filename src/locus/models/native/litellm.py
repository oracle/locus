# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""LiteLLM-backed model provider.

`LiteLLMModel` is the unified Locus entry point for all three Oracle
Generative AI Infrastructure inference surfaces:

1. **Native GenAI API** (``transport="oci_native"``, the default) — routes
   through ``litellm.acompletion(model="oci/...")``, which targets OCI's
   canonical ``POST /20231130/actions/chat`` (and ``/embedText``) with
   LiteLLM's vendor adapters for Cohere v1 transport (preamble +
   chatHistory) and the GENERIC apiFormat used by Grok, Llama, Gemini,
   and gpt-5.

2. **OpenAI Chat Completions compat** (``transport="oci_openai_v1"``) —
   routes through Locus's existing :class:`OCIChatCompletionsModel`, which
   targets OCI's ``POST /openai/v1/chat/completions`` (the V1 OpenAI-compat
   surface) with OCI Signature v1 RSA-SHA256 signing.

3. **OpenAI Responses API compat** (``transport="oci_responses"``) — routes
   through Locus's existing :class:`OCIResponsesModel`, which targets OCI's
   ``POST /openai/v1/responses`` (the newer server-stateful Responses API)
   with native support for ``previous_response_id`` multi-turn continuation.

All three paths converge on the same Locus interface (``complete`` /
``stream``), so callers pick the transport via config and the rest of the
agent loop doesn't care which surface served the request.

The native path also works with any of the 100+ non-OCI providers LiteLLM
supports — pass ``model="anthropic/claude-..."``, ``model="gpt-4o"``, etc.

Example — native (default)
--------------------------

    from locus.models.native.litellm import LiteLLMModel
    from locus.core.messages import Message

    model = LiteLLMModel(model="oci/xai.grok-4.20")  # OCI_* env vars supply auth
    resp = await model.complete(messages=[Message.user("Hello")])

Example — OCI OpenAI-compat (V1 shim)
-------------------------------------

    model = LiteLLMModel(
        model="openai.gpt-5-mini",
        transport="oci_openai_v1",
        oci_profile="DEFAULT",
        oci_compartment_id="ocid1.compartment.oc1..xxx",  # optional under profile=
    )
    resp = await model.complete(messages=[Message.user("Hello")])
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr

from locus.core.events import ModelChunkEvent
from locus.core.messages import Message, ToolCall
from locus.models.base import ModelConfig, ModelResponse


# Transports supported by LiteLLMModel.
LiteLLMTransport = Literal["oci_native", "oci_openai_v1", "oci_responses"]


# OCI signer object is opaque to Locus — we only forward it; we don't import
# the OCI SDK at module load. Keeps the SDK optional for users who only call
# non-OCI LiteLLM-backed providers.
_OciSigner = Any


# Models that reject ``temperature`` with HTTP 400 because they fix the
# temperature internally. Matched on stable prefixes so we don't need to bump
# the list every time a new minor version ships. As of 2026-05, OCI's hosted
# OpenAI deployments enforce ``temperature=1.0`` on the gpt-5 reasoning family,
# returning ``Unsupported value: 'temperature' does not support 0.7 with this
# model. Only the default (1) value is supported.``
_FIXED_TEMPERATURE_PREFIXES: tuple[str, ...] = (
    "oci/openai.gpt-5",
    "oci/openai.o1",
    "oci/openai.o3",
    "oci/openai.o4",
)


def _model_fixes_temperature(model_id: str) -> bool:
    """Return True if the named model rejects an explicit ``temperature``."""
    return any(model_id.startswith(p) for p in _FIXED_TEMPERATURE_PREFIXES)


class LiteLLMConfig(ModelConfig):
    """Configuration for `LiteLLMModel`.

    Under ``transport="oci_native"`` the ``model`` string follows LiteLLM's
    ``<provider>/<model>`` convention, e.g. ``"oci/xai.grok-4.20"``,
    ``"oci/openai.gpt-5-mini"``, ``"oci/cohere.command-latest"``. For non-OCI
    providers any LiteLLM model id works (``"gpt-4o"``,
    ``"anthropic/claude-sonnet-4-20250514"``, etc.).

    Under ``transport="oci_openai_v1"`` the ``model`` is the bare OCI model
    id (no ``oci/`` prefix), e.g. ``"openai.gpt-5-mini"``,
    ``"meta.llama-3.3-70b-instruct"`` — matching the form OCI's V1 shim
    expects.
    """

    transport: LiteLLMTransport = Field(
        default="oci_native",
        description=(
            "Which OCI inference surface to use. "
            "``oci_native`` routes through LiteLLM's ``oci/...`` provider "
            "against ``POST /20231130/actions/chat`` with vendor adapters. "
            "``oci_openai_v1`` routes through OCI's OpenAI Chat Completions "
            "shim at ``POST /openai/v1/chat/completions`` via Locus's "
            "``OCIChatCompletionsModel``. "
            "``oci_responses`` routes through OCI's OpenAI Responses API "
            "shim at ``POST /openai/v1/responses`` via Locus's "
            "``OCIResponsesModel`` (server-stateful, supports "
            "``previous_response_id`` multi-turn)."
        ),
    )

    # ----- Generic LiteLLM passthrough --------------------------------------
    api_key: str | None = Field(
        default=None,
        description=(
            "API key for non-OCI providers under ``transport=oci_native``. "
            "Ignored for ``oci/*`` models and for the ``oci_openai_v1`` "
            "transport (OCI auth is supplied through the ``oci_*`` fields, "
            "``oci_profile``, or ``OCI_*`` environment variables)."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description="Custom API base URL (e.g. a self-hosted LiteLLM proxy).",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description="LiteLLM-level retry count for transient 5xx / 429 errors.",
    )
    request_timeout: float | None = Field(
        default=None,
        description="Per-request timeout in seconds. ``None`` uses LiteLLM's default.",
    )
    drop_params: bool = Field(
        default=True,
        description=(
            "When True, pass ``drop_params=True`` to LiteLLM so OpenAI "
            "kwargs the target provider doesn't accept are silently "
            "stripped rather than raising. Recommended for cross-vendor "
            "agent loops."
        ),
    )

    # ----- OCI-native transport fields (all optional; env vars work too) ----
    # Used when ``transport="oci_native"``.
    oci_region: str | None = None
    oci_user: str | None = None
    oci_fingerprint: str | None = None
    oci_tenancy: str | None = None
    oci_key: str | None = None
    oci_key_file: str | None = None
    oci_compartment_id: str | None = None
    oci_serving_mode: str | None = Field(
        default=None,
        description='``"ON_DEMAND"`` (default in LiteLLM) or ``"DEDICATED"``.',
    )
    oci_endpoint_id: str | None = Field(
        default=None,
        description="OCID of the dedicated endpoint (used with ``oci_serving_mode='DEDICATED'``).",
    )
    oci_signer: _OciSigner = Field(
        default=None,
        description=(
            "OCI SDK Signer object (``oci.auth.signers.*``). Use this for "
            "Instance Principal, OKE Workload Identity, or any other "
            "non-API-key OCI auth path. When set, overrides the manual "
            "credentials above."
        ),
    )

    # ----- OCI-OpenAI-V1 transport fields -----------------------------------
    # Used when ``transport="oci_openai_v1"`` — forwarded straight into
    # ``OCIChatCompletionsModel``.
    oci_profile: str | None = Field(
        default=None,
        description=(
            "OCI config-file profile name (e.g. ``'DEFAULT'``, "
            "``'LUIGI_FRA_API'``). Mutually exclusive with ``oci_auth_type``. "
            "Only used under ``transport='oci_openai_v1'``."
        ),
    )
    oci_auth_type: str | None = Field(
        default=None,
        description=(
            "``'instance_principal'`` / ``'resource_principal'`` / "
            "``'security_token'`` / ``'delegation_token'``. Mutually "
            "exclusive with ``oci_profile``. Requires ``oci_compartment_id``."
        ),
    )
    oci_config_file: str = Field(
        default="~/.oci/config",
        description="Path to the OCI config file. Used with ``oci_profile``.",
    )
    oci_base_url: str | None = Field(
        default=None,
        description=(
            "Override the derived OCI OpenAI-compat endpoint URL (e.g. for a "
            "custom realm). Defaults to "
            "``https://inference.generativeai.<region>.oci.oraclecloud.com/openai/v1``."
        ),
    )

    # ----- OCI Responses transport (additional fields) ----------------------
    # Used when ``transport="oci_responses"`` — forwarded straight into
    # ``OCIResponsesModel`` on top of the V1 fields above.
    oci_project_ocid: str | None = Field(
        default=None,
        description=(
            "Optional OCI GenAI Project OCID. Most tenancies don't require "
            "it; ``OCIResponsesModel`` raises ``OCIProjectRequiredError`` "
            "at request time when the server signals it's needed."
        ),
    )
    oci_responses_store: bool = Field(
        default=True,
        description=(
            "When True (default), the OCI server persists responses and "
            "the agent uses ``previous_response_id`` for cheap multi-turn "
            "continuation. Set False for tenancies with Zero Data "
            "Retention (ZDR) enabled — the server rejects "
            "``previous_response_id`` otherwise."
        ),
    )
    oci_responses_request_timeout: float = Field(
        default=600.0,
        description="HTTP request timeout in seconds for the Responses transport.",
    )

    # ----- Reasoning / response_format passthrough --------------------------
    reasoning_effort: str | None = Field(
        default=None,
        description='``"low"`` | ``"medium"`` | ``"high"`` | ``"disable"``.',
    )


class LiteLLMModel(BaseModel):
    """Native Locus provider backed by `litellm.acompletion()`.

    Supports any model LiteLLM can route to. Designed first-class for the
    Oracle Generative AI Infrastructure catalog: pass ``oci/...`` models and
    LiteLLM handles RSA-SHA256 signing, vendor adapters, and the full OCI
    auth chain.
    """

    config: LiteLLMConfig

    # Cached LiteLLM-level kwargs computed once at init time.
    _completion_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    # Backend instance for the non-native transports (oci_openai_v1 or
    # oci_responses) — lazily constructed at __init__ time.
    _oci_backend: Any = PrivateAttr(default=None)

    def __init__(
        self,
        model: str = "oci/xai.grok-4.20",
        *,
        transport: LiteLLMTransport = "oci_native",
        api_key: str | None = None,
        api_base: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        # OCI native transport shortcuts.
        oci_region: str | None = None,
        oci_user: str | None = None,
        oci_fingerprint: str | None = None,
        oci_tenancy: str | None = None,
        oci_key: str | None = None,
        oci_key_file: str | None = None,
        oci_compartment_id: str | None = None,
        oci_serving_mode: str | None = None,
        oci_endpoint_id: str | None = None,
        oci_signer: _OciSigner = None,
        # OCI OpenAI-V1 + OCI Responses transport shortcuts.
        oci_profile: str | None = None,
        oci_auth_type: str | None = None,
        oci_config_file: str = "~/.oci/config",
        oci_base_url: str | None = None,
        oci_project_ocid: str | None = None,
        oci_responses_store: bool = True,
        oci_responses_request_timeout: float = 600.0,
        **kwargs: Any,
    ) -> None:
        config = LiteLLMConfig(
            model=model,
            transport=transport,
            api_key=api_key,
            api_base=api_base,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            oci_region=oci_region,
            oci_user=oci_user,
            oci_fingerprint=oci_fingerprint,
            oci_tenancy=oci_tenancy,
            oci_key=oci_key,
            oci_key_file=oci_key_file,
            oci_compartment_id=oci_compartment_id,
            oci_serving_mode=oci_serving_mode,
            oci_endpoint_id=oci_endpoint_id,
            oci_signer=oci_signer,
            oci_profile=oci_profile,
            oci_auth_type=oci_auth_type,
            oci_config_file=oci_config_file,
            oci_base_url=oci_base_url,
            oci_project_ocid=oci_project_ocid,
            oci_responses_store=oci_responses_store,
            oci_responses_request_timeout=oci_responses_request_timeout,
            **kwargs,
        )
        super().__init__(config=config)

        if config.transport == "oci_native":
            self._completion_kwargs = self._build_completion_kwargs()
        elif config.transport == "oci_openai_v1":
            self._completion_kwargs = {}
            self._oci_backend = self._build_v1_backend()
        elif config.transport == "oci_responses":
            self._completion_kwargs = {}
            self._oci_backend = self._build_responses_backend()
        else:
            raise ValueError(
                f"Unknown LiteLLMModel transport: {config.transport!r}. "
                "Expected one of: 'oci_native', 'oci_openai_v1', 'oci_responses'."
            )

    # ------------------------------------------------------------------ V1 backend

    def _build_v1_backend(self) -> Any:
        """Construct the OCIChatCompletionsModel that backs the V1 transport."""
        from locus.models.providers.oci import OCIChatCompletionsModel

        c = self.config
        # OCIChatCompletionsModel expects model id WITHOUT the ``oci/`` prefix.
        bare_model = c.model.removeprefix("oci/")

        kwargs: dict[str, Any] = {
            "model": bare_model,
            "max_tokens": c.max_tokens,
            "temperature": c.temperature,
            "config_file": c.oci_config_file,
        }
        if c.oci_profile is not None:
            kwargs["profile"] = c.oci_profile
        if c.oci_auth_type is not None:
            kwargs["auth_type"] = c.oci_auth_type
        if c.oci_compartment_id is not None:
            kwargs["compartment_id"] = c.oci_compartment_id
        if c.oci_region is not None:
            kwargs["region"] = c.oci_region
        if c.oci_base_url is not None:
            kwargs["base_url"] = c.oci_base_url

        return OCIChatCompletionsModel(**kwargs)

    def _build_responses_backend(self) -> Any:
        """Construct the OCIResponsesModel that backs the Responses transport."""
        from locus.models.providers.oci import OCIResponsesModel

        c = self.config
        # OCIResponsesModel expects model id WITHOUT the ``oci/`` prefix.
        bare_model = c.model.removeprefix("oci/")

        kwargs: dict[str, Any] = {
            "model": bare_model,
            "max_output_tokens": c.max_tokens,
            "temperature": c.temperature,
            "config_file": c.oci_config_file,
            "store": c.oci_responses_store,
            "request_timeout": c.oci_responses_request_timeout,
        }
        if c.oci_profile is not None:
            kwargs["profile"] = c.oci_profile
        if c.oci_auth_type is not None:
            kwargs["auth_type"] = c.oci_auth_type
        if c.oci_compartment_id is not None:
            kwargs["compartment_id"] = c.oci_compartment_id
        if c.oci_project_ocid is not None:
            kwargs["project_ocid"] = c.oci_project_ocid
        if c.oci_region is not None:
            kwargs["region"] = c.oci_region
        if c.oci_base_url is not None:
            kwargs["base_url"] = c.oci_base_url

        return OCIResponsesModel(**kwargs)

    # ------------------------------------------------------------------ helpers

    def _build_completion_kwargs(self) -> dict[str, Any]:
        """Compute the kwargs that get forwarded to ``litellm.acompletion`` on every call."""
        c = self.config
        kw: dict[str, Any] = {
            "model": c.model,
            "drop_params": c.drop_params,
            "num_retries": c.max_retries,
        }
        if c.api_key is not None:
            kw["api_key"] = c.api_key
        if c.api_base is not None:
            kw["api_base"] = c.api_base
        if c.request_timeout is not None:
            kw["timeout"] = c.request_timeout

        # OCI passthrough — only set keys that were explicitly supplied.
        # When unset, the LiteLLM OCI provider reads OCI_* env vars itself.
        for field in (
            "oci_region",
            "oci_user",
            "oci_fingerprint",
            "oci_tenancy",
            "oci_key",
            "oci_key_file",
            "oci_compartment_id",
            "oci_serving_mode",
            "oci_endpoint_id",
            "oci_signer",
        ):
            v = getattr(c, field)
            if v is not None:
                kw[field] = v

        return kw

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Locus `Message` list → OpenAI-shape dicts (LiteLLM's wire format)."""
        return [m.to_openai_format() for m in messages]

    @staticmethod
    def _coerce_tool_arguments(raw: Any) -> dict[str, Any]:
        """LiteLLM returns tool-call arguments as a JSON string; parse to dict for Locus.

        Falls back to an empty dict on malformed JSON rather than crashing the
        agent loop — the model can recover on the next turn.
        """
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}

    # ------------------------------------------------------------------ async API

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Run a non-streaming completion via the configured transport.

        ``tools`` is passed through unchanged — both LiteLLM (native path) and
        OCIChatCompletionsModel (V1 OpenAI-compat path) accept OpenAI-shape
        tool definitions.
        """
        if self._oci_backend is not None:
            # OCI OpenAI-V1 or OCI Responses transport: delegate.
            resp: ModelResponse = await self._oci_backend.complete(
                messages=messages, tools=tools, **kwargs
            )
            return resp

        import litellm  # type: ignore[import-not-found,unused-ignore]  # lazy: locus without litellm still imports

        # Per-call overrides take precedence over config defaults.
        params: dict[str, Any] = {
            **self._completion_kwargs,
            "messages": self._convert_messages(messages),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        # Skip ``temperature`` for models that fix it internally — sending
        # any non-default value returns a 400.
        if not _model_fixes_temperature(self.config.model):
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        if tools:
            params["tools"] = tools
            if "tool_choice" in kwargs:
                params["tool_choice"] = kwargs["tool_choice"]

        effort = kwargs.get("reasoning_effort", self.config.reasoning_effort)
        if effort is not None:
            params["reasoning_effort"] = effort

        # Pass through any other OpenAI-shape kwarg the caller supplied.
        for k in (
            "response_format",
            "top_p",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "user",
        ):
            if k in kwargs:
                params[k] = kwargs[k]

        response = await litellm.acompletion(**params)

        choice = response.choices[0]
        message = choice.message

        # Extract content + tool calls
        content: str | None = getattr(message, "content", None) or None
        tool_calls: list[ToolCall] = []
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        for tc in raw_tool_calls:
            fn = tc.function
            tool_calls.append(
                ToolCall(
                    id=tc.id or f"call_{len(tool_calls)}",
                    name=fn.name,
                    arguments=self._coerce_tool_arguments(fn.arguments),
                )
            )

        # Usage — including reasoning_tokens when the provider surfaces them.
        usage: dict[str, int] = {}
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            pt = getattr(raw_usage, "prompt_tokens", None)
            ct = getattr(raw_usage, "completion_tokens", None)
            tt = getattr(raw_usage, "total_tokens", None)
            if pt is not None:
                usage["prompt_tokens"] = int(pt)
            if ct is not None:
                usage["completion_tokens"] = int(ct)
            if tt is not None:
                usage["total_tokens"] = int(tt)
            details = getattr(raw_usage, "completion_tokens_details", None)
            rt = getattr(details, "reasoning_tokens", None) if details is not None else None
            if rt is not None:
                usage["reasoning_tokens"] = int(rt)

        return ModelResponse(
            message=Message.assistant(content=content, tool_calls=tool_calls),
            usage=usage,
            stop_reason=choice.finish_reason,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        """Stream a completion as `ModelChunkEvent`s.

        Under ``transport="oci_native"`` LiteLLM yields OpenAI-shape
        ``ModelResponseStream`` chunks. Under ``transport="oci_openai_v1"``
        the underlying ``OCIChatCompletionsModel.stream()`` is delegated to.
        In both cases each chunk is converted to a `ModelChunkEvent` with
        content and/or progressive tool-call deltas, and a terminal
        ``ModelChunkEvent(done=True)`` is always emitted last.
        """
        if self._oci_backend is not None:
            async for ev in self._oci_backend.stream(messages=messages, tools=tools, **kwargs):
                yield ev
            return

        import litellm  # type: ignore[import-not-found,unused-ignore]

        params: dict[str, Any] = {
            **self._completion_kwargs,
            "messages": self._convert_messages(messages),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }
        if not _model_fixes_temperature(self.config.model):
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        if tools:
            params["tools"] = tools
            if "tool_choice" in kwargs:
                params["tool_choice"] = kwargs["tool_choice"]

        effort = kwargs.get("reasoning_effort", self.config.reasoning_effort)
        if effort is not None:
            params["reasoning_effort"] = effort

        # Accumulate tool-call deltas across chunks (LiteLLM emits partial
        # tool-call fragments in OpenAI streaming format).
        tool_call_buffers: dict[int, dict[str, Any]] = {}

        stream = await litellm.acompletion(**params)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            text = getattr(delta, "content", None)
            if text:
                yield ModelChunkEvent(content=text)

            delta_tcs = getattr(delta, "tool_calls", None) or []
            for delta_tc in delta_tcs:
                idx = delta_tc.index if delta_tc.index is not None else 0
                buf = tool_call_buffers.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                if delta_tc.id:
                    buf["id"] = delta_tc.id
                fn = getattr(delta_tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        buf["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        buf["arguments"] += fn.arguments

        # Emit assembled tool calls (if any) on the terminal chunk.
        completed_tool_calls: list[ToolCall] = []
        for idx in sorted(tool_call_buffers.keys()):
            buf = tool_call_buffers[idx]
            if not buf.get("name"):
                continue
            completed_tool_calls.append(
                ToolCall(
                    id=buf["id"] or f"call_{idx}",
                    name=buf["name"],
                    arguments=self._coerce_tool_arguments(buf["arguments"]),
                )
            )

        yield ModelChunkEvent(
            done=True,
            tool_calls=completed_tool_calls or None,
        )

    async def close(self) -> None:
        """Close any backing clients.

        Under ``transport="oci_native"`` this is a no-op because LiteLLM owns
        its own httpx client lifecycle. Under ``transport="oci_openai_v1"``
        the underlying ``OCIChatCompletionsModel``'s httpx client is closed
        so the loop-bound connections shut down inside the same event loop
        that opened them.
        """
        if self._oci_backend is not None and hasattr(self._oci_backend, "close"):
            await self._oci_backend.close()

    async def __aenter__(self) -> LiteLLMModel:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
