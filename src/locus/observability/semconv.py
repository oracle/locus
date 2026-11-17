# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OpenTelemetry GenAI semantic conventions, as the telemetry hook emits them.

The names here follow the OpenTelemetry GenAI semantic conventions
(``open-telemetry/semantic-conventions-genai``, Development status, pinned
alongside semantic-conventions 1.45): an agent run is an ``invoke_agent``
span, every model call a ``chat`` span, every tool call an ``execute_tool``
span, with ``gen_ai.*`` attributes and the inference token counters. The
conventions are still moving, so the hook takes a version switch —
:data:`SEMCONV_GENAI` (the default) or :data:`SEMCONV_LEGACY` for the
``agent.invocation`` / ``tool.<name>`` / ``locus.*`` shape the SDK emitted
before — and this module is the one place the names live.

:func:`validate_spans` checks a finished trace against the conventions, so a
recorded trace can be asserted in a test and a backend's GenAI view can be
trusted to find what it expects.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit


# --- Convention switch -------------------------------------------------------

#: The GenAI semantic conventions, as published next to semantic-conventions
#: 1.45 (Development status). The default for ``TelemetryHook``.
SEMCONV_GENAI = "genai/1.45"
#: The span and metric names the SDK emitted before the conventions were
#: adopted: ``agent.invocation``, ``agent.iteration.<n>``, ``tool.<name>``
#: and the ``locus.*`` counters. No model-call spans.
SEMCONV_LEGACY = "locus-legacy"
SEMCONV_DEFAULT = SEMCONV_GENAI

_SEMCONV_ALIASES: dict[str, str] = {
    "genai": SEMCONV_GENAI,
    "genai/latest": SEMCONV_GENAI,
    "legacy": SEMCONV_LEGACY,
}
SUPPORTED_SEMCONV: tuple[str, ...] = (SEMCONV_GENAI, SEMCONV_LEGACY)


def resolve_semconv(value: str | None) -> str:
    """Normalise a ``semconv=`` argument to one of :data:`SUPPORTED_SEMCONV`.

    Raises:
        ValueError: An unknown convention name, listing the supported ones.
    """
    if value is None:
        return SEMCONV_DEFAULT
    key = value.strip().lower()
    key = _SEMCONV_ALIASES.get(key, key)
    if key not in SUPPORTED_SEMCONV:
        supported = ", ".join(repr(s) for s in SUPPORTED_SEMCONV)
        raise ValueError(f"unknown semconv {value!r}; supported: {supported}")
    return key


# --- Attribute names -----------------------------------------------------------

GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_AGENT_ID = "gen_ai.agent.id"
GEN_AI_AGENT_DESCRIPTION = "gen_ai.agent.description"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS = "gen_ai.usage.cache_write.input_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
GEN_AI_TOKEN_MODALITY = "gen_ai.token.modality"  # noqa: S105 — attribute name, not a secret
ERROR_TYPE = "error.type"
SERVER_ADDRESS = "server.address"
SERVER_PORT = "server.port"

#: Every ``gen_ai.*`` attribute the hook may set. The validator flags any
#: other ``gen_ai.`` key as a misspelling.
GEN_AI_ATTRIBUTES: frozenset[str] = frozenset(
    {
        GEN_AI_OPERATION_NAME,
        GEN_AI_PROVIDER_NAME,
        GEN_AI_AGENT_NAME,
        GEN_AI_AGENT_ID,
        GEN_AI_AGENT_DESCRIPTION,
        GEN_AI_CONVERSATION_ID,
        GEN_AI_REQUEST_MODEL,
        GEN_AI_REQUEST_MAX_TOKENS,
        GEN_AI_REQUEST_TEMPERATURE,
        GEN_AI_REQUEST_TOP_P,
        GEN_AI_RESPONSE_ID,
        GEN_AI_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS,
        GEN_AI_USAGE_INPUT_TOKENS,
        GEN_AI_USAGE_OUTPUT_TOKENS,
        GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
        GEN_AI_TOOL_NAME,
        GEN_AI_TOOL_CALL_ID,
        GEN_AI_TOOL_TYPE,
        GEN_AI_TOOL_DESCRIPTION,
        GEN_AI_TOOL_CALL_ARGUMENTS,
        GEN_AI_TOOL_CALL_RESULT,
        GEN_AI_TOKEN_MODALITY,
    }
)

# --- Operation names and well-known values -------------------------------------

OP_INVOKE_AGENT = "invoke_agent"
OP_CHAT = "chat"
OP_EXECUTE_TOOL = "execute_tool"
OPERATIONS: frozenset[str] = frozenset({OP_INVOKE_AGENT, OP_CHAT, OP_EXECUTE_TOOL})

TOOL_TYPE_FUNCTION = "function"
#: ``gen_ai.token.modality`` values. The conventions require ``unknown`` when
#: the modality cannot be determined; a provider's usage names the others.
TOKEN_MODALITY_TEXT = "text"  # noqa: S105 — attribute value, not a secret
TOKEN_MODALITY_AUDIO = "audio"  # noqa: S105 — attribute value, not a secret
TOKEN_MODALITY_IMAGE = "image"  # noqa: S105 — attribute value, not a secret
TOKEN_MODALITY_VIDEO = "video"  # noqa: S105 — attribute value, not a secret
TOKEN_MODALITY_UNKNOWN = "unknown"  # noqa: S105 — attribute value, not a secret
#: ``error.type`` when nothing more specific is known.
ERROR_TYPE_OTHER = "_OTHER"

#: ``gen_ai.provider.name`` for OCI Generative AI. The conventions let an
#: instrumentation define its own value when no well-known one applies; this
#: one follows the ``<cloud>.<service>`` shape of the well-known values.
PROVIDER_OCI_GENERATIVE_AI = "oci.generative_ai"

#: Locus model-string prefixes mapped to ``gen_ai.provider.name`` values.
#: Well-known values where the conventions define one; a prefix that names
#: a provider the conventions do not list is used as it is.
_PROVIDER_BY_PREFIX: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "oci": PROVIDER_OCI_GENERATIVE_AI,
    "cohere": "cohere",
    "deepseek": "deepseek",
    "gemini": "gcp.gemini",
    "groq": "groq",
    "mistral": "mistral",
    "ollama": "ollama",
    "perplexity": "perplexity",
    "xai": "x_ai",
}

#: The span name under which an iteration of the ReAct loop is filed in the
#: GenAI conventions. The conventions have no iteration concept; this span
#: is the SDK's own, kept low-cardinality (the index is an attribute).
ITERATION_SPAN_NAME = "locus.iteration"

#: The instrumentation scope ``TelemetryHook`` records its spans and metrics under.
TELEMETRY_SCOPE = "locus.hooks.telemetry"

# --- Metric names ----------------------------------------------------------------

METRIC_INFERENCE_DURATION = "gen_ai.client.inference.duration"
METRIC_INPUT_TOKENS = "gen_ai.client.inference.usage.input_tokens"
METRIC_OUTPUT_TOKENS = "gen_ai.client.inference.usage.output_tokens"
METRIC_INVOKE_AGENT_DURATION = "gen_ai.invoke_agent.duration"
METRIC_INVOKE_AGENT_INFERENCE_CALLS = "gen_ai.invoke_agent.inference_calls"
METRIC_INVOKE_AGENT_TOOL_CALLS = "gen_ai.invoke_agent.tool_calls"
METRIC_EXECUTE_TOOL_DURATION = "gen_ai.execute_tool.duration"

#: Explicit bucket boundaries the conventions recommend, in seconds.
DURATION_BUCKETS_S: tuple[float, ...] = (
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
)  # fmt: skip
INVOKE_AGENT_DURATION_BUCKETS_S: tuple[float, ...] = (
    0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8, 409.6,
    900, 1800, 3600, 7200, 14400, 43200, 86400,
)  # fmt: skip
CALL_COUNT_BUCKETS: tuple[float, ...] = (1, 2, 4, 8, 16, 32, 64, 128)


# --- Model identity --------------------------------------------------------------


def request_model(model: Any) -> str | None:
    """The ``gen_ai.request.model`` value for a model string or model object.

    A ``"provider:model"`` string yields its model half; an object yields its
    ``config.model`` (every SDK model class), else a ``model_id`` or ``model``
    string attribute. ``None`` when nothing fits.
    """
    if model is None:
        return None
    if isinstance(model, str):
        _, _, name = model.partition(":")
        return name or model
    config = getattr(model, "config", None)
    configured = getattr(config, "model", None)
    if isinstance(configured, str) and configured:
        return configured
    for attr in ("model_id", "model", "model_name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def provider_name(model: Any) -> str | None:
    """The ``gen_ai.provider.name`` value for a model string or model object.

    Strings map by prefix (``oci:`` → ``oci.generative_ai``, ``xai:`` →
    ``x_ai``, a prefix the conventions do not list as it is). Objects map by
    class: the OCI model classes, ``AnthropicModel`` and ``OpenAIModel``. An
    ``OpenAIModel`` behind a ``base_url`` the SDK's provider table knows
    (Groq, DeepSeek, Mistral, …) maps to that provider's value; any other
    OpenAI-compatible endpoint is ``openai`` — the wire flavour — with
    ``server.address`` telling endpoints apart. Pass ``provider_name=`` to the
    hook to override. ``None`` for a model class the SDK cannot place.
    """
    if model is None:
        return None
    if isinstance(model, str):
        prefix, sep, _ = model.partition(":")
        if not sep:
            return None
        return _PROVIDER_BY_PREFIX.get(prefix, prefix)
    names = {cls.__name__ for cls in type(model).__mro__}
    if names & {"OCIModel", "OCIChatCompletionsModel", "OCIResponsesModel"}:
        return PROVIDER_OCI_GENERATIVE_AI
    if "AnthropicModel" in names:
        return "anthropic"
    if "OpenAIModel" in names:
        base_url = getattr(getattr(model, "config", None), "base_url", None)
        if not base_url:
            return "openai"
        return _provider_for_host(str(base_url)) or "openai"
    return None


def _provider_for_host(base_url: str) -> str | None:
    """The provider behind an OpenAI-compatible ``base_url``, when the SDK knows the host."""
    parts = urlsplit(base_url)
    host = parts.hostname
    if not host:
        return None
    if host == "api.openai.com" or host.endswith(".openai.com"):
        return "openai"
    try:
        from locus.models.providers import COMPATIBLE_PROVIDERS  # noqa: PLC0415
    except ImportError:  # pragma: no cover - the table ships with the SDK
        return None
    # Host and port together: several local servers default to ``localhost``
    # on different ports, and the host alone would name the wrong one.
    netloc = parts.netloc.lower()
    for spec in COMPATIBLE_PROVIDERS:
        default = getattr(spec, "base_url", None)
        if default and urlsplit(default).netloc.lower() == netloc:
            prefix = str(spec.prefix)
            return _PROVIDER_BY_PREFIX.get(prefix, prefix)
    return None


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def fallback_provider_name(model: Any) -> str | None:
    """A last-resort ``gen_ai.provider.name``: the model object's class, in snake case.

    ``ScriptedModel`` becomes ``scripted_model``. Keeps the attribute the
    conventions require on every ``chat`` span present for custom model
    classes the SDK cannot place; ``None`` for strings and ``None``.
    """
    if model is None or isinstance(model, str):
        return None
    return _CAMEL.sub("_", type(model).__name__).lower()


def server_address(model: Any) -> tuple[str | None, int | None]:
    """``server.address`` / ``server.port`` for a model object with a ``base_url``."""
    base_url = getattr(getattr(model, "config", None), "base_url", None)
    if not isinstance(base_url, str) or not base_url:
        return None, None
    parts = urlsplit(base_url)
    if not parts.hostname:
        return None, None
    return parts.hostname, parts.port


_ERROR_TYPE = re.compile(r"^\s*([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Timeout))\b")


def error_type(error: str | BaseException | None) -> str:
    """The ``error.type`` value for a tool or model failure.

    An exception yields its class name; a string that starts with one
    (``"ValueError: bad id"``) yields that name; anything else is the
    conventions' fallback ``_OTHER``.
    """
    if isinstance(error, BaseException):
        return type(error).__name__
    if isinstance(error, str):
        match = _ERROR_TYPE.match(error)
        if match:
            return match.group(1).rsplit(".", 1)[-1]
    return ERROR_TYPE_OTHER


# --- Trace validation --------------------------------------------------------------

_SPAN_KIND_NAMES = {"INTERNAL", "CLIENT", "SERVER", "PRODUCER", "CONSUMER"}
_EXPECTED_KIND = {OP_INVOKE_AGENT: "INTERNAL", OP_CHAT: "CLIENT", OP_EXECUTE_TOOL: "INTERNAL"}


@dataclass
class SpanRecord:
    """The part of a finished span the validator and fixtures work from.

    ``scope`` is the instrumentation scope that produced the span (the
    hook's is :data:`TELEMETRY_SCOPE`); ``None`` when the source did not say.
    """

    name: str
    kind: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    status: str
    attributes: dict[str, Any] = field(default_factory=dict)
    status_description: str | None = None
    scope: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "status": self.status,
            "status_description": self.status_description,
            "attributes": dict(self.attributes),
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SpanRecord:
        return cls(
            name=str(data["name"]),
            kind=str(data.get("kind", "INTERNAL")),
            trace_id=str(data["trace_id"]),
            span_id=str(data["span_id"]),
            parent_span_id=(
                str(data["parent_span_id"]) if data.get("parent_span_id") is not None else None
            ),
            status=str(data.get("status", "UNSET")),
            status_description=data.get("status_description"),
            attributes=dict(data.get("attributes") or {}),
            scope=str(data["scope"]) if data.get("scope") is not None else None,
        )

    @classmethod
    def from_readable_span(cls, span: Any) -> SpanRecord:
        """Build a record from an OpenTelemetry SDK ``ReadableSpan``."""
        ctx = span.get_span_context()
        parent = getattr(span, "parent", None)
        kind = getattr(span, "kind", None)
        status = getattr(span, "status", None)
        status_code = getattr(status, "status_code", None)
        attributes = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dict(span.attributes or {}).items()
        }
        scope = getattr(getattr(span, "instrumentation_scope", None), "name", None)
        return cls(
            name=span.name,
            kind=getattr(kind, "name", str(kind)),
            trace_id=f"{ctx.trace_id:032x}",
            span_id=f"{ctx.span_id:016x}",
            parent_span_id=f"{parent.span_id:016x}" if parent is not None else None,
            status=getattr(status_code, "name", "UNSET"),
            status_description=getattr(status, "description", None),
            attributes=attributes,
            scope=str(scope) if scope is not None else None,
        )


def records_from_spans(spans: Iterable[Any]) -> list[SpanRecord]:
    """Convert SDK ``ReadableSpan`` objects (or dicts) to :class:`SpanRecord`."""
    out: list[SpanRecord] = []
    for span in spans:
        if isinstance(span, SpanRecord):
            out.append(span)
        elif isinstance(span, Mapping):
            out.append(SpanRecord.from_dict(span))
        else:
            out.append(SpanRecord.from_readable_span(span))
    return out


def dump_spans(spans: Iterable[Any]) -> str:
    """Serialise spans as the JSON a recorded-trace fixture holds."""
    return json.dumps([r.to_dict() for r in records_from_spans(spans)], indent=2, sort_keys=True)


def load_spans(text: str) -> list[SpanRecord]:
    """Read spans back from :func:`dump_spans` output."""
    return [SpanRecord.from_dict(d) for d in json.loads(text)]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_spans(
    spans: Iterable[Any],
    *,
    require_usage: bool = True,
    scope: str | None = TELEMETRY_SCOPE,
) -> list[str]:
    """Check a finished trace against the GenAI semantic conventions.

    Returns the problems found, one line each; an empty list means the trace
    conforms. Checked: one trace id; one root (an ``invoke_agent`` span, or
    one whose parent lies outside the recorded set when the run inherited a
    caller's trace); every non-iteration span carries a known
    ``gen_ai.operation.name`` with the kind and name the conventions give it;
    the required attributes of each operation are present and well typed;
    ``error.type`` is set on every span that ended in error; and every
    ``gen_ai.*`` key is one the conventions define.

    Args:
        spans: SDK ``ReadableSpan`` objects, :class:`SpanRecord` or dicts.
        require_usage: Insist on ``gen_ai.usage.input_tokens`` /
            ``output_tokens`` on ``chat`` and ``invoke_agent`` spans. The
            conventions only recommend them, but the hook sets them whenever
            the provider reports usage, so a trace from a real model has them.
        scope: When the spans come from more than one instrumentation scope,
            only those of this one are checked — by default the telemetry
            hook's own, so the spans a web framework or the ``mcp`` package
            wrote into the same exporter (``POST /``, ``MCP send tools/call
            …``) are not counted as foreign roots or extra traces. Spans from
            a single scope are checked as they are, whatever it is called
            (a hook with its own ``tracer_name``), and spans whose scope is
            unknown (records loaded from a fixture) are always kept. ``None``
            checks everything given.
    """
    records = records_from_spans(spans)
    filtered = False
    if scope is not None and len({r.scope for r in records if r.scope is not None}) > 1:
        records = [r for r in records if r.scope is None or r.scope == scope]
        filtered = True
    problems: list[str] = []
    if not records:
        return ["no spans recorded" + (f" in scope {scope!r}" if filtered else "")]

    trace_ids = {r.trace_id for r in records}
    if len(trace_ids) != 1:
        problems.append(f"spans belong to {len(trace_ids)} traces, expected 1")
    ids = {r.span_id for r in records}
    roots = [r for r in records if r.parent_span_id is None or r.parent_span_id not in ids]
    if len(roots) != 1:
        problems.append(f"{len(roots)} root spans, expected 1: {[r.name for r in roots]}")
    for root in roots:
        if root.attributes.get(GEN_AI_OPERATION_NAME) != OP_INVOKE_AGENT:
            problems.append(f"root span {root.name!r} is not an invoke_agent span")

    for r in records:
        where = f"span {r.name!r}"
        for key in r.attributes:
            if key.startswith("gen_ai.") and key not in GEN_AI_ATTRIBUTES:
                problems.append(f"{where}: unknown attribute {key!r}")
        if r.kind not in _SPAN_KIND_NAMES:
            problems.append(f"{where}: unknown span kind {r.kind!r}")
        if r.status == "ERROR" and ERROR_TYPE not in r.attributes:
            problems.append(f"{where}: status ERROR without {ERROR_TYPE}")

        operation = r.attributes.get(GEN_AI_OPERATION_NAME)
        if operation is None:
            if r.name != ITERATION_SPAN_NAME:
                problems.append(f"{where}: missing {GEN_AI_OPERATION_NAME}")
            elif r.parent_span_id is None:
                problems.append(f"{where}: iteration span has no parent")
            continue
        if operation not in OPERATIONS:
            problems.append(f"{where}: unknown operation {operation!r}")
            continue
        expected_kind = _EXPECTED_KIND[operation]
        if r.kind != expected_kind:
            problems.append(f"{where}: kind {r.kind}, {operation} spans are {expected_kind}")

        if operation == OP_INVOKE_AGENT:
            agent = r.attributes.get(GEN_AI_AGENT_NAME)
            expected = f"{OP_INVOKE_AGENT} {agent}" if agent else OP_INVOKE_AGENT
            if r.name != expected:
                problems.append(f"{where}: expected name {expected!r}")
            if require_usage:
                _check_usage(r, where, problems)
        elif operation == OP_CHAT:
            if GEN_AI_PROVIDER_NAME not in r.attributes:
                problems.append(f"{where}: missing {GEN_AI_PROVIDER_NAME}")
            model = r.attributes.get(GEN_AI_REQUEST_MODEL)
            expected = f"{OP_CHAT} {model}" if model else OP_CHAT
            if r.name != expected:
                problems.append(f"{where}: expected name {expected!r}")
            reasons = r.attributes.get(GEN_AI_RESPONSE_FINISH_REASONS)
            if reasons is not None and not (
                isinstance(reasons, list) and all(isinstance(x, str) for x in reasons)
            ):
                problems.append(f"{where}: {GEN_AI_RESPONSE_FINISH_REASONS} must be a string list")
            if require_usage and r.status != "ERROR":
                _check_usage(r, where, problems)
        elif operation == OP_EXECUTE_TOOL:
            tool = r.attributes.get(GEN_AI_TOOL_NAME)
            if not isinstance(tool, str) or not tool:
                problems.append(f"{where}: missing {GEN_AI_TOOL_NAME}")
            elif r.name != f"{OP_EXECUTE_TOOL} {tool}":
                problems.append(f"{where}: expected name {f'{OP_EXECUTE_TOOL} {tool}'!r}")
            call_id = r.attributes.get(GEN_AI_TOOL_CALL_ID)
            if call_id is not None and not isinstance(call_id, str):
                problems.append(f"{where}: {GEN_AI_TOOL_CALL_ID} must be a string")
    return problems


def _check_usage(r: SpanRecord, where: str, problems: list[str]) -> None:
    for key in (GEN_AI_USAGE_INPUT_TOKENS, GEN_AI_USAGE_OUTPUT_TOKENS):
        value = r.attributes.get(key)
        if value is None:
            problems.append(f"{where}: missing {key}")
        elif not _is_int(value) or value < 0:
            problems.append(f"{where}: {key} must be a non-negative integer")


def assert_valid_spans(
    spans: Iterable[Any],
    *,
    require_usage: bool = True,
    scope: str | None = TELEMETRY_SCOPE,
) -> None:
    """Raise ``AssertionError`` listing every problem :func:`validate_spans` finds."""
    problems = validate_spans(spans, require_usage=require_usage, scope=scope)
    if problems:
        raise AssertionError(
            "trace does not follow the GenAI conventions:\n  " + "\n  ".join(problems)
        )


__all__ = [
    "assert_valid_spans",
    "CALL_COUNT_BUCKETS",
    "dump_spans",
    "DURATION_BUCKETS_S",
    "ERROR_TYPE",
    "error_type",
    "ERROR_TYPE_OTHER",
    "fallback_provider_name",
    "GEN_AI_AGENT_DESCRIPTION",
    "GEN_AI_AGENT_ID",
    "GEN_AI_AGENT_NAME",
    "GEN_AI_ATTRIBUTES",
    "GEN_AI_CONVERSATION_ID",
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_PROVIDER_NAME",
    "GEN_AI_REQUEST_MAX_TOKENS",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_REQUEST_TEMPERATURE",
    "GEN_AI_REQUEST_TOP_P",
    "GEN_AI_RESPONSE_FINISH_REASONS",
    "GEN_AI_RESPONSE_ID",
    "GEN_AI_RESPONSE_MODEL",
    "GEN_AI_TOKEN_MODALITY",
    "GEN_AI_TOOL_CALL_ARGUMENTS",
    "GEN_AI_TOOL_CALL_ID",
    "GEN_AI_TOOL_CALL_RESULT",
    "GEN_AI_TOOL_DESCRIPTION",
    "GEN_AI_TOOL_NAME",
    "GEN_AI_TOOL_TYPE",
    "GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS",
    "GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    "INVOKE_AGENT_DURATION_BUCKETS_S",
    "ITERATION_SPAN_NAME",
    "load_spans",
    "METRIC_EXECUTE_TOOL_DURATION",
    "METRIC_INFERENCE_DURATION",
    "METRIC_INPUT_TOKENS",
    "METRIC_INVOKE_AGENT_DURATION",
    "METRIC_INVOKE_AGENT_INFERENCE_CALLS",
    "METRIC_INVOKE_AGENT_TOOL_CALLS",
    "METRIC_OUTPUT_TOKENS",
    "OP_CHAT",
    "OP_EXECUTE_TOOL",
    "OP_INVOKE_AGENT",
    "OPERATIONS",
    "provider_name",
    "PROVIDER_OCI_GENERATIVE_AI",
    "records_from_spans",
    "request_model",
    "resolve_semconv",
    "SEMCONV_DEFAULT",
    "SEMCONV_GENAI",
    "SEMCONV_LEGACY",
    "SERVER_ADDRESS",
    "server_address",
    "SERVER_PORT",
    "SpanRecord",
    "SUPPORTED_SEMCONV",
    "TELEMETRY_SCOPE",
    "TOKEN_MODALITY_AUDIO",
    "TOKEN_MODALITY_IMAGE",
    "TOKEN_MODALITY_TEXT",
    "TOKEN_MODALITY_UNKNOWN",
    "TOKEN_MODALITY_VIDEO",
    "TOOL_TYPE_FUNCTION",
    "validate_spans",
]
