# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""
Shared configuration for Locus notebooks.

Locus is built first-class on **Oracle Cloud Infrastructure (OCI)
Generative AI**, so the notebooks default to OCI when an OCI profile
is reachable, fall back to a built-in mock model otherwise, and switch
to any other live provider — OpenAI, Anthropic — with one environment
variable. That means:

  - On a developer laptop with ``~/.oci/config`` present, every notebook
    runs against live OCI GenAI with zero extra setup.
  - On a clean machine (no OCI config, no env vars), notebooks still run
    end-to-end against the mock model so you can read the output and
    understand the shape before authenticating.
  - Set ``LOCUS_MODEL_PROVIDER`` + the matching credentials to force a
    specific provider.

See ``docs/how-to/oci-models.md`` for OCI specifics.

Environment Variables:
    LOCUS_MODEL_PROVIDER   - "oci" (default when OCI is reachable), "mock",
                              "openai", or "anthropic". When unset,
                              we auto-detect: OCI if ``~/.oci/config`` has at
                              least one usable profile, else mock.
    LOCUS_MODEL_ID         - Model identifier (provider-specific)

    # OCI GenAI — every LOCUS_OCI_* variable below falls back to its
    # OCI_* equivalent (the OCI CLI standard) when unset, so once you've
    # run ``oci session authenticate --profile-name X`` and exported the
    # resulting ``OCI_PROFILE``, every locus notebook just works.
    LOCUS_OCI_PROFILE      - OCI config profile name. Falls back to
                              OCI_PROFILE. When neither is set the harness
                              picks [DEFAULT], or the only named profile.
    LOCUS_OCI_AUTH_TYPE    - "api_key", "security_token",
                              "instance_principal", "resource_principal".
                              Falls back to OCI_AUTH_TYPE.
    LOCUS_OCI_REGION       - OCI region. Falls back to OCI_REGION.
    LOCUS_OCI_COMPARTMENT  - Compartment OCID. Falls back to
                              OCI_COMPARTMENT. Auto-derived from the
                              profile's tenancy when unset.
    LOCUS_OCI_ENDPOINT     - Service endpoint URL. Falls back to
                              OCI_ENDPOINT. Only honored by the SDK
                              transport (OCIModel).
    LOCUS_OCI_TRANSPORT    - "v1" or "sdk" — force a specific transport.
                              Falls back to OCI_TRANSPORT. By default
                              Cohere R-series and DAC endpoints → "sdk"
                              (OCIModel), everything else → "v1"
                              (OCIChatCompletionsModel).

    # OpenAI
    OPENAI_API_KEY         - OpenAI API key

    # Anthropic
    ANTHROPIC_API_KEY      - Anthropic API key

Examples:
    # Run with mock (default - no credentials needed):
    python examples/notebook_02_basic_agent.py

    # Run with OCI GenAI (V1 transport, OpenAI-compatible endpoint):
    export LOCUS_MODEL_PROVIDER=oci
    export LOCUS_MODEL_ID=openai.gpt-5.5
    export LOCUS_OCI_PROFILE=MY_PROFILE
    python examples/notebook_02_basic_agent.py

    # Run with OCI GenAI (SDK transport, required for Cohere R-series):
    export LOCUS_MODEL_PROVIDER=oci
    export LOCUS_MODEL_ID=cohere.command-r-plus
    export LOCUS_OCI_PROFILE=MY_PROFILE
    python examples/notebook_02_basic_agent.py

    # Run with OpenAI:
    export LOCUS_MODEL_PROVIDER=openai
    export OPENAI_API_KEY=sk-...
    python examples/notebook_02_basic_agent.py

    # Run with Anthropic:
    export LOCUS_MODEL_PROVIDER=anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/notebook_02_basic_agent.py

    # Pick a specific model for the selected provider:
    export LOCUS_MODEL_ID=gpt-4o-mini
    python examples/notebook_02_basic_agent.py
"""

import ast
import configparser
import operator
import os
import re
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from locus.core.events import ModelChunkEvent
from locus.core.messages import Message
from locus.models.base import ModelResponse


def _oci_env(name: str, default: str | None = None) -> str | None:
    """Read an OCI config setting from the env with a consistent fallback chain.

    Closes #218. Notebooks historically read ``LOCUS_OCI_*`` (the harness
    namespace) but the OCI CLI and most other tooling speak ``OCI_*``.
    Newcomers got bitten by having to maintain both. We now look up
    ``LOCUS_OCI_<name>`` first (keeps existing behaviour for users who
    already export the namespaced form), then fall back to the
    OCI-CLI-standard ``OCI_<name>``, then ``default``. This way a user
    who just typed ``oci session authenticate --profile-name DEFAULT``
    can run any notebook without re-exporting variables.
    """
    return os.environ.get(f"LOCUS_OCI_{name}") or os.environ.get(f"OCI_{name}") or default


class MockModel(BaseModel):
    """
    Mock model for testing notebooks without API calls.

    Returns predetermined responses for common prompts.
    """

    max_tokens: int = 100
    temperature: float = 0.7

    # Simulated responses, keyed on words that show up in the notebooks'
    # prompts. First match wins, so the specific keys come before the generic
    # ones — a payments chargeback and a phishing alert are both "triage", and
    # answering either with the other's verdict is how an offline run stops
    # reading like the real thing and starts reading like a broken example.
    _responses: dict[str, str] = {
        # Payments triage (notebook 02). Three distinct patterns, because a
        # queue where every transaction gets the same verdict teaches nothing.
        "declines": (
            "Consistent with card testing: a run of small declines from one BIN "
            "across unrelated merchants, then an authorization. Block the BIN "
            "and review the merchants it touched."
        ),
        "never disputed": (
            "Looks like an established recurring subscription — same amount, "
            "same merchant, nine months, no disputes. No action; leave it alone."
        ),
        "chargeback": (
            "Possible friendly fraud: delivery was signed for at the billing "
            "address and the account has prior undisputed orders there. Gather "
            "the delivery signature and order history, then represent."
        ),
        # Deployment readiness (notebook 03). Keyed on the go/no-go language
        # the release-engineer prompts use.
        "pre-deploy": (
            "No-go on search: the image is unsigned and only 2 of 5 replicas "
            "are healthy with the health check failing. Fix both before shipping."
        ),
        "deploy": (
            "Go on checkout: the image is signed and current, and all 6 replicas are healthy."
        ),
        "go/no-go": ("No-go: the readiness report lists blockers that have to clear first."),
        # Security triage (the SOC/IR notebooks).
        "default": "This is a mock response for testing purposes.",
        "triage": "Escalate: the indicators line up with an active phishing campaign.",
        "phishing": "Classic phishing markers (lookalike domain, urgent lure) — treat as malicious.",
        "alert": "Alert assessment: likely true positive. Recommend containment.",
        "severity": "Severity: HIGH — the exposure is reachable and exploitable.",
        "ioc": "Indicator enriched: multiple vendor detections, first seen 2 days ago.",
        "abstain": "Insufficient grounded evidence — abstaining rather than guessing.",
    }

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Return a mock response, calling a tool when one is bound.

        A mock that only ever returns prose makes every tool-centric example
        print ``Tool calls made: 0``, which is the opposite of what those
        notebooks exist to show. So on the first turn of a tool-bound run this
        emits a real ``ToolCall`` with arguments synthesised from the tool's
        own JSON schema; once a result comes back it answers in prose, which
        also terminates the loop.
        """
        if tools and not self._has_tool_result(messages):
            # Original case, not lowered: argument synthesis reads proper nouns
            # out of the prompt so the printed trace says "Paris", not "sample".
            call = self._synth_tool_call(self._last_user_text(messages), tools)
            if call is not None:
                return ModelResponse(
                    message=Message.assistant(content=None, tool_calls=[call]),
                    usage={"prompt_tokens": 10, "completion_tokens": 20},
                    stop_reason="tool_calls",
                )

        response = self._get_response(self._last_user_text(messages).lower(), tools)
        return ModelResponse(
            message=Message.assistant(content=response),
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            stop_reason="end_turn",
        )

    @staticmethod
    def _last_user_text(messages: list[Message]) -> str:
        """The most recent user turn, or the last message if there is none."""
        for msg in reversed(messages):
            if getattr(msg.role, "value", msg.role) == "user":
                return msg.content or ""
        return (messages[-1].content or "") if messages else ""

    @staticmethod
    def _has_tool_result(messages: list[Message]) -> bool:
        """Whether a tool has already answered in this run.

        Without this the mock would call the same tool every turn and only
        stop at the iteration cap.
        """
        return any(getattr(m.role, "value", m.role) == "tool" for m in messages)

    def _synth_tool_call(self, prompt: str, tools: list[dict[str, Any]]) -> Any:
        """Build a ToolCall for the tool that best matches the prompt.

        Preference goes to a tool whose name shares a word with the prompt —
        ``get_weather`` for "what's the weather in Tokyo" — so the chosen tool
        reads as a deliberate decision rather than a coin toss. Falls back to
        the first bound tool, because demonstrating the mechanism matters more
        than picking correctly.
        """
        from locus.core.messages import ToolCall  # local: keeps import cost off the top

        specs = [self._tool_spec(t) for t in tools]
        specs = [s for s in specs if s.get("name")]
        if not specs:
            return None

        words = {w for w in re.split(r"\W+", prompt.lower()) if len(w) > 2}
        best = next(
            (
                s
                for s in specs
                if words & {w for w in re.split(r"\W+", s["name"].lower()) if len(w) > 2}
            ),
            specs[0],
        )
        return ToolCall(name=best["name"], arguments=self._synth_arguments(best, prompt))

    @staticmethod
    def _tool_spec(tool: dict[str, Any]) -> dict[str, Any]:
        """Normalise the two shapes a tool payload arrives in.

        OpenAI nests under ``function``; Anthropic-style payloads are flat.
        """
        inner = tool.get("function")
        return inner if isinstance(inner, dict) else tool

    @staticmethod
    def _synth_arguments(spec: dict[str, Any], prompt: str) -> dict[str, Any]:
        """Fill the tool's required parameters with schema-valid values.

        Only required parameters are supplied, so defaults keep their meaning.
        String arguments reuse the prompt's last capitalised word when there is
        one — "Weather in Tokyo?" yields ``{"city": "Tokyo"}``, which makes the
        printed trace legible instead of full of placeholders.
        """
        schema = spec.get("parameters") or spec.get("input_schema") or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []

        proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", prompt)
        sample_text = proper_nouns[-1] if proper_nouns else "sample"

        defaults: dict[str, Any] = {
            "string": sample_text,
            "integer": 1,
            "number": 1.0,
            "boolean": True,
            "array": [],
            "object": {},
        }
        args: dict[str, Any] = {}
        for name in required:
            prop = props.get(name) or {}
            if enum := prop.get("enum"):
                args[name] = enum[0]
            else:
                args[name] = defaults.get(prop.get("type", "string"), sample_text)
        return args

    def _get_response(self, prompt: str, tools: list[dict[str, Any]] | None) -> str:
        """Get appropriate response based on prompt content."""
        for keyword, response in self._responses.items():
            if keyword in prompt:
                return response
        return self._responses["default"]

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        """Stream mock response in chunks."""
        response = await self.complete(messages, tools, **kwargs)
        content = response.content or ""

        # Yield in small chunks
        chunk_size = 10
        for i in range(0, len(content), chunk_size):
            yield ModelChunkEvent(content=content[i : i + chunk_size])
        yield ModelChunkEvent(done=True)


def resolved_provider() -> str:
    """The provider ``get_model()`` will actually use.

    ``LOCUS_MODEL_PROVIDER`` wins when set; otherwise the same auto-detection
    ``get_model()`` runs (OCI when a usable profile exists, else ``mock``).
    Notebooks that behave differently under the mock model must ask this,
    not read the environment variable, or they misreport a live OCI run.
    """
    return os.environ.get("LOCUS_MODEL_PROVIDER", "").lower() or _auto_detect_provider()


def check_structured_output_capable() -> None:
    """Exit cleanly with guidance if the current model cannot produce JSON.

    Guards against MockModel (plain text).
    """
    if resolved_provider() != "mock":
        return

    print(
        "\n⚠  This notebook requires structured-output (JSON schema) support.\n"
        "   MockModel returns plain text and cannot demonstrate these features.\n\n"
        "   Run with a model that supports constrained decoding, for example:\n\n"
        "     export LOCUS_MODEL_PROVIDER=oci LOCUS_MODEL_ID=openai.gpt-4o   # an OCI profile\n"
        "     export LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o      # OPENAI_API_KEY\n"
        f"     python {Path(sys.argv[0]).name}\n"
    )
    sys.exit(0)


def _oci_config_profiles(config_path: Path) -> list[str]:
    """Return the section names in an OCI config file.

    The OCI CLI puts the unnamed default section under ``[DEFAULT]``
    explicitly, which :mod:`configparser` does not surface in
    ``.sections()``. We re-add it manually so callers see the full picture.

    Returns an empty list when the file is missing, unreadable, or
    malformed — callers use this as a probe and rely on it never raising.
    """
    if not config_path.exists():
        return []
    parser = configparser.ConfigParser()
    try:
        parser.read(config_path)
    except (OSError, configparser.Error):
        return []
    sections = list(parser.sections())
    if parser.defaults():
        sections.insert(0, "DEFAULT")
    return sections


def _resolve_oci_profile() -> str:
    """Pick the OCI config profile name the harness should use.

    Resolution order:

    1. ``LOCUS_OCI_PROFILE`` / ``OCI_PROFILE`` env var if set — no
       validation, the user named it and the SDK will tell them if it
       doesn't exist.
    2. ``[DEFAULT]`` section in ``~/.oci/config`` → ``"DEFAULT"``.
    3. Exactly one non-default section in ``~/.oci/config`` → that section.
    4. Otherwise raise a clear error listing the sections found.

    The returned profile name works with all three OCI transports —
    :class:`OCIChatCompletionsModel` (V1, ``profile=``),
    :class:`OCIModel` (SDK, ``profile_name=``), and
    :class:`OCIResponsesModel` (Responses, ``profile=``) — since they
    all delegate to the same ``oci.config.from_file`` lookup.

    Closes the long-standing footgun in #276 where a user with multiple
    profiles and no ``[DEFAULT]`` would get an opaque OCI SDK trace
    instead of a "set OCI_PROFILE=…" pointer.
    """
    explicit = _oci_env("PROFILE")
    if explicit:
        return explicit

    config_path = Path.home() / ".oci" / "config"
    sections = _oci_config_profiles(config_path)

    if "DEFAULT" in sections:
        return "DEFAULT"
    if len(sections) == 1:
        return sections[0]
    if not sections:
        msg = (
            f"No usable profiles found in {config_path}. "
            "Run `oci session authenticate --profile-name <name>` or add a "
            "[<name>] section, then set OCI_PROFILE=<name>."
        )
        raise RuntimeError(msg)
    names = ", ".join(sections)
    msg = (
        f"{config_path} has multiple profiles ({names}) and no [DEFAULT] "
        "section. Pick one by setting OCI_PROFILE=<name> (or "
        "LOCUS_OCI_PROFILE=<name>) before running the notebook."
    )
    raise RuntimeError(msg)


def _auto_detect_provider() -> str:
    """Pick a sensible default provider when ``LOCUS_MODEL_PROVIDER`` is unset.

    Prefer OCI — that is the platform Locus is built on — and only fall
    back to ``mock`` when an OCI profile clearly isn't available.
    Workload-identity env vars short-circuit to OCI (the harness can't
    inspect those; trust the user). Otherwise we look for at least one
    usable profile in ``~/.oci/config``. With no OCI config and no
    workload-identity hints we fall back to the bundled mock so the
    notebook still runs end-to-end.

    Never raises — a wrong guess here surfaces later as a friendly
    error from :func:`_resolve_oci_profile`.
    """
    if os.environ.get("LOCUS_OCI_AUTH_TYPE") in ("instance_principal", "resource_principal"):
        return "oci"
    if os.environ.get("OCI_AUTH_TYPE") in ("instance_principal", "resource_principal"):
        return "oci"
    if _oci_config_profiles(Path.home() / ".oci" / "config"):
        # Only claim OCI when the profile resolves unambiguously ([DEFAULT],
        # a single named section, or an explicit OCI_PROFILE). A config with
        # several profiles and no default would otherwise turn a notebook
        # that was never pointed at OCI into a RuntimeError; explicitly
        # setting LOCUS_MODEL_PROVIDER=oci still surfaces that friendly
        # error from :func:`_resolve_oci_profile`.
        try:
            _resolve_oci_profile()
        except RuntimeError:
            return "mock"
        return "oci"
    return "mock"


def get_model(**kwargs: Any) -> Any:
    """Return the configured model for the current notebook.

    Reads ``LOCUS_MODEL_PROVIDER`` first. If it isn't set, auto-detects:
    OCI when an OCI profile is reachable, otherwise the bundled mock.

    Args:
        **kwargs: Override any model parameters (max_tokens, temperature, …).
            Pass ``model_id="..."`` to use a specific model id without
            changing ``LOCUS_MODEL_ID``.
    """
    provider = os.environ.get("LOCUS_MODEL_PROVIDER", "").lower() or _auto_detect_provider()

    if provider == "mock":
        kwargs.pop("model_id", None)  # MockModel ignores model_id
        return MockModel(**kwargs)
    elif provider == "oci":
        return _get_oci_model(**kwargs)
    elif provider == "openai":
        return _get_openai_model(**kwargs)
    elif provider == "anthropic":
        return _get_anthropic_model(**kwargs)
    else:
        return _get_registry_model(provider, **kwargs)


def get_model_b(**kwargs: Any) -> Any:
    """Secondary model slot — typically a cheaper/faster variant for
    triage, routing, or color commentary in multi-agent notebooks.

    Reads ``LOCUS_MODEL_ID_B`` (the secondary model slot).
    Falls back to ``LOCUS_MODEL_ID`` (= slot A) when unset, so notebooks
    that call ``get_model_b()`` still work in plain CLI runs where only
    one model is configured.
    """
    kwargs.setdefault(
        "model_id",
        os.environ.get("LOCUS_MODEL_ID_B") or os.environ.get("LOCUS_MODEL_ID", ""),
    )
    return get_model(**kwargs)


def get_model_c(**kwargs: Any) -> Any:
    """Tertiary model slot — same fall-through rules as :func:`get_model_b`,
    typically used for a judge / critic role distinct from both A and B."""
    kwargs.setdefault(
        "model_id",
        os.environ.get("LOCUS_MODEL_ID_C") or os.environ.get("LOCUS_MODEL_ID", ""),
    )
    return get_model(**kwargs)


def _pick_oci_transport(model_id: str) -> str:
    """Pick the right OCI transport for a model id.

    Cohere R-series models need the OCI SDK's proprietary chat shape and
    are routed through ``OCIModel``. Everything else (OpenAI / Meta / xAI
    / Mistral / Gemini, and non-R Cohere) goes through
    ``OCIChatCompletionsModel`` against ``/openai/v1/chat/completions``.

    ``LOCUS_OCI_TRANSPORT=v1|sdk`` overrides the automatic choice.
    """
    forced = _oci_env("TRANSPORT")
    if forced in ("v1", "sdk"):
        return forced
    lowered = model_id.lower()
    # DAC endpoint OCIDs and Cohere R-series both need the SDK
    # transport (DedicatedServingMode for the former, the proprietary
    # Cohere chat shape for the latter).
    if lowered.startswith(("ocid1.generativeaiendpoint.", "cohere.command-r")):
        return "sdk"
    return "v1"


def _get_oci_model(**kwargs: Any) -> Any:
    """Get an OCI GenAI model — picks V1 vs SDK transport per model family."""
    model_id = kwargs.pop("model_id", os.environ.get("LOCUS_MODEL_ID", "openai.gpt-5.5-2026-04-23"))
    transport = _pick_oci_transport(model_id)
    if transport == "v1":
        return _get_oci_v1_model(model_id, **kwargs)
    return _get_oci_sdk_model(model_id, **kwargs)


def _get_oci_v1_model(model_id: str, **kwargs: Any) -> Any:
    """Build an OCIChatCompletionsModel against /openai/v1/chat/completions."""
    from locus.models.providers.oci import OCIChatCompletionsModel

    region = _oci_env("REGION", "us-chicago-1") or "us-chicago-1"
    compartment = _oci_env("COMPARTMENT")
    auth_type = _oci_env("AUTH_TYPE", "") or ""

    if auth_type in ("instance_principal", "resource_principal"):
        if not compartment:
            msg = f"LOCUS_OCI_COMPARTMENT is required when LOCUS_OCI_AUTH_TYPE={auth_type}"
            raise ValueError(msg)
        return OCIChatCompletionsModel(
            model=model_id,
            auth_type=auth_type,
            compartment_id=compartment,
            region=region,
            **kwargs,
        )

    # Default: profile-based auth. compartment auto-derived from the
    # profile's tenancy unless overridden.
    profile = _resolve_oci_profile()
    return OCIChatCompletionsModel(
        model=model_id,
        profile=profile,
        compartment_id=compartment,
        region=region,
        **kwargs,
    )


def _get_oci_sdk_model(model_id: str, **kwargs: Any) -> Any:
    """Build an OCIModel against /20231130/actions/v1/chat (SDK transport)."""
    from locus.models import OCIAuthType, OCIModel

    auth_type_str = _oci_env("AUTH_TYPE", "api_key") or "api_key"
    if auth_type_str in ("instance_principal", "resource_principal"):
        # Workload identity: profile_name is required by the SDK signature
        # but unused at signing time. Honour an explicit override if the
        # user set one, else fall back to "DEFAULT" to keep the call shape.
        profile = _oci_env("PROFILE", "DEFAULT") or "DEFAULT"
    else:
        profile = _resolve_oci_profile()
    compartment = _oci_env("COMPARTMENT")
    endpoint = _oci_env("ENDPOINT")
    # The OCI profile's home region is often *not* the GenAI region
    # (e.g. MY_PROFILE's us-ashburn-1 vs GenAI in us-chicago-1). Derive
    # the endpoint from LOCUS_OCI_REGION / OCI_REGION when no explicit
    # endpoint is set so cross-tenancy / cross-region session tokens
    # still hit the right service.
    if not endpoint:
        region = _oci_env("REGION")
        if region:
            endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    auth_type_map = {
        "api_key": OCIAuthType.API_KEY,
        "security_token": OCIAuthType.SECURITY_TOKEN,
        "session_token": OCIAuthType.SECURITY_TOKEN,
        "instance_principal": OCIAuthType.INSTANCE_PRINCIPAL,
        "resource_principal": OCIAuthType.RESOURCE_PRINCIPAL,
    }
    auth_type = auth_type_map.get(auth_type_str, OCIAuthType.API_KEY)

    return OCIModel(
        model_id=model_id,
        profile_name=profile,
        auth_type=auth_type,
        compartment_id=compartment,
        service_endpoint=endpoint,
        **kwargs,
    )


def _get_openai_model(**kwargs: Any) -> Any:
    """Get OpenAI model."""
    from locus.models import OpenAIModel

    model_id = kwargs.pop("model_id", os.environ.get("LOCUS_MODEL_ID", "gpt-4o"))
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")

    return OpenAIModel(
        model=model_id,
        api_key=api_key,
        **kwargs,
    )


def _get_registry_model(provider: str, **kwargs: Any) -> Any:
    """Any other provider the model registry knows — openrouter, groq, gemini, ollama, …

    The provider reads its own credentials; ``LOCUS_MODEL_ID`` names the model.
    """
    from locus.models import get_model as registry_get_model
    from locus.models.registry import list_providers

    if provider not in list_providers():
        raise ValueError(
            f"Unknown model provider: {provider}. Use 'mock' or one of {list_providers()}."
        )
    model_id = kwargs.pop("model_id", None) or os.environ.get("LOCUS_MODEL_ID")
    if not model_id:
        raise ValueError(f"LOCUS_MODEL_ID is required with LOCUS_MODEL_PROVIDER={provider}")
    return registry_get_model(f"{provider}:{model_id}", **kwargs)


def _get_anthropic_model(**kwargs: Any) -> Any:
    """Get Anthropic model."""
    from locus.models.native.anthropic import AnthropicModel

    model_id = kwargs.pop("model_id", os.environ.get("LOCUS_MODEL_ID", "claude-sonnet-5-5"))
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable required")

    return AnthropicModel(
        model=model_id,
        api_key=api_key,
        **kwargs,
    )


def get_embedder(**kwargs: Any) -> Any:
    """Return the embedding model for the RAG notebooks.

    Mirrors :func:`get_model`: the id comes from ``LOCUS_EMBEDDING_MODEL`` so
    the notebooks can be pointed at a self-hosted embedding endpoint (vLLM,
    Ollama, LM Studio, TEI) instead of only api.openai.com. Provider
    credentials and ``OPENAI_BASE_URL`` are read by the client as usual.
    """
    from locus.rag import OpenAIEmbeddings

    model_id = kwargs.pop("model", None) or os.environ.get(
        "LOCUS_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    return OpenAIEmbeddings(model=model_id, **kwargs)


def get_embedding_dimension(default: int = 1536) -> int:
    """Vector width of the configured embedder.

    Stores need this up front, and it is not 1536 for every model — bge-small
    is 384, text-embedding-3-large is 3072. ``LOCUS_EMBEDDING_DIM`` keeps the
    store and the embedder in agreement when either is swapped.
    """
    return int(os.environ.get("LOCUS_EMBEDDING_DIM", default))


_SAFE_MATH_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_MATH_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_math_eval(expression: str) -> float:
    """Evaluate plain arithmetic for the notebooks' ``calculate`` tools.

    AST-only: numbers and the arithmetic operators above, no names, calls,
    or attribute access, so a model-supplied expression cannot turn the
    calculator into a sandbox escape. Raises ``ValueError`` (or
    ``SyntaxError`` / ``ZeroDivisionError``) on anything else.
    """
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_MATH_BIN_OPS:
            return _SAFE_MATH_BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_MATH_UNARY_OPS:
            return _SAFE_MATH_UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    return _eval(tree)


def print_config():
    """Print current configuration for debugging."""
    provider = os.environ.get("LOCUS_MODEL_PROVIDER", "").lower() or "mock"
    model_id = os.environ.get("LOCUS_MODEL_ID", "(default)")

    print(f"Model Provider: {provider}")

    if provider == "mock":
        print("Using mock model (no API calls)")
    else:
        print(f"Model ID: {model_id}")
