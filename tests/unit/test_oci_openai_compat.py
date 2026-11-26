# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the OCI OpenAI-compat model.

Verifies:
- Auth-mode validation (exactly one of profile/auth_type).
- Profile path builds an AsyncOpenAI with an OCIRequestSigner-wrapped
  httpx client.
- ``auth_type`` paths require ``compartment_id`` and dispatch to the
  correct signer.
- Region default and base-URL derivation.
- Inherited :class:`OpenAIModel` parsing still works.

No live OCI calls — the openai SDK and OCI signers are mocked.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from locus.core.messages import Message
from locus.models.providers.oci.openai_compat import (
    DEFAULT_OCI_GENAI_REGION,
    OCIChatCompletionsConfig,
    OCIChatCompletionsModel,
    build_oci_openai_base_url,
)


COMPARTMENT_OCID = "ocid1.compartment.oc1..aaaaaaaaexample"


class TestBuildOCIChatCompletionsBaseURL:
    def test_default_region(self):
        assert build_oci_openai_base_url(DEFAULT_OCI_GENAI_REGION) == (
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1"
        )

    def test_other_region(self):
        assert build_oci_openai_base_url("eu-frankfurt-1") == (
            "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com/openai/v1"
        )


class TestAuthModeValidation:
    def test_no_auth_mode_raises(self):
        with pytest.raises(ValueError, match="specify exactly one"):
            OCIChatCompletionsModel(model="openai.gpt-5.5")

    def test_both_auth_modes_raises(self):
        with pytest.raises(ValueError, match="specify exactly one"):
            OCIChatCompletionsModel(
                model="openai.gpt-5.5",
                profile="MY_PROFILE",
                auth_type="instance_principal",
                compartment_id=COMPARTMENT_OCID,
            )

    def test_unknown_auth_type_raises(self):
        with pytest.raises(ValueError, match="auth_type must be one of"):
            OCIChatCompletionsModel(
                model="openai.gpt-5.5",
                auth_type="federated_user",
                compartment_id=COMPARTMENT_OCID,
            )

    def test_auth_type_without_compartment_raises(self):
        with pytest.raises(ValueError, match="compartment_id is required"):
            OCIChatCompletionsModel(
                model="openai.gpt-5.5",
                auth_type="instance_principal",
            )

    def test_auth_mode_error_includes_remediation(self):
        """The error message must spell out the two valid call shapes.

        Regression target: the message used to be
            "specify exactly one of profile=, auth_type="
        which left users guessing what to pass. The expanded message
        names the OCI config section, the auth_type values, and the
        compartment_id requirement so the error is self-fixing.
        """
        with pytest.raises(ValueError) as excinfo:
            OCIChatCompletionsModel(model="openai.gpt-5.5")
        msg = str(excinfo.value)
        # Self-fix breadcrumbs the user can act on.
        assert "~/.oci/config" in msg
        assert "DEFAULT" in msg
        assert "instance_principal" in msg
        assert "compartment_id" in msg


class TestProfileMode:
    def test_config_set(self):
        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value={"tenancy": COMPARTMENT_OCID},
        ):
            model = OCIChatCompletionsModel(model="openai.gpt-5.5", profile="MY_PROFILE")
        assert model.config.profile == "MY_PROFILE"
        assert model.config.auth_type is None
        assert model.config.compartment_id == COMPARTMENT_OCID
        assert model.config.region == DEFAULT_OCI_GENAI_REGION

    def test_compartment_auto_derived_from_profile_tenancy(self):
        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value={"tenancy": "ocid1.tenancy.oc1..fromprofile"},
        ):
            model = OCIChatCompletionsModel(model="openai.gpt-5.5", profile="MY_PROFILE")
        assert model.config.compartment_id == "ocid1.tenancy.oc1..fromprofile"

    def test_explicit_compartment_overrides_auto_derive(self):
        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value={"tenancy": "ocid1.tenancy.oc1..fromprofile"},
        ) as mock_load:
            model = OCIChatCompletionsModel(
                model="openai.gpt-5.5",
                profile="MY_PROFILE",
                compartment_id="ocid1.compartment.oc1..explicit",
            )
        # When explicit, no need to load the profile.
        mock_load.assert_not_called()
        assert model.config.compartment_id == "ocid1.compartment.oc1..explicit"

    def test_profile_load_failure_does_not_crash_init(self):
        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            side_effect=FileNotFoundError("no config file"),
        ):
            # init should still succeed (compartment ends up None).
            model = OCIChatCompletionsModel(model="openai.gpt-5.5", profile="MISSING")
        assert model.config.compartment_id is None

    def test_env_var_overrides_profile_tenancy(self, monkeypatch):
        """``OCI_COMPARTMENT`` env var beats the profile-tenancy fallback.

        Common case: MY_PROFILE's home tenancy lacks GenAI policy, but the
        user has access to GenAI inference in the target tenancy's compartment.
        Setting ``OCI_COMPARTMENT`` should make the model target the
        compartment with the policy.
        """
        env_compartment = "ocid1.compartment.oc1..env-set"
        monkeypatch.setenv("OCI_COMPARTMENT", env_compartment)
        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value={"tenancy": "ocid1.tenancy.oc1..from-profile"},
        ) as mock_load:
            model = OCIChatCompletionsModel(model="openai.gpt-4o-mini", profile="ANY")
        # Env var takes precedence; profile load is short-circuited.
        mock_load.assert_not_called()
        assert model.config.compartment_id == env_compartment

    def test_compartment_id_arg_beats_env_var(self, monkeypatch):
        """Explicit ``compartment_id=`` arg wins over ``OCI_COMPARTMENT`` env."""
        monkeypatch.setenv("OCI_COMPARTMENT", "ocid1.compartment.oc1..env-set")
        explicit = "ocid1.compartment.oc1..explicit"
        model = OCIChatCompletionsModel(
            model="openai.gpt-4o-mini",
            profile="ANY",
            compartment_id=explicit,
        )
        assert model.config.compartment_id == explicit

    def test_oci_compartment_id_env_alias(self, monkeypatch):
        """Both ``OCI_COMPARTMENT`` and ``OCI_COMPARTMENT_ID`` work."""
        monkeypatch.delenv("OCI_COMPARTMENT", raising=False)
        monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..alias")
        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value={"tenancy": "ocid1.tenancy.oc1..ignored"},
        ):
            model = OCIChatCompletionsModel(model="openai.gpt-4o-mini", profile="ANY")
        assert model.config.compartment_id == "ocid1.compartment.oc1..alias"

    def test_client_uses_signer_http_client(self):
        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value={"tenancy": COMPARTMENT_OCID},
        ):
            model = OCIChatCompletionsModel(model="openai.gpt-5.5", profile="MY_PROFILE")

        signer = MagicMock(name="signer")
        with (
            patch(
                "locus.models.providers.oci.openai_compat._build_signer_from_profile",
                return_value=signer,
            ) as mock_build,
            patch("openai.AsyncOpenAI") as mock_async_openai,
            patch("httpx.AsyncClient") as mock_httpx_client,
        ):
            _ = model.client
            mock_build.assert_called_once_with("MY_PROFILE", "~/.oci/config")
            mock_httpx_client.assert_called_once()
            kwargs = mock_async_openai.call_args.kwargs
            assert kwargs["api_key"] == "not-used"
            assert kwargs["base_url"] == build_oci_openai_base_url(DEFAULT_OCI_GENAI_REGION)
            assert "http_client" in kwargs

    def test_custom_config_file_passed_to_signer(self):
        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value={"tenancy": COMPARTMENT_OCID},
        ):
            model = OCIChatCompletionsModel(
                model="openai.gpt-5.5",
                profile="MY_PROFILE",
                config_file="/tmp/oci-config",
            )
        with (
            patch(
                "locus.models.providers.oci.openai_compat._build_signer_from_profile",
                return_value=MagicMock(),
            ) as mock_build,
            patch("openai.AsyncOpenAI"),
            patch("httpx.AsyncClient"),
        ):
            _ = model.client
            mock_build.assert_called_once_with("MY_PROFILE", "/tmp/oci-config")


class TestAuthTypeMode:
    def test_instance_principal(self):
        model = OCIChatCompletionsModel(
            model="openai.gpt-5.5",
            auth_type="instance_principal",
            compartment_id=COMPARTMENT_OCID,
        )
        assert model.config.auth_type == "instance_principal"
        assert model.config.compartment_id == COMPARTMENT_OCID

        with (
            patch(
                "locus.models.providers.oci.openai_compat._build_instance_principal_signer",
                return_value=MagicMock(),
            ) as mock_build,
            patch("openai.AsyncOpenAI"),
            patch("httpx.AsyncClient"),
        ):
            _ = model.client
            mock_build.assert_called_once_with()

    def test_resource_principal(self):
        model = OCIChatCompletionsModel(
            model="openai.gpt-5.5",
            auth_type="resource_principal",
            compartment_id=COMPARTMENT_OCID,
        )
        assert model.config.auth_type == "resource_principal"

        with (
            patch(
                "locus.models.providers.oci.openai_compat._build_resource_principal_signer",
                return_value=MagicMock(),
            ) as mock_build,
            patch("openai.AsyncOpenAI"),
            patch("httpx.AsyncClient"),
        ):
            _ = model.client
            mock_build.assert_called_once_with()


class TestRefreshCallableFor:
    """``_refresh_callable_for`` is the bridge that prevents the
    "instance principal token expires at ~15min, every subsequent
    GenAI call 401s, only a pod restart recovers" production failure
    mode. Mirroring oracle-samples/oci-genai-auth-python, it returns a
    callable that *rebuilds* a fresh signer — instance / resource
    principal from the metadata service, session token from disk —
    rather than mutating one in place, and ``None`` for static signers
    (user API key). The OCIRequestSigner refresh path is dead code
    without it."""

    def test_instance_principal_returns_rebuild_builder(self):
        from locus.models.providers.oci.openai_compat import (
            _build_instance_principal_signer,
            _refresh_callable_for,
        )

        # Rebuild, not in-place: the callable re-mints a brand-new signer
        # from IMDS on each refresh (the in-place refresh_security_token
        # path is what rots in long-lived processes).
        cb = _refresh_callable_for(MagicMock(), auth_type="instance_principal")
        assert cb is _build_instance_principal_signer

    def test_resource_principal_returns_rebuild_builder(self):
        from locus.models.providers.oci.openai_compat import (
            _build_resource_principal_signer,
            _refresh_callable_for,
        )

        cb = _refresh_callable_for(MagicMock(), auth_type="resource_principal")
        assert cb is _build_resource_principal_signer

    def test_static_signer_returns_none(self):
        from locus.models.providers.oci.openai_compat import _refresh_callable_for

        # User-principal API-key signer: no auth_type, no profile — the
        # API key doesn't expire so refresh is a no-op.
        signer = MagicMock(spec=["do_request_sign"])
        assert _refresh_callable_for(signer) is None


class TestClientWiresRefreshSigner:
    """The ``client`` property must construct ``OCIRequestSigner`` with
    a ``refresh_signer`` callback for token-based signers AND with a
    shorter ``refresh_interval`` than the upstream 1-hour default —
    otherwise instance-principal federation tokens (which expire on
    the order of 15-30 minutes) silently rot inside the http client."""

    def test_instance_principal_passes_rebuild_refresh_to_signing_wrapper(self):
        model = OCIChatCompletionsModel(
            model="openai.gpt-5.5",
            auth_type="instance_principal",
            compartment_id=COMPARTMENT_OCID,
        )

        with (
            patch(
                "locus.models.providers.oci.openai_compat._build_instance_principal_signer",
            ) as mock_build,
            patch("locus.models.providers.oci._signing.OCIRequestSigner") as mock_sig_ctor,
            patch("openai.AsyncOpenAI"),
            patch("httpx.AsyncClient"),
        ):
            _ = model.client
            kwargs = mock_sig_ctor.call_args.kwargs
            assert kwargs["compartment_id"] == COMPARTMENT_OCID
            # Rebuild-on-refresh: the refresh callable IS the instance-
            # principal builder, so auth_flow re-mints a fresh signer from
            # IMDS on 401 + on the periodic timer (mirrors oracle-samples/
            # oci-genai-auth-python) rather than mutating a cached
            # federation client in place.
            assert kwargs["refresh_signer"] is mock_build
            # 600s is tighter than the upstream 3600s default — short
            # enough that proactive refresh beats the typical 15-30min
            # federation-token TTL.
            assert kwargs["refresh_interval"] == 600.0

    def test_user_principal_signer_gets_no_refresh_callback(self):
        """API-key signers don't expire; passing a no-op refresh
        wastes a clock check on every request, but the constructor
        contract is None=disabled. Confirm the dispatch sends None."""
        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value={"tenancy": COMPARTMENT_OCID},
        ):
            model = OCIChatCompletionsModel(model="openai.gpt-5.5", profile="MY_PROFILE")

        # Static signer — no refresh_security_token attribute.
        static_signer = MagicMock(spec=["do_request_sign"])
        with (
            patch(
                "locus.models.providers.oci.openai_compat._build_signer_from_profile",
                return_value=static_signer,
            ),
            patch("locus.models.providers.oci._signing.OCIRequestSigner") as mock_sig_ctor,
            patch("openai.AsyncOpenAI"),
            patch("httpx.AsyncClient"),
        ):
            _ = model.client
            assert mock_sig_ctor.call_args.kwargs["refresh_signer"] is None


class TestBaseURLAndRegion:
    def test_explicit_base_url_wins(self):
        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value={"tenancy": COMPARTMENT_OCID},
        ):
            model = OCIChatCompletionsModel(
                model="openai.gpt-5.5",
                profile="MY_PROFILE",
                base_url="https://custom.example.com/v1",
            )
        assert model.config.base_url == "https://custom.example.com/v1"

    def test_other_region(self):
        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value={"tenancy": COMPARTMENT_OCID},
        ):
            model = OCIChatCompletionsModel(
                model="openai.gpt-5.5",
                profile="MY_PROFILE",
                region="eu-frankfurt-1",
            )
        assert model.config.base_url == (
            "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com/openai/v1"
        )


class TestClientCaching:
    def test_client_built_once(self):
        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value={"tenancy": COMPARTMENT_OCID},
        ):
            model = OCIChatCompletionsModel(model="openai.gpt-5.5", profile="MY_PROFILE")
        with (
            patch(
                "locus.models.providers.oci.openai_compat._build_signer_from_profile",
                return_value=MagicMock(),
            ),
            patch("openai.AsyncOpenAI") as mock_async_openai,
            patch("httpx.AsyncClient"),
        ):
            mock_async_openai.return_value = MagicMock()
            first = model.client
            second = model.client
            assert first is second
            assert mock_async_openai.call_count == 1


class TestConfigInheritsOpenAIFields:
    def test_seed_and_stop_sequences(self):
        config = OCIChatCompletionsConfig(
            model="openai.gpt-5.5",
            seed=42,
            stop_sequences=["STOP"],
        )
        assert config.seed == 42
        assert config.stop_sequences == ["STOP"]


class TestCompleteEndToEndMocked:
    """Confirms inherited complete() still works through the OCI subclass."""

    @pytest.mark.asyncio
    async def test_complete(self):
        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value={"tenancy": COMPARTMENT_OCID},
        ):
            model = OCIChatCompletionsModel(model="openai.gpt-5.5", profile="MY_PROFILE")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hi from OCI"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        model._client = mock_client

        result = await model.complete([Message.user("Hi")])
        assert result.message.content == "Hi from OCI"
        assert result.stop_reason == "stop"


class TestReasoningEffort:
    """``reasoning_effort=`` is folded into ``extra_body`` for the chat endpoint."""

    def _model(self, **kwargs):
        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value={"tenancy": COMPARTMENT_OCID},
        ):
            return OCIChatCompletionsModel(
                model="openai.gpt-6-luna", profile="MY_PROFILE", **kwargs
            )

    def test_default_sends_nothing(self):
        assert self._model().config.extra_body is None

    def test_kwarg_lands_in_extra_body(self):
        assert self._model(reasoning_effort="none").config.extra_body == {
            "reasoning_effort": "none"
        }

    def test_explicit_extra_body_key_wins_and_other_keys_survive(self):
        model = self._model(
            reasoning_effort="none", extra_body={"reasoning_effort": "low", "seed": 7}
        )
        assert model.config.extra_body == {"reasoning_effort": "low", "seed": 7}


class TestToolsRefusedUnderReasoning:
    """The chat endpoint's 400 for function tools under reasoning is answered with 'none'."""

    _MESSAGE = (
        "Function tools with reasoning_effort are not supported for gpt-6-luna in "
        "/v1/chat/completions. To use function tools, use /v1/responses or set "
        "reasoning_effort to 'none'"
    )
    _TOOLS = [
        {
            "type": "function",
            "function": {"name": "stock", "parameters": {"type": "object", "properties": {}}},
        }
    ]

    def _model(self, **kwargs):
        with patch(
            "locus.models.providers.oci.openai_compat._load_profile_config",
            return_value={"tenancy": COMPARTMENT_OCID},
        ):
            return OCIChatCompletionsModel(
                model="openai.gpt-6-luna", profile="MY_PROFILE", **kwargs
            )

    @staticmethod
    def _refusal(message: str):
        import httpx
        import openai

        response = httpx.Response(
            400, request=httpx.Request("POST", "https://example.invalid/v1/chat/completions")
        )
        return openai.BadRequestError(
            message, response=response, body={"code": "400", "message": message}
        )

    @staticmethod
    def _response(text: str = "17 in stock"):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = text
        response.choices[0].message.tool_calls = None
        response.choices[0].finish_reason = "stop"
        response.usage.prompt_tokens = 5
        response.usage.completion_tokens = 3
        return response

    def _wire(self, model, *outcomes):
        create = AsyncMock(side_effect=list(outcomes))
        model._client = MagicMock()
        model._client.chat.completions.create = create
        return create

    @pytest.mark.asyncio
    async def test_refusal_is_retried_with_none_and_logged_once(self, caplog):
        model = self._model()
        create = self._wire(model, self._refusal(self._MESSAGE), self._response(), self._response())
        with caplog.at_level(logging.WARNING, logger="locus.models.providers.oci"):
            result = await model.complete([Message.user("Hi")], tools=self._TOOLS)
        assert result.message.content == "17 in stock"
        assert create.call_count == 2
        assert create.call_args_list[1].kwargs["extra_body"]["reasoning_effort"] == "none"
        notices = [r for r in caplog.records if "reasoning_effort='none'" in r.getMessage()]
        assert len(notices) == 1
        assert notices[0].levelno == logging.WARNING
        assert "OCIChatCompletionsModel" in notices[0].getMessage()
        # From now on tools go out with "none" at once, and nothing is said again.
        caplog.clear()
        await model.complete([Message.user("Again")], tools=self._TOOLS)
        assert create.call_count == 3
        assert create.call_args_list[2].kwargs["extra_body"]["reasoning_effort"] == "none"
        assert not [r for r in caplog.records if "reasoning_effort" in r.getMessage()]

    @pytest.mark.asyncio
    async def test_other_400_is_raised(self):
        import openai

        model = self._model()
        create = self._wire(model, self._refusal("Unsupported OpenAI operation"))
        with pytest.raises(openai.BadRequestError):
            await model.complete([Message.user("Hi")], tools=self._TOOLS)
        assert create.call_count == 1

    @pytest.mark.asyncio
    async def test_without_tools_the_refusal_is_raised(self):
        import openai

        model = self._model()
        create = self._wire(model, self._refusal(self._MESSAGE))
        with pytest.raises(openai.BadRequestError):
            await model.complete([Message.user("Hi")])
        assert create.call_count == 1

    @pytest.mark.asyncio
    async def test_already_none_is_not_retried(self):
        import openai

        model = self._model(reasoning_effort="none")
        create = self._wire(model, self._refusal(self._MESSAGE))
        with pytest.raises(openai.BadRequestError):
            await model.complete([Message.user("Hi")], tools=self._TOOLS)
        assert create.call_count == 1

    # --- a model that refuses 'none' as well -------------------------------

    _NO_NONE = (
        "Unsupported value: 'reasoning_effort' does not support 'none' with this model. "
        "Supported values are: 'low', 'medium', 'high', and 'xhigh'."
    )

    @pytest.mark.asyncio
    async def test_a_model_that_refuses_none_too_says_to_use_responses(self):
        from locus.models.failover import FailoverReason, classify
        from locus.models.providers.oci import OCIToolsNeedResponsesError

        model = self._model()
        create = self._wire(
            model,
            self._refusal(self._MESSAGE),
            self._refusal(self._NO_NONE),
            self._refusal(self._NO_NONE),
        )
        with pytest.raises(OCIToolsNeedResponsesError) as raised:
            await model.complete([Message.user("Hi")], tools=self._TOOLS)
        assert create.call_count == 2
        assert 'transport="responses"' in str(raised.value)
        assert raised.value.model == "openai.gpt-6-luna"
        # The same request fails the same way: not something to retry.
        assert classify(raised.value).reason is FailoverReason.FORMAT_ERROR
        # The instance now sends "none" at once; the answer is the same error.
        with pytest.raises(OCIToolsNeedResponsesError):
            await model.complete([Message.user("Again")], tools=self._TOOLS)
        assert create.call_count == 3

    @pytest.mark.asyncio
    async def test_refusing_none_without_the_tools_refusal_is_raised_as_it_is(self):
        import openai

        # The caller chose "none"; nothing says tools are the reason.
        model = self._model(reasoning_effort="none")
        create = self._wire(model, self._refusal(self._NO_NONE))
        with pytest.raises(openai.BadRequestError):
            await model.complete([Message.user("Hi")], tools=self._TOOLS)
        assert create.call_count == 1

    # --- the streaming call takes the same decision -------------------------

    @staticmethod
    def _stream(text: str = "17 in stock"):
        """A streamed turn: two content deltas, then the finish chunk."""
        from types import SimpleNamespace

        def chunk(content, finish=None):
            delta = SimpleNamespace(content=content, tool_calls=None)
            choice = SimpleNamespace(delta=delta, finish_reason=finish)
            return SimpleNamespace(model="openai.gpt-6-luna", usage=None, choices=[choice])

        head, tail = text[: len(text) // 2], text[len(text) // 2 :]
        chunks = [chunk(head), chunk(tail), chunk(None, finish="stop")]

        async def turn():
            for c in chunks:
                yield c

        return turn()

    @staticmethod
    async def _text_of(events) -> str:
        return "".join([e.content async for e in events if e.content])

    @pytest.mark.asyncio
    async def test_streamed_refusal_is_retried_with_none_and_logged_once(self, caplog):
        model = self._model()
        create = self._wire(model, self._refusal(self._MESSAGE), self._stream(), self._stream())
        with caplog.at_level(logging.WARNING, logger="locus.models.providers.oci"):
            text = await self._text_of(model.stream([Message.user("Hi")], tools=self._TOOLS))
        assert text == "17 in stock"
        assert create.call_count == 2
        assert create.call_args_list[1].kwargs["extra_body"]["reasoning_effort"] == "none"
        notices = [r for r in caplog.records if "reasoning_effort='none'" in r.getMessage()]
        assert len(notices) == 1
        assert "OCIChatCompletionsModel" in notices[0].getMessage()
        # The next streamed turn with tools goes out with "none" at once, silently.
        caplog.clear()
        text = await self._text_of(model.stream([Message.user("Again")], tools=self._TOOLS))
        assert text == "17 in stock"
        assert create.call_count == 3
        assert create.call_args_list[2].kwargs["extra_body"]["reasoning_effort"] == "none"
        assert not [r for r in caplog.records if "reasoning_effort" in r.getMessage()]

    @pytest.mark.asyncio
    async def test_streamed_other_400_is_raised(self):
        import openai

        model = self._model()
        create = self._wire(model, self._refusal("Unknown parameter: 'seed'"))
        with pytest.raises(openai.BadRequestError):
            await self._text_of(model.stream([Message.user("Hi")], tools=self._TOOLS))
        assert create.call_count == 1

    @pytest.mark.asyncio
    async def test_streamed_refusal_without_tools_is_raised(self):
        import openai

        model = self._model()
        create = self._wire(model, self._refusal(self._MESSAGE))
        with pytest.raises(openai.BadRequestError):
            await self._text_of(model.stream([Message.user("Hi")]))
        assert create.call_count == 1

    @pytest.mark.asyncio
    async def test_a_turn_that_streams_nothing_yields_nothing(self):
        """The guard for a turn that ends before its first event."""
        model = self._model()

        async def silent(*_args, **_kwargs):
            return
            yield  # pragma: no cover — makes this an async generator

        with patch.object(type(model), "_stream_oci_turn", silent):
            events = [e async for e in model.stream([Message.user("Hi")], tools=self._TOOLS)]
        assert events == []
