# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for OCI Dedicated AI Cluster (DAC) endpoint support.

Covers:

- ``get_model("oci:ocid1.generativeaiendpoint.<region>....")`` routes
  to ``OCIModel`` (SDK transport), not ``OCIChatCompletionsModel`` (V1).
- ``OCIClient.get_serving_mode()`` returns ``DedicatedServingMode`` for
  endpoint OCIDs and ``OnDemandServingMode`` for plain model ids.
- ``GenericProvider.parse_stream_chunk()`` correctly extracts text +
  tool-call deltas from the SSE event format the SDK emits.
- ``CohereProvider.parse_stream_chunk()`` does the same for Cohere's
  chunk shape.
- ``examples/config.py`` ``_pick_oci_transport()`` returns ``"sdk"``
  for DAC OCIDs.

Tests skip cleanly when the ``oci`` SDK isn't installed — the
provider routing tests fall through to the V1 transport in that case
(which is itself testable without the SDK).
"""

from __future__ import annotations

import pytest


# Generic placeholder OCIDs — never use real tenancy / endpoint OCIDs
# in test fixtures (CLAUDE.md privacy rule). These match the OCI shape
# (``ocid1.<resource_type>.oc1.<region>.<id>``) but the id portion is
# obviously synthetic.
_FAKE_DAC_OCID = (
    "ocid1.generativeaiendpoint.oc1.uk-london-1."
    "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab"
)
_FAKE_COMPARTMENT = "ocid1.compartment.oc1..abcdef1234567890abcdef1234567890abcdef1234567890abcdef"


# ---------------------------------------------------------------------------
# Routing: get_model("oci:ocid...")
# ---------------------------------------------------------------------------


class TestModelRegistryRoutesDACToSDK:
    def test_dac_endpoint_routes_to_oci_model(self) -> None:
        pytest.importorskip("oci")
        from locus.models import get_model
        from locus.models.providers.oci import OCIModel

        # The registry needs an auth path to actually instantiate the
        # SDK client. ``profile_name`` keeps the constructor pure
        # (deferred client creation), so we don't have to mock OCI.
        model = get_model(
            f"oci:{_FAKE_DAC_OCID}",
            compartment_id=_FAKE_COMPARTMENT,
            profile_name="DEFAULT",
        )
        assert isinstance(model, OCIModel)
        assert model.config.model_id == _FAKE_DAC_OCID

    def test_cohere_r_still_routes_to_oci_model(self) -> None:
        # Regression check: the new DAC rule shouldn't change Cohere
        # R-series routing.
        pytest.importorskip("oci")
        from locus.models import get_model
        from locus.models.providers.oci import OCIModel

        model = get_model(
            "oci:cohere.command-r-plus-08-2024",
            compartment_id=_FAKE_COMPARTMENT,
            profile_name="DEFAULT",
        )
        assert isinstance(model, OCIModel)

    def test_non_dac_non_cohere_still_routes_to_v1(self) -> None:
        # gpt-style on-demand models continue to use the V1 transport.
        pytest.importorskip("oci")
        from locus.models import get_model
        from locus.models.providers.oci import OCIChatCompletionsModel

        model = get_model("oci:openai.gpt-5.5", profile="DEFAULT")
        assert isinstance(model, OCIChatCompletionsModel)


# ---------------------------------------------------------------------------
# Serving-mode selection (already existed but worth a focused test)
# ---------------------------------------------------------------------------


class TestServingModeForDACOCIDs:
    def test_endpoint_ocid_yields_dedicated_serving_mode(self) -> None:
        pytest.importorskip("oci")
        from oci.generative_ai_inference import models as oci_models

        from locus.models.providers.oci.client import (
            OCIAuthType,
            OCIClient,
            OCIClientConfig,
        )

        # Don't instantiate the actual OCI client; use the helper directly.
        cfg = OCIClientConfig(
            auth_type=OCIAuthType.API_KEY,
            profile_name="DEFAULT",
            compartment_id=_FAKE_COMPARTMENT,
        )
        # OCIClient.__init__ defers SDK client creation until first use,
        # so this is safe without mocking.
        client = OCIClient.__new__(OCIClient)
        client.config = cfg

        mode = client.get_serving_mode(_FAKE_DAC_OCID)
        assert isinstance(mode, oci_models.DedicatedServingMode)
        assert mode.endpoint_id == _FAKE_DAC_OCID

    def test_plain_model_id_yields_on_demand_serving_mode(self) -> None:
        pytest.importorskip("oci")
        from oci.generative_ai_inference import models as oci_models

        from locus.models.providers.oci.client import (
            OCIAuthType,
            OCIClient,
            OCIClientConfig,
        )

        cfg = OCIClientConfig(
            auth_type=OCIAuthType.API_KEY,
            profile_name="DEFAULT",
            compartment_id=_FAKE_COMPARTMENT,
        )
        client = OCIClient.__new__(OCIClient)
        client.config = cfg

        mode = client.get_serving_mode("cohere.command-r-plus")
        assert isinstance(mode, oci_models.OnDemandServingMode)
        assert mode.model_id == "cohere.command-r-plus"


# ---------------------------------------------------------------------------
# Streaming chunk parsers
# ---------------------------------------------------------------------------


class TestGenericProviderStreamChunks:
    def test_parses_text_delta(self) -> None:
        pytest.importorskip("oci")
        from locus.models.providers.oci.models import GenericProvider

        chunk = {
            "message": {
                "content": [
                    {"type": "TEXT", "text": "Hello, "},
                ]
            },
        }
        text, tool_calls, is_done = GenericProvider().parse_stream_chunk(chunk)
        assert text == "Hello, "
        assert tool_calls == []
        assert not is_done

    def test_parses_finish_reason(self) -> None:
        pytest.importorskip("oci")
        from locus.models.providers.oci.models import GenericProvider

        chunk = {"finishReason": "stop"}
        text, tool_calls, is_done = GenericProvider().parse_stream_chunk(chunk)
        assert text == ""
        assert tool_calls == []
        assert is_done is True

    def test_parses_tool_call_delta(self) -> None:
        pytest.importorskip("oci")
        from locus.models.providers.oci.models import GenericProvider

        chunk = {
            "message": {"toolCalls": [{"id": "tc-1", "name": "lookup", "arguments": '{"q":"x"}'}]},
        }
        text, tool_calls, is_done = GenericProvider().parse_stream_chunk(chunk)
        assert text == ""
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "lookup"
        assert tool_calls[0].arguments == {"q": "x"}
        assert tool_calls[0].id == "tc-1"

    def test_handles_malformed_tool_args_gracefully(self) -> None:
        pytest.importorskip("oci")
        from locus.models.providers.oci.models import GenericProvider

        chunk = {
            "message": {"toolCalls": [{"name": "x", "arguments": "not-json"}]},
        }
        _, tool_calls, _ = GenericProvider().parse_stream_chunk(chunk)
        assert len(tool_calls) == 1
        assert tool_calls[0].arguments == {}  # Falls back, doesn't raise


class TestCohereProviderStreamChunks:
    def test_parses_text_delta(self) -> None:
        pytest.importorskip("oci")
        from locus.models.providers.oci.models import CohereProvider

        chunk = {"text": "world"}
        text, tool_calls, is_done = CohereProvider().parse_stream_chunk(chunk)
        assert text == "world"
        assert tool_calls == []
        assert not is_done

    def test_parses_tool_call_on_final_event(self) -> None:
        pytest.importorskip("oci")
        from locus.models.providers.oci.models import CohereProvider

        chunk = {
            "finishReason": "stop",
            "toolCalls": [{"name": "search", "parameters": {"q": "tokyo"}}],
        }
        text, tool_calls, is_done = CohereProvider().parse_stream_chunk(chunk)
        assert text == ""
        assert is_done is True
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search"
        assert tool_calls[0].arguments == {"q": "tokyo"}


# ---------------------------------------------------------------------------
# examples/config.py transport routing
# ---------------------------------------------------------------------------


class TestExamplesConfigPicksTransport:
    def test_dac_ocid_picks_sdk(self) -> None:
        # The examples/config.py module isn't on the package path; load
        # it as a script for this test. Doing it here instead of a
        # conftest fixture keeps the dependency narrow.
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "examples" / "config.py"
        spec = importlib.util.spec_from_file_location("examples_config", path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module._pick_oci_transport(_FAKE_DAC_OCID) == "sdk"
        assert module._pick_oci_transport("cohere.command-r-plus-08-2024") == "sdk"
        assert module._pick_oci_transport("openai.gpt-5.5") == "v1"
