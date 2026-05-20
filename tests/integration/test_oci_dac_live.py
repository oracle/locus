# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Live integration tests for OCI Dedicated AI Cluster (DAC) endpoints.

These tests fire real inference requests against a configured DAC
endpoint and skip cleanly when the credentials / OCID aren't
configured. Activation requires:

- ``OCI_DAC_ENDPOINT_OCID`` — the ``ocid1.generativeaiendpoint....``
  OCID for the DAC.
- ``OCI_DAC_COMPARTMENT_ID`` — compartment OCID where the endpoint
  lives.
- ``OCI_DAC_REGION`` — the region hosting the endpoint
  (e.g. ``uk-london-1``).
- ``OCI_PROFILE`` — a profile in ``~/.oci/config`` with permission to
  invoke the endpoint.
- ``oci`` SDK installed (``pip install -e ".[oci]"``).

Example:

  export OCI_DAC_ENDPOINT_OCID="ocid1.generativeaiendpoint.oc1.uk-london-1...."
  export OCI_DAC_COMPARTMENT_ID="ocid1.compartment.oc1...."
  export OCI_DAC_REGION="uk-london-1"
  export OCI_PROFILE="MY_DAC_PROFILE"
  pytest tests/integration/test_oci_dac_live.py -v

The OCIDs are intentionally read from env vars rather than checked
into the repo (CLAUDE.md privacy rule). Each test asserts the smallest
useful invariant for the layer under test, so the tests stay
informative regardless of which model is wired behind the DAC
(qwen, llama, etc.).
"""

from __future__ import annotations

import os

import pytest


# Skip everything if the DAC endpoint isn't configured.
_DAC_OCID = os.environ.get("OCI_DAC_ENDPOINT_OCID")
_DAC_COMPARTMENT = os.environ.get("OCI_DAC_COMPARTMENT_ID")
_DAC_REGION = os.environ.get("OCI_DAC_REGION", "us-chicago-1")
_OCI_PROFILE = os.environ.get("OCI_PROFILE", "DEFAULT")


pytestmark = pytest.mark.skipif(
    not (_DAC_OCID and _DAC_COMPARTMENT),
    reason=(
        "OCI DAC endpoint not configured. Set OCI_DAC_ENDPOINT_OCID + "
        "OCI_DAC_COMPARTMENT_ID + OCI_DAC_REGION + OCI_PROFILE to run."
    ),
)


@pytest.fixture
def dac_model() -> object:
    """Build an OCIModel pointed at the configured DAC endpoint."""
    pytest.importorskip("oci")
    from locus.models.providers.oci import OCIAuthType, OCIModel

    # The DAC endpoint OCID *is* the model_id for routing; OCIClient's
    # get_serving_mode() recognises the prefix and returns
    # DedicatedServingMode(endpoint_id=...).
    # Detect auth type from environment — default to API_KEY but honour
    # security_token profiles (e.g. BOAT-OC1) so the fixture works without
    # requiring a separate API-key profile just for DAC tests.
    auth_type_str = os.environ.get("OCI_AUTH_TYPE", "api_key").lower()
    auth_type = (
        OCIAuthType.SECURITY_TOKEN if auth_type_str == "security_token" else OCIAuthType.API_KEY
    )

    return OCIModel(
        model_id=_DAC_OCID,  # type: ignore[arg-type]  # filtered above
        compartment_id=_DAC_COMPARTMENT,
        profile_name=_OCI_PROFILE,
        auth_type=auth_type,
        # The SDK derives the service endpoint from the region.
        service_endpoint=(f"https://inference.generativeai.{_DAC_REGION}.oci.oraclecloud.com"),
        max_tokens=128,
    )


@pytest.mark.asyncio
async def test_dac_complete_returns_content(dac_model: object) -> None:
    """Non-streaming chat against the DAC returns a non-empty response."""
    from locus.core.messages import Message

    response = await dac_model.complete(  # type: ignore[attr-defined]
        messages=[Message.user("Reply with the single word 'OK'.")],
        tools=None,
    )
    assert response.message is not None
    content = response.message.content or ""
    assert content.strip(), (
        f"DAC complete() returned empty content. "
        f"usage={response.usage}, stop_reason={response.stop_reason}"
    )


@pytest.mark.asyncio
async def test_dac_stream_yields_chunks(dac_model: object) -> None:
    """Streaming chat against the DAC yields at least one content chunk
    and a final done event.

    Robust to endpoints that reject ``is_stream`` — the OCIModel.stream()
    fallback path emits a single content chunk with the full response
    in that case, which still satisfies the assertions.
    """
    from locus.core.events import ModelChunkEvent
    from locus.core.messages import Message

    chunks: list[ModelChunkEvent] = []
    done = False
    async for event in dac_model.stream(  # type: ignore[attr-defined]
        messages=[Message.user("Count from 1 to 3, one number per line.")],
        tools=None,
    ):
        chunks.append(event)
        if event.done:
            done = True

    assert done, "stream never emitted a done=True event"
    content_chunks = [c for c in chunks if c.content]
    assert content_chunks, f"stream emitted no content chunks. total_events={len(chunks)}"
    full_content = "".join(c.content or "" for c in content_chunks)
    assert full_content.strip(), "stream content is empty after concat"


@pytest.mark.asyncio
async def test_dac_via_get_model_routes_to_oci_model(dac_model: object) -> None:
    """``get_model("oci:<DAC OCID>")`` returns an ``OCIModel`` instance
    rather than ``OCIChatCompletionsModel`` (V1 transport can't speak DAC)."""
    from locus.models import get_model
    from locus.models.providers.oci import OCIModel

    model = get_model(
        f"oci:{_DAC_OCID}",
        compartment_id=_DAC_COMPARTMENT,
        profile_name=_OCI_PROFILE,
        service_endpoint=(f"https://inference.generativeai.{_DAC_REGION}.oci.oraclecloud.com"),
    )
    assert isinstance(model, OCIModel)
