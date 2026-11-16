# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus.observability.exporters.oci_apm`` — configuration only, no network."""

from __future__ import annotations

import pytest

from locus.observability.exporters import (
    ENV_OCI_APM_ENDPOINT,
    ENV_OCI_APM_PRIVATE_DATA_KEY,
    OCIAPMExport,
    oci_apm,
)


ENDPOINT = "https://aaaabbbb.apm-agt.eu-frankfurt-1.oci.oraclecloud.com"
KEY = "ZZZZ0000EXAMPLEKEY"


def _otlp_http_available() -> bool:
    try:
        import opentelemetry.exporter.otlp.proto.http.trace_exporter  # noqa: F401
    except ImportError:
        return False
    return True


class TestOciApmSettings:
    def test_endpoints_and_header(self) -> None:
        export = oci_apm(ENDPOINT, KEY)
        assert export.traces_endpoint == f"{ENDPOINT}/20200101/opentelemetry/private/v1/traces"
        assert export.metrics_endpoint == f"{ENDPOINT}/20200101/opentelemetry/private/v1/metrics"
        assert dict(export.headers) == {"Authorization": f"dataKey {KEY}"}
        assert export.timeout == 10.0

    def test_accepts_host_without_scheme_and_trailing_slash(self) -> None:
        bare = ENDPOINT.removeprefix("https://")
        assert oci_apm(bare + "/", KEY).traces_endpoint == oci_apm(ENDPOINT, KEY).traces_endpoint

    def test_accepts_a_full_otlp_url_and_normalises_it(self) -> None:
        export = oci_apm(f"{ENDPOINT}/20200101/opentelemetry/private/v1/traces", KEY)
        assert export.traces_endpoint == f"{ENDPOINT}/20200101/opentelemetry/private/v1/traces"
        assert export.metrics_endpoint.endswith("/v1/metrics")

    def test_rejects_plain_http(self) -> None:
        with pytest.raises(ValueError, match="https"):
            oci_apm("http://example.invalid", KEY)

    def test_rejects_an_endpoint_with_an_unknown_path(self) -> None:
        with pytest.raises(ValueError, match="host alone"):
            oci_apm(f"{ENDPOINT}/somewhere/else", KEY)

    def test_missing_values_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_OCI_APM_ENDPOINT, raising=False)
        monkeypatch.delenv(ENV_OCI_APM_PRIVATE_DATA_KEY, raising=False)
        with pytest.raises(ValueError, match=ENV_OCI_APM_ENDPOINT):
            oci_apm(None, KEY)
        with pytest.raises(ValueError, match=ENV_OCI_APM_PRIVATE_DATA_KEY):
            oci_apm(ENDPOINT, None)
        with pytest.raises(ValueError, match="timeout"):
            oci_apm(ENDPOINT, KEY, timeout=0)

    def test_environment_fallbacks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_OCI_APM_ENDPOINT, ENDPOINT)
        monkeypatch.setenv(ENV_OCI_APM_PRIVATE_DATA_KEY, f" {KEY} ")
        export = oci_apm()
        assert export.traces_endpoint.startswith(ENDPOINT)
        assert export.headers["Authorization"] == f"dataKey {KEY}"

    def test_repr_redacts_the_key(self) -> None:
        text = repr(oci_apm(ENDPOINT, KEY))
        assert KEY not in text
        assert "dataKey ***" in text
        assert isinstance(oci_apm(ENDPOINT, KEY), OCIAPMExport)


@pytest.mark.skipif(not _otlp_http_available(), reason="OTLP/HTTP exporter not installed")
class TestOciApmExporters:
    def test_span_exporter_is_otlp_http_with_the_header(self) -> None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = oci_apm(ENDPOINT, KEY).span_exporter()
        assert isinstance(exporter, OTLPSpanExporter)
        assert exporter._endpoint == f"{ENDPOINT}/20200101/opentelemetry/private/v1/traces"
        assert exporter._client._headers["authorization"] == f"dataKey {KEY}"
        exporter.shutdown()

    def test_metric_exporter_is_otlp_http_with_the_header(self) -> None:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        exporter = oci_apm(ENDPOINT, KEY).metric_exporter()
        assert isinstance(exporter, OTLPMetricExporter)
        assert exporter._endpoint == f"{ENDPOINT}/20200101/opentelemetry/private/v1/metrics"
        assert exporter._client._headers["authorization"] == f"dataKey {KEY}"
        exporter.shutdown()

    def test_processors_and_reader_types(self) -> None:
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

        export = oci_apm(ENDPOINT, KEY)
        batch = export.span_processor()
        simple = export.span_processor(batch=False)
        assert isinstance(batch, BatchSpanProcessor)
        assert isinstance(simple, SimpleSpanProcessor)
        batch.shutdown()
        simple.shutdown()
        reader = export.metric_reader(interval_ms=5_000)
        assert isinstance(reader, PeriodicExportingMetricReader)
        reader.shutdown()

    def test_install_builds_a_resource_with_the_service_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``install`` sets the globals; here they are captured instead of set."""
        from opentelemetry import trace as otel_trace

        captured: list[object] = []
        monkeypatch.setattr(otel_trace, "set_tracer_provider", captured.append)
        tracer_provider, meter_provider = oci_apm(ENDPOINT, KEY).install(
            service_name="support-agent",
            resource_attributes={"deployment.environment.name": "test"},
            metrics=False,
        )
        assert captured == [tracer_provider]
        assert meter_provider is None
        attributes = tracer_provider.resource.attributes
        assert attributes["service.name"] == "support-agent"
        assert attributes["deployment.environment.name"] == "test"
        tracer_provider.shutdown()
