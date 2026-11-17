# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``locus.observability.exporters.oci_apm`` — configuration only, no network."""

from __future__ import annotations

from typing import Any

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

    def test_rejects_an_empty_endpoint(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            oci_apm("  ", KEY)

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


class _Globals:
    """Stand-ins for the OpenTelemetry globals, so ``install`` never touches the real ones."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry import trace as otel_trace

        self.tracer_provider: object = otel_trace.ProxyTracerProvider()
        self.meter_provider: object = otel_metrics.get_meter_provider().__class__()
        monkeypatch.setattr(otel_trace, "get_tracer_provider", lambda: self.tracer_provider)
        monkeypatch.setattr(otel_trace, "set_tracer_provider", self._set_tracer_provider)
        monkeypatch.setattr(otel_metrics, "get_meter_provider", lambda: self.meter_provider)
        monkeypatch.setattr(otel_metrics, "set_meter_provider", self._set_meter_provider)
        self.set_calls = 0

    def _set_tracer_provider(self, provider: object) -> None:
        self.tracer_provider = provider
        self.set_calls += 1

    def _set_meter_provider(self, provider: object) -> None:
        self.meter_provider = provider
        self.set_calls += 1


def _processors(tracer_provider: Any) -> list[Any]:
    return list(tracer_provider._active_span_processor._span_processors)


def _readers(meter_provider: Any) -> list[Any]:
    return list(meter_provider._measurement_consumer._reader_storages)


@pytest.mark.skipif(not _otlp_http_available(), reason="OTLP/HTTP exporter not installed")
class TestInstall:
    def test_sets_the_globals_when_none_are_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.trace import TracerProvider

        globals_ = _Globals(monkeypatch)
        tracer_provider, meter_provider = oci_apm(ENDPOINT, KEY).install(
            service_name="support-agent",
            resource_attributes={"deployment.environment.name": "test"},
        )
        assert isinstance(tracer_provider, TracerProvider)
        assert isinstance(meter_provider, MeterProvider)
        assert globals_.tracer_provider is tracer_provider
        assert globals_.meter_provider is meter_provider
        attributes = tracer_provider.resource.attributes
        assert attributes["service.name"] == "support-agent"
        assert attributes["deployment.environment.name"] == "test"
        assert len(_processors(tracer_provider)) == 1
        assert len(_readers(meter_provider)) == 1
        tracer_provider.shutdown()
        meter_provider.shutdown()

    def test_metrics_off_installs_no_meter_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        globals_ = _Globals(monkeypatch)
        tracer_provider, meter_provider = oci_apm(ENDPOINT, KEY).install(metrics=False)
        assert meter_provider is None
        assert globals_.meter_provider.__class__.__name__ == "_ProxyMeterProvider"
        tracer_provider.shutdown()

    def test_joins_the_providers_already_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A global provider is kept and given the processor / reader; nothing is replaced."""
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        globals_ = _Globals(monkeypatch)
        existing_tp = TracerProvider(resource=Resource.create({"service.name": "first"}))
        existing_mp = MeterProvider()
        globals_.tracer_provider, globals_.meter_provider = existing_tp, existing_mp

        tracer_provider, meter_provider = oci_apm(ENDPOINT, KEY).install(service_name="second")
        assert tracer_provider is existing_tp
        assert meter_provider is existing_mp
        assert globals_.set_calls == 0
        assert tracer_provider.resource.attributes["service.name"] == "first"
        assert len(_processors(tracer_provider)) == 1
        assert len(_readers(meter_provider)) == 1
        existing_tp.shutdown()
        existing_mp.shutdown()

    def test_a_second_install_for_the_same_domain_adds_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _Globals(monkeypatch)
        first = oci_apm(ENDPOINT, KEY).install()
        second = oci_apm(ENDPOINT, KEY).install()
        assert second == first
        assert len(_processors(first[0])) == 1
        assert len(_readers(first[1])) == 1
        # Another domain on the same providers gets its own processor and reader.
        other = oci_apm("https://other.apm-agt.eu-frankfurt-1.oci.oraclecloud.com", KEY).install()
        assert other == first
        assert len(_processors(first[0])) == 2
        assert len(_readers(first[1])) == 2
        first[0].shutdown()
        first[1].shutdown()

    def test_a_meter_provider_that_takes_no_reader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ``add_metric_reader`` on the global: a clear error, or ``force`` for an own one."""
        from opentelemetry.metrics import NoOpMeterProvider
        from opentelemetry.sdk.metrics import MeterProvider

        globals_ = _Globals(monkeypatch)
        globals_.meter_provider = NoOpMeterProvider()
        with pytest.raises(RuntimeError, match="force=True"):
            oci_apm(ENDPOINT, KEY).install()
        tracer_provider, meter_provider = oci_apm(ENDPOINT, KEY).install(force=True)
        assert isinstance(meter_provider, MeterProvider)
        assert globals_.meter_provider.__class__ is NoOpMeterProvider  # not made global
        assert len(_readers(meter_provider)) == 1
        tracer_provider.shutdown()
        meter_provider.shutdown()

    def test_a_tracer_provider_that_takes_no_processor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from opentelemetry.trace import NoOpTracerProvider

        globals_ = _Globals(monkeypatch)
        globals_.tracer_provider = NoOpTracerProvider()
        with pytest.raises(RuntimeError, match="span processor"):
            oci_apm(ENDPOINT, KEY).install(metrics=False)


# ---------------------------------------------------------------------------
# The installation: what install() returns flushes and shuts down what it added
# ---------------------------------------------------------------------------


def _recorders(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """Fake exporters on ``OCIAPMExport`` recording what reached them; no OTLP needed."""
    from opentelemetry.sdk.metrics.export import MetricExporter
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class _Spans(SpanExporter):
        def __init__(self) -> None:
            self.spans: list[Any] = []
            self.shut = False

        def export(self, spans: Any) -> Any:
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            self.shut = True

    class _Metrics(MetricExporter):
        def __init__(self) -> None:
            super().__init__()
            self.exports: list[Any] = []
            self.shut = False

        def export(self, metrics_data: Any, timeout_millis: float = 10_000, **kwargs: Any) -> Any:
            from opentelemetry.sdk.metrics.export import MetricExportResult

            self.exports.append(metrics_data)
            return MetricExportResult.SUCCESS

        def force_flush(self, timeout_millis: float = 10_000) -> bool:
            return True

        def shutdown(self, timeout_millis: float = 30_000, **kwargs: Any) -> None:
            self.shut = True

        def instrument_names(self) -> set[str]:
            return {
                metric.name
                for data in self.exports
                for rm in data.resource_metrics
                for sm in rm.scope_metrics
                for metric in sm.metrics
            }

    spans, metrics = _Spans(), _Metrics()
    monkeypatch.setattr(OCIAPMExport, "span_exporter", lambda self: spans)
    monkeypatch.setattr(OCIAPMExport, "metric_exporter", lambda self: metrics)
    return spans, metrics


class _Atexit:
    def __init__(self) -> None:
        self.registered: list[Any] = []
        self.unregistered: list[Any] = []

    def register(self, fn: Any) -> Any:
        self.registered.append(fn)
        return fn

    def unregister(self, fn: Any) -> None:
        self.unregistered.append(fn)


class TestInstallation:
    def test_unpacks_and_compares_as_the_pair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from locus.observability.exporters import OCIAPMInstallation

        _Globals(monkeypatch)
        _recorders(monkeypatch)
        installation = oci_apm(ENDPOINT, KEY).install(batch=False)
        assert isinstance(installation, OCIAPMInstallation)
        tracer_provider, meter_provider = installation
        assert installation == (tracer_provider, meter_provider)
        assert installation.tracer_provider is tracer_provider is installation[0]
        assert installation.meter_provider is meter_provider is installation[1]
        assert installation.added_to_existing_providers is False
        installation.shutdown()

    def test_shutdown_flushes_the_reader_added_to_an_existing_meter_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The SDK's ``MeterProvider.shutdown()`` visits constructor-time readers
        only; the installation's reaches the one install() added."""
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        from opentelemetry.sdk.trace import TracerProvider

        from locus.observability import exporters as exporters_mod

        fake_atexit = _Atexit()
        monkeypatch.setattr(exporters_mod, "atexit", fake_atexit)
        globals_ = _Globals(monkeypatch)
        existing_tp = TracerProvider()
        existing_mp = MeterProvider(metric_readers=[InMemoryMetricReader()])
        globals_.tracer_provider, globals_.meter_provider = existing_tp, existing_mp
        spans, metrics = _recorders(monkeypatch)

        installation = oci_apm(ENDPOINT, KEY).install(batch=False)
        assert installation.added_to_existing_providers is True
        assert fake_atexit.registered == [installation.shutdown]

        existing_mp.get_meter("test").create_counter("requests.total").add(1)
        existing_tp.get_tracer("test").start_span("probe").end()
        assert [s.name for s in spans.spans] == ["probe"]
        assert metrics.exports == []

        installation.shutdown()
        assert "requests.total" in metrics.instrument_names()
        assert metrics.shut is True
        # The span processor belongs to the provider that was already there:
        # flushed here, shut down by that provider's own shutdown().
        assert spans.shut is False
        assert fake_atexit.unregistered == [installation.shutdown]

        installation.shutdown()  # idempotent
        assert len(metrics.exports) == 1
        existing_tp.shutdown()
        assert spans.shut is True
        existing_mp.shutdown()

    def test_force_flush_exports_the_added_reader_now(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.trace import TracerProvider

        from locus.observability import exporters as exporters_mod

        monkeypatch.setattr(exporters_mod, "atexit", _Atexit())
        globals_ = _Globals(monkeypatch)
        globals_.tracer_provider, globals_.meter_provider = TracerProvider(), MeterProvider()
        _, metrics = _recorders(monkeypatch)
        installation = oci_apm(ENDPOINT, KEY).install(batch=False)
        globals_.meter_provider.get_meter("test").create_counter("tokens").add(3)
        assert installation.force_flush() is True
        assert "tokens" in metrics.instrument_names()
        installation.shutdown()
        globals_.tracer_provider.shutdown()
        globals_.meter_provider.shutdown()

    def test_shutdown_of_providers_it_created_shuts_them_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from locus.observability import exporters as exporters_mod

        fake_atexit = _Atexit()
        monkeypatch.setattr(exporters_mod, "atexit", fake_atexit)
        _Globals(monkeypatch)
        spans, metrics = _recorders(monkeypatch)
        installation = oci_apm(ENDPOINT, KEY).install(batch=False)
        assert fake_atexit.registered == []  # the SDK's own atexit covers created providers
        installation.meter_provider.get_meter("test").create_counter("calls").add(1)
        installation.tracer_provider.get_tracer("test").start_span("probe").end()
        installation.shutdown()
        assert spans.shut is True
        assert metrics.shut is True
        assert "calls" in metrics.instrument_names()
        installation.shutdown()  # idempotent, no second export
        assert len(metrics.exports) == 1

    def test_register_atexit_is_explicit_and_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from locus.observability import exporters as exporters_mod

        fake_atexit = _Atexit()
        monkeypatch.setattr(exporters_mod, "atexit", fake_atexit)
        _Globals(monkeypatch)
        _recorders(monkeypatch)
        installation = oci_apm(ENDPOINT, KEY).install(batch=False)
        installation.register_atexit()
        installation.register_atexit()
        assert fake_atexit.registered == [installation.shutdown]
        installation.shutdown()
        assert fake_atexit.unregistered == [installation.shutdown]
        installation.register_atexit()  # after shutdown: nothing to register
        assert len(fake_atexit.registered) == 1

    def test_a_second_install_for_the_same_domain_has_nothing_of_its_own(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from locus.observability import exporters as exporters_mod

        monkeypatch.setattr(exporters_mod, "atexit", _Atexit())
        _Globals(monkeypatch)
        _, metrics = _recorders(monkeypatch)
        first = oci_apm(ENDPOINT, KEY).install(batch=False)
        second = oci_apm(ENDPOINT, KEY).install(batch=False)
        assert second == first
        assert second.added_to_existing_providers is False
        assert second.force_flush() is True
        second.shutdown()  # nothing added by this call: nothing to stop
        assert metrics.shut is False
        first.shutdown()
        assert metrics.shut is True

    def test_force_flush_collects_from_a_reader_without_force_flush(self) -> None:
        from locus.observability.exporters import OCIAPMInstallation

        class _CollectOnly:
            def __init__(self) -> None:
                self.collected: list[float] = []

            def collect(self, timeout_millis: float = 10_000) -> None:
                self.collected.append(timeout_millis)

        reader = _CollectOnly()
        installation = OCIAPMInstallation(object(), object(), metric_reader=reader)
        assert installation.force_flush(timeout_millis=250) is True
        assert reader.collected == [250]
