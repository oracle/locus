# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Exporter recipes for the backends an agent's traces land in.

The telemetry hook emits standard OpenTelemetry spans and metrics; any OTLP
backend receives them once an exporter is configured. Most backends need
nothing beyond the SDK's environment variables (``OTEL_EXPORTER_OTLP_ENDPOINT``
and friends). OCI Application Performance Monitoring needs two things the
environment variables do not express together — the APM domain's data
upload endpoint with the OpenTelemetry paths appended, and the private data
key as a ``dataKey`` authorization header — so :func:`oci_apm` builds them.

Nothing here is imported until asked for: the OTLP exporter classes load
inside the methods, so the module imports with the ``telemetry`` extra alone.
"""

from __future__ import annotations

import atexit
import logging
import os
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


_logger = logging.getLogger(__name__)

#: Environment variables :func:`oci_apm` reads when not given the values.
ENV_OCI_APM_ENDPOINT = "OCI_APM_DATA_UPLOAD_ENDPOINT"
ENV_OCI_APM_PRIVATE_DATA_KEY = "OCI_APM_PRIVATE_DATA_KEY"

#: The OpenTelemetry collector path prefix on an APM data upload endpoint.
_OTLP_PREFIX = "/20200101/opentelemetry/private"
_TRACES_PATH = f"{_OTLP_PREFIX}/v1/traces"
_METRICS_PATH = f"{_OTLP_PREFIX}/v1/metrics"


#: The OTLP endpoints each provider already exports to, so that a second
#: ``install()`` for the same domain adds no second processor or reader.
_INSTALLED: weakref.WeakKeyDictionary[Any, set[str]] = weakref.WeakKeyDictionary()


def _is_installed(provider: Any, endpoint: str) -> bool:
    return endpoint in _INSTALLED.get(provider, set())


def _mark_installed(provider: Any, endpoint: str) -> None:
    _INSTALLED.setdefault(provider, set()).add(endpoint)


def _global_tracer_provider(otel_trace: Any) -> Any | None:
    """The global ``TracerProvider`` an application set, or ``None`` for the API's placeholder."""
    provider = otel_trace.get_tracer_provider()
    return None if isinstance(provider, otel_trace.ProxyTracerProvider) else provider


def _global_meter_provider(otel_metrics: Any) -> Any | None:
    """The global ``MeterProvider`` an application set, or ``None`` for the API's placeholder."""
    provider = otel_metrics.get_meter_provider()
    # The placeholder is private to the API package; it is the one provider
    # that forwards to whatever is set later.
    return None if type(provider).__name__ == "_ProxyMeterProvider" else provider


def _base_endpoint(value: str) -> str:
    """The APM data upload endpoint, with any OpenTelemetry path stripped.

    Accepts the endpoint as the console shows it (``https://<id>.apm-agt.<region>
    .oci.oraclecloud.com``) or a full OTLP URL built from it.
    """
    text = value.strip()
    if not text:
        raise ValueError("OCI APM data upload endpoint is empty")
    if "://" not in text:
        text = f"https://{text}"
    parts = urlsplit(text)
    if parts.scheme != "https" or not parts.netloc:
        raise ValueError(f"OCI APM data upload endpoint must be an https URL, got {value!r}")
    path = parts.path.rstrip("/")
    marker = path.find("/20200101/")
    if marker >= 0:
        path = path[:marker]
    if path:
        raise ValueError(
            f"OCI APM data upload endpoint should be the host alone, got path {parts.path!r}"
        )
    return f"https://{parts.netloc}"


# A tuple subclass cannot declare a non-empty ``__slots__``, and the instance
# dict is where the bookkeeping below lives: the pair stays a pair.
class OCIAPMInstallation(tuple[Any, Any]):  # noqa: SLOT001
    """What one :meth:`OCIAPMExport.install` call set up.

    A ``(tracer_provider, meter_provider)`` pair — it unpacks as before —
    that also knows what the call created, so :meth:`force_flush` and
    :meth:`shutdown` reach it. That matters for a ``MeterProvider`` that was
    already in place: OpenTelemetry's ``MeterProvider.force_flush()`` and
    ``shutdown()`` visit only the readers given at construction, so the
    reader ``install()`` adds to an existing provider would otherwise never
    export its last interval. When something was added to a provider already
    in place, :meth:`shutdown` is also registered with :mod:`atexit`, so the
    exit flush happens without any user code.
    """

    _span_processor: Any | None
    _metric_reader: Any | None
    _owns_tracer_provider: bool
    _owns_meter_provider: bool
    _atexit: Callable[[], None] | None
    _shut: bool

    def __new__(
        cls,
        tracer_provider: Any,
        meter_provider: Any | None,
        *,
        span_processor: Any | None = None,
        metric_reader: Any | None = None,
        owns_tracer_provider: bool = False,
        owns_meter_provider: bool = False,
    ) -> OCIAPMInstallation:
        self = super().__new__(cls, (tracer_provider, meter_provider))
        self._span_processor = span_processor
        self._metric_reader = metric_reader
        self._owns_tracer_provider = owns_tracer_provider
        self._owns_meter_provider = owns_meter_provider
        self._atexit = None
        self._shut = False
        return self

    @property
    def tracer_provider(self) -> Any:
        """The ``TracerProvider`` the export goes through."""
        return self[0]

    @property
    def meter_provider(self) -> Any | None:
        """The ``MeterProvider`` the export goes through; ``None`` with ``metrics=False``."""
        return self[1]

    @property
    def added_to_existing_providers(self) -> bool:
        """Whether the call joined providers already in place rather than creating them."""
        return (self._span_processor is not None and not self._owns_tracer_provider) or (
            self._metric_reader is not None and not self._owns_meter_provider
        )

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        """Export what the processor and reader of this call hold now.

        Works on them directly, so the reader added to an existing
        ``MeterProvider`` — which that provider's own ``force_flush()``
        skips — exports too. ``True`` when every flush reported success.
        """
        ok = True
        if self._span_processor is not None:
            ok = bool(self._span_processor.force_flush(int(timeout_millis))) and ok
        if self._metric_reader is not None:
            flush = getattr(self._metric_reader, "force_flush", None)
            if callable(flush):
                ok = bool(flush(timeout_millis=timeout_millis)) and ok
            else:
                self._metric_reader.collect(timeout_millis=timeout_millis)
        return ok

    def shutdown(self, timeout_millis: float = 30_000) -> None:
        """Flush and stop what this call set up; safe to call more than once.

        Providers created by the call are shut down (their ``shutdown()``
        covers the processor and reader they were built with). On providers
        that were already in place, the metric reader the call added is shut
        down here — its final collection exports the last interval — and the
        span processor it added is flushed and left to that provider's own
        ``shutdown()``, which does visit added processors.
        """
        if self._shut:
            return
        self._shut = True
        if self._atexit is not None:
            atexit.unregister(self._atexit)
            self._atexit = None
        if self._owns_tracer_provider:
            self[0].shutdown()
        elif self._span_processor is not None:
            self._span_processor.force_flush(int(timeout_millis))
        if self._owns_meter_provider and self[1] is not None:
            self[1].shutdown(timeout_millis=timeout_millis)
        elif self._metric_reader is not None:
            self._metric_reader.shutdown(timeout_millis=timeout_millis)

    def register_atexit(self) -> None:
        """Register :meth:`shutdown` with :mod:`atexit`; ``shutdown()`` unregisters it.

        :meth:`OCIAPMExport.install` does this when it added to providers
        already in place. Harmless to call again: one registration is kept.
        """
        if self._atexit is None and not self._shut:
            handler = self.shutdown
            self._atexit = handler
            atexit.register(handler)


@dataclass(frozen=True)
class OCIAPMExport:
    """OTLP/HTTP export settings for one OCI APM domain.

    Attributes:
        traces_endpoint: The OTLP traces URL on the domain's data upload endpoint.
        metrics_endpoint: The OTLP metrics URL on the same endpoint.
        headers: The ``Authorization: dataKey …`` header both exporters send.
        timeout: Export timeout in seconds.
    """

    traces_endpoint: str
    metrics_endpoint: str
    headers: Mapping[str, str]
    timeout: float = 10.0

    def __repr__(self) -> str:  # the data key is a credential
        return (
            f"OCIAPMExport(traces_endpoint={self.traces_endpoint!r}, "
            f"metrics_endpoint={self.metrics_endpoint!r}, headers={{'Authorization': "
            f"'dataKey ***'}}, timeout={self.timeout!r})"
        )

    def span_exporter(self) -> Any:
        """An OTLP/HTTP (protobuf) span exporter aimed at the domain."""
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(
            endpoint=self.traces_endpoint, headers=dict(self.headers), timeout=int(self.timeout)
        )

    def span_processor(self, *, batch: bool = True) -> Any:
        """A span processor wrapping :meth:`span_exporter`.

        ``batch=True`` (the default) exports in the background; pass
        ``False`` for a simple processor that exports each span as it ends,
        which is what a short script or a test wants.
        """
        from opentelemetry.sdk.trace.export import (  # noqa: PLC0415
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )

        exporter = self.span_exporter()
        return BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)

    def metric_exporter(self) -> Any:
        """An OTLP/HTTP (protobuf) metric exporter aimed at the domain."""
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # noqa: PLC0415
            OTLPMetricExporter,
        )

        return OTLPMetricExporter(
            endpoint=self.metrics_endpoint, headers=dict(self.headers), timeout=int(self.timeout)
        )

    def metric_reader(self, *, interval_ms: float = 60_000) -> Any:
        """A periodic reader exporting through :meth:`metric_exporter`."""
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader  # noqa: PLC0415

        return PeriodicExportingMetricReader(
            self.metric_exporter(), export_interval_millis=interval_ms
        )

    def install(
        self,
        *,
        service_name: str = "locus-agent",
        resource_attributes: Mapping[str, Any] | None = None,
        metrics: bool = True,
        batch: bool = True,
        force: bool = False,
    ) -> OCIAPMInstallation:
        """Export to the domain through the global providers.

        With no global ``TracerProvider`` set yet, one is created with a
        Resource naming the service and set as the global; the same for the
        ``MeterProvider`` when ``metrics`` is on. OpenTelemetry refuses to
        replace a global provider once set, so a provider already in place
        — from an earlier call, another library, or the SDK's own
        configuration — is kept and given this domain's span processor and
        metric reader instead; its Resource (and so the service name) stays
        as it was. A second call for the same domain adds nothing: each
        provider gets one processor and one reader per endpoint.

        Args:
            service_name: ``service.name`` on the Resource — what APM shows
                as the service — when a provider is created here.
            resource_attributes: Further Resource attributes
                (``deployment.environment.name``, say), likewise.
            metrics: Also export the token and duration metrics.
            batch: Batch span export (``False`` exports each span as it ends).
            force: When the global ``MeterProvider`` in place cannot take a
                reader (an implementation without ``add_metric_reader``),
                build a ``MeterProvider`` of this domain's own instead of
                raising. That provider is *not* the global one — it is
                returned, and only instruments created from it reach the
                domain. The ``TracerProvider`` is never forced: every SDK
                tracer provider takes a span processor.

        Returns:
            An :class:`OCIAPMInstallation`: the ``(TracerProvider,
            MeterProvider | None)`` the export goes through — the global
            providers in every case but ``force`` — with ``force_flush()``
            and ``shutdown()`` that reach what this call added. Call its
            ``shutdown()`` before the process exits so the last batch and the
            last metrics interval are exported; when the call added to
            providers already in place, that ``shutdown()`` is also registered
            with :mod:`atexit`, since a ``MeterProvider`` does not flush a
            reader added after its construction on its own.

        Raises:
            RuntimeError: A global provider is set that cannot take this
                domain's processor or reader (and, for metrics, ``force`` is
                off).
        """
        from opentelemetry import metrics as otel_metrics  # noqa: PLC0415
        from opentelemetry import trace as otel_trace  # noqa: PLC0415
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415

        attributes: dict[str, Any] = {"service.name": service_name}
        if resource_attributes:
            attributes.update(resource_attributes)
        resource = Resource.create(attributes)

        span_processor: Any | None = None
        owns_tracer_provider = False
        tracer_provider = _global_tracer_provider(otel_trace)
        if tracer_provider is None:
            span_processor = self.span_processor(batch=batch)
            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(span_processor)
            _mark_installed(tracer_provider, self.traces_endpoint)
            otel_trace.set_tracer_provider(tracer_provider)
            owns_tracer_provider = True
        elif not _is_installed(tracer_provider, self.traces_endpoint):
            if not hasattr(tracer_provider, "add_span_processor"):
                raise RuntimeError(
                    f"the global TracerProvider ({type(tracer_provider).__name__}) takes no "
                    "span processor; add the exporter from span_exporter() to it yourself"
                )
            span_processor = self.span_processor(batch=batch)
            tracer_provider.add_span_processor(span_processor)
            _mark_installed(tracer_provider, self.traces_endpoint)

        meter_provider: Any | None = None
        metric_reader: Any | None = None
        owns_meter_provider = False
        if metrics:
            from opentelemetry.sdk.metrics import MeterProvider  # noqa: PLC0415

            meter_provider = _global_meter_provider(otel_metrics)
            if meter_provider is None:
                metric_reader = self.metric_reader()
                meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
                _mark_installed(meter_provider, self.metrics_endpoint)
                otel_metrics.set_meter_provider(meter_provider)
                owns_meter_provider = True
            elif not _is_installed(meter_provider, self.metrics_endpoint):
                if hasattr(meter_provider, "add_metric_reader"):
                    # The provider's own force_flush() / shutdown() skip a
                    # reader added here; the installation's reach it.
                    metric_reader = self.metric_reader()
                    meter_provider.add_metric_reader(metric_reader)
                    _mark_installed(meter_provider, self.metrics_endpoint)
                elif force:
                    _logger.warning(
                        "OCI APM: the global MeterProvider (%s) takes no metric reader; the "
                        "MeterProvider returned exports to the domain but is not the global one",
                        type(meter_provider).__name__,
                    )
                    metric_reader = self.metric_reader()
                    meter_provider = MeterProvider(
                        resource=resource, metric_readers=[metric_reader]
                    )
                    _mark_installed(meter_provider, self.metrics_endpoint)
                    owns_meter_provider = True
                else:
                    raise RuntimeError(
                        f"the global MeterProvider ({type(meter_provider).__name__}) takes no "
                        "metric reader after construction; pass force=True for a provider of "
                        "this domain's own (not global), or metrics=False"
                    )
        installation = OCIAPMInstallation(
            tracer_provider,
            meter_provider,
            span_processor=span_processor,
            metric_reader=metric_reader,
            owns_tracer_provider=owns_tracer_provider,
            owns_meter_provider=owns_meter_provider,
        )
        if installation.added_to_existing_providers:
            installation.register_atexit()
        return installation


def oci_apm(
    data_upload_endpoint: str | None = None,
    private_data_key: str | None = None,
    *,
    timeout: float = 10.0,
) -> OCIAPMExport:
    """Export settings for an OCI Application Performance Monitoring domain.

    Args:
        data_upload_endpoint: The domain's *Data upload endpoint* as the
            console shows it. Defaults to ``OCI_APM_DATA_UPLOAD_ENDPOINT``.
        private_data_key: One of the domain's *private* data keys. Defaults
            to ``OCI_APM_PRIVATE_DATA_KEY``. Public keys are for browser and
            mobile agents, not for a server-side SDK.
        timeout: Export timeout in seconds.

    Returns:
        An :class:`OCIAPMExport`; call ``install()`` on it (it sets the global
        providers, or joins the ones already set), or wire its
        ``span_processor()`` / ``metric_reader()`` into providers you own.

    Raises:
        ValueError: No endpoint or key, or an endpoint that is not an https host.

    Example::

        from locus.observability.exporters import oci_apm

        apm = oci_apm().install(service_name="support-agent")
        agent = Agent(model="oci:openai.gpt-4.1", hooks=[TelemetryHook()])
        ...
        apm.shutdown()  # flushes the last batch and the last metrics interval
    """
    endpoint = data_upload_endpoint or os.environ.get(ENV_OCI_APM_ENDPOINT)
    key = private_data_key or os.environ.get(ENV_OCI_APM_PRIVATE_DATA_KEY)
    if not endpoint:
        raise ValueError(
            f"OCI APM data upload endpoint missing: pass data_upload_endpoint= or set "
            f"{ENV_OCI_APM_ENDPOINT}"
        )
    if not key:
        raise ValueError(
            f"OCI APM private data key missing: pass private_data_key= or set "
            f"{ENV_OCI_APM_PRIVATE_DATA_KEY}"
        )
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    base = _base_endpoint(endpoint)
    return OCIAPMExport(
        traces_endpoint=f"{base}{_TRACES_PATH}",
        metrics_endpoint=f"{base}{_METRICS_PATH}",
        headers={"Authorization": f"dataKey {key.strip()}"},
        timeout=timeout,
    )


__all__ = [
    "ENV_OCI_APM_ENDPOINT",
    "ENV_OCI_APM_PRIVATE_DATA_KEY",
    "OCIAPMExport",
    "OCIAPMInstallation",
    "oci_apm",
]
