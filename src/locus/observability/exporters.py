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

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


#: Environment variables :func:`oci_apm` reads when not given the values.
ENV_OCI_APM_ENDPOINT = "OCI_APM_DATA_UPLOAD_ENDPOINT"
ENV_OCI_APM_PRIVATE_DATA_KEY = "OCI_APM_PRIVATE_DATA_KEY"

#: The OpenTelemetry collector path prefix on an APM data upload endpoint.
_OTLP_PREFIX = "/20200101/opentelemetry/private"
_TRACES_PATH = f"{_OTLP_PREFIX}/v1/traces"
_METRICS_PATH = f"{_OTLP_PREFIX}/v1/metrics"


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
    ) -> tuple[Any, Any | None]:
        """Set the global tracer (and meter) provider to export to the domain.

        Args:
            service_name: ``service.name`` on the Resource — what APM shows
                as the service.
            resource_attributes: Further Resource attributes
                (``deployment.environment.name``, say).
            metrics: Also install a ``MeterProvider`` for the token and
                duration metrics.
            batch: Batch span export (``False`` exports each span as it ends).

        Returns:
            The ``(TracerProvider, MeterProvider | None)`` installed. Call
            ``shutdown()`` on them before the process exits so the last batch
            is flushed.
        """
        from opentelemetry import metrics as otel_metrics  # noqa: PLC0415
        from opentelemetry import trace as otel_trace  # noqa: PLC0415
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415

        attributes: dict[str, Any] = {"service.name": service_name}
        if resource_attributes:
            attributes.update(resource_attributes)
        resource = Resource.create(attributes)

        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(self.span_processor(batch=batch))
        otel_trace.set_tracer_provider(tracer_provider)

        meter_provider: Any | None = None
        if metrics:
            from opentelemetry.sdk.metrics import MeterProvider  # noqa: PLC0415

            meter_provider = MeterProvider(resource=resource, metric_readers=[self.metric_reader()])
            otel_metrics.set_meter_provider(meter_provider)
        return tracer_provider, meter_provider


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
        An :class:`OCIAPMExport`; call ``install()`` on it, or wire its
        ``span_processor()`` / ``metric_reader()`` into providers you own.

    Raises:
        ValueError: No endpoint or key, or an endpoint that is not an https host.

    Example::

        from locus.observability.exporters import oci_apm

        tracer_provider, meter_provider = oci_apm().install(
            service_name="support-agent"
        )
        agent = Agent(model="oci:openai.gpt-4.1", hooks=[TelemetryHook()])
        ...
        tracer_provider.shutdown()
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
    "oci_apm",
]
