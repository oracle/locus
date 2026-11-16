# Observability: exporting Locus traces and metrics

`TelemetryHook` emits OpenTelemetry spans and metrics under the GenAI
semantic conventions — one `invoke_agent` span per run, a `chat` span per
model call, an `execute_tool` span per tool call, `gen_ai.*` attributes and
the `gen_ai.client.inference.usage.*` token counters. Any OTLP backend
receives them; this directory holds the OCI Application Performance
Monitoring pieces. The recipes for Jaeger, Grafana Tempo with Prometheus,
and SigNoz are on the
[Observability page](https://locusagents.ai/concepts/observability/#export).

## OCI Application Performance Monitoring

1. **An APM domain.** Console → Observability & Management → Application
   Performance Monitoring → Administration → *Create APM domain* (a free
   domain is enough to start; one per tenancy), or:

   ```bash
   oci apm-control-plane apm-domain create \
       --compartment-id <compartment OCID> --display-name locus-agents --is-free-tier true
   ```

2. **Its data upload endpoint and a private data key.** Both are on the
   domain's page (Administration → Data keys), or:

   ```bash
   oci apm-control-plane apm-domain get --apm-domain-id <domain OCID> \
       --query 'data."data-upload-endpoint"' --raw-output
   oci apm-control-plane data-key list --apm-domain-id <domain OCID> --data-key-type PRIVATE \
       --query 'data[0].value' --raw-output
   ```

   Private keys are for server-side SDKs; public keys are for browser and
   mobile agents and are not accepted by `oci_apm`.

3. **Export.** Put the two values in the environment and let `oci_apm`
   build the exporters and set the providers (an application that has
   already set its global providers keeps them: `install()` adds the
   domain's span processor and metric reader to them instead):

   ```bash
   export OCI_APM_DATA_UPLOAD_ENDPOINT=https://<domain>.apm-agt.<region>.oci.oraclecloud.com
   export OCI_APM_PRIVATE_DATA_KEY=<private data key>
   ```

   ```python
   from locus.agent import Agent
   from locus.hooks.builtin import TelemetryHook
   from locus.observability.exporters import oci_apm

   tracer_provider, meter_provider = oci_apm().install(service_name="support-agent")

   agent = Agent(model="oci:openai.gpt-4.1", name="support-desk", tools=[...], hooks=[TelemetryHook()])
   result = agent.run_sync("Look up order CS-1042 and reply with its status.")

   tracer_provider.shutdown()   # flushes the last batch
   meter_provider.shutdown()
   ```

   The exporters post OTLP/HTTP (protobuf) to
   `<endpoint>/20200101/opentelemetry/private/v1/traces` and `/v1/metrics`
   with `Authorization: dataKey <key>`.

4. **Look.** Trace Explorer (Application Performance Monitoring → Trace
   Explorer, the domain selected) lists the run as the `invoke_agent
   support-desk` span under service `support-agent`, with its `chat` and
   `execute_tool` children; the `gen_ai.*` attributes are span dimensions:

   ```text
   show (spans) gen_ai.request.model, sum(gen_ai.usage.input_tokens), sum(gen_ai.usage.output_tokens)
   where OperationName like 'chat%' group by gen_ai.request.model
   ```

## The dashboard

`oci_apm_dashboard.json` is a Management Dashboard import file with six
tiles, each a Trace Explorer query over the spans the hook exports:

| Tile | Spans | Shows |
|---|---|---|
| Agent runs | `invoke_agent*` | runs and failures over time |
| Run latency by agent | `invoke_agent*` | count, average and p95 duration, failures per agent |
| Tool latency and errors | `execute_tool*` | calls, average and p95 duration, errors per `gen_ai.tool.name` |
| Token spend by model | `chat*` | model calls, input and output tokens per `gen_ai.request.model` and provider |
| Token spend over time | `chat*` | input and output tokens in 5-minute buckets |
| Model call latency | `chat*` | average and p95 duration, errors per model |

Replace the `${compartmentId}` placeholders with the OCID of the
compartment that holds the APM domain (and `${apmDomainId}` with the
domain's OCID), then import:

```bash
sed -e 's|${compartmentId}|<compartment OCID>|g' \
    -e 's|${apmDomainId}|<APM domain OCID>|g' \
    oci_apm_dashboard.json > dashboard.json
oci management-dashboard dashboard import --from-json file://dashboard.json
```

The dashboard then appears under Observability & Management → Dashboards.
The queries are the portable part: each tile's `queryText` runs as it is
in Trace Explorer, so a tile can be rebuilt by hand from it if a console
version renders widgets differently.
