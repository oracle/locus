# Vendor integrations

Worked integrations with the external systems an agentic-cybersecurity
workflow actually touches. Each is a real client for a real vendor, written
to the same convention so the cookbook stays runnable on a clean machine.

## The bring-your-own-credentials convention

Every integration reads its credential from the environment:

- **Credential set** → the live API is called.
- **Credential unset** → a deterministic, benign sample is returned (RFC 5737
  documentation addresses, `*.example` domains, the EICAR test hash), so the
  example runs offline with no account and no network.

The return *shape* is identical on both paths, so an agent's downstream
reasoning doesn't change between the offline demo and a live deployment.

| Module | Vendor shape | Credential(s) | Offline behavior |
|---|---|---|---|
| `threat_intel.py` | VirusTotal / GreyNoise | `VT_API_KEY` | sample IOC reputation |
| `siem_query.py` | Splunk / Elastic | `SIEM_URL`, `SIEM_TOKEN` | sample events |
| `gpu_probe_dispatch.py` | RunPod / Lambda | `RUNPOD_API_KEY` or `LAMBDA_API_KEY` | deterministic feature vector |

Run any module standalone to see the offline path:

```bash
python examples/integrations/threat_intel.py
python examples/integrations/siem_query.py
python examples/integrations/gpu_probe_dispatch.py
```

Or see them handed to a triage agent end-to-end in
`examples/notebook_70_vendor_integrations.py`.

## GPU probe dispatch — the probe artifact

Inference fingerprinting (see `notebook_27`) measures streaming-timing
features *where the hardware is*. `dispatch_timing_probe()` orchestrates the
GPU-cloud lifecycle; you supply the probe that does the measuring. It must
emit a JSON feature vector:

```json
{"ttft_ms_p50": 38.2, "itl_ms_mean": 11.4, "itl_cv": 0.07, "tps_mean": 87.6}
```

- **RunPod** — package the probe as a container image, set `_PROBE_IMAGE`,
  and the dispatcher creates a pod, waits for its output, parses the feature
  vector, and terminates the pod.
- **Lambda** — the probe uploads its feature JSON to a result sink (an S3
  object or small HTTP endpoint); set `LAMBDA_PROBE_RESULT_URL` and the
  dispatcher launches an instance, polls the sink, and terminates the
  instance.

The live paths are illustrative integration code: the control flow is
complete and honest, but the probe artifact is yours to provide. Without
credentials, `dispatch_timing_probe()` returns the offline sample.
