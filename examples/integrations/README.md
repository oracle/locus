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

| Module | Vendor shape | Credential(s) | Offline behavior | Live-path status |
|---|---|---|---|---|
| `threat_intel.py` | VirusTotal / GreyNoise | `VT_API_KEY` | sample IOC reputation | ⚠ written to VT v3 shape, **untested live** |
| `siem_query.py` | Splunk / Elastic | `SIEM_URL`, `SIEM_TOKEN` | sample events | ⚠ illustrative endpoint, **untested live** |
| `gpu_probe_dispatch.py` | offline reference only — the two live GPU clouds are **split** below | — | deterministic feature vector | reference dispatch; live lifecycle lives in `tulip-integrations` |
| ↳ RunPod (`tulip_integrations.compute.runpod`) | RunPod GPU pod | `RUNPOD_API_KEY` (+ `RUNPOD_PROBE_IMAGE`) | deterministic feature vector | ⚠ pod lifecycle real, needs a probe **image** you supply — **untested live** |
| ↳ Lambda Cloud (`tulip_integrations.compute.lambda_cloud`) | Lambda Cloud GPU instance | `LAMBDA_API_KEY` (+ `LAMBDA_REGION`, `LAMBDA_PROBE_RESULT_URL`) | deterministic feature vector | ⚠ launch/poll lifecycle real, needs a result **sink** you supply — **untested live** |
| `remote_timing.py` | any OpenAI-compatible endpoint | `OPENAI_API_KEY` | deterministic feature vector | ✓ **verified live** vs OpenAI gpt-4o-mini (measurement real; classifier still mock) |

**Honesty note:** only the offline sample paths and the timing math are
verified in CI. The live API branches are written to each vendor's
documented shape but have not been run against the real services — treat
them as starting points to validate with your own credentials, not as
battle-tested clients. The GPU dispatch is the least real: it can launch
hardware but has no probe to run until the CUDA artifact (clusiana) exists.

Run any module standalone to see the offline path:

```bash
python examples/integrations/threat_intel.py
python examples/integrations/siem_query.py
python examples/integrations/gpu_probe_dispatch.py
```

Or see them handed to a triage agent end-to-end in
`examples/notebook_70_vendor_integrations.py`.

## GPU probe dispatch — two separate providers

Inference fingerprinting (see `notebook_27`) can measure streaming-timing
features *where the hardware is* — by renting a GPU, running a co-located
probe against the target endpoint, and reading the feature vector the probe
emits:

```json
{"ttft_ms_p50": 38.2, "itl_ms_mean": 11.4, "itl_cv": 0.07, "tps_mean": 87.6}
```

RunPod and Lambda are **two different GPU clouds with two different
result-collection mechanisms**, so they are two separate modules in
`tulip-integrations` (`tulip_integrations.compute.runpod` and
`…compute.lambda_cloud`), behind two separate install extras
(`tulip-integrations[compute-runpod]` and `[compute-lambda]`). Core (this
SDK) ships only `dispatch_timing_probe_reference()` — the credential-free
*offline* dispatch that returns the sample vector. You pick a provider
explicitly; you never get one by accident.

**RunPod — pod + container image.** `runpod_probe(endpoint)` uses the RunPod
SDK to create a GPU pod from a **container image** (`RUNPOD_PROBE_IMAGE`,
default `tuliplabs/timing-probe:latest`), waits for the pod's output, parses
the feature vector out of it, and terminates the pod in a `finally`. The
probe *is* the image — you build and publish it. Credential: `RUNPOD_API_KEY`.

**Lambda Cloud — instance + result sink.** `lambda_probe(endpoint)` launches
a Lambda Cloud GPU instance over `httpx` (no extra SDK), then **polls a result
sink** the probe uploads its feature JSON to (`LAMBDA_PROBE_RESULT_URL` — an
S3 object or a small HTTP endpoint), and terminates the instance in a
`finally`. The probe runs on the instance and pushes its result out-of-band;
Lambda has no "wait for pod output" call, hence the sink. Credentials:
`LAMBDA_API_KEY` (+ `LAMBDA_REGION`).

Both live paths are honest integration code — the provision → probe → tear-down
control flow is complete — but **the probe artifact is yours to supply** (the
container image for RunPod, the sink-uploading probe for Lambda), and **the
live path is billable** (it spins up an H100-class GPU). Neither has been run
end-to-end against a real account, so both are flagged `UNVERIFIED LIVE PATH`
in source. With no credentials set, each returns the deterministic offline
sample, so the cookbook runs for free. Gate the live path behind explicit
approval and a provider spend limit.
