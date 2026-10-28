# Model & hardware fingerprinting via timing side-channels

A CISO is about to sign a $2M/year contract with a vendor claiming to serve
"GPT-4o on H100 clusters." Before signing, she wants to verify: is the model
actually GPT-4o (or a smaller substitute), and is it running on H100 (or shared
consumer-grade GPU)?

Timing side-channels are the only non-intrusive signal available to a buyer with
black-box API access: TTFT (time-to-first-token), inter-token latency, and tail
percentiles differ significantly across model families and hardware classes
because of differences in FLOPs, KV-cache behaviour, and memory bandwidth.
``measure_endpoint_timing`` collects timed token streams into a feature vector;
``fingerprint_to_finding`` classifies it and grounds the result — a grounded
``FingerprintFinding`` if feature coverage clears the threshold, an
``Abstention`` if too few features were observed. The ``AuditTrail`` records the
probe and its grounding decision as JSONL — evidence for a vendor SLA dispute or
a procurement audit.

Runs offline by default. Set ``OPENAI_API_KEY`` (and optional ``TIMING_BASE_URL``
for any OpenAI-compatible endpoint) to use the live streaming path.

Run it:

    python packages/locus-sdk-security/examples/notebook_88_model_fingerprint.py

## Source

````python
--8<-- "examples/notebook_88_model_fingerprint.py"
````
