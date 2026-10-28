# Live vendor integrations — PII discovery, data map, scan dispatch

The earlier notebooks used inline mock tools to keep the focus on agent
mechanics. Real privacy work calls real systems: a data-classification
feed to score an identifier, a data catalog to pull the records behind a
subject request, a scanning cloud to run a PII-discovery probe over a
data store. This notebook works through three vendor integrations and hands two of them
to a data-subject-request (DSAR) triage agent.

The classification and data-map integrations follow one convention: read
the vendor credential from the environment and take the live path when
it's set; otherwise return a deterministic, synthetic sample so the
notebooks run offline with no account. The offline sample uses the
return shape the live call is meant to produce, so the agent code doesn't
change when you replace a stub with the real vendor call.

- ``scan_for_pii`` — BigID/OneTrust-shaped identifier classification
  (``DATAMAP_API_KEY``).
- ``query_data_map`` — Collibra/Atlan-shaped data-catalog search
  (``DATAMAP_URL`` + ``DATAMAP_TOKEN``).
- ``scan_dataset_reference`` — an offline reference PII-discovery scan over
  a named data store. In this notebook it always returns its deterministic
  sample and does not read ``SCANNER_API_KEY``; a live version would send
  the scan to a classification cloud.

Run it:

```
.venv/bin/python examples/notebook_80_vendor_integrations.py
```

Without an OCI profile the model falls back to the bundled mock, and every vendor tool falls back to its offline sample, so this runs end-to-end with no credentials.
Set ``DATAMAP_API_KEY``, or ``DATAMAP_URL`` + ``DATAMAP_TOKEN``, to move the
classification or data-map tool onto its live path; in this notebook that
path is a stub that raises ``NotImplementedError`` until you add the vendor
call. The scan always stays on its offline reference path.

Prerequisites:

- The Agent-with-tools notebook.
- The specialist agents notebook (RIGHTSIZER) — introduces the ``Specialist`` class the STEWARD triage agent here is built on.

## Source

````python
--8<-- "examples/notebook_80_vendor_integrations.py"
````
