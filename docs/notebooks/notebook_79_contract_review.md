# Vendor DPA and Data-Privacy Review

Reviewing a data processor's Data Processing Agreement involves multiple
stakeholders working in parallel, then a back-and-forth negotiation
phase, then sign-off.

{{ locus_diagram contract-review }}

The processor here is a marketing-analytics SaaS that ingests your
customers' personal data, so a weak erasure SLA or a vague
international-transfer mechanism is a real compliance gap, not a
paperwork detail: the vendor's breach-notification window must fit
inside the 72-hour GDPR Art. 33 timeline the controller owes its own
supervisory authority. The DPO — the privacy office's data-protection
officer — writes the typed sign-off.

- `Send`: three reviewers run concurrently.
- `add_conditional_edges` with cycles enabled: negotiation can loop
  back to re-review when terms change. Hard cap of four review rounds
  (three negotiation passes).
- `interrupt()`: negotiation step pauses for human counsel to edit terms.
- `output_schema=ContractDecision`: typed terminal artifact.

Run it (OCI Generative AI with an OCI profile, otherwise the bundled mock model; set `LOCUS_MODEL_PROVIDER` to force a provider):

    python examples/notebook_79_contract_review.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_79_contract_review.py

Pin a model that supports structured output for the `ContractDecision` schema:

    LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o \
      python examples/notebook_79_contract_review.py

## Source

````python
--8<-- "examples/notebook_79_contract_review.py"
````
