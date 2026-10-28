# Customer Support Concession Approval

Support organizations gate costly concessions behind a tier-based escalation chain: the more a concession costs and the more precedent it sets, the more approvals it takes.

{{ locus_diagram concession-approval }}

Each approval gate is a separate `interrupt()` so a reviewer can come
back to it later. The terminal node, the decision recorder (the
`scribe-decision-recorder` agent in the source), emits a typed
`ConcessionDecision` Pydantic model that files into the concessions
ledger without parsing. A large refund or contract make-good commits
real money and sets a precedent other customers will cite, so the impact
step is where you weigh the customer's standing and lifetime value
against the cost of the concession and the downside of a denial (churn,
escalation, public complaint).

- Risk-tier router is a plain conditional edge — no DSL, no policy file.
- Each gate is its own node — easy to add a tier, easy to re-order,
  easy to swap a human gate for an automated rule.
- `output_schema=ConcessionDecision` keeps the terminal artifact typed.

Run it (OCI Generative AI with an OCI profile, otherwise the bundled mock model; set `LOCUS_MODEL_PROVIDER` to force a provider):

    python examples/notebook_78_concession_approval.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_78_concession_approval.py

Pin a strong-enough model for the structured `ConcessionDecision` schema:

    LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o \
      python examples/notebook_78_concession_approval.py

## Source

````python
--8<-- "examples/notebook_78_concession_approval.py"
````
