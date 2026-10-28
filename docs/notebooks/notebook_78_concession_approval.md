# Customer-Support Concession Approval

Real support orgs gate costly concessions behind a tier-based escalation chain::

    Ticket intake (case history on file)
       │
       ▼
    Ticket analyst  (summarises what the customer is asking for and why)
       │
       ▼
    Impact analyst  (assesses concession cost + precedent + churn risk)
       │
       ▼
    Risk-tier router  ── score < 25 ──> auto-approve (small credit)
                      ── 25–49     ──> support-manager approval (interrupt)
                      ── 50–74     ──> manager + billing approval (two interrupts)
                      ── >= 75     ──> manager + billing + director approval (three interrupts)
       │
       ▼
    Decision recorder  (emits structured ConcessionDecision)

Each approval gate is a separate `interrupt()` so a reviewer can come
back to it later. The terminal node is SCRIBE, the support org's case
recorder: it emits a typed `ConcessionDecision` Pydantic model that files
into the concessions ledger without parsing. A large refund or contract
make-good spends real money and sets a precedent other customers will
cite, so the impact step is where you weigh the customer's standing and
lifetime value against the cost of the concession and the downside of a
denial (churn, escalation, public complaint).

- Risk-tier router is a plain conditional edge — no DSL, no policy file.
- Each gate is its own node — easy to add a tier, easy to re-order,
  easy to swap a human gate for an automated rule.
- `output_schema=ConcessionDecision` keeps the terminal artifact typed.

Run it (defaults to the bundled mock model; set `LOCUS_MODEL_PROVIDER` to `openai` / `anthropic` for a live model):

    python examples/notebook_78_concession_approval.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_78_concession_approval.py

Pin a strong-enough model for the structured ConcessionDecision schema:

    LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_78_concession_approval.py

## Source

````python
--8<-- "examples/notebook_78_concession_approval.py"
````
