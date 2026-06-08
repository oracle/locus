# Procurement Approval

Real procurement workflows have a threshold-based escalation chain::

    Request submitted
       │
       ▼
    Justifier  (drafts business justification)
       │
       ▼
    Vendor analyst  (validates vendor + pricing)
       │
       ▼
    Tier router   ── < $1k     ──> auto-approve
                  ── $1k-$10k  ──> manager approval (interrupt)
                  ── $10k-$100k──> manager + finance approval (two interrupts)
                  ── > $100k   ──> manager + finance + CFO approval (three interrupts)
       │
       ▼
    PO generator  (emits structured PurchaseOrder)

Each approval gate is a separate `interrupt()` so a reviewer can come
back to it later. The workflow ends with a typed `PurchaseOrder`
Pydantic model that can be filed into an ERP without parsing.

- Tier router is a plain conditional edge — no DSL, no policy file.
- Each gate is its own node — easy to add a tier, easy to re-order,
  easy to swap a human gate for an automated rule.
- `output_schema=PurchaseOrder` keeps the terminal artifact typed.

Run it (defaults to the bundled mock model; set `TULIP_MODEL_PROVIDER` to `openai` / `anthropic` for a live model):

    python examples/notebook_64_procurement_approval.py

Offline:

    TULIP_MODEL_PROVIDER=mock python examples/notebook_64_procurement_approval.py

Pin a strong-enough model for the structured PurchaseOrder schema:

    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_64_procurement_approval.py

## Source

```python
--8<-- "examples/notebook_64_procurement_approval.py"
```
