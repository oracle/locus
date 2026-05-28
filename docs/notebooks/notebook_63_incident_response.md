# Incident Response

Models the loop a real on-call engineer runs when a page fires::

    Page fires
      │
      └──> Triage  ──>  scatter to 3 parallel investigators
                          ├── log analyst
                          ├── metric analyst
                          └── trace analyst
                          ▼
                   Synthesizer (root-cause hypothesis)
                          │
                          ▼
            Severity gate ─── critical? ──> page humans (interrupt)
                          │                     │
                          │                  approve mitigation? yes/no
                          │                     │
                          ▼                     ▼
                       Mitigator <──────────────┘
                          │
                          ▼
                       Postmortem (structured)

- `Send`: fan out to 3 investigator Agents in parallel.
- `add_conditional_edges`: severity-based routing decides
  auto-mitigate vs escalate to a human.
- `interrupt()`: critical severity pauses for explicit human approval
  before any mitigation runs.
- `output_schema=Postmortem`: the final report is a typed Pydantic
  instance, ready to file into a runbook database.

Run it (OCI Generative AI is the default — the notebook harness picks it when `~/.oci/config` exists; set `OCI_PROFILE` to choose a non-default profile):

    python examples/notebook_63_incident_response.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_63_incident_response.py

Pin a strong-enough model for the structured postmortem schema:

    LOCUS_MODEL_ID=openai.gpt-4.1 python examples/notebook_63_incident_response.py

## Source

```python
--8<-- "examples/notebook_63_incident_response.py"
```
