# A CI security gate for AI agents

You ship an AI agent. Every PR can quietly weaken its defenses — a prompt tweak,
a new tool, a model swap. This is a **regression gate** you drop into CI: it runs
``assure(target)`` (the OWASP-ASI suite as a grounded guardrail-coverage check)
and **fails the build** if coverage drops below a bar. Same idea as a coverage
gate or a failing test — but for AI security.

The gate's verdict is grounded: coverage is computed from observed probe
outcomes, and the finding names exactly which categories slipped through. In real
CI you assess your one agent and exit non-zero on failure:

```python
posture = await assure(Target.endpoint(MY_AGENT_URL))
if posture[0].confidence < THRESHOLD:
    raise SystemExit(1)        # block the merge
```

This notebook runs offline and shows the gate catching a regression: a "previous"
build that passes and a "candidate" that fails the bar.

Run it:

    python packages/locus-sdk-security/examples/notebook_85_ci_security_gate.py

## Source

````python
--8<-- "examples/notebook_85_ci_security_gate.py"
````
