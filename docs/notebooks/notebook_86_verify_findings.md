# Verify findings before you act on them

A Locus ``Evidence`` is a *claim*. Before you act on it, ``verify()`` puts it through an
independent skeptic that challenges the evidence and scores confidence. A
well-grounded finding **survives**; an unsupported one is **refuted**, with the
objections listed. ``verify()`` only returns the verdict: gate the action on
``survives`` so a refuted finding never reaches it. The default skeptic grades
the evidence a finding carries; it does not check that the references are real.

``verify()`` takes a Locus ``Evidence`` *or* a finding-shaped dict (``title``,
``severity``, ``gsar_score``, ``evidence_refs``, ``confidence``), so a finding
produced outside Locus goes through the same challenge. The
notebook verifies a real grounded finding (survives), a fabricated one with no
evidence (refuted), and an external finding that has references but was never
grounded (refuted).

Runs fully offline.

Run it:

    python packages/locus-sdk-security/examples/notebook_86_verify_findings.py

See also: [Agentic AI-security](../concepts/agentic-ai-security.md) ·
[SecurityContext](../concepts/security-context.md).

## Source

<!-- The line ranges match examples/notebook_86_verify_findings.py in locus-sdk
v2.16.0: they skip the module docstring (lines 5-20), which the prose above covers,
and the closing print (lines 61-65), which claims more than verify() does.
Re-check them whenever the SDK the site builds against changes. -->
````python
--8<-- "examples/notebook_86_verify_findings.py:1:4"
--8<-- "examples/notebook_86_verify_findings.py:22:60"
--8<-- "examples/notebook_86_verify_findings.py:66:"
````
