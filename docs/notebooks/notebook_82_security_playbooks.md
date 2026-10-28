# SOC playbooks over the security toolset

The SDK ships curated IR / SOC playbooks (``phishing_triage``,
``nist_800_61_ir``, ``ransomware_containment``, ``cloud_posture_audit``) and the
agent-ready security adapters they drive — ``security_toolset()``: IOC
enrichment, SIEM search, EDR forensics, vuln / posture scanning, inference
fingerprinting. A playbook pins an investigation to its steps in order via the
``PlaybookEnforcer``; the toolset gives the agent the tools each step names.

This notebook lists the bundled playbooks, wires one onto an ``Agent`` with the
security toolset and runs it, then walks the ``PlaybookEnforcer`` deterministically
to show the step gate.

Runs on OCI Generative AI with an OCI profile, otherwise offline on the bundled mock model with no credentials and no network; ``LOCUS_MODEL_PROVIDER`` with its key forces another provider.

Run it:

    python packages/locus-sdk-security/examples/notebook_82_security_playbooks.py

## Source

````python
--8<-- "examples/notebook_82_security_playbooks.py"
````
