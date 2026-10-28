# SOC alert triage with SIEM-grounded verdicts

Every verdict the triage agent in this notebook proposes goes through
``ground_report``: a proposed finding is filed as an ``Evidence`` only if the
evidence the agent cited clears the [GSAR](../concepts/gsar.md) threshold. When
the agent proposes a finding without grounded evidence, the result is an
``Abstention`` — nothing is filed, and the alert is left for an analyst to
review manually.

The false-positive case matters most. When the agent sees benign noise and
correctly finds no corroborating evidence, an ``Abstention`` is the right output:
a finding without evidence is not filed. Four SIEM alerts arrive in a one-hour
window — a phishing click, lateral movement, a C2 beacon, and benign process
noise — yielding three grounded ``Evidence`` results and one ``Abstention``.

Runs fully offline on mock SIEM, EDR, and threat-intelligence adapters. Swap
them for ``security_toolset(siem=True, edr=True, threat_intel=True)`` and a real
model to run against a live environment.

Run it:

    python packages/locus-sdk-security/examples/notebook_87_soc_alert_triage.py

See also: [GSAR — typed grounding](../concepts/gsar.md) ·
[``ground_report``](../concepts/cloud-posture.md#the-grounding-split) ·
[Agentic AI-security](../concepts/agentic-ai-security.md).

## Output

Running it offline — no credentials, bundled mock model — prints the triage an analyst would otherwise do:

```text

────────────────────────────────────────────────────────────
Alert ALT-001: Phishing link clicked by user jsmith
  [FINDING ] HIGH     Phishing link clicked by user jsmith
             grounded @ 1.00
             evidence : ['siem:ALT-001:related_events']
             taxonomy : —

────────────────────────────────────────────────────────────
Alert ALT-002: PsExec lateral movement from WKSTN-04 to SRV-FINANCE
  [FINDING ] CRITICAL PsExec lateral movement from WKSTN-04 to SRV-FINANCE
             grounded @ 1.00
             evidence : ['siem:ALT-002:related_events', 'edr:WKSTN-04:process_tree']
             taxonomy : —

────────────────────────────────────────────────────────────
Alert ALT-003: Outbound connection to known C2 IP 203.0.113.99
  [FINDING ] HIGH     Outbound connection to known C2 IP 203.0.113.99
             grounded @ 1.00
             evidence : ['siem:ALT-003:related_events', 'intel:203.0.113.99:malicious']
             taxonomy : —

────────────────────────────────────────────────────────────
Alert ALT-004: Unusual process: msiexec.exe spawned by Teams.exe
  [ABSTAIN ] Unusual process: msiexec.exe spawned by Teams.exe
             reason   : withheld (replan): grounding below the proceed threshold (1 ungrounded of 1 claims)
             gsar     : 0.00 (below threshold)

════════════════════════════════════════════════════════════
Triage complete: 3 findings filed, 1 abstentions.
The abstention(s) above are alerts a commodity AI SOC would have filed as findings.
Here they were withheld because no corroborating evidence was cited — reducing
analyst noise and false-positive rate.
```
<!-- notebook-output:end -->

## Source

````python
--8<-- "examples/notebook_87_soc_alert_triage.py"
````
