# Investigate an incident with SecurityContext

The point of ``SecurityContext`` is that you reason in *domains*, not vendors.
The whole investigation below never says "Splunk" or "Okta" or "CrowdStrike" — it
says logs, identity, threat-intel, endpoint, actions. Swap a real vendor in by
injecting a provider (``SecurityContext(logs=SplunkLogs())``); the investigation
code does not change.

The notebook walks a suspected account compromise across six domains — search the
logs, score the user's risk, enrich an indicator, pull the host timeline — then
proposes containment and **gates it through policy** before acting. One
investigation, six domains, zero vendor names.

Runs offline on the bundled reference providers.

Run it:

    python packages/locus-sdk-security/examples/notebook_89_investigate_with_ctx.py

See also: [SecurityContext](../concepts/security-context.md).

## Source

````python
--8<-- "examples/notebook_89_investigate_with_ctx.py"
````
