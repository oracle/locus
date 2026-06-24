#!/usr/bin/env python
"""GSAR ablation: same model, WITH vs WITHOUT Tulip's grounding gate.

This is the SDK-attributable experiment. The model is held constant; the only
variable is whether the model's verdict passes through GSAR grounding. We feed
SOC 'finding candidates' — half real (a tool output directly supports the claim),
half traps (a plausible claim whose only support is inference). We measure the
hallucinated-finding rate (shipping a trap) with and without GSAR.

  Pipeline A (no Tulip):  ship iff the model says finding=true (trust the model).
  Pipeline B (Tulip GSAR): take the model's TYPED evidence -> Partition ->
                           ground_finding(); ship iff it clears the threshold.

Run: ANTHROPIC_API_KEY / OPENAI_API_KEY in env.
"""
from __future__ import annotations

import json
import os
import re
import sys

from tulip.reasoning.gsar import Partition
from tulip.security import Severity, ground_finding, inference_claim, is_finding, tool_match

# (id, context, label) — label "ship" = evidence truly supports; "trap" = abstain.
ITEMS = [
    ("expired-tls", "tls_scan output for 192.0.2.10:443 -> not_after=2026-05-30 (today 2026-06-24; expired). Claim: 'expired TLS certificate on 192.0.2.10:443'.", "ship"),
    ("vt-malicious", "VirusTotal v3: 198.51.100.5 flagged malicious by 7/90 engines. Claim: '198.51.100.5 is a malicious IP'.", "ship"),
    ("impossible-travel", "Auth log: user mallory two successful sign-ins 4 minutes apart from Toronto then Minsk. Claim: 'account compromise (impossible travel) for mallory'.", "ship"),
    ("osv-malware", "OSV lookup: dependency left-pad@1.0.0 matches advisory GHSA-xxxx (confirmed malware). Claim: 'supply-chain malware in left-pad@1.0.0'.", "ship"),
    ("edr-exec", "CrowdStrike EDR: host ws-42 ran `powershell -enc <b64>` that downloaded from a domain VirusTotal flags 12/90 malicious. Claim: 'malicious code execution on ws-42'.", "ship"),
    ("cdn-exfil", "Firewall log: outbound 443 connection from ws-7 to 151.101.0.0 (a well-known CDN range). Claim: 'data exfiltration to a C2 server'.", "trap"),
    ("offhours-login", "Auth log: user jsmith logged in successfully at 02:14 local time from his usual corporate IP. Claim: 'jsmith account is compromised'.", "trap"),
    ("pdf-download", "Proxy log: ws-9 downloaded invoice_april.pdf from a partner domain. No AV/sandbox result available. Claim: 'malware delivery via invoice_april.pdf'.", "trap"),
    ("clean-domain-phish", "VirusTotal: login.acme-portal.com flagged by 0/90 engines; domain registered 5 years ago; valid TLS. Claim: 'login.acme-portal.com is a phishing domain'.", "trap"),
    ("highcpu-miner", "Monitoring: server db-3 at 95% CPU for 20 minutes. No process/hash/network evidence collected. Claim: 'db-3 is infected with a cryptominer'.", "trap"),
]

PROMPT = """You are a SOC analyst. Decide whether the stated CLAIM is a CONFIRMED security finding.
Reply with STRICT JSON only:
{"finding": true|false, "severity": "low|medium|high|critical",
 "evidence": [{"text": "<short>", "type": "TOOL_MATCH|INFERENCE|CONTRADICTED"}]}
TOOL_MATCH = a tool/log output directly supports the claim.
INFERENCE  = your own reasoning/assumption, not directly observed.
CONTRADICTED = evidence arguing against the claim.
CONTEXT:
"""


def call_anthropic(model: str, prompt: str) -> str:
    import anthropic
    c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    r = c.messages.create(model=model, max_tokens=600, messages=[{"role": "user", "content": prompt}])
    return r.content[0].text


def call_openai(model: str, prompt: str) -> str:
    import openai
    c = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tok = {"max_completion_tokens": 600} if model.startswith(("gpt-5", "o1", "o3", "o4")) else {"max_tokens": 600}
    r = c.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], **tok)
    return r.choices[0].message.content or ""


def parse(txt: str):
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    return json.loads(m.group(0)) if m else None


def gsar_ships(a: dict) -> bool:
    ev = a.get("evidence", []) or []
    grounded = [tool_match(e.get("text", ""), f"ev:{i}") for i, e in enumerate(ev) if e.get("type") == "TOOL_MATCH"]
    ungrounded = [inference_claim(e.get("text", ""), f"ev:{i}") for i, e in enumerate(ev) if e.get("type") != "TOOL_MATCH"]
    part = Partition(grounded=grounded, ungrounded=ungrounded)
    res = ground_finding(title="finding", description="ablation", severity=Severity.HIGH,
                         asset="asset", remediation="n/a", partition=part)
    return is_finding(res)


def run(model: str, caller) -> dict:
    A_fp = B_fp = A_tp = B_tp = traps = ships = errs = 0
    for tid, ctx, label in ITEMS:
        try:
            a = parse(caller(model, PROMPT + ctx))
        except Exception as e:  # noqa: BLE001
            print(f"  [err {tid}] {e}", file=sys.stderr); errs += 1; continue
        if not a:
            errs += 1; continue
        a_ship = bool(a.get("finding"))
        b_ship = gsar_ships(a)
        if label == "trap":
            traps += 1; A_fp += a_ship; B_fp += b_ship
        else:
            ships += 1; A_tp += a_ship; B_tp += b_ship
    return dict(model=model, traps=traps, ships=ships, A_fp=A_fp, B_fp=B_fp, A_tp=A_tp, B_tp=B_tp, errs=errs)


def report(r: dict) -> None:
    t, s = max(r["traps"], 1), max(r["ships"], 1)
    print(f"\n### {r['model']}  (errs={r['errs']})")
    print(f"  WITHOUT GSAR (bare model):  hallucinated findings {r['A_fp']}/{r['traps']} traps "
          f"({r['A_fp']/t*100:.0f}%) | detection {r['A_tp']}/{r['ships']} ({r['A_tp']/s*100:.0f}%)")
    print(f"  WITH GSAR (Tulip):          hallucinated findings {r['B_fp']}/{r['traps']} traps "
          f"({r['B_fp']/t*100:.0f}%) | detection {r['B_tp']}/{r['ships']} ({r['B_tp']/s*100:.0f}%)")
    print(f"  >>> GSAR cut hallucinated findings by {r['A_fp'] - r['B_fp']} (of {r['traps']}), "
          f"detection delta {r['B_tp'] - r['A_tp']}")


if __name__ == "__main__":
    out = []
    if os.environ.get("OPENAI_API_KEY"):
        out.append(run("gpt-4o", call_openai))
    if os.environ.get("ANTHROPIC_API_KEY"):
        out.append(run("claude-sonnet-4-6", call_anthropic))
    print("=" * 60)
    print("GSAR ABLATION — same model, with vs without grounding")
    print("=" * 60)
    for r in out:
        report(r)
    json.dump(out, open("ablation_report.json", "w"), indent=2)
    print("\nwrote ablation_report.json")
