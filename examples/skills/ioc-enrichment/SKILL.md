---
name: ioc-enrichment
description: Use this skill when triaging an indicator of compromise (IP, domain, URL, or file hash) — enriches it against intel sources and returns a grounded verdict.
allowed-tools: lookup_ioc enrich_domain kb_search
license: Apache-2.0
metadata:
  author: tuliplabs
  domain: security
  version: "1.0"
---

# IOC enrichment & triage

When handed an indicator of compromise, walk these steps in order. Each
step's output is required in the final response — no shortcuts.

## 1. Classify the indicator

State the indicator type in one line: IPv4/IPv6, domain, URL, or file
hash (MD5/SHA-1/SHA-256). If the value is malformed or ambiguous, say so
and ask before enriching.

## 2. Enrich against intel

- For a **domain or URL**: registrar age, category, and reputation
  (`enrich_domain`). A domain registered in the last few days is a
  newly-observed-domain (NOD) signal.
- For an **IP / domain / hash**: vendor detections and first-seen date
  (`lookup_ioc`). Note how many independent vendors flag it.
- Pull related context from the knowledge base (`kb_search`) — known
  campaigns, ATT&CK techniques, prior sightings.

## 3. Score and verdict

Weigh the evidence and assign one of: **malicious**, **suspicious**,
**benign**, or **unknown**. Cite every signal that moved the verdict.
If the evidence is thin, return **unknown** and say what you'd need —
never guess a verdict the evidence doesn't support.

## 4. Recommend an action

- **malicious** → block/sinkhole the indicator, hunt for related
  activity, open or escalate a case.
- **suspicious** → watchlist + enrich further before acting.
- **benign / unknown** → close with rationale, or request more data.

## Anti-patterns

- ❌ Don't declare an indicator malicious on a single low-confidence
  signal — say how many vendors and which sources agree.
- ❌ Don't act on documentation-range or RFC 5737 addresses
  (`198.51.100.0/24`, `192.0.2.0/24`) as if they were live.
- ❌ Don't skip the verdict + action block "because it's obviously bad."
