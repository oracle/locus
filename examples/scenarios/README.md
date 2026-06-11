# Threat → defense scenarios

A threat-indexed catalog of AI-security scenarios. Each gist is small,
standalone, and **runnable offline with no credentials** — it states one
threat, shows an agent hitting it, and shows the Tulip defense stopping it.
Together they map **every** item in the three catalogues Tulip encodes
(`tulip.security.taxonomy`) to at least one runnable example.

```bash
python examples/scenarios/run_all.py        # run every gist, assert all pass
python examples/scenarios/prompt_injection.py   # or run one
```

Each gist's defense is one of three kinds:
- **primitive** — a built-in SDK control (`url_safety`/`path_safety`, `GuardrailsHook`, `ground_finding`/`ground_fingerprint`);
- **pattern** — an allowlist/audit pattern with SDK taxonomy + wiring points (where there is no single built-in);
- both, where they stack.

## OWASP LLM Top 10 (2025)

| ID | Risk | Gist | Defense |
|----|------|------|---------|
| LLM01 | Prompt Injection | `prompt_injection.py` | GuardrailsHook content patterns (primitive) |
| LLM02 | Sensitive Information Disclosure | `sensitive_disclosure.py` | GuardrailsHook PII redaction (primitive) |
| LLM03 | Supply Chain | `supply_chain.py` | provenance allowlist (pattern) |
| LLM04 | Data & Model Poisoning | `memory_poisoning.py` | `ground_finding` abstention (primitive) |
| LLM05 | Improper Output Handling | `improper_output_handling.py` | GuardrailsHook at output→sink (primitive) |
| LLM06 | Excessive Agency | `excessive_agency.py`, `tool_abuse.py` | allow_only_tools; url/path safety (primitive) |
| LLM07 | System Prompt Leakage | `sensitive_disclosure.py` | secret-egress content block (primitive) |
| LLM08 | Vector & Embedding Weaknesses | `memory_poisoning.py` | grounding over retrieved claims (primitive) |
| LLM09 | Misinformation | `misinformation_trust.py` | `ground_finding` abstention (primitive) |
| LLM10 | Unbounded Consumption | `model_extraction.py` | rate-limit / coverage abstention (primitive + pattern) |

## OWASP ASI Top 10 — Agentic (2026)

| ID | Risk | Gist | Defense |
|----|------|------|---------|
| ASI01 | Agent Goal Hijack | `prompt_injection.py` | content guardrail at tool boundary (primitive) |
| ASI02 | Tool Misuse | `tool_abuse.py` | `is_safe_url` / `safe_resolve` (primitive) |
| ASI03 | Identity & Privilege Abuse | `excessive_agency.py` | deny-by-default allowlist (primitive) |
| ASI04 | Agentic Supply Chain | `supply_chain.py` | provenance allowlist (pattern) |
| ASI05 | Unexpected Code Execution | `code_execution.py` | block_dangerous_tools (primitive) |
| ASI06 | Memory & Context Poisoning | `memory_poisoning.py` | `ground_finding` abstention (primitive) |
| ASI07 | Insecure Inter-Agent Communication | `inter_agent_comms.py` | A2A bearer auth + peer allowlist (primitive + pattern) |
| ASI08 | Cascading Failures | `cascading_failures.py` | grounding gate between stages (primitive) |
| ASI09 | Human-Agent Trust Exploitation | `misinformation_trust.py` | abstain on ungrounded directives (primitive) |
| ASI10 | Rogue Agents | `rogue_agent.py` | mandate allowlist + audit trail (pattern) |

## MITRE ATLAS

| ID | Technique | Gist |
|----|-----------|------|
| AML.T0043 | Craft Adversarial Data | `model_extraction.py` |
| AML.T0051 | LLM Prompt Injection | `prompt_injection.py` |
| AML.T0054 | LLM Jailbreak | `prompt_injection.py` |
| AML.T0020 | Poison Training Data | `memory_poisoning.py` |
| AML.T0018 | Backdoor ML Model | `supply_chain.py` |
| AML.T0040 | AI Model Inference API Access | `model_extraction.py` |
| AML.T0024 | Exfiltration via Inference API | `model_extraction.py` |
| AML.T0086 | Exfiltration via Agent Tool Invocation | `inter_agent_comms.py` |
| AML.T0110 | AI Agent Tool Poisoning | `supply_chain.py` |
| AML.T0048 | External Harms | `code_execution.py` |

Every ID in `tulip.security.taxonomy` (`AtlasTechnique`, `OwaspLLM`,
`OwaspASI`) appears above — coverage is complete and `run_all.py` keeps it
runnable. `model_extraction.py` uses the real streaming timing probe in
[`../integrations/remote_timing.py`](../integrations/remote_timing.py)
(offline sample with no key).
