# Research

Findings from building the SDK's control layer — the policy gate, approvals and
audit trail that decide which of an agent's actions may run.
Everything here comes from running code against real catalogs, real tools and
real models. The method and the scoring code are public and runnable against
your own endpoint; where a result depends on a model or a corpus we do not
redistribute, the page says so, and where a result is a single illustration
rather than a benchmark, it says that too. The evaluations, and the training
of Clusiana-Admit-4B, ran on the authors' own accounts, API keys and rented
GPUs.

## Related work

The SDK's grounding layer draws on **[GSAR: Typed Grounding for Hallucination
Detection and Recovery in Multi-Agent LLMs](https://arxiv.org/abs/2604.23366)**
(F. A. Kamelhar, 2026), a paper separate from this project. It uses a judge to
partition claims into grounded / ungrounded / contradicted / complementary
against typed evidence, scores the partition, and routes the result to
proceed, regenerate, replan or abstain; the paper evaluates it with multiple
LLM judges on FEVER.

The SDK implements the method as [GSAR](../concepts/gsar.md);
`ground_finding()` returns evidence *or* an abstention, so an ungrounded claim
is not something the caller can accidentally ship.

## Findings

**[The family of harm your agent policy cannot see](policy-blindness.md)** — a
risk policy tends to encode one family of consequence and stay silent about the
others, and the silence survives code review, its own tests, validation against
the real tool catalog, and training.

Three independent instances, one of them measured: **Clusiana-Admit-4B** on
3,139 distinct held-out items reaches a 1.88% false-allow rate. Row-weighted
across the 8,989-row split, those errors cluster by family — 5.45% on execution
and 5.32% on egress against 0.00% on destruction. Includes a
seven-model comparison against GPT-5 and Claude Opus/Sonnet/Haiku on identical
rows, and the reason the headline number from that comparison is misleading.
