# Decisions: pick from a list, with a probability

`tulip.decision` asks a model questions with a fixed list of answers and gets
back a probability for every answer, from one forward pass.

## Why not just generate

Much of what an agent decides is not writing. *Is this message meant for me?*
*Which of these thirteen commands is it?* *Does this reply ask for a home
address?* A model that writes answers such a question in a sentence, slowly,
and says nothing about how sure it is. The caller then parses text, and has no
number to put a threshold on.

A **decision model** answers the way a classifier does. Every listed answer gets
a probability, read from the logits at one position. Nothing is generated past
the first token. On a GPU a small trained head answers in a few milliseconds;
quantised on a CPU, in tens of milliseconds. And the probability is the thing a
threshold can be set on, and certified.

## Fields, answers, decisions

```python
from tulip.decision import Choice, LogprobDecider, Score, YesNo

intent = Choice("intent", "What does the speaker want the companion to do?",
                ("come", "follow", "build", "give", "chat"))
personal = YesNo("personal", "Does it ask for or share personal details?")
upset = Score("upset", "How upset is the speaker?", 5)      # levels "1".."5"

decider = LogprobDecider("http://127.0.0.1:8000/v1", model="my-head")
decision = await decider.decide("can u build me a castle", [intent, personal, upset])

decision["intent"].label          # "build"
decision["intent"].probability    # 0.97
decision["intent"].margin         # top minus runner-up
decision["personal"].p_yes        # 0.003
decision["upset"].expected        # probability-weighted level in [0, 1]
decision["intent"].coverage       # mass the listed letters had before renormalising
```

`coverage` is worth watching. It is the share of the model's probability that
went to the listed answers at all. Near 1, the model answered the question. Low,
it wanted to say something else, and the renormalised distribution is a model
out of its depth, whatever its argmax.

A provider that cannot read an answer raises `DecisionError`. It never guesses:
a caller thresholding a probability would act on the guess.

## The wire format

The format is a contract, so anyone can train a head for it and any server can
serve the head. Each field of an input is one chat request:

```text
system: You answer one question about the input. Reply with the letter of one listed answer and nothing else.

user:   [input]
        <the input>

        [question]
        <the question>

        [answers]
        A) <first answer>
        B) <second answer>
        ...

        Answer with one letter.
```

`YesNo` is always `A) yes`, `B) no`. Up to 26 answers. The input comes first, so
the fields asked of one input share a prefix and a server with prefix caching
computes it once. `render()` and `SYSTEM_PROMPT` produce it, and a unit test pins
it character for character.

To train a head: render each labelled example with `render()`, put the answer's
letter as the assistant turn, and fine-tune. A head trained this way needs no
glue code to be served.

## The free, local default: any OpenAI-compatible server

`LogprobDecider` sends, per field:

```json
{"model": "my-head",
 "messages": [{"role": "system", "content": "<SYSTEM_PROMPT>"},
              {"role": "user", "content": "<render(...)>"}],
 "max_tokens": 1, "temperature": 0, "logprobs": true, "top_logprobs": 20}
```

and reads the letters from `choices[0].logprobs.content[0].top_logprobs`.
Whitespace around a token is ignored, so `" A"` counts for `A`. Fields of one
input go concurrently. `extra_body` adds options under those keys and cannot
replace them; a Qwen3 head wants `{"chat_template_kwargs": {"enable_thinking":
False}}`, so the first token is the answer and not the start of a thought.

It needs nothing but a server that returns logprobs:

```bash
vllm serve ./my-head --served-model-name my-head --enable-prefix-caching
llama-server -m my-head-q4_k_m.gguf --port 8080          # a CPU is enough for 0.6B
```

A general instruction model works too, zero-shot, through the same rendered
question. A head trained for the questions answers them more sharply.

## In the admission gate

A decision model never grants anything. Two seams, both escalate-only:

**An admit head as a `ControlAdvisor`.** `approve()` consults an advisor only
when policy already allows, and takes `max(policy, model)` on `allow <
require_human < deny`.

```python
from tulip.control import approve
from tulip.decision import DecisionAdvisor

advisor = DecisionAdvisor(decider)                    # argmax of allow / require_human / deny
advisor = DecisionAdvisor(decider, hold_at=0.12)      # or: hold when 1 - P(allow) >= 0.12
decision = approve(action, policy=policy, advisor=advisor)
```

A head that is down, slow to error or off its list is no opinion, and the
policy decision stands.

**Safety heads as verification.** `verification_from_decision()` turns yes/no
answers into the `VerificationResult` that `ControlPolicy.require_verification_score`
weighs:

```python
from tulip.decision import verification_from_decision

safety = await decider.decide(reply_text, [personal, meeting, unkind])
verdict = verification_from_decision(safety, {"personal": 0.21, "meeting": 0.30, "unkind": 0.18})
approve(say_action, policy=policy, verdict=verdict)
```

A head at or past its threshold is a fatal refutation, and `approve()` denies.
Below every threshold, `confidence` is `1 - max(P(yes) / threshold)`, so the
policy's verification bar decides how near a threshold is near enough to want a
person.

## Thresholds belong to weights

A threshold is worth something only if it was chosen on held-out data the head
never trained on, and stated with a bound: *with probability at least 1 - delta,
the miss rate at this threshold is at most alpha*. A Neyman–Pearson order
statistic over the held-out positives gives exactly that, with no assumption
about the model. The bound holds for the weights it was measured on and nothing
else. Ship the threshold with the model; recalibrate when the model changes;
and note that it covers ordinary traffic, not an adversary searching against
the score. The escalate-only seams are what hold there.

## Per tenant

`TenantDecisionRouter` is the same protocol for many tenants:

```python
from tulip.decision import LogprobDecider, TenantDecisionRouter, per_tenant_trails

router = TenantDecisionRouter(
    lambda tenant: LogprobDecider(url, model=f"head-{tenant}") if tenant in trained else None,
    audit_for=per_tenant_trails("audit/"),   # one hash-chained file per tenant
)
decision = await router.decide(text, fields, tenant="acme")
advisor = DecisionAdvisor(router.for_tenant("acme"))
```

- A tenant is answered only by its own head: its own model, or its own LoRA
  adapter on a shared base (vLLM serves an adapter under its own model name).
- An unknown tenant is refused. A `public=` head may stand in only when you
  name one, it must be trained on no tenant's data, and the record says it was
  used.
- Every decision lands on that tenant's audit chain: field names, labels,
  probabilities, model, latency. Not the input, unless `record_text=True` — it
  may be a customer's words, or a child's.
- If the record cannot be written, the decision is not returned.

`per_tenant_trails()` is the zero-infra form of the audit side. A hosted
deployment backs the same `audit_for` with each tenant's chain in a
row-level-secured database, and serves the heads behind the same protocol.
