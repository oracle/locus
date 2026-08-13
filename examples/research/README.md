# Research evals

The scripts behind
[tulipagents.ai/research/policy-blindness](https://tulipagents.ai/research/policy-blindness/).
Published so the numbers on that page can be checked rather than believed.

| script | what it measures |
|---|---|
| `family_eval.py` | 64 hand-written cases, six per consequence family, under two policies — the out-of-distribution probe |
| `heldout_eval.py` | the full 8,989-row held-out split, per source and per consequence family |
| `frontier_eval.py` | GPT-5 / GPT-5-mini on identical rows, at two reasoning budgets |
| `anthropic_eval.py` | Claude Opus 5 / Sonnet 5 / Haiku 4.5 on identical rows |

## Running them

`family_eval.py` needs only an OpenAI-chat-compatible endpoint:

```bash
python examples/research/family_eval.py http://127.0.0.1:8010
```

The other three read a `sample600.json` in the working directory — a
stratified sample of a held-out split, in the chat format
`{"messages": [...], "source": "..."}` where the final assistant turn is the
gold verdict. Build one from whatever corpus you are evaluating; ours is not
redistributed here.

## Two things that will bite you

**Reasoning models silently return nothing at small token budgets.** GPT-5 and
Claude Opus 5 spend the whole allowance reasoning and answer with an empty
string. Our first Opus run scored 21.79% with 445 of 638 responses
unparseable; at an adequate budget the same model on the same rows scored
68.50% with one. Always report the unparseable count — `frontier_eval.py` and
`anthropic_eval.py` both do.

**Accuracy is the wrong headline.** These corpora are ~51% `require_human`, so
a constant predictor scores 51%. The numbers that matter are false-allow rate
(of actions that should have been held, how many ran), over-hold rate, and
hold recall. All four scripts print those.
