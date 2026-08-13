"""Frontier baselines on the identical held-out rows Clusiana was scored on.

GPT-5 is a reasoning model: with a small token budget it spends the whole
allowance on reasoning and returns empty content, so it is run twice --
once at minimal reasoning (10 tokens, Clusiana's operating point) and once
at default reasoning (~138 tokens) -- and completion tokens are recorded,
because a 4B model answering in 8 tokens and a frontier model answering in
138 are not the same product even at equal accuracy.
"""

import asyncio
import json
import os
import pathlib
import sys

import httpx


RISKY = {"require_human", "deny"}
with pathlib.Path("sample600.json").open() as _fh:
    SAMPLE = json.load(_fh)


async def one(client, sem, row, model, key, extra):
    msgs = [m for m in row["messages"] if m["role"] != "assistant"]
    gold = row["messages"][-1]["content"].strip().lower()
    body = {"model": model, "messages": msgs, "max_completion_tokens": 2000, **extra}
    toks = 0
    async with sem:
        for a in range(4):
            try:
                r = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=180,
                )
                r.raise_for_status()
                j = r.json()
                t = (j["choices"][0]["message"].get("content") or "").strip()
                toks = j.get("usage", {}).get("completion_tokens", 0)
                pred = t.split()[0].rstrip(".,:").lower() if t.split() else "?"
                break
            except Exception:
                if a == 3:
                    pred = "?"
                await asyncio.sleep(2 + a * 2)
    return {
        "source": row["source"],
        "gold": gold,
        "pred": pred,
        "correct": pred == gold,
        "tokens": toks,
    }


async def run(name, model, key, extra, conc=10):
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient() as c:
        res = await asyncio.gather(*[one(c, sem, r, model, key, extra) for r in SAMPLE])
    n = len(res)
    risky = [r for r in res if r["gold"] in RISKY]
    benign = [r for r in res if r["gold"] == "allow"]
    held = [r for r in res if r["gold"] == "require_human"]
    bad = sum(1 for r in res if r["pred"] not in {"allow", "require_human", "deny"})
    print(
        f"{name:26} acc {sum(r['correct'] for r in res) / n:6.2%}  "
        f"false-allow {sum(1 for r in risky if r['pred'] == 'allow') / len(risky):6.2%}  "
        f"over-hold {sum(1 for r in benign if r['pred'] != 'allow') / len(benign):6.2%}  "
        f"hold-recall {sum(1 for r in held if r['pred'] == 'require_human') / len(held):6.2%}  "
        f"avg-tokens {sum(r['tokens'] for r in res) / n:6.1f}  unparseable {bad}",
        flush=True,
    )
    pathlib.Path(f"frontier_{name.replace(' ', '_')}.json").write_text(json.dumps(res, indent=1))


async def main():
    key = os.environ["OPENAI_API_KEY"]
    print(f"same {len(SAMPLE)} held-out rows, stratified by source\n", flush=True)
    await run("gpt-5 minimal-reasoning", "gpt-5", key, {"reasoning_effort": "minimal"})
    await run("gpt-5 default-reasoning", "gpt-5", key, {})
    await run("gpt-5-mini minimal", "gpt-5-mini", key, {"reasoning_effort": "minimal"})


asyncio.run(main())
