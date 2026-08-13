"""Anthropic frontier baselines on the identical held-out rows."""

import asyncio
import json
import os
import pathlib

import httpx


RISKY = {"require_human", "deny"}
with pathlib.Path("sample600.json").open() as _fh:
    SAMPLE = json.load(_fh)
URL = "https://api.anthropic.com/v1/messages"


async def one(client, sem, row, model, key):
    msgs = [m for m in row["messages"] if m["role"] != "assistant"]
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    user = [{"role": "user", "content": next(m["content"] for m in msgs if m["role"] == "user")}]
    gold = row["messages"][-1]["content"].strip().lower()
    body = {"model": model, "system": system, "messages": user, "max_tokens": 512}
    toks = 0
    async with sem:
        for a in range(4):
            try:
                r = await client.post(
                    URL,
                    json=body,
                    headers={
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    timeout=180,
                )
                r.raise_for_status()
                j = r.json()
                t = "".join(b.get("text", "") for b in j.get("content", [])).strip()
                toks = j.get("usage", {}).get("output_tokens", 0)
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


async def run(name, model, key, conc=8):
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient() as c:
        res = await asyncio.gather(*[one(c, sem, r, model, key) for r in SAMPLE])
    n = len(res)
    risky = [r for r in res if r["gold"] in RISKY]
    benign = [r for r in res if r["gold"] == "allow"]
    held = [r for r in res if r["gold"] == "require_human"]
    bad = sum(1 for r in res if r["pred"] not in {"allow", "require_human", "deny"})
    print(
        f"{name:22} acc {sum(r['correct'] for r in res) / n:6.2%}  "
        f"false-allow {sum(1 for r in risky if r['pred'] == 'allow') / len(risky):6.2%}  "
        f"over-hold {sum(1 for r in benign if r['pred'] != 'allow') / len(benign):6.2%}  "
        f"hold-recall {sum(1 for r in held if r['pred'] == 'require_human') / len(held):6.2%}  "
        f"avg-tokens {sum(r['tokens'] for r in res) / n:5.1f}  unparseable {bad}",
        flush=True,
    )
    pathlib.Path(f"frontier_{name.replace(' ', '_')}.json").write_text(json.dumps(res, indent=1))


async def main():
    key = os.environ["ANTHROPIC_API_KEY"]
    print(f"same {len(SAMPLE)} held-out rows\n", flush=True)
    await run("claude-sonnet-5", "claude-sonnet-5", key)
    await run("claude-opus-5", "claude-opus-5", key)
    await run("claude-haiku-4-5", "claude-haiku-4-5-20251001", key)


if __name__ == "__main__":
    asyncio.run(main())
