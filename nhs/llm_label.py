"""Label the rows of a task once with an LLM (OpenAI-compatible chat API) and cache the answers.

Experiments never call the API: samplers buy labels from this cache, one LLM call per label,
so many methods and seeds can be compared for the price of a single labeling pass.

Usage:
  python -m nhs.llm_label --task lts_shark --provider openai --model gpt-4o-mini --limit 50   # try a few
  python -m nhs.llm_label --task lts_shark --provider openai --model gpt-4o-mini              # all rows
  python -m nhs.llm_label --task lts_shark --model gpt-4o-mini --compile
      -> data/lts_shark/labels_llm_gpt-4o-mini.npy, used with label_source: llm_gpt-4o-mini

API keys come from OPENAI_API_KEY / DEEPSEEK_API_KEY (process environment or, on Windows, the
user environment in the registry, so keys set with `setx` work without a restart). Never commit keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .paths import CACHE, CONFIGS, DATA

PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
}
# Extra request fields per provider. DeepSeek models think by default; thinking tokens are billed as
# output and a 10-token answer budget would be spent on them, so classification turns it off.
EXTRA_BODY = {
    "deepseek": {"thinking": {"type": "disabled"}},
}
# CNY per 1M tokens (peak, off-peak) from api-docs.deepseek.com/zh-cn/quick_start/pricing, Nov 2026.
# Off-peak = outside Mon-Fri 9:00-12:00 and 14:00-18:00 Beijing time. Used only to print an estimate.
PRICES_CNY = {
    "deepseek-flash": {"hit": (0.04, 0.02), "miss": (2.0, 1.0), "out": (8.0, 4.0)},
    "deepseek-v4-pro": {"hit": (0.30, 0.15), "miss": (9.0, 4.5), "out": (27.0, 13.5)},
}


def get_key(var: str) -> str | None:
    value = os.environ.get(var)
    if value:
        return value
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                return winreg.QueryValueEx(k, var)[0]
        except OSError:
            return None
    return None


def parse_label(answer: str) -> int:
    a = answer.strip().lower()
    if "not a relevant" in a or a.startswith("no") or a.startswith("0") or a.startswith("false"):
        return 0
    if "relevant animal" in a or a.startswith("yes") or a.startswith("1") or a.startswith("true"):
        return 1
    return -1


def chat(base_url: str, key: str, model: str, prompt: str, extra: dict | None = None,
         retries: int = 6, timeout: int = 60) -> tuple[str, dict]:
    """One chat completion; returns (answer text, usage dict)."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 10,
        **(extra or {}),
    }).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    delay = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(base_url + "/chat/completions", data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"] or "", data.get("usage", {})
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def cache_path(task: str, model: str, template: str):
    h = hashlib.sha1(template.encode()).hexdigest()[:8]
    return CACHE / "llm_labels" / task / f"{model.replace('/', '_')}__{h}.jsonl"


def read_cache(path) -> dict[int, dict]:
    out = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[int(r["row_id"])] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--provider", choices=sorted(PROVIDERS), default="openai")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", help="override the provider's API base URL")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="label at most this many new rows (for a trial run)")
    ap.add_argument("--max-chars", type=int, default=2000, help="truncate long texts")
    ap.add_argument("--compile", action="store_true", help="write labels_llm_<model>.npy from the cache")
    args = ap.parse_args()

    template = (CONFIGS / "prompts" / f"{args.task}.txt").read_text(encoding="utf-8")
    meta = json.loads((DATA / args.task / "meta.json").read_text(encoding="utf-8"))
    src = meta.get("emb_from") or args.task
    with open(DATA / src / "texts.jsonl", encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]
    path = cache_path(args.task, args.model, template)
    done = read_cache(path)

    if args.compile:
        labels = np.full(len(texts), -1, dtype=np.int8)
        for i, r in done.items():
            labels[i] = r["label"]
        out = DATA / args.task / f"labels_llm_{args.model.replace('/', '_')}.npy"
        np.save(out, labels)
        y = np.load(DATA / args.task / "y.npy")
        known = (y >= 0) & (labels >= 0)
        tp = int(((y == 1) & (labels == 1) & known).sum())
        fp = int(((y == 0) & (labels == 1) & known).sum())
        fn = int(((y == 1) & (labels == 0) & known).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"labeled {int((labels >= 0).sum())}/{len(texts)} rows, unparsed answers: {int((labels < 0).sum())}")
        print(f"LLM vs gold on {int(known.sum())} rows: precision={p:.3f} recall={r:.3f} F1={f1:.3f}")
        print(f"wrote {out}")
        return

    base_url, var = PROVIDERS[args.provider]
    base_url = args.base_url or base_url
    key = get_key(var)
    if not key:
        sys.exit(f"Set {var} first (e.g. PowerShell: setx {var} \"<key>\", then open a new terminal).")
    todo = [i for i in range(len(texts)) if i not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} rows cached, labeling {len(todo)} rows with {args.provider}/{args.model}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = EXTRA_BODY.get(args.provider)
    tok = {"hit": 0, "miss": 0, "out": 0}
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        futs = {
            pool.submit(chat, base_url, key, args.model, template.replace("{text}", texts[i][: args.max_chars]), extra): i
            for i in todo
        }
        for n, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            try:
                answer, usage = fut.result()
            except Exception as e:  # keep going; failed rows are retried on the next run
                print(f"row {i}: {e}", flush=True)
                continue
            hit = int(usage.get("prompt_cache_hit_tokens", 0))
            tok["hit"] += hit
            tok["miss"] += int(usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit))
            tok["out"] += int(usage.get("completion_tokens", 0))
            f.write(json.dumps({"row_id": i, "label": parse_label(answer), "raw": answer}, ensure_ascii=False) + "\n")
            if n % 200 == 0:
                f.flush()
                print(f"{n}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)
    print(f"finished in {time.time() - t0:.0f}s; cache: {path}")
    print(f"tokens: input cache-hit {tok['hit']:,}, input cache-miss {tok['miss']:,}, output {tok['out']:,}")
    price = PRICES_CNY.get(args.model)
    if price:
        for k, name in ((0, "peak"), (1, "off-peak")):
            cost = sum(tok[t] * price[t][k] for t in tok) / 1e6
            print(f"estimated cost at {name} prices: {cost:.2f} CNY")


if __name__ == "__main__":
    main()
