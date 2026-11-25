"""Full-label reference: a proxy trained on ALL pool rows with gold labels (no label budget).

For each (task, positive rate, seed) of a config, train the logistic-regression proxy on every pool row
and score it on the same held-out test set as the samplers. ``best_threshold_f1`` uses the best decision
threshold chosen on the test set itself, an optimistic measure of how separable the classes are in the
embedding space.

This is NOT a strict upper bound: under extreme imbalance, balanced class weights on the full pool
over-predict positives (low precision at threshold 0.5), and a proxy trained on a well-chosen sample can
score higher (observed on 20 Newsgroups). A low ``best_threshold_f1`` means the embedding itself cannot
separate the classes, so no sampler can rescue the proxy (observed on Toxic Conversations).

Usage:
  python -m nhs.upper_bound --config configs/main_toxic.yaml   ->  results/<exp>/upper_bound.csv
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from .metrics import macro_f1, prf
from .paths import DEFAULT_MODEL, RESULTS
from .proxy import fit_lr, predict_proba
from .run import load_config, pr_key
from .tasks import build_context


def best_threshold_f1(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    fn = int((y == 1).sum()) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(f1.max()) if len(f1) else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = RESULTS / cfg["exp"] / "upper_bound.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, tcfg in cfg["tasks"].items():
        for pos_rate in (tcfg or {}).get("pos_rates", [None]):
            for seed in cfg.get("seeds", [0]):
                ctx = build_context(task, pos_rate, seed, cfg.get("embed_model", DEFAULT_MODEL), "gold",
                                    float(cfg.get("test_frac", 0.3)))
                if not ctx.has_full_gold:
                    continue
                train = np.flatnonzero(ctx.pool_mask & (ctx.y >= 0))
                t0 = time.time()
                model = fit_lr(ctx.emb[train], ctx.y[train])
                ev = ctx.eval_mask & (ctx.y >= 0)
                score = predict_proba(model, ctx.emb[ev])
                yt = ctx.y[ev]
                yp = (score >= 0.5).astype(int)
                p, r, f1 = prf(yt, yp)
                row = {
                    "task": task, "pos_rate": pr_key(pos_rate), "seed": seed,
                    "n_train": len(train), "n_pos_train": int(ctx.y[train].sum()),
                    "proxy_precision": p, "proxy_recall": r, "proxy_f1": f1,
                    "proxy_macro_f1": macro_f1(yt, yp), "best_threshold_f1": best_threshold_f1(yt, score),
                    "fit_s": time.time() - t0,
                }
                rows.append(row)
                print(f"{task:14s} pr={pr_key(pos_rate):7s} s={seed} train={len(train)} "
                      f"f1={f1:.3f} best_thr_f1={row['best_threshold_f1']:.3f}", flush=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
