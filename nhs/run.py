"""Run an experiment grid defined in a YAML or JSON config.

Usage:
  python -m nhs.run --config configs/main_20ng.yaml

Every run: a sampler spends ``budget`` labels in batches of ``batch_size``; a logistic-regression
proxy is trained on the bought labels and evaluated against gold labels on rows it did not
see. Results are appended to results/<exp>/runs.csv and traces.csv; finished runs are skipped,
so an interrupted grid can be resumed by running the same command again.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .labelers import Labeler
from .metrics import macro_f1, prf
from .paths import DEFAULT_MODEL, RESULTS, ROOT
from .proxy import fit_lr, predict_proba
from .samplers import make_sampler, resolve
from .tasks import Context, build_context

RUN_FIELDS = [
    "run_id", "exp", "task", "pos_rate", "seed", "budget", "batch_size", "method", "base_method",
    "params", "label_source", "flip_rate", "test_frac", "n_rows", "n_pos", "base_rate",
    "pos_found", "pos_found_gold", "pos_per_call", "subtypes_covered", "n_subtypes", "subtype_coverage",
    "proxy_trained", "proxy_precision", "proxy_recall", "proxy_f1", "proxy_macro_f1", "n_eval",
    "e2e_precision", "e2e_recall", "e2e_f1",
    "t_setup_ms", "t_select_ms_mean", "t_update_ms_mean", "t_round_ms_mean", "t_final_train_ms", "t_total_s",
    "extra_llm_calls", "embed_model", "git_commit",
]
TRACE_FIELDS = ["run_id", "round", "used", "cum_pos", "cum_pos_gold", "cum_subtypes"]


def load_config(path: str) -> dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return ""


def pr_key(pos_rate) -> str:
    return "natural" if pos_rate is None else f"{pos_rate:g}"


def run_one(ctx: Context, method: str, params: dict | None, budget: int, batch_size: int,
            seed: int, labeler: Labeler):
    """Run one sampler on one context. Returns (summary dict, per-round trace, labeled rows)."""
    sampler = make_sampler(method, params)
    rng = np.random.default_rng([seed, 1009])
    t_start = time.perf_counter()
    sampler.setup(ctx, budget, rng)
    t_setup = (time.perf_counter() - t_start) * 1000
    n = len(ctx.emb)
    labeled = np.zeros(n, dtype=bool)
    L: list[int] = []
    Y: list[int] = []
    trace, t_sel, t_upd = [], [], []
    while len(L) < budget:
        k = min(batch_size, budget - len(L))
        t0 = time.perf_counter()
        idx = np.asarray(sampler.select(k), dtype=np.int64)
        t1 = time.perf_counter()
        if len(idx) != k or len(np.unique(idx)) != k:
            raise RuntimeError(f"{method}: selected {len(idx)} rows ({len(np.unique(idx))} unique), expected {k}")
        if labeled[idx].any() or not ctx.pool_mask[idx].all():
            raise RuntimeError(f"{method}: selected an already labeled row or a row outside the pool")
        y = labeler.label(idx)
        t2 = time.perf_counter()
        sampler.update(idx, y)
        t3 = time.perf_counter()
        labeled[idx] = True
        L.extend(idx.tolist())
        Y.extend(y.tolist())
        t_sel.append((t1 - t0) * 1000)
        t_upd.append((t3 - t2) * 1000)
        Li = np.asarray(L)
        if ctx.has_full_gold:
            gold_pos = Li[ctx.y[Li] == 1]
            st = ctx.subtype[gold_pos]
            cum_gold, cum_st = int(gold_pos.size), int(np.unique(st[st >= 0]).size)
        else:
            cum_gold, cum_st = "", ""
        trace.append({"round": len(trace) + 1, "used": len(L), "cum_pos": int(np.sum(Y)),
                      "cum_pos_gold": cum_gold, "cum_subtypes": cum_st})

    Li = np.asarray(L)
    Yi = np.asarray(Y)
    t0 = time.perf_counter()
    model = fit_lr(ctx.emb[Li], Yi)
    t_final = (time.perf_counter() - t0) * 1000

    ev = (ctx.eval_mask.copy() if ctx.eval_mask is not None else ~labeled) & (ctx.y >= 0)
    yt = ctx.y[ev]
    yp = (predict_proba(model, ctx.emb[ev]) >= 0.5).astype(int) if model is not None else np.zeros(len(yt), int)
    p, r, f1 = prf(yt, yp)
    e2e = ("", "", "")
    if ctx.has_full_gold:  # query result over the whole table: bought labels + proxy elsewhere
        pred = (predict_proba(model, ctx.emb) >= 0.5).astype(int) if model is not None else np.zeros(n, int)
        pred[Li] = Yi
        e2e = prf(ctx.y, pred)
    if ctx.has_full_gold:
        all_st = ctx.subtype[(ctx.y == 1) & ctx.pool_mask]
        n_sub = int(np.unique(all_st[all_st >= 0]).size)
        covered = trace[-1]["cum_subtypes"]
        pos_gold = trace[-1]["cum_pos_gold"]
        coverage = covered / n_sub if n_sub else ""
    else:
        n_sub, covered, pos_gold, coverage = "", "", "", ""
    row = {
        "pos_found": int(Yi.sum()),
        "pos_found_gold": pos_gold,
        "pos_per_call": float(Yi.sum()) / budget,
        "subtypes_covered": covered,
        "n_subtypes": n_sub,
        "subtype_coverage": coverage,
        "proxy_trained": model is not None,
        "proxy_precision": p,
        "proxy_recall": r,
        "proxy_f1": f1,
        "proxy_macro_f1": macro_f1(yt, yp),
        "n_eval": int(ev.sum()),
        "e2e_precision": e2e[0],
        "e2e_recall": e2e[1],
        "e2e_f1": e2e[2],
        "t_setup_ms": t_setup,
        "t_select_ms_mean": float(np.mean(t_sel)),
        "t_update_ms_mean": float(np.mean(t_upd)),
        "t_round_ms_mean": float(np.mean(np.asarray(t_sel) + np.asarray(t_upd))),
        "t_final_train_ms": t_final,
        "t_total_s": time.perf_counter() - t_start,
        "extra_llm_calls": sampler.extra_llm_calls,
    }
    return row, trace, Li


def _append(path: Path, fields: list[str], rows: list[dict]):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def _done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["run_id"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", nargs="*", help="only run these methods from the config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg["exp"]
    out = RESULTS / exp
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_snapshot.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    runs_path, traces_path = out / "runs.csv", out / "traces.csv"
    done = _done_ids(runs_path)

    model_name = cfg.get("embed_model", DEFAULT_MODEL)
    batch_size = int(cfg.get("batch_size", 25))
    budgets = cfg.get("budgets", [500])
    seeds = cfg.get("seeds", [0])
    label_source = cfg.get("label_source", "gold")
    flips = cfg.get("flip_rates", [0.0])
    test_frac = float(cfg.get("test_frac", 0.3))
    methods = {m: p for m, p in cfg["methods"].items() if not args.methods or m in args.methods}
    commit = git_commit()

    for task, tcfg in cfg["tasks"].items():
        for pos_rate in (tcfg or {}).get("pos_rates", [None]):
            for seed in seeds:
                ctx = None
                for flip in flips:
                    for budget in budgets:
                        for m, params in methods.items():
                            base, full = resolve(m, params)
                            ph = hashlib.sha1(json.dumps(full, sort_keys=True).encode()).hexdigest()[:8]
                            run_id = (f"{task}|{pr_key(pos_rate)}|b{budget}|{m}|{ph}|s{seed}|{label_source}"
                                      f"|f{flip:g}|t{test_frac:g}")
                            if run_id in done:
                                continue
                            if ctx is None:
                                ctx = build_context(task, pos_rate, seed, model_name, label_source, test_frac)
                            labeler = Labeler(ctx.labels, flip, seed)
                            summary, trace, _ = run_one(ctx, m, params, budget, batch_size, seed, labeler)
                            n_pos = int((ctx.y == 1).sum())
                            if ctx.has_full_gold:
                                base_rate = n_pos / max(1, int((ctx.y >= 0).sum()))
                            else:  # gold only on the test set: report the pool rate under the labeler
                                base_rate = float((ctx.labels[ctx.pool_mask] == 1).mean())
                            row = {
                                "run_id": run_id, "exp": exp, "task": task, "pos_rate": pr_key(pos_rate),
                                "seed": seed, "budget": budget, "batch_size": batch_size, "method": m,
                                "base_method": base, "params": json.dumps(full, sort_keys=True),
                                "label_source": label_source, "flip_rate": flip, "test_frac": test_frac,
                                "n_rows": len(ctx.y),
                                "n_pos": n_pos, "base_rate": base_rate,
                                "embed_model": model_name, "git_commit": commit, **summary,
                            }
                            _append(runs_path, RUN_FIELDS, [row])
                            _append(traces_path, TRACE_FIELDS, [{"run_id": run_id, **t} for t in trace])
                            done.add(run_id)
                            cov = f"{summary['subtypes_covered']}/{summary['n_subtypes']}" if summary["n_subtypes"] != "" else "-"
                            print(
                                f"{task:14s} pr={pr_key(pos_rate):7s} s={seed} b={budget} {m:18s} "
                                f"pos={summary['pos_found']:4d} types={cov:5s} f1={summary['proxy_f1']:.3f} "
                                f"e2e={summary['e2e_f1'] if summary['e2e_f1'] == '' else round(summary['e2e_f1'], 3)} "
                                f"ms/round={summary['t_round_ms_mean']:.1f}",
                                flush=True,
                            )


if __name__ == "__main__":
    main()
