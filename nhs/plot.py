"""Figures and a markdown summary from results/<exp>/runs.csv and traces.csv.

Usage:
  python -m nhs.plot --exp main_20ng main_toxic
Writes results/<exp>/figures/*.png and results/<exp>/summary.md.
Only needs pandas + matplotlib (no torch).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Fixed categorical order (reference palette, light mode). Colour follows the method, never its rank.
COLORS = {
    "qsb": "#2a78d6",
    "random": "#eb6834",
    "topk": "#1baf7a",
    "keyword": "#eda100",
    "seals_mlp": "#e87ba4",
    "lts_lr": "#008300",
}
LABELS = {
    "qsb": "QSB (ours)",
    "random": "Random (Chung)",
    "topk": "Top-K (Chung)",
    "keyword": "Keyword (KBS)",
    "seals_mlp": "SEALS-MLP",
    "lts_lr": "LTS-LR",
}
ORDER = list(COLORS)
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TASK_NAMES = {
    "20ng_comp": "20 Newsgroups (comp.*)",
    "toxic_conv": "Toxic Conversations",
    "jigsaw_threat": "Jigsaw: threat",
    "jigsaw_severe": "Jigsaw: threat / identity hate / severe",
    "lts_shark": "LTS shark ads",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT2,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.color": TEXT2,
    "ytick.color": TEXT2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "font.size": 9,
})


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _methods(df):
    present = list(dict.fromkeys(df["method"]))
    return [m for m in ORDER if m in present] + [m for m in present if m not in ORDER]


def _color(m):
    return COLORS.get(m, TEXT2)


def _label(m):
    return LABELS.get(m, m)


def load(exp: str):
    d = RESULTS / exp
    runs = pd.read_csv(d / "runs.csv")
    traces = pd.read_csv(d / "traces.csv") if (d / "traces.csv").exists() else None
    return runs, traces


def fig_f1_vs_rate(runs, out: Path):
    tasks = list(dict.fromkeys(runs["task"]))
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        df = runs[runs.task == task]
        x_of = df.groupby("pos_rate")["base_rate"].mean()
        for m in _methods(df):
            g = df[df.method == m].groupby("pos_rate")["proxy_f1"].agg(["mean", "std"])
            g["x"] = x_of.reindex(g.index)
            g = g.sort_values("x")
            ax.errorbar(g["x"], g["mean"], yerr=g["std"].fillna(0), color=_color(m), label=_label(m),
                        linewidth=2, marker="o", markersize=5, capsize=2, elinewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("share of positives in the table (log scale)")
        ax.set_ylabel("proxy F1 on unseen rows")
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:g}%"))
        ax.set_title(TASK_NAMES.get(task, task), loc="left")
        _style(ax)
    axes[0][0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "f1_vs_posrate.png", dpi=200)
    plt.close(fig)


def fig_pos_curves(runs, traces, out: Path):
    if traces is None:
        return
    t = traces.merge(runs[["run_id", "task", "pos_rate", "method", "base_rate"]], on="run_id")
    t["found"] = pd.to_numeric(t["cum_pos_gold"], errors="coerce").fillna(t["cum_pos"])
    panels = t.groupby(["task", "pos_rate"])["base_rate"].mean().sort_values(ascending=False)
    panels = panels.reset_index().sort_values(["task", "base_rate"], ascending=[True, False])
    n = len(panels)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows), squeeze=False)
    for ax, (_, pr) in zip(axes.ravel(), panels.iterrows()):
        df = t[(t.task == pr.task) & (t.pos_rate == pr.pos_rate)]
        for m in _methods(df):
            g = df[df.method == m].groupby("used")["found"].mean()
            ax.plot(g.index, g.values, color=_color(m), linewidth=2, label=_label(m))
        ax.set_title(f"{TASK_NAMES.get(pr.task, pr.task)}, {pr.base_rate * 100:.2g}% positives", loc="left", fontsize=9)
        ax.set_xlabel("LLM labels used")
        ax.set_ylabel("positives found")
        _style(ax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0][0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "positives_found_curves.png", dpi=200)
    plt.close(fig)


def fig_coverage(runs, out: Path):
    df = runs[pd.to_numeric(runs["n_subtypes"], errors="coerce") > 1].copy()
    if df.empty:
        return
    df["cov"] = pd.to_numeric(df["subtype_coverage"], errors="coerce")
    groups = df.groupby(["task", "pos_rate"])["base_rate"].mean().reset_index().sort_values(["task", "base_rate"], ascending=[True, False])
    methods = _methods(df)
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(groups), 3.2))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        vals = [df[(df.task == g.task) & (df.pos_rate == g.pos_rate) & (df.method == m)]["cov"].mean() for _, g in groups.iterrows()]
        xs = np.arange(len(groups)) + (j - (len(methods) - 1) / 2) * w
        ax.bar(xs, vals, width=w * 0.92, color=_color(m), label=_label(m))
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([f"{TASK_NAMES.get(g.task, g.task)}\n{g.base_rate * 100:.2g}% positives" for _, g in groups.iterrows()], fontsize=8)
    ax.set_ylabel("share of positive subtypes found")
    ax.set_ylim(0, 1.05)
    _style(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out / "subtype_coverage.png", dpi=200)
    plt.close(fig)


def fig_timing(runs, out: Path):
    g = runs.groupby("method")["t_round_ms_mean"].mean()
    methods = [m for m in _methods(runs) if m in g.index]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh([_label(m) for m in methods], [g[m] for m in methods], color=[_color(m) for m in methods], height=0.6)
    for i, m in enumerate(methods):
        ax.text(g[m] * 1.08, i, f"{g[m]:.1f} ms", va="center", fontsize=8, color=TEXT2)
    ax.set_xscale("log")
    ax.set_xlabel("time per round: select + update (ms, log scale)")
    ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "time_per_round.png", dpi=200)
    plt.close(fig)


def _ms(s: pd.Series, fmt="{:.3f}") -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "-"
    return (fmt + " ± " + fmt).format(s.mean(), s.std(ddof=1) if len(s) > 1 else 0.0)


def summary_md(runs, exp: str, out_md: Path, ref: str = "qsb"):
    lines = [f"# Results: {exp}", "",
             f"Mean ± std over seeds. Budget = {sorted(runs.budget.unique())} LLM labels. "
             "F1 is the proxy's F1 against gold labels on rows it did not label (LTS shark: expert rows). "
             "End-to-end F1 uses bought labels for labeled rows and the proxy for the rest.", ""]
    for (task, pr), df in runs.groupby(["task", "pos_rate"], sort=False):
        lines += [f"## {TASK_NAMES.get(task, task)}, pos_rate = {pr} "
                  f"({df.base_rate.mean() * 100:.3g}% positives, {int(df.n_pos.mean())} of {int(df.n_rows.mean())} rows)", ""]
        lines += ["| method | positives found | subtypes found | proxy F1 | precision | recall | end-to-end F1 | ms / round |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in _methods(df):
            r = df[df.method == m]
            sub = "-" if r["n_subtypes"].isna().all() or (pd.to_numeric(r["n_subtypes"], errors="coerce") <= 1).all() \
                else f"{pd.to_numeric(r.subtypes_covered, errors='coerce').mean():.1f} / {int(pd.to_numeric(r.n_subtypes).max())}"
            lines.append(
                f"| {_label(m)} | {_ms(r.pos_found, '{:.1f}')} | {sub} | {_ms(r.proxy_f1)} | {_ms(r.proxy_precision)} | "
                f"{_ms(r.proxy_recall)} | {_ms(r.e2e_f1)} | {pd.to_numeric(r.t_round_ms_mean).mean():.1f} |")
        if ref in set(df.method):
            a = df[df.method == ref].set_index("seed")
            notes = []
            for m in _methods(df):
                if m == ref:
                    continue
                b = df[df.method == m].set_index("seed")
                common = a.index.intersection(b.index)
                if len(common) == 0:
                    continue
                win_f1 = int((a.loc[common, "proxy_f1"] > b.loc[common, "proxy_f1"]).sum())
                win_pos = int((a.loc[common, "pos_found"] > b.loc[common, "pos_found"]).sum())
                notes.append(f"{_label(m)}: higher F1 in {win_f1}/{len(common)} seeds, more positives in {win_pos}/{len(common)}")
            lines += ["", f"Paired by seed, {_label(ref)} vs " + "; ".join(notes) + "."]
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--ref", default="qsb", help="method used for the paired comparison in summary.md")
    args = ap.parse_args()
    for exp in args.exp:
        runs, traces = load(exp)
        out = RESULTS / exp / "figures"
        out.mkdir(parents=True, exist_ok=True)
        fig_f1_vs_rate(runs, out)
        fig_pos_curves(runs, traces, out)
        fig_coverage(runs, out)
        fig_timing(runs, out)
        ref = args.ref if args.ref in set(runs.method) else runs.method.iloc[0]
        summary_md(runs, exp, RESULTS / exp / "summary.md", ref)
        print(f"[{exp}] figures -> {out}, summary -> {RESULTS / exp / 'summary.md'}")


if __name__ == "__main__":
    main()
