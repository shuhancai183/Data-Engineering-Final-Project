"""Tests on a small synthetic table (no downloads needed).

Run: python -m pytest tests  (or: python tests/test_core.py)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nhs.labelers import Labeler  # noqa: E402
from nhs.metrics import macro_f1, prf  # noqa: E402
from nhs.run import run_one  # noqa: E402
from nhs.tasks import Context, subset_indices  # noqa: E402

METHODS = {
    "random": {},
    "topk": {},
    "keyword": {"keywords": ["alpha0", "alpha1", "alpha2"]},
    "seals_mlp": {},
    "lts_lr": {},
    "qsb": {},
}


def make_toy(n=3000, d=16, n_pos=30, seed=0, exclude=0) -> Context:
    """Negatives spread over 7 blobs; positives in 3 blobs near the query (3 subtypes)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(10, d)) * 2.0
    comp = np.concatenate([rng.integers(3, 10, size=n - n_pos), np.arange(n_pos) % 3])
    emb = centers[comp] + rng.normal(size=(n, d))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    y = (comp < 3).astype(np.int64)
    subtype = np.where(y == 1, comp, -1)
    q = centers[:3].mean(0)
    q /= np.linalg.norm(q)
    texts = []
    for c in comp:
        words = [f"w{c}_{j}" for j in rng.integers(0, 20, size=8)]
        if c < 3 and rng.random() < 0.5:
            words.append(f"alpha{c}")
        texts.append(" ".join(words))
    perm = rng.permutation(n)
    pool = np.ones(n, dtype=bool)
    pool[:exclude] = False
    return Context(
        task="toy", pos_rate=None, seed=seed, emb=emb[perm].astype(np.float32), q=q.astype(np.float32),
        texts=[texts[i] for i in perm], y=y[perm], subtype=subtype[perm], labels=y[perm],
        pool_mask=pool, cache_clusters=False,
    )


def test_subset_indices_rate_and_strata():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(9000, int), np.ones(1000, int)]
    subtype = np.r_[-np.ones(9000, int), rng.integers(0, 4, size=1000)]
    idx = subset_indices(y, subtype, 0.01, seed=3)
    rate = y[idx].mean()
    assert abs(rate - 0.01) < 0.002
    assert set(subtype[idx][y[idx] == 1]) == {0, 1, 2, 3}
    assert np.array_equal(idx, subset_indices(y, subtype, 0.01, seed=3))
    assert not np.array_equal(idx, subset_indices(y, subtype, 0.01, seed=4))
    assert len(subset_indices(y, subtype, None, 0)) == len(y)


def test_samplers_respect_budget_pool_and_determinism():
    for m, params in METHODS.items():
        ctx = make_toy(exclude=200)
        row, trace, L = run_one(ctx, m, params, budget=200, batch_size=25, seed=1, labeler=Labeler(ctx.labels))
        assert len(L) == 200 and len(np.unique(L)) == 200, m
        assert ctx.pool_mask[L].all(), m
        assert trace[-1]["used"] == 200, m
        ctx2 = make_toy(exclude=200)
        _, _, L2 = run_one(ctx2, m, params, budget=200, batch_size=25, seed=1, labeler=Labeler(ctx2.labels))
        assert np.array_equal(L, L2), f"{m} is not deterministic"


def test_samplers_never_read_gold_labels():
    for m, params in METHODS.items():
        ctx = make_toy()
        _, _, L = run_one(ctx, m, params, budget=150, batch_size=25, seed=2, labeler=Labeler(ctx.labels))
        ctx2 = make_toy()
        rng = np.random.default_rng(9)
        ctx2.y = rng.permutation(ctx2.y)
        ctx2.subtype = rng.permutation(ctx2.subtype)
        _, _, L2 = run_one(ctx2, m, params, budget=150, batch_size=25, seed=2, labeler=Labeler(ctx2.labels))
        assert np.array_equal(L, L2), f"{m} reads gold labels"


def test_qsb_finds_more_positives_than_random():
    found = {"random": [], "qsb": []}
    for s in range(3):
        for m in found:
            ctx = make_toy(seed=s)
            row, _, _ = run_one(ctx, m, {}, budget=200, batch_size=25, seed=s, labeler=Labeler(ctx.labels))
            found[m].append(row["pos_found"])
    assert np.mean(found["qsb"]) > 2 * np.mean(found["random"]), found


def test_flip_rate_is_consistent():
    lab = np.r_[np.zeros(500, int), np.ones(500, int)]
    a = Labeler(lab, flip_rate=0.2, seed=1)
    b = Labeler(lab, flip_rate=0.2, seed=1)
    assert np.array_equal(a.labels, b.labels)
    assert 0.15 < (a.labels != lab).mean() < 0.25


def test_metrics():
    assert prf([1, 1, 0, 0], [1, 0, 1, 0]) == (0.5, 0.5, 0.5)
    assert prf([0, 0], [0, 0]) == (0.0, 0.0, 0.0)
    assert abs(macro_f1([1, 0], [1, 0]) - 1.0) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
