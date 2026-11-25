"""Samplers decide which rows are sent to the (LLM) labeler, within a fixed budget.

A sampler sees embeddings, the query embedding, texts, the pool mask and the labels it has
bought so far. It never reads ``ctx.y`` or ``ctx.subtype`` (gold labels, used for evaluation).

Methods
  random     uniform sample (default in Chung et al.)
  topk       rows most similar to the query embedding (Top-K in Chung et al., Table 10)
  keyword    rows matching a keyword list (knowledge-based sampling, KBS, in LTS)
  seals_mlp  SEALS: candidates = kNN of labeled rows, pick most-likely positive (LR)
  lts_lr     LTS with BERT replaced by logistic regression: LDA clusters, Thompson sampling,
             reward = validation-F1 improvement, half predicted positives per batch
  qsb        Query-Seeded Bandit (ours): k-means clusters, query prior, reward = positive found,
             per-slot Thompson sampling, in-cluster LR score + kNN bonus, 15% random
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .metrics import prf
from .paths import CONFIGS
from .proxy import fit_lr, predict_proba


class Sampler:
    def __init__(self, **params):
        self.params = params
        self.extra_llm_calls = 0

    def setup(self, ctx, budget: int, rng: np.random.Generator):
        self.ctx, self.budget, self.rng = ctx, budget, rng
        self.N = len(ctx.emb)
        self.avail = ctx.pool_mask.copy()
        self.L_idx: list[int] = []
        self.L_y: list[int] = []

    def select(self, k: int) -> np.ndarray:
        raise NotImplementedError

    def update(self, idx, y):
        idx = np.asarray(idx, dtype=np.int64)
        self.avail[idx] = False
        self.L_idx.extend(idx.tolist())
        self.L_y.extend(np.asarray(y).astype(int).tolist())

    # ------------------------------------------------------------------ helpers
    def _random(self, k: int, mask: np.ndarray) -> np.ndarray:
        cand = np.flatnonzero(mask)
        k = min(int(k), len(cand))
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        return self.rng.choice(cand, size=k, replace=False)

    def _fill(self, chosen, k: int) -> np.ndarray:
        """Deduplicate ``chosen`` and top it up with random available rows."""
        chosen = list(dict.fromkeys(int(i) for i in chosen))[:k]
        if len(chosen) < k:
            mask = self.avail.copy()
            mask[chosen] = False
            chosen += self._random(k - len(chosen), mask).tolist()
        return np.asarray(chosen, dtype=np.int64)

    def _query_order(self) -> np.ndarray:
        return np.argsort(-(self.ctx.emb @ self.ctx.q), kind="stable")


class RandomSampler(Sampler):
    def select(self, k):
        return self._fill([], k)


class TopKSampler(Sampler):
    def setup(self, ctx, budget, rng):
        super().setup(ctx, budget, rng)
        self.order = self._query_order()
        self.ptr = 0

    def select(self, k):
        out = []
        while len(out) < k and self.ptr < len(self.order):
            i = int(self.order[self.ptr])
            self.ptr += 1
            if self.avail[i]:
                out.append(i)
        return self._fill(out, k)


def load_keywords(task: str, path: str | None = None) -> list[str]:
    p = Path(path) if path else CONFIGS / "keywords" / f"{task}.txt"
    words = [w.strip() for w in p.read_text(encoding="utf-8").splitlines()]
    return [w for w in words if w and not w.startswith("#")]


class KeywordSampler(Sampler):
    """Uniform sample among rows that contain at least one keyword; random once exhausted."""

    def setup(self, ctx, budget, rng):
        super().setup(ctx, budget, rng)
        kws = self.params.get("keywords") or load_keywords(ctx.task, self.params.get("keywords_file"))
        key = ("keyword_hits", tuple(kws))
        if key not in ctx.cache:
            rx = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(w) for w in kws), re.IGNORECASE)
            ctx.cache[key] = np.fromiter((bool(rx.search(t)) for t in ctx.texts), bool, len(ctx.texts))
        self.hits = ctx.cache[key]

    def select(self, k):
        return self._fill(self._random(k, self.hits & self.avail), k)


class SealsSampler(Sampler):
    """SEALS (Coleman et al., AAAI 2022) with the MLP (most-likely positive) criterion.

    No labeled positives exist at the start of a query, so the first ``seed_frac * budget``
    labels come from the query Top-K (same seed budget as QSB)."""

    def setup(self, ctx, budget, rng):
        super().setup(ctx, budget, rng)
        p = {"seed_frac": 0.1, "knn_k": 100, **self.params}
        self.n_seed = int(round(p["seed_frac"] * budget))
        self.knn_k = int(p["knn_k"])
        self.order = self._query_order()
        self.ptr = 0
        self.qsim = ctx.emb @ ctx.q
        self.in_pool = np.zeros(self.N, dtype=bool)
        self.model = None

    def select(self, k):
        if len(self.L_idx) < self.n_seed:
            out = []
            while len(out) < k and self.ptr < len(self.order):
                i = int(self.order[self.ptr])
                self.ptr += 1
                if self.avail[i]:
                    out.append(i)
            return self._fill(out, k)
        cand = np.flatnonzero(self.in_pool & self.avail)
        if cand.size == 0:
            return self._fill([], k)
        score = predict_proba(self.model, self.ctx.emb[cand]) if self.model is not None else self.qsim[cand]
        return self._fill(cand[np.argsort(-score, kind="stable")[:k]], k)

    def update(self, idx, y):
        super().update(idx, y)
        self.in_pool[self.ctx.neighbors(idx, self.knn_k)] = True
        self.model = fit_lr(self.ctx.emb[self.L_idx], self.L_y)


QSB_DEFAULTS = dict(
    cluster_method="kmeans",  # kmeans | lda
    n_clusters=50,
    prior="query",  # query | none
    a0=1.0,
    b0=1.0,
    prior_strength=5.0,
    reward="positive",  # positive | f1_gain
    per_slot=True,  # Thompson draw per batch slot (True) or one cluster per batch (False, LTS)
    in_cluster="score",  # score (LR prob + kNN bonus) | lts_half (half predicted positive, random)
    knn_bonus=0.5,
    knn_k=50,
    seed_frac=0.1,
    seed_cap=3,
    random_frac=0.15,
    val_frac=0.0,
    decay=1.0,
    f1_baseline=0.5,
)

LTS_LR_DEFAULTS = dict(
    QSB_DEFAULTS,
    cluster_method="lda",
    n_clusters=10,
    prior="none",
    a0=0.5,
    b0=0.5,
    prior_strength=0.0,
    reward="f1_gain",
    per_slot=False,
    in_cluster="lts_half",
    knn_bonus=0.0,
    seed_frac=0.0,
    seed_cap=0,
    random_frac=0.0,
    val_frac=0.1,
    decay=0.99,
)

PRESETS = {"qsb": QSB_DEFAULTS, "lts_lr": LTS_LR_DEFAULTS}


class ClusterBandit(Sampler):
    """Cluster-level Thompson sampling; covers both LTS-LR and QSB through its parameters."""

    def setup(self, ctx, budget, rng):
        super().setup(ctx, budget, rng)
        p = self.cfg = dict(self.params)
        if p["in_cluster"] == "lts_half" and p["per_slot"]:
            raise ValueError("in_cluster='lts_half' needs per_slot=False")
        cl = self.cl = ctx.clusters(p["cluster_method"], int(p["n_clusters"]))
        K = self.K = int(cl.max()) + 1
        self.members = [np.flatnonzero(cl == c) for c in range(K)]
        self.qsim = ctx.emb @ ctx.q
        if p["prior"] == "query":
            M = sp.csr_matrix((np.ones(self.N, dtype=np.float32), (cl, np.arange(self.N))), shape=(K, self.N))
            cent = np.asarray(M @ ctx.emb)
            cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-12)
            s = cent @ ctx.q
            pn = (s - s.min()) / (s.max() - s.min() + 1e-12)
            self.alpha0 = p["a0"] + p["prior_strength"] * pn
            self.beta0 = p["b0"] + p["prior_strength"] * (1.0 - pn)
        else:
            self.alpha0 = np.full(K, float(p["a0"]))
            self.beta0 = np.full(K, float(p["b0"]))
        self.wins = np.zeros(K)
        self.losses = np.zeros(K)

        pool = np.flatnonzero(ctx.pool_mask)
        self.is_val = np.zeros(self.N, dtype=bool)
        val_idx = np.empty(0, dtype=np.int64)
        if p["reward"] == "f1_gain":
            n_val = min(len(pool), max(2, int(round(p["val_frac"] * budget))))
            val_idx = rng.choice(pool, size=n_val, replace=False)
            self.is_val[val_idx] = True
        self.val_lab: dict[int, int] = {}

        seeds = []
        n_seed = int(round(p["seed_frac"] * budget))
        if n_seed > 0:
            cap = int(p["seed_cap"]) if p["seed_cap"] > 0 else n_seed
            cnt = np.zeros(K, dtype=int)
            for i in np.argsort(-self.qsim, kind="stable"):
                if not ctx.pool_mask[i] or self.is_val[i] or cnt[cl[i]] >= cap:
                    continue
                seeds.append(int(i))
                cnt[cl[i]] += 1
                if len(seeds) >= n_seed:
                    break
        self.queue = [int(i) for i in val_idx] + seeds
        self.model = None
        self.f1_base = float(p["f1_baseline"])
        self.nbr = np.zeros(self.N, dtype=bool)
        self.pulls: list[int] = []
        q = self.qsim
        self.qnorm = (q - q.min()) / (q.max() - q.min() + 1e-12)

    def _scores(self) -> np.ndarray:
        base = predict_proba(self.model, self.ctx.emb) if self.model is not None else self.qnorm
        return base + float(self.cfg["knn_bonus"]) * self.nbr

    def select(self, k):
        p = self.cfg
        out: list[int] = []
        while self.queue and len(out) < k:
            i = self.queue.pop(0)
            if self.avail[i]:
                out.append(i)
        self.pulls = []
        if len(out) >= k:
            return np.asarray(out, dtype=np.int64)
        rem = k - len(out)
        n_band = rem - int(round(p["random_frac"] * rem))
        avail = self.avail.copy()
        avail[out] = False
        cnt = np.bincount(self.cl[avail], minlength=self.K)
        alpha = self.alpha0 + self.wins
        beta = self.beta0 + self.losses
        score = self._scores() if p["in_cluster"] == "score" else None

        if p["per_slot"]:
            ranked: dict[int, np.ndarray] = {}
            ptr: dict[int, int] = {}
            for _ in range(n_band):
                active = np.flatnonzero(cnt > 0)
                if active.size == 0:
                    break
                c = int(active[np.argmax(self.rng.beta(alpha[active], beta[active]))])
                if c not in ranked:
                    rows = self.members[c][avail[self.members[c]]]
                    ranked[c] = rows[np.argsort(-score[rows], kind="stable")] if score is not None else self.rng.permutation(rows)
                    ptr[c] = 0
                rows = ranked[c]
                while ptr[c] < len(rows) and not avail[rows[ptr[c]]]:
                    ptr[c] += 1
                if ptr[c] >= len(rows):
                    cnt[c] = 0
                    continue
                i = int(rows[ptr[c]])
                ptr[c] += 1
                out.append(i)
                avail[i] = False
                cnt[c] -= 1
                self.pulls.append(c)
        else:
            active = np.flatnonzero(cnt > 0)
            if active.size:
                c = int(active[np.argmax(self.rng.beta(alpha[active], beta[active]))])
                rows = self.members[c][avail[self.members[c]]]
                if p["in_cluster"] == "score":
                    pick = rows[np.argsort(-score[rows], kind="stable")[:n_band]]
                elif self.model is None:
                    pick = self.rng.choice(rows, size=min(n_band, len(rows)), replace=False)
                else:  # LTS: up to half predicted positives, rest predicted negatives, random within each
                    pr = predict_proba(self.model, self.ctx.emb[rows])
                    pos, neg = rows[pr >= 0.5], rows[pr < 0.5]
                    n_pos = min(len(pos), n_band // 2)
                    n_neg = min(len(neg), n_band - n_pos)
                    n_pos = min(len(pos), n_band - n_neg)
                    pick = np.concatenate([
                        self.rng.choice(pos, size=n_pos, replace=False),
                        self.rng.choice(neg, size=n_neg, replace=False),
                    ]).astype(np.int64)
                out += [int(i) for i in pick]
                self.pulls = [c] * len(pick)
        return self._fill(out, k)

    def update(self, idx, y):
        super().update(idx, y)
        p = self.cfg
        idx = np.asarray(idx, dtype=np.int64)
        y = np.asarray(y).astype(int)
        if p["decay"] != 1.0:
            self.wins *= p["decay"]
            self.losses *= p["decay"]
        val = self.is_val[idx]
        for i, v in zip(idx[val], y[val]):
            self.val_lab[int(i)] = int(v)
        tr_idx, tr_y = idx[~val], y[~val]
        if p["reward"] == "positive":
            np.add.at(self.wins, self.cl[tr_idx], tr_y)
            np.add.at(self.losses, self.cl[tr_idx], 1 - tr_y)

        L = np.asarray(self.L_idx)
        Ly = np.asarray(self.L_y)
        keep = ~self.is_val[L]
        new = fit_lr(self.ctx.emb[L[keep]], Ly[keep])
        if p["reward"] == "f1_gain":
            if self.pulls and self.val_lab:
                vi = np.fromiter(self.val_lab.keys(), dtype=np.int64)
                vy = np.fromiter(self.val_lab.values(), dtype=np.int64)
                pred = (predict_proba(new, self.ctx.emb[vi]) >= 0.5).astype(int) if new is not None else np.zeros(len(vi), int)
                f1 = prf(vy, pred)[2]
                r = 1.0 if f1 > self.f1_base else 0.0
                if r:
                    self.f1_base = f1
                    self.model = new
                for c in self.pulls:
                    self.wins[c] += r
                    self.losses[c] += 1.0 - r
            if self.model is None:
                self.model = new
        else:
            self.model = new

        if p["knn_bonus"] > 0:
            newpos = tr_idx[tr_y == 1]
            if newpos.size:
                self.nbr[self.ctx.neighbors(newpos, int(p["knn_k"]))] = True


SIMPLE = {
    "random": RandomSampler,
    "topk": TopKSampler,
    "keyword": KeywordSampler,
    "seals_mlp": SealsSampler,
}


def resolve(name: str, params: dict | None) -> tuple[str, dict]:
    """Return (base method, full parameter dict) for a method entry in a config."""
    params = dict(params or {})
    base = params.pop("base", name)
    if base in PRESETS:
        unknown = set(params) - set(PRESETS[base])
        if unknown:
            raise KeyError(f"unknown parameters for {name}: {sorted(unknown)}")
        return base, {**PRESETS[base], **params}
    if base in SIMPLE:
        return base, params
    raise KeyError(f"unknown method {name} (base {base})")


def make_sampler(name: str, params: dict | None = None) -> Sampler:
    base, full = resolve(name, params)
    return ClusterBandit(**full) if base in PRESETS else SIMPLE[base](**full)
