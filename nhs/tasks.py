"""Task loading, positive-rate control and the per-run context object."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from .paths import CACHE, DATA, DEFAULT_MODEL, model_short


@dataclass
class TaskData:
    name: str
    meta: dict
    texts: list
    y: np.ndarray
    subtype: np.ndarray
    emb: np.ndarray
    q: np.ndarray
    eval_mask: np.ndarray | None
    emb_src: str


@lru_cache(maxsize=4)
def load_texts(src: str) -> list:
    with open(DATA / src / "texts.jsonl", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


@lru_cache(maxsize=2)
def load_task(name: str, model: str = DEFAULT_MODEL) -> TaskData:
    d = DATA / name
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    src = meta.get("emb_from") or name
    short = model_short(model)
    y = np.load(d / "y.npy").astype(np.int64)
    subtype = np.load(d / "subtype.npy").astype(np.int64)
    emb = np.load(DATA / src / f"emb_{short}.npy").astype(np.float32, copy=False)
    q = np.load(d / f"query_emb_{short}.npy").astype(np.float32)
    ev = np.load(d / "eval_mask.npy") if (d / "eval_mask.npy").exists() else None
    return TaskData(name, meta, load_texts(src), y, subtype, emb, q, ev, src)


def subset_indices(y: np.ndarray, subtype: np.ndarray, pos_rate: float | None, seed: int) -> np.ndarray:
    """Keep every negative and a subtype-stratified random subset of positives so that
    n_pos / (n_pos + n_neg) ~= pos_rate. ``pos_rate=None`` keeps all rows."""
    if pos_rate is None:
        return np.arange(len(y))
    if (y < 0).any():
        raise ValueError("positive-rate control needs gold labels on every row")
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    target = max(1, int(round(pos_rate * len(neg) / (1.0 - pos_rate))))
    if target > len(pos):
        raise ValueError(f"pos_rate={pos_rate} is above the natural rate {len(pos) / len(y):.4f}")
    rng = np.random.default_rng([seed, 7919])
    st = subtype[pos]
    types, counts = np.unique(st, return_counts=True)
    share = counts / counts.sum() * target
    alloc = np.floor(share).astype(int)
    rest = target - int(alloc.sum())
    if rest > 0:
        alloc[np.argsort(-(share - alloc), kind="stable")[:rest]] += 1
    if target >= len(types):  # keep at least one positive of every subtype
        for j in np.flatnonzero(alloc == 0):
            donor = int(np.argmax(alloc))
            alloc[donor] -= 1
            alloc[j] += 1
    alloc = np.minimum(alloc, counts)
    chosen = [rng.choice(pos[st == t], size=a, replace=False) for t, a in zip(types, alloc) if a > 0]
    return np.sort(np.concatenate([neg, *chosen]))


@dataclass
class Context:
    """Everything one experimental run may use.

    Samplers may read ``emb``, ``q``, ``texts``, ``pool_mask`` and call ``neighbors`` /
    ``clusters``. ``y`` and ``subtype`` are gold labels for evaluation only; labels reach a
    sampler exclusively through the Labeler.
    """

    task: str
    pos_rate: float | None
    seed: int
    emb: np.ndarray
    q: np.ndarray
    texts: list
    y: np.ndarray
    subtype: np.ndarray
    labels: np.ndarray
    pool_mask: np.ndarray
    eval_mask: np.ndarray | None = None
    has_full_gold: bool = True
    idx: np.ndarray | None = None
    emb_src: str | None = None
    cache_clusters: bool = True
    cache: dict = field(default_factory=dict)

    def neighbors(self, rows, k: int) -> np.ndarray:
        """Flat array with the k nearest neighbours (cosine) of each row, excluding itself."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return np.empty(0, dtype=np.int64)
        k = int(min(k, len(self.emb) - 1))
        sims = self.emb @ self.emb[rows].T
        sims[rows, np.arange(len(rows))] = -np.inf
        nb = np.argpartition(-sims, k - 1, axis=0)[:k]
        return nb.T.ravel()

    def clusters(self, method: str, n_clusters: int) -> np.ndarray:
        """Cluster id of every row, fitted on this (sub)table only and cached on disk."""
        key = (method, n_clusters)
        if key in self.cache:
            return self.cache[key]
        path = None
        if self.cache_clusters:
            pr = "nat" if self.pos_rate is None else f"{self.pos_rate:g}"
            path = CACHE / "clusters" / self.task / f"{pr}_s{self.seed}_{method}{n_clusters}.npy"
            if path.exists():
                self.cache[key] = np.load(path)
                return self.cache[key]
        if method == "kmeans":
            lab = _kmeans(self.emb, n_clusters, self.seed)
        elif method == "lda":
            lab = _lda(self, n_clusters)
        else:
            raise ValueError(f"unknown cluster method {method}")
        lab = np.unique(lab, return_inverse=True)[1].astype(np.int64)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, lab)
        self.cache[key] = lab
        return lab


def _kmeans(emb: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    k = min(k, len(emb))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3, max_iter=100)
    return km.fit_predict(emb)


def _doc_term(ctx: Context):
    from sklearn.feature_extraction.text import CountVectorizer

    if ctx.idx is None or ctx.emb_src is None:  # synthetic contexts in tests
        return CountVectorizer(min_df=1).fit_transform(ctx.texts)
    path = CACHE / "dtm" / f"{ctx.emb_src}.npz"
    if path.exists():
        X = sp.load_npz(path)
    else:
        vec = CountVectorizer(max_features=5000, stop_words="english", min_df=5, max_df=0.5)
        X = vec.fit_transform(load_texts(ctx.emb_src)).tocsr()
        path.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path, X)
    return X[ctx.idx]


def _lda(ctx: Context, k: int) -> np.ndarray:
    from sklearn.decomposition import LatentDirichletAllocation

    X = _doc_term(ctx)
    rng = np.random.default_rng([ctx.seed, 31])
    fit_rows = rng.choice(X.shape[0], size=min(X.shape[0], 10000), replace=False)
    lda = LatentDirichletAllocation(
        n_components=k, learning_method="online", max_iter=10, batch_size=1024, random_state=ctx.seed
    )
    lda.fit(X[fit_rows])
    return lda.transform(X).argmax(axis=1)


def holdout_mask(y: np.ndarray, subtype: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Stratified random test set (negatives, and positives per subtype) that is never sampled."""
    rng = np.random.default_rng([seed, 2027])
    mask = np.zeros(len(y), dtype=bool)
    strata = np.where(y == 1, subtype, -2)
    for s in np.unique(strata):
        rows = np.flatnonzero(strata == s)
        n = int(round(frac * len(rows)))
        if n:
            mask[rng.choice(rows, size=n, replace=False)] = True
    return mask


def build_context(task: str, pos_rate: float | None = None, seed: int = 0,
                  model: str = DEFAULT_MODEL, label_source: str = "gold", test_frac: float = 0.3) -> Context:
    """Sub-table for (task, pos_rate, seed). Rows are split into a sampling pool and a fixed test set:
    the expert-labeled rows when the task has them, otherwise a stratified random ``test_frac``."""
    td = load_task(task, model)
    if label_source == "gold":
        labels = td.y
    else:  # cached LLM labels, e.g. label_source="llm_gpt-4o-mini" -> data/<task>/labels_llm_gpt-4o-mini.npy
        labels = np.load(DATA / task / f"labels_{label_source}.npy").astype(np.int64)
    if td.eval_mask is not None and pos_rate is not None:
        # fixed expert test set (lts_shark): lower the rate inside the pool only, using the labels the
        # labeler returns there; the expert test rows are kept unchanged
        pool_idx = np.flatnonzero(~td.eval_mask)
        lab = labels[pool_idx]
        keep = subset_indices(lab, np.where(lab == 1, 0, -1), pos_rate, seed)
        idx = np.sort(np.concatenate([pool_idx[keep], np.flatnonzero(td.eval_mask)]))
    else:
        idx = subset_indices(td.y, td.subtype, pos_rate, seed)
    if td.eval_mask is not None:
        eval_mask = td.eval_mask[idx]
    elif test_frac > 0:
        eval_mask = holdout_mask(td.y[idx], td.subtype[idx], test_frac, seed)
    else:
        eval_mask = None
    pool = np.ones(len(idx), dtype=bool) if eval_mask is None else ~eval_mask
    return Context(
        task=task,
        pos_rate=pos_rate,
        seed=seed,
        emb=td.emb[idx],
        q=td.q,
        texts=[td.texts[i] for i in idx],
        y=td.y[idx],
        subtype=td.subtype[idx],
        labels=labels[idx],
        pool_mask=pool,
        eval_mask=eval_mask,
        has_full_gold=bool(td.meta.get("has_full_gold", True)),
        idx=idx,
        emb_src=td.emb_src,
    )
