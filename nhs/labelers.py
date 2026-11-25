"""Labelers: the only way a sampler learns labels. Every returned label counts as one LLM call."""
from __future__ import annotations

import numpy as np


class Labeler:
    """Returns pre-computed labels for requested rows.

    ``labels`` are what the LLM would answer: gold labels (simulated perfect LLM) or labels
    cached from a real LLM run (see nhs.llm_label). ``flip_rate`` flips a fixed random subset of
    the labels to simulate a weaker LLM; the same row always gets the same answer.
    """

    def __init__(self, labels, flip_rate: float = 0.0, seed: int = 0):
        lab = np.asarray(labels, dtype=np.int64).copy()
        if flip_rate > 0:
            rng = np.random.default_rng([seed, 4242])
            flip = (rng.random(len(lab)) < flip_rate) & (lab >= 0)
            lab[flip] = 1 - lab[flip]
        self.labels = lab
        self.calls = 0

    def label(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        out = self.labels[idx]
        if (out < 0).any():
            raise ValueError(f"{int((out < 0).sum())} requested rows have no label")
        self.calls += len(idx)
        return out.copy()
