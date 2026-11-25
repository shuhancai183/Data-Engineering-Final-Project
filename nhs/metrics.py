"""Classification metrics with explicit zero-division handling."""
from __future__ import annotations

import numpy as np


def prf(y_true, y_pred) -> tuple[float, float, float]:
    """Precision, recall and F1 of the positive class (0 when undefined)."""
    t = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    tp = int(((t == 1) & (p == 1)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    fn = int(((t == 1) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def macro_f1(y_true, y_pred) -> float:
    """Mean F1 over the positive and the negative class."""
    t = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    return (prf(t, p)[2] + prf(1 - t, 1 - p)[2]) / 2
