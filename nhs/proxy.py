"""The proxy model: logistic regression on embeddings with balanced class weights (as in Chung et al.)."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_lr(X: np.ndarray, y, C: float = 1.0):
    """Fit the proxy; returns None when the labels contain a single class."""
    y = np.asarray(y)
    if y.size == 0 or y.min() == y.max():
        return None
    model = LogisticRegression(class_weight="balanced", max_iter=1000, C=C)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """P(positive) for every row of X (fast path for a fitted binary LogisticRegression)."""
    z = X @ model.coef_.ravel() + model.intercept_[0]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
