"""Batch-level drift metrics: does a new batch still look like the data the model was built on?

- PSI (population stability index), averaged over features: a standard monitoring metric
  in industry (rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift).
- Domain-classifier AUC (classifier two-sample test): train a classifier to tell reference
  from new data. AUC ≈ 0.5 means indistinguishable; AUC → 1 means the data has moved.
- Alarm rate: share of the batch flagged by a calibrated anomaly detector.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


def psi(ref: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    """Mean PSI across columns, using reference-quantile bins."""
    vals = []
    for j in range(ref.shape[1]):
        edges = np.unique(np.quantile(ref[:, j], np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(ref[:, j], edges)[0] / len(ref) + 1e-6
        q = np.histogram(new[:, j], edges)[0] / len(new) + 1e-6
        vals.append(np.sum((q - p) * np.log(q / p)))
    return float(np.mean(vals))


def domain_auc(ref: np.ndarray, new: np.ndarray, seed: int = 0) -> float:
    """Cross-validated AUC of a classifier separating `ref` from `new` (balanced by subsampling)."""
    rng = np.random.default_rng(seed)
    n = min(len(ref), len(new))
    a = ref[rng.choice(len(ref), n, replace=False)]
    b = new[rng.choice(len(new), n, replace=False)]
    X = np.vstack([a, b])
    y = np.r_[np.zeros(n), np.ones(n)]
    proba = cross_val_predict(HistGradientBoostingClassifier(random_state=seed), X, y, cv=5, method="predict_proba")
    return float(roc_auc_score(y, proba[:, 1]))
