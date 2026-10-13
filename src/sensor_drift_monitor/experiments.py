"""The two studies reported in the README.

1. Fault benchmark: inject each synthetic fault into held-out *healthy* data from the
   reference period and measure how well each detector finds it (ROC-AUC, plus recall
   and precision at an alarm threshold calibrated for a ~1% false-alarm rate).

2. Drift study: freeze a gas classifier and the monitors on batches 1-2 (months 1-10), then
   replay batches 3-10 (months 11-36) batch by batch. Do the label-free drift signals warn us *before*
   we would otherwise notice that the classifier's accuracy is collapsing?
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import BATCH_MONTHS, FEATURES
from .detectors import PCASPE, PCAT2, all_detectors
from .drift import domain_auc, psi
from .faults import FAULTS, inject

REFERENCE_BATCHES = (1, 2)


def reference_split(df: pd.DataFrame, seed: int = 0):
    ref = df[df.batch.isin(REFERENCE_BATCHES)]
    train, rest = train_test_split(ref, test_size=0.4, random_state=seed, stratify=ref.gas)
    calib, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest.gas)
    return train, calib, test


def fault_benchmark(df: pd.DataFrame, seeds=(0, 1, 2, 3, 4), rate: float = 0.2, quantile: float = 0.99) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        train, calib, test = reference_split(df, seed)
        Xtr, Xcal, Xte = (d[FEATURES].to_numpy() for d in (train, calib, test))
        for det in all_detectors(seed):
            det.fit(Xtr)
            det.calibrate(Xcal, quantile)
            for fault in FAULTS:
                Xf, y = inject(Xte, Xtr, fault, rate=rate, seed=seed)
                s = det.score(Xf)
                pred = s > det.threshold
                rows.append({
                    "seed": seed, "detector": det.name, "fault": fault,
                    "roc_auc": roc_auc_score(y, s),
                    "recall": recall_score(y, pred),
                    "precision": precision_score(y, pred, zero_division=0),
                })  # fmt: skip
    return pd.DataFrame(rows)


def drift_study(df: pd.DataFrame, seed: int = 0, quantile: float = 0.99) -> pd.DataFrame:
    train, calib, _ = reference_split(df, seed)
    Xtr = train[FEATURES].to_numpy()
    clf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1).fit(Xtr, train.gas)
    monitors = {"spe": PCASPE().fit(Xtr), "t2": PCAT2().fit(Xtr)}
    for m in monitors.values():
        m.calibrate(calib[FEATURES].to_numpy(), quantile)
    # First row: held-out data from the reference period itself (the "no drift" baseline).
    segments = [("1-2 (held-out)", calib)] + [
        (str(b), df[df.batch == b]) for b in range(max(REFERENCE_BATCHES) + 1, 11)
    ]
    rows = []
    for label, batch in segments:
        X = batch[FEATURES].to_numpy()
        rows.append({
            "batch": label,
            "months": "1-10" if label.startswith("1-2") else BATCH_MONTHS[int(label)],
            "n": len(batch),
            "gas_accuracy": float((clf.predict(X) == batch.gas).mean()),
            "psi": psi(Xtr, X),
            "domain_auc": domain_auc(Xtr, X, seed),
            "spe_alarm_rate": float(monitors["spe"].alarms(X).mean()),
            "t2_alarm_rate": float(monitors["t2"].alarms(X).mean()),
        })  # fmt: skip
    return pd.DataFrame(rows)


def drift_correlations(study: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between each label-free drift signal and the classifier's accuracy."""
    out = []
    for m in ["psi", "domain_auc", "spe_alarm_rate", "t2_alarm_rate"]:
        rho, p = spearmanr(study[m], study.gas_accuracy)
        out.append({"signal": m, "spearman_rho_vs_accuracy": rho, "p_value": p})
    return pd.DataFrame(out)


def summarize_benchmark(res: pd.DataFrame) -> pd.DataFrame:
    return res.groupby(["fault", "detector"])[["roc_auc", "recall", "precision"]].mean().round(3).unstack("detector")


def first_alarm_batch(study: pd.DataFrame, psi_limit: float = 0.25, alarm_limit: float = 0.05) -> dict:
    """When would each rule have fired, versus when accuracy actually fell below 80%?"""
    after = study.iloc[1:]  # skip the reference row

    def first(mask):
        hit = after[mask]
        return hit.batch.iloc[0] if len(hit) else None

    return {
        "psi_above_limit": first(after.psi > psi_limit),
        "spe_alarms_above_limit": first(after.spe_alarm_rate > alarm_limit),
        "t2_alarms_above_limit": first(after.t2_alarm_rate > alarm_limit),
        "accuracy_below_80pct": first(after.gas_accuracy < 0.8),
    }


__all__ = ["drift_correlations", "drift_study", "fault_benchmark", "first_alarm_batch", "summarize_benchmark"]
