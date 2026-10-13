"""Synthetic sensor faults for benchmarking detectors.

Real fault labels are rare in field data, so a standard practice is to inject
realistic, physically motivated faults into healthy data and measure how well each
detector recovers them. Each fault corrupts all 8 features of one randomly chosen
physical sensor, mirroring how a single failing element looks in a sensor array.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .data import FEATS_PER_SENSOR, N_SENSORS


def _sensor_slice(rng: np.random.Generator) -> slice:
    s = rng.integers(N_SENSORS)
    return slice(s * FEATS_PER_SENSOR, (s + 1) * FEATS_PER_SENSOR)


def stuck(x: np.ndarray, ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sensor frozen at a stale reading (taken from an unrelated healthy sample)."""
    sl = _sensor_slice(rng)
    x = x.copy()
    x[sl] = ref[rng.integers(len(ref)), sl]
    return x


def dropout(x: np.ndarray, ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sensor stops responding: all features read zero."""
    sl = _sensor_slice(rng)
    x = x.copy()
    x[sl] = 0.0
    return x


def spike(x: np.ndarray, ref: np.ndarray, rng: np.random.Generator, k: float = 6.0) -> np.ndarray:
    """Transient electrical spike: +k healthy standard deviations on one sensor."""
    sl = _sensor_slice(rng)
    x = x.copy()
    x[sl] += k * ref[:, sl].std(axis=0) * rng.choice([-1, 1])
    return x


def gain(x: np.ndarray, ref: np.ndarray, rng: np.random.Generator, factor: float = 0.4) -> np.ndarray:
    """Sensitivity loss (poisoned element): response scaled down to `factor`."""
    sl = _sensor_slice(rng)
    x = x.copy()
    x[sl] *= factor
    return x


def noise(x: np.ndarray, ref: np.ndarray, rng: np.random.Generator, k: float = 2.0) -> np.ndarray:
    """Loose contact / EMI: heavy additive noise on one sensor."""
    sl = _sensor_slice(rng)
    x = x.copy()
    x[sl] += rng.normal(0, k * ref[:, sl].std(axis=0))
    return x


FAULTS: dict[str, Callable] = {"stuck": stuck, "dropout": dropout, "spike": spike, "gain": gain, "noise": noise}


def inject(
    X: np.ndarray, ref: np.ndarray, fault: str, rate: float = 0.1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt a random `rate` fraction of rows of X with `fault`. Returns (X_corrupted, is_fault)."""
    rng = np.random.default_rng(seed)
    is_fault = rng.random(len(X)) < rate
    Xc = X.copy()
    for i in np.flatnonzero(is_fault):
        Xc[i] = FAULTS[fault](X[i], ref, rng)
    return Xc, is_fault
