"""Anomaly detectors with a common interface: fit on healthy data, score new samples
(higher = more anomalous), and calibrate an alarm threshold from healthy scores.

- RobustZ:        max |robust z-score| across features (median/MAD); a strong, explainable baseline
- PCA-SPE:        squared prediction error (Q statistic) from a PCA model: catches samples that
                  break the learned correlation structure (e.g. one sensor disagreeing with the rest)
- PCA-T2:         Hotelling's T^2 inside the PCA subspace: catches samples that keep the structure
                  but move far along it (e.g. the whole array drifting). SPE + T^2 together are the
                  classic multivariate statistical process monitoring (MSPM) pair
- IsolationForest: tree-based outlier isolation (scikit-learn)
- Autoencoder:    small PyTorch MLP autoencoder; reconstruction error as the score
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch import nn


class Detector:
    name = "base"

    def fit(self, X: np.ndarray) -> Detector:
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def calibrate(self, X_healthy: np.ndarray, quantile: float = 0.99) -> float:
        """Alarm threshold = `quantile` of scores on healthy validation data (sets the false-alarm rate)."""
        self.threshold = float(np.quantile(self.score(X_healthy), quantile))
        return self.threshold

    def alarms(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) > self.threshold

    def explain(self, x: np.ndarray, top: int = 3) -> list[tuple[int, float]]:
        """Indices of the features contributing most to a sample's anomaly score (default: robust z)."""
        z = np.abs((x - self._med) / self._mad)
        idx = np.argsort(z)[::-1][:top]
        return [(int(i), float(z[i])) for i in idx]

    def _fit_robust_stats(self, X: np.ndarray) -> None:
        self._med = np.median(X, axis=0)
        self._mad = 1.4826 * np.median(np.abs(X - self._med), axis=0) + 1e-9


class RobustZ(Detector):
    name = "robust-z"

    def fit(self, X):
        self._fit_robust_stats(X)
        return self

    def score(self, X):
        return np.abs((X - self._med) / self._mad).max(axis=1)


class PCASPE(Detector):
    name = "pca-spe"

    def __init__(self, variance: float = 0.95):
        self.variance = variance

    def fit(self, X):
        self._fit_robust_stats(X)
        self.scaler = StandardScaler().fit(X)
        self.pca = PCA(n_components=self.variance, svd_solver="full").fit(self.scaler.transform(X))
        return self

    def score(self, X):
        Z = self.scaler.transform(X)
        recon = self.pca.inverse_transform(self.pca.transform(Z))
        return ((Z - recon) ** 2).sum(axis=1)


class PCAT2(PCASPE):
    name = "pca-t2"

    def score(self, X):
        T = self.pca.transform(self.scaler.transform(X))
        return (T**2 / self.pca.explained_variance_).sum(axis=1)


class IForest(Detector):
    name = "isolation-forest"

    def __init__(self, n_estimators: int = 300, seed: int = 0):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed)

    def fit(self, X):
        self._fit_robust_stats(X)
        self.scaler = StandardScaler().fit(X)
        self.model.fit(self.scaler.transform(X))
        return self

    def score(self, X):
        return -self.model.score_samples(self.scaler.transform(X))


class Autoencoder(Detector):
    name = "autoencoder"

    def __init__(self, bottleneck: int = 12, epochs: int = 200, lr: float = 1e-3, seed: int = 0):
        self.bottleneck, self.epochs, self.lr, self.seed = bottleneck, epochs, lr, seed

    def fit(self, X):
        torch.manual_seed(self.seed)
        self._fit_robust_stats(X)
        self.scaler = StandardScaler().fit(X)
        Z = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        d = Z.shape[1]
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Linear(64, self.bottleneck), nn.GELU(),
            nn.Linear(self.bottleneck, 64), nn.GELU(), nn.Linear(64, d),
        )  # fmt: skip
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        loader = torch.utils.data.DataLoader(Z, batch_size=128, shuffle=True)
        self.net.train()
        for _ in range(self.epochs):
            for xb in loader:
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.net(xb), xb)
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def score(self, X):
        Z = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        return ((self.net(Z) - Z) ** 2).sum(dim=1).numpy()


def all_detectors(seed: int = 0) -> list[Detector]:
    return [RobustZ(), PCASPE(), PCAT2(), IForest(seed=seed), Autoencoder(seed=seed)]
