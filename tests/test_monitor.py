"""Fast tests on synthetic data shaped like the real dataset (no download needed)."""

import numpy as np
import pytest

from sensor_drift_monitor import data, faults
from sensor_drift_monitor.detectors import PCASPE, PCAT2, Autoencoder, IForest, RobustZ
from sensor_drift_monitor.drift import domain_auc, psi

D = len(data.FEATURES)


@pytest.fixture(scope="module")
def healthy():
    """Correlated 'sensor array' data: 6 latent factors drive 128 features."""
    rng = np.random.default_rng(0)
    W = rng.normal(size=(6, D))
    return rng.normal(size=(800, 6)) @ W + 0.1 * rng.normal(size=(800, D))


def test_parse_line_handles_concentration_suffix():
    y, x = data.parse_line("3;50.000000 1:2.5 128:-1.0")
    assert y == 3 and x[0] == 2.5 and x[127] == -1.0 and x.shape == (128,)


def test_sensor_columns():
    cols = data.sensor_columns(2)
    assert len(cols) == 8 and all(c.startswith("s02_") for c in cols)


@pytest.mark.parametrize("fault", list(faults.FAULTS))
def test_inject_changes_exactly_one_sensor(healthy, fault):
    X, ref = healthy[:50], healthy[50:]
    Xf, is_fault = faults.inject(X, ref, fault, rate=1.0, seed=1)
    assert is_fault.all()
    changed = np.abs(Xf - X) > 1e-12
    sensors_touched = changed.reshape(len(X), data.N_SENSORS, data.FEATS_PER_SENSOR).any(axis=2).sum(axis=1)
    assert (sensors_touched <= 1).all()


@pytest.mark.parametrize("det", [RobustZ(), PCASPE(), IForest(n_estimators=50), Autoencoder(epochs=20)])
def test_detectors_flag_spikes_and_respect_false_alarm_rate(healthy, det):
    train, calib, test = healthy[:400], healthy[400:600], healthy[600:]
    det.fit(train)
    det.calibrate(calib, quantile=0.95)
    assert det.alarms(test).mean() < 0.15  # false alarms on healthy data stay near 5%
    Xf, y = faults.inject(test, train, "spike", rate=0.5, seed=0)
    s = det.score(Xf)
    assert s[y].mean() > s[~y].mean()


def test_pca_spe_catches_broken_correlation(healthy):
    """One sensor disagreeing with the others is invisible to per-feature stats but obvious to SPE."""
    det = PCASPE().fit(healthy[:600])
    Xf, y = faults.inject(healthy[600:], healthy[:600], "stuck", rate=0.5, seed=0)
    s = det.score(Xf)
    assert s[y].mean() > 3 * s[~y].mean()


def test_pca_t2_catches_in_subspace_excursions(healthy):
    """Samples that keep the correlation structure but move far along it: T^2 sees them, SPE does not."""
    t2, spe = PCAT2().fit(healthy[:600]), PCASPE().fit(healthy[:600])
    test = healthy[600:]
    # Push every sample 6 standard deviations along the first principal direction (in raw units).
    pc1 = t2.pca.components_[0] * t2.scaler.scale_
    far = test + 6 * np.sqrt(t2.pca.explained_variance_[0]) * pc1
    assert t2.score(far).mean() > 5 * t2.score(test).mean()
    assert spe.score(far).mean() < 1.5 * spe.score(test).mean()


def test_drift_metrics(healthy):
    same = psi(healthy[:400], healthy[400:]), domain_auc(healthy[:400], healthy[400:])
    shifted = healthy[400:] * 0.7 + 1.0
    moved = psi(healthy[:400], shifted), domain_auc(healthy[:400], shifted)
    assert same[0] < 0.1 and same[1] < 0.65
    assert moved[0] > 0.25 and moved[1] > 0.95
