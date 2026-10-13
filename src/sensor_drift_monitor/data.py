"""Loader for the UCI Gas Sensor Array Drift dataset (Vergara et al., 2012).

13,910 measurements from an array of 16 metal-oxide gas sensors exposed to 6 gases,
collected in 10 batches over 36 months. Each sensor contributes 8 features
(steady-state resistance change, its normalized form, and 3+3 exponential moving
averages of the rise and decay transients), giving 128 features per sample.

Because the sensors age and drift, a model fitted on early batches gradually stops
matching later data, which makes it a realistic testbed for monitoring sensor health in the field.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GASES = {1: "ethanol", 2: "ethylene", 3: "ammonia", 4: "acetaldehyde", 5: "acetone", 6: "toluene"}
N_SENSORS, FEATS_PER_SENSOR = 16, 8
FEATURE_KINDS = ["dR", "dR_norm", "ema_rise_0.001", "ema_rise_0.01", "ema_rise_0.1",
                 "ema_decay_0.001", "ema_decay_0.01", "ema_decay_0.1"]  # fmt: skip
FEATURES = [f"s{s:02d}_{k}" for s in range(1, N_SENSORS + 1) for k in FEATURE_KINDS]
# Months covered by each batch, from the dataset description.
BATCH_MONTHS = {
    1: "1-2",
    2: "3-10",
    3: "11-13",
    4: "14-15",
    5: "16",
    6: "17-20",
    7: "21",
    8: "22-23",
    9: "24-30",
    10: "36",
}
UCI_URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"


def parse_line(line: str) -> tuple[int, np.ndarray]:
    head, *pairs = line.split()
    label = int(head.split(";")[0])  # some releases append ";concentration" to the label
    x = np.zeros(len(FEATURES))
    for p in pairs:
        i, v = p.split(":")
        x[int(i) - 1] = float(v)
    return label, x


def load(data_dir: str | Path = "data/Dataset") -> pd.DataFrame:
    """Return one row per measurement with columns: batch, gas, gas_name, + 128 features."""
    data_dir = Path(data_dir)
    if not (data_dir / "batch1.dat").exists():
        raise FileNotFoundError(f"Dataset not found in {data_dir}. Download it with: python scripts/download_data.py")
    rows, labels, batches = [], [], []
    for b in range(1, 11):
        for line in (data_dir / f"batch{b}.dat").read_text().splitlines():
            if line.strip():
                y, x = parse_line(line)
                rows.append(x)
                labels.append(y)
                batches.append(b)
    df = pd.DataFrame(np.vstack(rows), columns=FEATURES)
    df.insert(0, "gas_name", [GASES[y] for y in labels])
    df.insert(0, "gas", labels)
    df.insert(0, "batch", batches)
    return df


def sensor_columns(sensor: int) -> list[str]:
    """Feature column indices (0-based, within FEATURES) belonging to one physical sensor (1-16)."""
    start = (sensor - 1) * FEATS_PER_SENSOR
    return FEATURES[start : start + FEATS_PER_SENSOR]
