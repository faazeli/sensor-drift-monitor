"""Download and unpack the UCI Gas Sensor Array Drift dataset (CC BY 4.0) into data/Dataset."""

import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(out)
    print(f"Done: {sorted(p.name for p in (out / 'Dataset').iterdir())}")
