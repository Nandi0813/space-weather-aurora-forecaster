"""Train and back-test a 3-hour-ahead Kp forecaster from solar-wind data.

Data:
  - Solar wind: NASA OMNI2 hourly (ACE/DSCOVR at L1, time-shifted to Earth's bow shock).
  - Target Kp: GFZ Potsdam definitive Kp archive (kp.gfz.de).

Output (consumed by the web app in docs/):
  - docs/model.json     Ridge coefficients + scaler, run live in the browser
  - docs/backtest.json  metrics and storm back-test series
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data")
DOCS = Path("docs")
YEARS = range(2010, 2026)
TRAIN_END = "2023-01-01"  # train: 2010-2022, test: 2023-2025 (includes May 2024)
STORMS = {  # back-test windows, all inside the test period
    "April 2023": ("2023-04-22", "2023-04-26"),
    "May 2024 (Gannon)": ("2024-05-09", "2024-05-14"),
    "October 2024": ("2024-10-09", "2024-10-13"),
}
FEATURES = ["kp_prev", "v", "n", "bt", "bz", "bz_min", "vbs", "newell"]
STORM_KP = 5.0  # G1 threshold

# OMNI2 0-based column -> (name, fill value)
OMNI_COLS = {8: ("bt", 999.9), 15: ("by", 999.9), 16: ("bz", 999.9), 23: ("n", 999.9), 24: ("v", 9999.0)}


def cached(url: str, name: str) -> Path:
    path = DATA / name
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def load_omni(year: int) -> pd.DataFrame:
    path = cached(f"https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat", f"omni2_{year}.dat")
    raw = np.loadtxt(path)
    t = pd.to_datetime(raw[:, 0].astype(int).astype(str), format="%Y") + pd.to_timedelta(
        (raw[:, 1] - 1) * 24 + raw[:, 2], unit="h"
    )
    df = pd.DataFrame({name: raw[:, col] for col, (name, _) in OMNI_COLS.items()}, index=t)
    for name, fill in OMNI_COLS.values():
        df.loc[df[name] >= fill, name] = np.nan
    return df


def load_kp() -> pd.Series:
    path = cached(
        f"https://kp.gfz.de/app/json/?start={YEARS[0]}-01-01T00:00:00Z&end={YEARS[-1] + 1}-01-01T00:00:00Z&index=Kp",
        "kp_gfz.json",
    )
    d = json.loads(path.read_text())
    return pd.Series(d["Kp"], index=pd.to_datetime(d["datetime"]).tz_localize(None), name="kp")


def newell(v, bt, by, bz):
    """Newell et al. (2007) coupling function dPhi/dt, scaled to O(1-10)."""
    theta = np.arctan2(by, bz)
    return 1e-3 * np.abs(v) ** (4 / 3) * bt ** (2 / 3) * np.abs(np.sin(theta / 2)) ** (8 / 3)


def make_features(sw: pd.DataFrame, kp: pd.Series) -> pd.DataFrame:
    """One row per 3-hour Kp bin t: solar wind of the 3 hours before t and the Kp of the previous bin.

    Target is Kp of bin [t, t+3h). Everything used is known at time t, so there is no leakage.
    The browser does exactly the same on hourly means of the live 1-minute feed (docs/app.js).
    """
    h = sw.copy()
    h["vbs"] = h["v"] * np.clip(-h["bz"], 0, None) / 1000  # mV/m
    h["newell"] = newell(h["v"], h["bt"], h["by"], h["bz"])
    # 3-hour window [t-3h, t): label each window by its right edge
    agg = h.resample("3h", label="right", closed="left").agg(
        {"v": "mean", "n": "mean", "bt": "mean", "bz": ["mean", "min"], "vbs": "mean", "newell": "mean"}
    )
    agg.columns = ["v", "n", "bt", "bz", "bz_min", "vbs", "newell"]
    df = agg.join(kp.rename("kp"), how="inner")
    df["kp_prev"] = kp.shift(1).reindex(df.index)
    return df.dropna()


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    storm = y >= STORM_KP
    hit, miss = np.sum(storm & (p >= STORM_KP)), np.sum(storm & (p < STORM_KP))
    fa, cn = np.sum(~storm & (p >= STORM_KP)), np.sum(~storm & (p < STORM_KP))
    # Heidke skill score: 0 = no better than chance, 1 = perfect
    hss = 2 * (hit * cn - fa * miss) / ((hit + miss) * (miss + cn) + (hit + fa) * (fa + cn))
    return {
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse_storm": float(np.sqrt(np.mean((y[storm] - p[storm]) ** 2))),
        "pod": float(hit / (hit + miss)),
        "far": float(fa / max(hit + fa, 1)),
        "hss": float(hss),
        "n": int(len(y)),
        "n_storm": int(storm.sum()),
    }


def main():
    sw = pd.concat([load_omni(y) for y in YEARS])
    df = make_features(sw, load_kp())
    train, test = df[df.index < TRAIN_END], df[df.index >= TRAIN_END]
    print(f"train {len(train)} bins, test {len(test)} bins")

    # Storms are rare (~4% of bins); up-weight them so the fit is not dominated by quiet time.
    # Costs some overall RMSE but buys storm detection (see README).
    w = np.exp(train["kp"] / 2)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
        train[FEATURES], train["kp"], ridge__sample_weight=w
    )
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(
        train[FEATURES], train["kp"], sample_weight=w
    )
    preds = {
        "persistence": test["kp_prev"].to_numpy(),
        "ridge": np.clip(ridge.predict(test[FEATURES]), 0, 9),
        "gbm": np.clip(hgb.predict(test[FEATURES]), 0, 9),
    }
    results = {name: metrics(test["kp"], p) for name, p in preds.items()}
    for name, m in results.items():
        print(f"{name:12s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, float)))

    storms = {}
    for name, (a, b) in STORMS.items():
        mask = (test.index >= a) & (test.index < b)
        storms[name] = {
            "time": [t.isoformat() + "Z" for t in test.index[mask]],
            "observed": test["kp"][mask].round(2).tolist(),
            **{k: np.round(p[mask], 2).tolist() for k, p in preds.items()},
            "metrics": {k: metrics(test["kp"][mask], p[mask]) for k, p in preds.items()},
        }

    scaler, lin = ridge.named_steps["standardscaler"], ridge.named_steps["ridge"]
    DOCS.mkdir(exist_ok=True)
    (DOCS / "model.json").write_text(
        json.dumps(
            {
                "features": FEATURES,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "coef": lin.coef_.tolist(),
                "intercept": float(lin.intercept_),
                "trained_on": f"{YEARS[0]}-01-01..{TRAIN_END}",
            },
            indent=1,
        )
    )
    (DOCS / "backtest.json").write_text(
        json.dumps({"test_period": f"{TRAIN_END}..{YEARS[-1] + 1}-01-01", "overall": results, "storms": storms})
    )
    print("wrote docs/model.json, docs/backtest.json")


if __name__ == "__main__":
    main()
