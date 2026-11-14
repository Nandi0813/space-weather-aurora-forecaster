import numpy as np
import pandas as pd

from train import make_features, metrics, newell


def test_newell_zero_for_northward_imf():
    assert newell(400, 5, 0, 5) == 0
    assert newell(400, 5, 0, -5) > newell(400, 5, 5, 0) > 0


def test_features_use_only_past_data():
    t = pd.date_range("2024-01-01", periods=12, freq="h")
    sw = pd.DataFrame({"bt": 5.0, "by": 0.0, "bz": np.arange(12.0), "n": 5.0, "v": 400.0}, index=t)
    kp = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2024-01-01", periods=4, freq="3h"))
    df = make_features(sw, kp)
    row = df.loc["2024-01-01 06:00"]
    assert row["bz"] == np.mean([3, 4, 5])  # hours 03,04,05 only
    assert row["kp_prev"] == 2.0 and row["kp"] == 3.0


def test_metrics_perfect_forecast():
    y = np.array([1, 2, 6, 7, 3])
    m = metrics(y, y)
    assert m["rmse"] == 0 and m["hss"] == 1 and m["pod"] == 1 and m["far"] == 0
