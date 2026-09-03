"""DailyAnomalyDetector 单元测试（unittest）。"""

import unittest

import numpy as np
import pandas as pd

from wtclean import DailyAnomalyDetector

CUT_IN, RATED_WS, RATED_PWR = 3.0, 12.0, 2000.0


def _clean_power(ws):
    p = np.where(ws < RATED_WS, RATED_PWR * (ws - CUT_IN) / (RATED_WS - CUT_IN), RATED_PWR)
    return np.clip(p, 0, RATED_PWR)


def _synth(days=30, seed=0, curtail_days=(), cap=1200.0):
    """构造理想功率曲线 + 噪声；在 ``curtail_days`` 把功率封顶模拟限电天。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    rows = []
    for d in dates:
        for i in range(144):
            rows.append({"ts": d + pd.Timedelta(minutes=10 * i), "ws": rng.uniform(2.5, 20.0)})
    df = pd.DataFrame(rows)
    df["pw"] = _clean_power(df["ws"].values) + rng.normal(0, 20, len(df))
    curtailed = df["ts"].dt.normalize().isin(dates[list(curtail_days)])
    if curtailed.any():
        df.loc[curtailed, "pw"] = (
            np.minimum(_clean_power(df.loc[curtailed, "ws"].values), cap)
            + rng.normal(0, 20, int(curtailed.sum()))
        )
    return df, dates


def _det(df, **kw):
    return DailyAnomalyDetector(
        "ws", "pw", "ts", df,
        cut_in_speed=CUT_IN, rated_wind_speed=RATED_WS, rated_power=RATED_PWR, **kw
    )


class TestDailyAnomalyDetector(unittest.TestCase):
    def test_partition(self):
        df, _ = _synth()
        normal, abnormal = _det(df).detect()
        self.assertEqual(len(normal) + len(abnormal), len(df))
        self.assertTrue(set(normal.index).isdisjoint(set(abnormal.index)))

    def test_detects_curtailment_day(self):
        df, dates = _synth(curtail_days=(5, 6, 7))
        det = _det(df)
        det.detect()
        expected = set(dates[[5, 6, 7]].tolist())
        self.assertTrue(expected.issubset(det.abnormal_days_))

    def test_normal_day_not_flagged(self):
        df, dates = _synth()
        det = _det(df)
        det.detect()
        self.assertNotIn(dates[10], det.abnormal_days_)

    def test_curtailment_points_labeled_abnormal(self):
        df, dates = _synth(curtail_days=(5,))
        det = _det(df)
        normal, abnormal = _det(df).detect()
        day_mask = pd.to_datetime(df["ts"]).dt.normalize() == dates[5]
        day_abnormal = set(abnormal.index) & set(df.index[day_mask])
        self.assertTrue(len(day_abnormal) > 0)

    def test_missing_column_raises(self):
        df, _ = _synth()
        with self.assertRaises(ValueError):
            DailyAnomalyDetector("bad", "pw", "ts", df, CUT_IN, RATED_WS, RATED_PWR)

    def test_empty_raises(self):
        df, _ = _synth(days=1)
        with self.assertRaises(ValueError):
            _det(df.iloc[0:0])

    def test_invalid_params(self):
        df, _ = _synth()
        with self.assertRaises(ValueError):
            _det(df, low_power_ratio=1.5)
        with self.assertRaises(ValueError):
            _det(df, bin_width=0)

    def test_auto_estimate_params(self):
        df, _ = _synth()
        det = DailyAnomalyDetector("ws", "pw", "ts", df)
        normal, abnormal = det.detect()
        self.assertTrue(len(normal) + len(abnormal) == len(df))
        self.assertGreater(det.rated_power, 0)


if __name__ == "__main__":
    unittest.main()
