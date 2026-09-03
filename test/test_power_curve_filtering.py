"""PowerCurveFiltering 单元测试（unittest，无 pytest 依赖）。"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from wtclean import PowerCurveFiltering

CUT_IN = 3.0
RATED_WS = 12.0
RATED_PWR = 2000.0


def _clean_df(n=2000, seed=0):
    """构造一条理想功率曲线（切入~额定线性爬升，额定以上平台）+ 小噪声。"""
    rng = np.random.default_rng(seed)
    ws = rng.uniform(CUT_IN, 25.0, n)
    power = np.where(
        ws < RATED_WS,
        RATED_PWR * (ws - CUT_IN) / (RATED_WS - CUT_IN),
        RATED_PWR,
    )
    power += rng.normal(0, 20, n)
    return pd.DataFrame({"ws": ws, "pw": power})


def _pcf(df, **kwargs):
    return PowerCurveFiltering(
        windspeed_label="ws",
        power_label="pw",
        df=df,
        cut_in_speed=CUT_IN,
        rated_wind_speed=RATED_WS,
        rated_power=RATED_PWR,
        **kwargs,
    )


class TestPowerCurveFiltering(unittest.TestCase):
    def test_partition_and_outlier_detection(self):
        df = _clean_df()
        df["is_outlier"] = False
        # 注入两个明确异常：高风低功率、高风停机
        df.loc[len(df)] = {"ws": 20.0, "pw": 100.0, "is_outlier": True}
        df.loc[len(df)] = {"ws": 15.0, "pw": 0.0, "is_outlier": True}

        normal_df, abnormal_df = _pcf(df).process()

        self.assertEqual(len(normal_df) + len(abnormal_df), len(df))
        self.assertTrue(set(normal_df.index).isdisjoint(set(abnormal_df.index)))

        outlier_idx = set(df.index[df["is_outlier"]].tolist())
        self.assertTrue(outlier_idx.issubset(set(abnormal_df.index)))
        self.assertTrue(outlier_idx.isdisjoint(set(normal_df.index)))

    def test_duplicate_index_no_expansion(self):
        df = _clean_df(n=500)
        # 人为制造重复 index，旧实现 loc[list] 会膨胀
        df.index = np.arange(len(df)) // 2

        normal_df, abnormal_df = _pcf(df).process()

        self.assertEqual(len(normal_df) + len(abnormal_df), len(df))

    def test_high_wind_low_power_removed(self):
        df = _clean_df()
        df.loc[len(df)] = {"ws": 20.0, "pw": 0.5 * RATED_PWR}
        _, abnormal_df = _pcf(df).process()
        self.assertIn(len(df) - 1, abnormal_df.index.tolist())

    def test_downtime_removed(self):
        df = _clean_df()
        df.loc[len(df)] = {"ws": 10.0, "pw": 0.0}
        _, abnormal_df = _pcf(df).process()
        self.assertIn(len(df) - 1, abnormal_df.index.tolist())

    def test_return_fig_saves_file(self):
        df = _clean_df(n=300)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "nested", "dir", "turbine_pc.png")
            _pcf(df, return_fig=True, image_path=out).process()
            self.assertTrue(os.path.exists(out))

    def test_invalid_params(self):
        df = _clean_df(n=100)

        with self.assertRaises(ValueError):
            PowerCurveFiltering("bad_col", "pw", df, CUT_IN, RATED_WS, RATED_PWR)
        with self.assertRaises(ValueError):
            PowerCurveFiltering("ws", "pw", df, 0, RATED_WS, RATED_PWR)
        with self.assertRaises(ValueError):
            PowerCurveFiltering("ws", "pw", df, CUT_IN, CUT_IN, RATED_PWR)
        with self.assertRaises(ValueError):
            PowerCurveFiltering(
                "ws", "pw", df, CUT_IN, RATED_WS, RATED_PWR, low_power_ratio=1.5
            )
        with self.assertRaises(ValueError):
            PowerCurveFiltering(
                "ws", "pw", df, CUT_IN, RATED_WS, RATED_PWR, bin_interval=0
            )

    def test_empty_df(self):
        df = _clean_df(n=10).iloc[0:0]
        with self.assertRaises(ValueError):
            _pcf(df)

    def test_removal_history_recorded(self):
        df = _clean_df(n=500)
        pcf = _pcf(df, filter_cycle=3)
        pcf.process()
        self.assertEqual(len(pcf.removal_history), 3)


if __name__ == "__main__":
    unittest.main()
