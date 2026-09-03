"""wtclean.utils.filters.unresponsive_flag 单元测试（unittest）。"""

import unittest

import numpy as np
import pandas as pd

from wtclean.utils.filters import unresponsive_flag


class TestUnresponsiveFlag(unittest.TestCase):
    def setUp(self):
        # 连续重复段：[1,1,1] 冻结、[2,2] 不够、[3,3,3,3] 冻结
        self.s = pd.Series([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])

    def test_threshold_default(self):
        out = unresponsive_flag(self.s.to_frame("ws"), cols="ws")
        self.assertEqual(
            out.tolist(), [True, True, True, False, False, True, True, True, True]
        )

    def test_threshold_boundary(self):
        df = self.s.to_frame("ws")
        self.assertEqual(
            unresponsive_flag(df, cols="ws", threshold=4).tolist(),
            [False, False, False, False, False, True, True, True, True],
        )
        # 恰好等于 threshold 的段被判
        self.assertEqual(
            unresponsive_flag(df, cols="ws", threshold=3).tolist()[:3],
            [True, True, True],
        )

    def test_nan_breaks_run(self):
        df = pd.DataFrame({"ws": [np.nan, np.nan, 1.0, 1.0, 1.0]})
        self.assertEqual(
            unresponsive_flag(df, cols="ws", threshold=3).tolist(),
            [False, False, True, True, True],
        )

    def test_no_freeze_all_false(self):
        df = pd.DataFrame({"ws": [1.0, 2.0, 3.0, 4.0]})
        self.assertFalse(unresponsive_flag(df, cols="ws", threshold=3).any())

    def test_empty_series(self):
        df = pd.DataFrame({"ws": pd.Series([], dtype=float)})
        out = unresponsive_flag(df, cols="ws", threshold=3)
        self.assertEqual(len(out), 0)

    def test_single_col_returns_series(self):
        df = pd.DataFrame(
            {"ws": [1.0, 1.0, 1.0, 2.0, 2.0], "pw": [5.0, 6.0, 7.0, 8.0, 9.0]}
        )
        out = unresponsive_flag(df, cols="ws", threshold=3)
        self.assertIsInstance(out, pd.Series)
        self.assertEqual(out.tolist(), [True, True, True, False, False])

    def test_multiple_cols_returns_dataframe(self):
        df = pd.DataFrame(
            {"ws": [1.0, 1.0, 1.0], "pw": [5.0, 5.0, 5.0], "other": [1.0, 2.0, 3.0]}
        )
        out = unresponsive_flag(df, cols=["ws", "pw", "other"], threshold=3)
        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(list(out.columns), ["ws", "pw", "other"])
        self.assertTrue(out["ws"].all())
        self.assertTrue(out["pw"].all())
        self.assertFalse(out["other"].any())

    def test_index_aligned(self):
        idx = pd.RangeIndex(start=10, stop=19)
        df = pd.DataFrame(
            {"ws": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]}, index=idx
        )
        out = unresponsive_flag(df, cols="ws", threshold=3)
        self.assertTrue(out.index.equals(idx))
        self.assertEqual(out.iloc[:3].tolist(), [True, True, True])
        self.assertTrue(out.loc[10])

    def test_invalid_threshold(self):
        df = self.s.to_frame("ws")
        with self.assertRaises(ValueError):
            unresponsive_flag(df, cols="ws", threshold=1)
        with self.assertRaises(ValueError):
            unresponsive_flag(df, cols="ws", threshold=0)
        with self.assertRaises(TypeError):
            unresponsive_flag(df, cols="ws", threshold=2.5)
        with self.assertRaises(TypeError):
            unresponsive_flag(df, cols="ws", threshold=True)

    def test_missing_column(self):
        df = pd.DataFrame({"ws": [1.0, 1.0, 1.0]})
        with self.assertRaises(ValueError):
            unresponsive_flag(df, cols="bad", threshold=3)


if __name__ == "__main__":
    unittest.main()
