"""
数据质量粗洗辅助函数（逐点布尔掩码）。

本模块封装与模型无关的单点/时间序列质量判定，供上层检测器按需组合。
与 OpenOA ``utils/filters.py`` 思路对齐，但只保留本项目需要且实现更简洁的算子。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def unresponsive_flag(
    data: pd.DataFrame,
    cols: str | list[str],
    threshold: int = 3,
) -> pd.Series | pd.DataFrame:
    """标记连续 ``threshold`` 个时间步读数完全不变的"冻结/卡死"点。

    判断的是"相邻时间步读数是否完全相同"（游程长度 >= ``threshold``），
    因此 ``data`` **必须已按时间升序排列**；本函数只按位置相邻，不感知时间戳。
    NaN 视为断点：既不被计入连续长度，也不被标记为冻结（缺失数据由上层单独处理）。

    Args:
        data: 待判定的数据框。
        cols: 待判定的列；单个列名字符串返回布尔 Series，列名列表返回布尔 DataFrame。
        threshold: 连续不变的最小时间步数（>=2），默认 3。连续 ``threshold`` 个采样点
            读数完全相同即整段判为冻结。

    Returns:
        与 ``data`` 索引对齐的布尔掩码：``cols`` 为字符串时是 Series，
        ``cols`` 为列表时是 DataFrame（列为 ``cols``），True 表示冻结/卡死点。

    Raises:
        TypeError: ``threshold`` 非整数。
        ValueError: ``threshold`` < 2，或 ``cols`` 中有列不存在。
    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int, np.integer)):
        raise TypeError(f"threshold 必须是整数，got {type(threshold).__name__}")
    threshold = int(threshold)
    if threshold < 2:
        raise ValueError(f"threshold 必须 >= 2，got {threshold}")

    if isinstance(cols, str):
        cols = [cols]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"DataFrame 缺少列: {missing}")

    flags = {c: _unresponsive_series(data[c], threshold) for c in cols}

    if len(cols) == 1:
        return flags[cols[0]]
    return pd.DataFrame(flags)


def _unresponsive_series(s: pd.Series, threshold: int) -> pd.Series:
    """对单个序列做游程编码判定：连续 ``threshold`` 个相同读数整段判为冻结。"""
    vals = s.to_numpy()

    if len(vals) == 0:
        return pd.Series(False, index=s.index, dtype=bool)

    # 相邻不同（含 NaN 边界，NaN != NaN 为 True）处为游程断点
    changed = np.ones(len(vals), dtype=bool)
    changed[1:] = vals[1:] != vals[:-1]
    run_id = np.cumsum(changed)

    # 每个游程的长度
    _, counts = np.unique(run_id, return_counts=True)
    run_len = np.repeat(counts, counts)

    flag = run_len >= threshold
    return pd.Series(flag, index=s.index, dtype=bool)
