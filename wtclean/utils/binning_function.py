"""
Function for calculating the median power and median absolute deviation (MAD) of power
for each wind speed bin of a given turbine.
"""

import numpy as np
import pandas as pd


def binning_func(
    turbine_data: pd.DataFrame,
    windspeed_label: str,
    power_label: str,
    bin_interval: float = 0.5,
) -> pd.DataFrame:
    """Computes per-bin statistics (sample count, median wind speed, median power, MAD).

    Bins wind speed into integer groups of width ``bin_interval`` using
    ``floor(ws / bin_interval)``, which decouples the bin count from ``bin_interval`` and
    covers all wind speeds without out-of-range drops.

    Note:
        The MAD is intentionally kept raw (no 1.4826 scaling); ``z_coeff`` in
        ``PowerCurveFiltering`` is therefore an empirical MAD multiplier, not a
        standard-deviation-equivalent z-score.

    Note:
        Bins with fewer than 3 samples, or whose power has zero spread (MAD == 0), get
        ``pwr_bin_mad = NaN`` so the filtering layer treats them as "undecidable" rather
        than flagging every point (e.g. the rated-power plateau).

    Args:
        turbine_data: DataFrame containing the ``windspeed_label`` and ``power_label`` columns.
        windspeed_label: Column name of the wind speed.
        power_label: Column name of the active power.
        bin_interval: Wind speed bin width, default 0.5. Must be positive.

    Returns:
        DataFrame with columns ``windspeed_bin``, ``n``, ``windspeed_bin_median``,
        ``pwr_bin_median``, and ``pwr_bin_mad``.

    Raises:
        ValueError: If ``bin_interval`` is not positive.
    """
    if bin_interval <= 0:
        raise ValueError("bin_interval must be > 0")

    ws = turbine_data[windspeed_label]

    # NaN-safe integer bin label: floor(ws / bin_interval), NA preserved via Int64.
    bin_label = np.floor(ws / bin_interval).astype("Int64")

    binned = (
        turbine_data.assign(windspeed_bin=bin_label)
        .groupby("windspeed_bin", sort=True)
        .agg(
            n=(power_label, "size"),
            windspeed_bin_median=(windspeed_label, "median"),
            pwr_bin_median=(power_label, "median"),
            pwr_bin_mad=(power_label, lambda x: (x - x.median()).abs().median()),
        )
        .reset_index()
    )

    # Undecidable bins: too few samples, or zero spread (e.g. rated-power plateau).
    binned.loc[(binned["n"] < 3) | (binned["pwr_bin_mad"] == 0), "pwr_bin_mad"] = np.nan

    return binned[
        ["windspeed_bin", "n", "windspeed_bin_median", "pwr_bin_median", "pwr_bin_mad"]
    ]


def test_binning_func() -> None:
    """Self-check for ``binning_func`` against hand-computed expectations."""

    df = pd.DataFrame(
        {
            "ws": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.1, np.nan],
            "pw": [10, 20, 30, 40, 100, 100, 100, 50, 60, 5],
        }
    )

    out = binning_func(df, "ws", "pw", bin_interval=0.5)

    # 输出列与顺序
    assert list(out.columns) == [
        "windspeed_bin",
        "n",
        "windspeed_bin_median",
        "pwr_bin_median",
        "pwr_bin_mad",
    ]

    # bin 0: ws in [0.0, 0.5) -> n=4, 风速中位 0.25, 功率中位 25, MAD 10
    b0 = out[out["windspeed_bin"] == 0].iloc[0]
    assert b0["n"] == 4
    assert b0["windspeed_bin_median"] == 0.25
    assert b0["pwr_bin_median"] == 25
    assert b0["pwr_bin_mad"] == 10

    # bin 1: 功率恒定 -> MAD == 0 -> 置 NaN（不判定）
    b1 = out[out["windspeed_bin"] == 1].iloc[0]
    assert b1["n"] == 3
    assert b1["pwr_bin_median"] == 100
    assert pd.isna(b1["pwr_bin_mad"])

    # bin 2: n=2（<3）-> MAD 置 NaN（不判定）
    b2 = out[out["windspeed_bin"] == 2].iloc[0]
    assert b2["n"] == 2
    assert b2["pwr_bin_median"] == 55
    assert pd.isna(b2["pwr_bin_mad"])

    # NaN 风速的行被排除在统计之外（n 总和 = 9，而非 10）
    assert out["n"].sum() == 9

    # bin_interval <= 0 抛出 ValueError
    for bad in (0, -0.5):
        try:
            binning_func(df, "ws", "pw", bin_interval=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bin_interval={bad} 应抛出 ValueError")

    print("binning_func 自测通过：所有断言均满足")


if __name__ == "__main__":
    test_binning_func()
