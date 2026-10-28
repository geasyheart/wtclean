"""
按天拟合检测隐藏异常

对单台风机：先用整机数据拟合基准功率曲线，再按天统计该天的点相对基准曲线的
系统性偏移，从而把"水平聚集 / 限电 / 降载"这类单点统计方法标不出的异常天揪出来，
并把异常天内的点整体标记为异常。最终输出 normal / abnormal 两个数据子集。
"""

import numpy as np
import pandas as pd

from wtclean.power_curve import fit_power_curve
from wtclean.utils.machine_parameter_estimation import estimate_machine_parameters


class DailyAnomalyDetector:
    """按天拟合 + 基准曲线对比，检测单台风机的时间维度异常。

    Attributes:
        windspeed_label: 风速列名。
        power_label: 有功功率列名。
        timestamp_label: 时间戳列名。
        df: 单台风机的 SCADA DataFrame。
        cut_in_speed: 切入风速（厂商机型参数；缺省自动从 SCADA 反推）。
        rated_wind_speed: 额定风速；缺省自动反推。
        rated_power: 额定功率；缺省自动反推。
        cut_out_speed: 切出风速（可选）；给定后切出风速以上功率归零视为正常停机。
        low_power_ratio: 风速达到额定风速后，功率低于该比例×额定功率即视为限电异常。
        downtime_power: 功率低于该值且风速在切入以上视为停机异常。
        base_method: 基准曲线后端（gam / logistic / iec）。
    """

    def __init__(
        self,
        windspeed_label: str,
        power_label: str,
        timestamp_label: str,
        df: pd.DataFrame,
        cut_in_speed: float | None = None,
        rated_wind_speed: float | None = None,
        rated_power: float | None = None,
        cut_out_speed: float | None = None,
        low_power_ratio: float = 0.9,
        downtime_power: float = 1.0,
        base_method: str = "gam",
        **base_kwargs,
    ) -> None:
        for col in (windspeed_label, power_label, timestamp_label):
            if col not in df.columns:
                raise ValueError(f"DataFrame 缺少列: {col}")
        if df.empty:
            raise ValueError("DataFrame 为空")

        # 厂商物理参数缺省时从 SCADA 反推
        given = {
            "cut_in_speed": cut_in_speed,
            "rated_wind_speed": rated_wind_speed,
            "rated_power": rated_power,
        }
        est = {}
        if any(v is None for v in given.values()):
            est = estimate_machine_parameters(df, windspeed_label, power_label)
            print(
                f"[DailyAnomalyDetector] 厂商传入={given} 反推={est}，"
                f"实际采用={ {k: (given[k] if given[k] is not None else est[k]) for k in est} }"
            )
        self.cut_in_speed = (
            cut_in_speed if cut_in_speed is not None else est["cut_in_speed"]
        )
        self.rated_wind_speed = (
            rated_wind_speed
            if rated_wind_speed is not None
            else est["rated_wind_speed"]
        )
        self.rated_power = (
            rated_power if rated_power is not None else est["rated_power"]
        )

        if self.cut_in_speed <= 0:
            raise ValueError(f"cut_in_speed must be > 0, got {self.cut_in_speed}")
        if self.rated_wind_speed <= self.cut_in_speed:
            raise ValueError(
                f"rated_wind_speed must be > cut_in_speed, got "
                f"cut_in_speed={self.cut_in_speed}, rated_wind_speed={self.rated_wind_speed}"
            )
        if self.rated_power <= 0:
            raise ValueError(f"rated_power must be > 0, got {self.rated_power}")
        if not 0 < low_power_ratio <= 1:
            raise ValueError(
                f"low_power_ratio must be in (0, 1], got {low_power_ratio}"
            )

        self.windspeed_label = windspeed_label
        self.power_label = power_label
        self.timestamp_label = timestamp_label
        self.df = df
        self.cut_out_speed = cut_out_speed
        self.downtime_power = downtime_power
        self.low_power_ratio = low_power_ratio

        self.base_method = base_method
        self.base_kwargs = base_kwargs

    def detect(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """运行天级检测，返回 ``(normal_df, abnormal_df)``。

        Returns:
            按行划分原始 ``df`` 的正常 / 异常两个子集；被判异常的天可通过
            
        """
        work = self.df.copy()
        work["_row_id"] = np.arange(len(work))
        work["_date"] = (
            pd.to_datetime(work[self.timestamp_label], utc=True, errors="coerce")
            .dt.tz_localize(None)
            .dt.normalize()
        )

        ws = work[self.windspeed_label].to_numpy(dtype=float)
        pw = work[self.power_label].to_numpy(dtype=float)

        cut_out_eff = self.cut_out_speed if self.cut_out_speed is not None else np.inf

        # ---- 粗洗掩码（这些点直接判异常，不参与基准曲线 / 天级 / 残差分析）----
        # nan_mask：风速或功率缺失（NaN）的点，直接判为缺失异常，无法参与任何建模
        nan_mask = np.isnan(ws) | np.isnan(pw)
        # range_mask：数值超出物理合理范围的点（功率/风速为负，或功率远超额定）。
        # 1.2x 上限（OpenOA 惯例）：既兜住额定功率略超的情形，又容忍反推额定功率偏软
        range_mask = (pw < 0) | (ws < 0) | (pw > 1.2 * self.rated_power)
        if self.cut_out_speed is not None:
            # 切出风速以上功率非零视为异常（正常停机功率应归零）
            range_mask |= (ws > cut_out_eff) & (pw > self.downtime_power)

        # downtime_mask：停机异常——风速在切入风速以上（本应发电），功率却接近零
        # （低于 downtime_power），说明风机停机未发，属停机/故障类异常
        downtime_mask = (
            (pw <= self.downtime_power)
            & (ws >= self.cut_in_speed)
            & (ws <= cut_out_eff)
        )
        # curtail_mask：限电异常（高风低功率）——风速已达额定风速（应满发），
        # 功率却低于 low_power_ratio×额定功率，说明电网限电/降载，属弃电限电类异常
        curtail_mask = (
            (pw < self.low_power_ratio * self.rated_power)
            & (ws >= self.rated_wind_speed)
            & (ws <= cut_out_eff)
        )

        # 参与建模的点：非粗洗异常，且风速在 [0, cut_out] 内（排除正常停机尾段）
        analysis_mask = (~nan_mask) & (~range_mask) & (~downtime_mask) & (~curtail_mask)
        analysis_mask &= ws <= cut_out_eff

        if not analysis_mask.any():
            return self.df.iloc[0:0].copy(), self.df.copy()

        # ---- 整机基准曲线 ----
        self.base_curve_ = fit_power_curve(
            ws[analysis_mask],
            pw[analysis_mask],
            method=self.base_method,
            **self.base_kwargs,
        )

        clean_df = work[analysis_mask].copy()
        clean_df["_resid"] = clean_df[self.power_label] - self.base_curve_(
            clean_df[self.windspeed_label]
        )

        # ---- 天级判定：每天中位残差 + 每天点数都用 median ± 3×MAD ----
        k = 3.0
        raw_day_stats = clean_df.groupby("_date")["_resid"].agg(
            median="median", n="size"
        )

        median_n = raw_day_stats["n"].median()
        mad_n = (raw_day_stats["n"] - median_n).abs().median()
        count_mask = raw_day_stats["n"].between(
            median_n - k * mad_n, median_n + k * mad_n
        )
        day_stats = raw_day_stats.loc[count_mask]

        if day_stats.empty:
            return self.df.iloc[0:0].copy(), self.df.copy()

        median_resid = day_stats["median"].median()
        mad_resid = (day_stats["median"] - median_resid).abs().median()
        normal_mask = day_stats["median"].between(
            median_resid - k * mad_resid, median_resid + k * mad_resid
        )
        normal_dates = set(day_stats.index[normal_mask])
 
        normal_row_ids = clean_df.loc[
            clean_df["_date"].isin(normal_dates), "_row_id"
        ]
        is_normal = work["_row_id"].isin(normal_row_ids)
        normal_df = work.loc[is_normal, self.df.columns].copy()
        abnormal_df = work.loc[~is_normal, self.df.columns].copy()
        return normal_df, abnormal_df

