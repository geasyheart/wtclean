"""
按天拟合检测隐藏异常

对单台风机：先用整机数据拟合基准功率曲线，再按天统计该天的点相对基准曲线的
系统性偏移，从而把"水平聚集 / 限电 / 降载"这类单点统计方法标不出的异常天揪出来，
并把异常天内的点整体标记为异常。最终仍输出逐点 normal / abnormal 标签。
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
        low_power_ratio: 高风区功率低于该比例即视为限电/高风低功率异常。
        downtime_power: 功率低于该值且风速在切入以上视为停机异常。
        bin_width: 残差分箱宽度。
        min_day_points: 参与天级判定的最少样本数。
        min_bin_points: 残差 bin 参与判定的最少样本数。
        z_coeff: 逐点残差 MAD 倍数（控制逐点残差精洗的误杀率，越大保留越多）。
        day_z_coeff: 天级分数 MAD 倍数（控制"整天系统性偏移"判定的松紧；真实数据上天级
            偏移呈重尾连续分布、无天然分界，标出天数随该值平滑变化，建议按站点校准）。
        base_method: 基准曲线后端（gam / logistic / iec）。
    """

    def __init__(
        self,
        windspeed_label: str,
        power_label: str,
        timestamp_label: str,
        df: pd.DataFrame,
        #
        cut_in_speed: float | None = None,
        rated_wind_speed: float | None = None,
        rated_power: float | None = None,
        cut_out_speed: float | None = None,
        #
        low_power_ratio: float = 0.9,
        downtime_power: float = 1.0,
        bin_width: float = 0.5,
        min_day_points: int = 10,
        min_bin_points: int = 3,
        z_coeff: float = 3.5,
        day_z_coeff: float = 2.5,
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
        if bin_width <= 0:
            raise ValueError(f"bin_width must be > 0, got {bin_width}")
        if z_coeff <= 0 or day_z_coeff <= 0:
            raise ValueError("z_coeff and day_z_coeff must be > 0")
        if min_day_points < 1 or min_bin_points < 1:
            raise ValueError("min_day_points and min_bin_points must be >= 1")

        self.windspeed_label = windspeed_label
        self.power_label = power_label
        self.timestamp_label = timestamp_label
        self.df = df
        self.cut_out_speed = cut_out_speed
        self.low_power_ratio = low_power_ratio
        self.downtime_power = downtime_power
        self.bin_width = bin_width
        self.min_day_points = min_day_points
        self.min_bin_points = min_bin_points
        self.z_coeff = z_coeff
        self.day_z_coeff = day_z_coeff
        self.base_method = base_method
        self.base_kwargs = base_kwargs

    def detect(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """运行天维度 + 残差检测，返回 ``(normal_df, abnormal_df)``。

        Returns:
            按行划分原始 ``df`` 的正常 / 异常两个子集；异常类型可通过
            ``self.labels_`` 与 ``self.abnormal_days_`` 查看。
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

        # ---- 整机基准曲线 ----
        self.base_curve_ = fit_power_curve(
            ws[analysis_mask],
            pw[analysis_mask],
            method=self.base_method,
            **self.base_kwargs,
        )

        resid = np.full_like(pw, np.nan, dtype=float)
        finite = np.isfinite(ws) & np.isfinite(pw)
        resid[finite] = pw[finite] - self.base_curve_(ws[finite])
        work["_resid"] = resid

        # 逐点标签：normal=正常；其余为各粗洗异常类型（对应上面的掩码）
        labels = pd.Series("normal", index=work.index, dtype=object)
        labels[nan_mask] = "nan"  # 缺失：风速/功率为 NaN
        labels[range_mask] = "range"  # 越界：数值超出物理合理范围
        labels[downtime_mask] = "downtime"  # 停机：切入风速以上却功率近零
        labels[curtail_mask] = "curtailment"  # 限电：额定风速以上功率远低于额定

        clean = work[analysis_mask].copy()

        # ---- 天级判定：该天运行段（切入以上）的中位残差 = 系统性偏移 ----
        op = clean[clean[self.windspeed_label] >= self.cut_in_speed]
        day_stats = op.groupby("_date")["_resid"].agg(median="median", n="size")
        day_stats = day_stats[day_stats["n"] >= self.min_day_points]

        self.day_scores_ = day_stats
        if len(day_stats) >= 3:
            center = day_stats["median"].median()
            mad = (day_stats["median"] - center).abs().median()
            lo = center - self.day_z_coeff * mad
            abnormal_days = set(day_stats.index[day_stats["median"] < lo].tolist())
        else:
            lo = np.nan
            abnormal_days = set()

        self.abnormal_days_ = abnormal_days
        self.day_threshold_ = lo

        # ---- 逐点残差 bin-MAD 精洗（异方差）----
        clean = clean.assign(
            _bin=np.floor(clean[self.windspeed_label] / self.bin_width).astype("Int64")
        )
        bin_stats = (
            clean.groupby("_bin")["_resid"]
            .agg(n="size", med="median", mad=lambda x: (x - x.median()).abs().median())
            .reset_index()
        )
        bin_stats.loc[
            (bin_stats["n"] < self.min_bin_points) | (bin_stats["mad"] == 0), "mad"
        ] = np.nan
        clean = clean.merge(bin_stats[["_bin", "med", "mad"]], on="_bin", how="left")

        resid_outlier = (
            (clean["_resid"] - clean["med"]).abs() > self.z_coeff * clean["mad"]
        ) & clean["mad"].notna()

        resid_outlier_ids = clean.loc[resid_outlier, "_row_id"].tolist()
        temporal_ids = clean.loc[clean["_date"].isin(abnormal_days), "_row_id"].tolist()

        # temporal：时间维度异常——整天的运行段整体偏离基准曲线（限电/降载/停机方向），
        # 该天内的点整体标为异常（单点看可能落在正常带内，只有整条天曲线对比才现形）
        labels.loc[work["_row_id"].isin(set(temporal_ids))] = "temporal"
        # residual：逐点残差异常——单个点相对基准曲线的残差超出该风速 bin 的
        # z_coeff×MAD 带外（异方差离群点），仅在尚未被标记的点上生效
        labels.loc[
            work["_row_id"].isin(set(resid_outlier_ids)) & (labels == "normal")
        ] = "residual"

        self.labels_ = labels
        self.removal_counts_ = labels[labels != "normal"].value_counts().to_dict()

        normal_df = work[labels == "normal"].drop(
            columns=["_row_id", "_date", "_resid"]
        )
        abnormal_df = work[labels != "normal"].drop(
            columns=["_row_id", "_date", "_resid"]
        )
        return normal_df, abnormal_df
