import numpy as np
import pandas as pd

from wtclean.utils.binning_function import binning_func


def estimate_machine_parameters(
    df: pd.DataFrame,
    windspeed_label: str,
    power_label: str,
    bin_interval: float = 0.5,
    min_bin_samples: int = 20,
    plateau_ratio: float = 0.97,
    rated_power_ratio: float = 0.95,
    cutin_power_fraction: float = 0.01,
    consecutive_bins: int = 2,
) -> dict[str, float]:
    """从单台风机的 SCADA 风速-功率数据估计切入风速、额定风速与额定功率（三个全部估计）。

    Args:
        df: 含 ``windspeed_label`` 与 ``power_label`` 列的**单台风机** SCADA DataFrame。
        windspeed_label: 风速列名。
        power_label: 有功功率列名。
        bin_interval: 风速分箱宽度，默认 0.5。
        min_bin_samples: 参与统计的最小 bin 样本数，低于该值的 bin 被丢弃（抗稀疏）。
        plateau_ratio: 判定"额定平台"的比例阈值：bin 中位功率达到全风速
            ``plateau_ratio * 峰值中位功率`` 即视为平台点。
        rated_power_ratio: 额定风速的判定标准：bin 中位功率首次达到
            ``rated_power_ratio * 额定功率`` 所对应的风速。
        cutin_power_fraction: 切入风速的判定标准：bin 中位功率首次达到
            ``cutin_power_fraction * 额定功率`` 所对应的风速（观测口径的有效出力起始风速，
            非厂商切入风速定义）。
        consecutive_bins: 达标判定的连续 bin 数（≥1），且要求这些 bin 在风速上物理连续
            （相邻 bin 中位风速间隔 ≤ ``bin_interval * 1.1``），避免单个噪声 bin 造成误判。

    Returns:
        ``{"cut_in_speed": ..., "rated_wind_speed": ..., "rated_power": ...}``，
        三项全部由数据估计得出。

    Raises:
        ValueError: 列缺失、数据为空、或数据无法支撑估计（如额定平台无法判定、
            无法找到连续达标的风速段）。

    Example:
        >>> params = estimate_machine_parameters(g, "Ws_avg", "P_avg")
        >>> params  # {"cut_in_speed": 3.77, "rated_wind_speed": 12.72, "rated_power": 1985.3}
    """
    for col in (windspeed_label, power_label):
        if col not in df.columns:
            raise ValueError(f"DataFrame 缺少列: {col}")
    if df.empty:
        raise ValueError("DataFrame 为空，无法估计参数")
    if bin_interval <= 0:
        raise ValueError(f"bin_interval must be > 0, got {bin_interval}")
    if not 0 < plateau_ratio <= 1:
        raise ValueError(f"plateau_ratio must be in (0, 1], got {plateau_ratio}")
    if not 0 < rated_power_ratio <= 1:
        raise ValueError(
            f"rated_power_ratio must be in (0, 1], got {rated_power_ratio}"
        )
    if not 0 < cutin_power_fraction <= 1:
        raise ValueError(
            f"cutin_power_fraction must be in (0, 1], got {cutin_power_fraction}"
        )
    if consecutive_bins < 1:
        raise ValueError(f"consecutive_bins must be >= 1, got {consecutive_bins}")

    binned = binning_func(df, windspeed_label, power_label, bin_interval)
    binned = binned[binned["n"] >= min_bin_samples].sort_values("windspeed_bin_median")
    if len(binned) < 3:
        raise ValueError(
            f"样本过少：满足 n>={min_bin_samples} 的风速 bin 仅 {len(binned)} 个，无法估计参数"
        )

    xs = binned["windspeed_bin_median"].to_numpy(dtype=float)
    ys = binned["pwr_bin_median"].to_numpy(dtype=float)

    def _sustained_cross(target: float) -> float | None:
        """寻找中位功率曲线首次在风速上连续达到 target 的位置。

        连续条件同时要求：
        - 连续 ``consecutive_bins`` 个 bin 的功率均 >= target；
        - 这些 bin 在风速轴上物理连续（相邻中位风速间隔 ≤ bin_interval * 1.1）。

        若找不到满足条件的风速段，返回 None。
        """
        cond = ys >= target
        n = len(cond)

        for i in range(n):
            end = i + consecutive_bins
            if end > n:
                break

            # 检查窗口内风速是否物理连续
            if not np.all(np.diff(xs[i:end]) <= bin_interval * 1.1):
                continue

            # 检查窗口内功率是否全部达标
            if not cond[i:end].all():
                continue

            # 找到首个连续达标段，计算交叉风速
            if i == 0:
                return float(xs[0])

            x0, y0 = xs[i - 1], ys[i - 1]
            x1, y1 = xs[i], ys[i]

            # 前一个 bin 与当前 bin 间隔过大时，不做跨间隙插值，直接取当前 bin 中位风速
            if x1 - x0 > bin_interval * 1.1:
                return float(x1)

            if y1 == y0:
                return float(x1)

            # 线性插值得到目标功率对应的风速
            return float(x0 + (target - y0) / (y1 - y0) * (x1 - x0))

        return None

    # 额定功率：最高平台（>= plateau_ratio * 峰值）上 bin 中位功率的中位数。
    peak = float(np.nanmax(ys))
    plateau = ys >= plateau_ratio * peak
    if not plateau.any():
        raise ValueError("无法从数据中判定额定功率平台")
    rated_power = float(np.median(ys[plateau]))

    # 额定风速：中位功率首次连续达到 rated_power_ratio * 额定功率。
    crossing = _sustained_cross(rated_power_ratio * rated_power)
    if crossing is None:
        raise ValueError(
            f"无法从功率曲线中估计 rated_wind_speed："
            f"没有连续 {consecutive_bins} 个有效 bin 达到 "
            f"{rated_power_ratio:.0%} 额定功率"
        )
    rated_wind_speed = crossing

    # 切入风速：中位功率首次连续达到 cutin_power_fraction * 额定功率。
    # 设置一个最小阈值 1.0 kW，避免绝对零值附近噪声的影响。
    crossing = _sustained_cross(max(cutin_power_fraction * rated_power, 1.0))
    if crossing is None:
        raise ValueError(
            f"无法从功率曲线中估计 cut_in_speed："
            f"没有连续 {consecutive_bins} 个有效 bin 达到 "
            f"{cutin_power_fraction:.0%} 额定功率"
        )
    cut_in_speed = crossing

    return {
        "cut_in_speed": round(cut_in_speed, 2),
        "rated_wind_speed": round(rated_wind_speed, 2),
        "rated_power": round(rated_power, 2),
    }
