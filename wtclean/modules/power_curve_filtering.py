"""
该模块对 SCADA 数据应用迭代 bin + MAD（中位数绝对偏差）过滤，
在风速-功率曲线上把正常运行点与异常点分开。
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wtclean.utils.binning_function import binning_func


class PowerCurveFiltering:
    """对风速-功率曲线做迭代 bin + MAD 过滤。

    Attributes:
        windspeed_label: 风速列名。
        power_label: 有功功率列名。
        df: 单台风机的 SCADA DataFrame。
        cut_in_speed: 切入风速（厂商机型参数，必填）。
        rated_wind_speed: 额定风速，高于此风速功率应接近额定功率（厂商机型参数，必填）。
        rated_power: 风机额定功率（厂商机型参数，必填）。
        low_power_ratio: 额定功率的比例，高风区功率低于该比例即被粗洗。
        bin_interval: 风速分箱宽度。
        z_coeff: bin MAD 的倍数，用于界定正常带。
        filter_cycle: 迭代过滤轮数。
        return_fig: 是否保存功率曲线图。
        image_path: 保存图片的目录。

    """

    def __init__(
        self,
        windspeed_label: str,
        power_label: str,
        df: pd.DataFrame,
        cut_in_speed: float,
        rated_wind_speed: float,
        rated_power: float,
        low_power_ratio: float = 0.9,
        bin_interval: float = 0.5,
        z_coeff: float = 2.5,
        filter_cycle: int = 3,
        return_fig: bool = False,
        image_path: os.PathLike | str = "",
    ) -> None:

        for col in (windspeed_label, power_label):
            if col not in df.columns:
                raise ValueError(f"DataFrame 缺少列: {col}")
        if df.empty:
            raise ValueError("DataFrame 为空")

        if cut_in_speed <= 0:
            raise ValueError(f"cut_in_speed must be > 0, got {cut_in_speed}")
        if rated_wind_speed <= cut_in_speed:
            raise ValueError(
                f"rated_wind_speed must be > cut_in_speed, got "
                f"cut_in_speed={cut_in_speed}, rated_wind_speed={rated_wind_speed}"
            )
        if rated_power <= 0:
            raise ValueError(f"rated_power must be > 0, got {rated_power}")
        if not 0 < low_power_ratio <= 1:
            raise ValueError(f"low_power_ratio must be in (0, 1], got {low_power_ratio}")
        if bin_interval <= 0:
            raise ValueError(f"bin_interval must be > 0, got {bin_interval}")
        if z_coeff <= 0:
            raise ValueError(f"z_coeff must be > 0, got {z_coeff}")
        if filter_cycle < 1:
            raise ValueError(f"filter_cycle must be >= 1, got {filter_cycle}")

        self.windspeed_label = windspeed_label
        self.power_label = power_label
        self.df = df
        self.cut_in_speed = cut_in_speed
        self.rated_wind_speed = rated_wind_speed
        self.rated_power = rated_power
        self.low_power_ratio = low_power_ratio
        self.bin_interval = bin_interval
        self.z_coeff = z_coeff
        self.filter_cycle = filter_cycle
        self.return_fig = return_fig
        self.image_path = image_path

    def remove_downtime_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """剔除停机事件（风速达到切入风速及以上但功率接近零）。"""
        return df[
            ~(
                (df[self.power_label] <= 1)
                & (df[self.windspeed_label] >= self.cut_in_speed)
            )
        ]

    def remove_high_wind_low_power(self, df: pd.DataFrame) -> pd.DataFrame:
        """粗洗高风低功率点。

        达到额定风速及以上时风机本应接近额定功率；该区域内功率远低于
        ``low_power_ratio * rated_power`` 的点视为非正常工况（限电、降载、桨距/故障），
        为功率曲线基线建模而被剔除。这是粗洗，不是故障检测。
        """
        return df[
            ~(
                (df[self.power_label] < self.low_power_ratio * self.rated_power)
                & (df[self.windspeed_label] >= self.rated_wind_speed)
            )
        ]

    def secondary_filter(self, df: pd.DataFrame) -> list:
        """使用 bin + MAD 阈值（``z_coeff``）过滤单台风机数据。

        Returns:
            该台风机保留（正常）数据点的 ``_row_id`` 列表。
        """
        no_dt_per_turbine_df = df.copy()

        no_dt_per_turbine_df.loc[:, "windspeed_bin"] = np.floor(
            no_dt_per_turbine_df[self.windspeed_label] / self.bin_interval
        ).astype("Int64")

        if self.filter_cycle < 1:
            self.filter_cycle = 1

        self.removal_history: list[int] = []
        prev = len(no_dt_per_turbine_df)

        for i in range(int(self.filter_cycle)):
            binned_turb_df = binning_func(
                turbine_data=no_dt_per_turbine_df,
                windspeed_label=self.windspeed_label,
                power_label=self.power_label,
                bin_interval=self.bin_interval,
            )

            bins_cols = ["windspeed_bin_median", "pwr_bin_median", "pwr_bin_mad", "n"]
            if set(no_dt_per_turbine_df.columns).issuperset(set(bins_cols)):
                no_dt_per_turbine_df.drop(bins_cols, axis=1, inplace=True)

            no_dt_per_turbine_df = pd.merge(
                no_dt_per_turbine_df, binned_turb_df, how="left", on="windspeed_bin"
            )

            no_dt_per_turbine_df.loc[:, "pwr_low_thresh"] = (
                no_dt_per_turbine_df["pwr_bin_median"]
                - self.z_coeff * no_dt_per_turbine_df["pwr_bin_mad"]
            ).clip(lower=0)

            no_dt_per_turbine_df.loc[:, "pwr_high_thresh"] = (
                no_dt_per_turbine_df["pwr_bin_median"]
                + self.z_coeff * no_dt_per_turbine_df["pwr_bin_mad"]
            )

            in_band = (
                no_dt_per_turbine_df[self.power_label]
                > no_dt_per_turbine_df["pwr_low_thresh"]
            ) & (
                no_dt_per_turbine_df[self.power_label]
                < no_dt_per_turbine_df["pwr_high_thresh"]
            )
            below_cutin = no_dt_per_turbine_df[self.windspeed_label] < self.cut_in_speed
            unknown = no_dt_per_turbine_df["pwr_bin_mad"].isna()

            no_dt_per_turbine_df = no_dt_per_turbine_df[in_band | below_cutin | unknown]

            removed = prev - len(no_dt_per_turbine_df)
            self.removal_history.append(removed)
            print(
                f"iteration {i + 1}: removed {removed} "
                f"({removed / prev:.2%}), remaining {len(no_dt_per_turbine_df)}"
            )
            prev = len(no_dt_per_turbine_df)

        return no_dt_per_turbine_df["_row_id"].tolist()

    def process(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """运行完整过滤流程，返回正常与异常两个子集。

        Returns:
            元组 ``(normal_df, abnormal_df)``，按行划分原始 ``df``。
        """
        work = self.df.copy()
        work["_row_id"] = np.arange(len(work))

        no_dt_df = self.remove_downtime_events(work)

        # 从剩余的非停机数据中剔除高风低功率点
        no_dt_df = self.remove_high_wind_low_power(no_dt_df)

        normal_id_list = self.secondary_filter(no_dt_df)

        keep = work["_row_id"].isin(set(normal_id_list))

        assert keep.sum() == len(normal_id_list)

        normal_df = work[keep].drop(columns="_row_id")
        abnormal_df = work[~keep].drop(columns="_row_id")

        if self.return_fig:
            self.normal_df = normal_df.copy()
            self.abnormal_df = abnormal_df.copy()

            self.normal_df.loc[:, "Abnormal"] = "No"
            self.abnormal_df.loc[:, "Abnormal"] = "Yes"

            self.processed_data = pd.concat([self.normal_df, self.abnormal_df])

            if self.image_path and not os.path.exists(os.path.dirname(self.image_path)):
                os.makedirs(os.path.dirname(self.image_path), exist_ok=True)

            fig, ax = plt.subplots(figsize=(18, 6))
            for flag, color, label in (("No", "blue", "Normal"), ("Yes", "orange", "Abnormal")):
                sub = self.processed_data[self.processed_data["Abnormal"] == flag]
                ax.scatter(
                    sub[self.windspeed_label],
                    sub[self.power_label],
                    s=6,
                    c=color,
                    label=label,
                )
            ax.set_title("Operational power curve", fontsize=16)
            ax.legend()
            ax.set_xlabel("Wind Speed", fontsize=14)
            ax.set_ylabel("Power", fontsize=14)
            ax.tick_params(labelsize=14)
            
            fig.savefig(self.image_path)
            plt.close(fig)

        return normal_df, abnormal_df
