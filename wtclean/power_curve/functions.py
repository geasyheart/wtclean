"""
Ready-to-use power curve backends.

Every backend takes (wind speed, power) arrays and returns a callable
``f(x) -> y`` that evaluates the fitted power curve at arbitrary wind speeds.
The unified entry point is :func:`fit_power_curve`.
"""

from typing import Callable, Iterable

import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution

from wtclean.power_curve.parametric_forms import logistic5param
from wtclean.power_curve.parametric_optimize import (
    fit_parametric_power_curve,
    least_squares,
)


def IEC(
    windspeed,
    power,
    bin_width: float = 0.5,
    windspeed_start: float = 0.0,
    windspeed_end: float = 30.0,
    interpolate: bool = False,
) -> Callable:
    """IEC 61400-12-1 binned power curve.

    Power is set to zero outside [``windspeed_start``, ``windspeed_end``].

    Args:
        windspeed: wind speed data.
        power: power data.
        bin_width: wind speed bin width (default 0.5 m/s per the standard).
        windspeed_start: left edge of first bin.
        windspeed_end: right edge of last bin.
        interpolate: if True, linearly interpolate between bin points instead of
            assigning the bin average to every point in the bin.

    Returns:
        Callable ``f(x) -> y``.
    """
    windspeed = np.asarray(windspeed, dtype=float)
    power = np.asarray(power, dtype=float)

    n_bins = int(np.ceil((windspeed_end - windspeed_start) / bin_width)) + 1
    bins = np.append(np.linspace(windspeed_start, windspeed_end, n_bins), [np.inf])
    p_bin = np.ones(len(bins) - 1) * np.nan

    for i in range(len(bins) - 1):
        idx = (windspeed >= bins[i]) & (windspeed < bins[i + 1])
        if idx.any():
            p_bin[i] = power[idx].mean()

    p_bin = pd.Series(p_bin).interpolate(method="linear").bfill().values

    def pc_iec_bin(x):
        x = np.asarray(x, dtype=float)
        p = np.zeros_like(x, dtype=float)
        for i in range(len(bins) - 1):
            idx = (x >= bins[i]) & (x < bins[i + 1])
            p[idx] = p_bin[i]
        p[(x < windspeed_start) | (x > windspeed_end)] = 0.0
        return p

    def pc_iec_interp(x):
        x = np.asarray(x, dtype=float)
        f = interp1d(
            bins[:-1] + bin_width / 2,
            p_bin,
            fill_value=(p_bin[0], p_bin[-1]),
            bounds_error=False,
        )
        p = f(x)
        p[(x < windspeed_start) | (x > windspeed_end)] = 0.0
        return p

    return pc_iec_interp if interpolate else pc_iec_bin


def logistic_5_parametric(
    windspeed,
    power,
    bounds=((1200, 1800), (-10, -1e-3), (1e-3, 30), (1e-3, 1), (1e-3, 10)),
) -> Callable:
    """Fit the 5-parameter logistic to observed data via least squares.

    Default bounds assume power in kW (roughly a 1.2-1.8 MW machine). Override
    ``bounds`` for other ratings (e.g. a 2 MW machine: ``((1500, 2200), ...)``).

    Args:
        windspeed: wind speed data.
        power: power data.
        bounds: parameter bounds for ``(a, b, c, d, g)``.

    Returns:
        Callable ``f(x) -> y``.
    """
    return fit_parametric_power_curve(
        windspeed,
        power,
        curve=logistic5param,
        optimization_algorithm=differential_evolution,
        cost_function=least_squares,
        bounds=bounds,
    )


def _fit_gam(
    windspeed, power, monotonic: bool, n_splines: int, lam: float
) -> LinearGAM:
    x = np.asarray(windspeed, dtype=float).reshape(-1, 1)
    y = np.asarray(power, dtype=float)
    constraint = "monotonic_inc" if monotonic else None
    return LinearGAM(s(0, n_splines=n_splines, lam=lam, constraints=constraint)).fit(
        x, y
    )


def gam(
    windspeed,
    power,
    n_splines: int | None = None,
    n_splines_grid: Iterable[int] = (10, 15, 20, 25, 30, 35),
    lam: float = 0.6,
    monotonic: bool = True,
) -> LinearGAM:
    """Fit a generalized additive model of power vs wind speed.

    Unlike a plain ``LinearGAM``, this enforces a monotonically increasing
    relationship (physical: more wind never reduces the ideal power output) and,
    by default, grid-searches ``n_splines`` minimizing generalized
    cross-validation (GCV). Both features make the fitted curve more faithful and
    robust than an unconstrained spline, which can bend to fit outlier clusters.

    Args:
        windspeed: wind speed data.
        power: power data.
        n_splines: fixed number of splines. If None, grid search over
            ``n_splines_grid`` is performed.
        n_splines_grid: candidate spline counts for the grid search.
        lam: smoothing penalty strength.
        monotonic: enforce a monotonically increasing curve.

    Returns:
        Fitted :class:`pygam.LinearGAM` (use ``.predict`` to evaluate).
    """
    if n_splines is not None:
        return _fit_gam(windspeed, power, monotonic, n_splines, lam)

    # x = np.asarray(windspeed, dtype=float).reshape(-1, 1)
    # y = np.asarray(power, dtype=float)

    best, best_gcv = None, np.inf
    for ns in n_splines_grid:
        try:
            model = _fit_gam(windspeed, power, monotonic, int(ns), lam)
            gcv = model.statistics_["GCV"]
        except Exception:
            continue
        if np.isfinite(gcv) and gcv < best_gcv:
            best, best_gcv = model, gcv

    if best is None:
        raise ValueError("GAM 拟合失败：无法在给定 n_splines_grid 上得到有限 GCV")

    return best


def fit_power_curve(windspeed, power, method: str = "gam", **kwargs) -> Callable:
    """Fit a power curve and return ``f(x) -> y``.

    Args:
        windspeed: wind speed data.
        power: power data.
        method: one of ``"gam"`` (default), ``"logistic"`` / ``"logistic5param"``,
            or ``"iec"``.
        **kwargs: forwarded to the selected backend (e.g. ``n_splines``, ``bin_width``).

    Returns:
        Callable ``f(x) -> y``.
    """
    if method == "gam":
        return gam(windspeed, power, **kwargs).predict
    if method in ("logistic", "logistic5param", "sigmoid5"):
        return logistic_5_parametric(windspeed, power, **kwargs)
    if method == "iec":
        return IEC(windspeed, power, **kwargs)
    raise ValueError(f"未知的功率曲线方法: {method!r}（可选 gam / logistic / iec）")
