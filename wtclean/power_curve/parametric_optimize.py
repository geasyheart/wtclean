"""
Curve-fitting routines for parametric power curves.
"""

from typing import Callable

import numpy as np


def least_squares(x, y):
    """Least-squares loss between two arrays."""
    return np.sum((np.asarray(x) - np.asarray(y)) ** 2)


def fit_parametric_power_curve(
    x,
    y,
    curve: Callable,
    optimization_algorithm: Callable,
    cost_function: Callable,
    bounds,
    return_params: bool = False,
):
    """Fit a parametric curve to (wind speed, power) data.

    Args:
        x: independent variable (wind speed).
        y: dependent variable (power).
        curve: parametric form ``f(x, *params)``.
        optimization_algorithm: ``scipy.optimize``-style minimizer ``f(fun, bounds)``.
        cost_function: ``f(predicted, observed) -> float``.
        bounds: parameter bounds as a sequence of (low, high) tuples.
        return_params: if True return ``(callable, fit)``, else only ``callable``.

    Returns:
        Callable ``f(x) -> y`` implementing the fitted curve (or tuple if ``return_params``).
    """

    def f(opt_params):
        return cost_function(curve(np.asarray(x), *opt_params), y)

    fit = optimization_algorithm(f, bounds)

    def fit_curve(x_2):
        return curve(np.asarray(x_2), *fit.x)

    if return_params:
        return fit_curve, fit
    return fit_curve
