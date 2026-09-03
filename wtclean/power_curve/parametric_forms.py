"""
Parametric power curve forms.

These are written in the style of ``scipy.optimize``: each curve is a callable
``f(x, *params) -> y`` suitable for passing to a minimizer.

The 5-parameter logistic is an asymmetric generalization of the sigmoid and
models real wind-turbine power curves more faithfully than a symmetric sigmoid
(cut-in knee and rated-power plateau are not symmetric).
"""

import numpy as np


def _power_curve(x, a, b, c, d, g):
    return d + (a - d) / (1 + (x / c) ** b) ** g


def logistic5param(x, a, b, c, d, g):
    """5-parameter logistic (asymmetric sigmoid) power curve.

    Args:
        x: input wind speed.
        a: minimum value asymptote.
        b: steepness of the curve.
        c: point of inflection.
        d: maximum value asymptote.
        g: asymmetry of the curve.

    Returns:
        Array of power values.
    """
    x = np.asarray(x, dtype=np.float64)
    res = np.ones_like(x, dtype=np.float64)
    if b < 0:
        res *= d
        dom = x != 0.0
    else:
        dom = slice(None)
    res[dom] = _power_curve(x[dom], a, b, c, d, g)
    return res


def _cap(y, lower, upper):
    y = np.where(y < lower, lower, y)
    y = np.where(y > upper, upper, y)
    return y
