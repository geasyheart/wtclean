"""wtclean.power_curve 统一接口单元测试（unittest）。"""

import unittest

import numpy as np

from wtclean.power_curve import IEC, fit_power_curve, gam, logistic_5_parametric

CUT_IN, RATED_WS, RATED_PWR = 3.0, 12.0, 2000.0


def _sample(n=400, seed=0):
    rng = np.random.default_rng(seed)
    ws = rng.uniform(CUT_IN, 25.0, n)
    power = np.where(ws < RATED_WS, RATED_PWR * (ws - CUT_IN) / (RATED_WS - CUT_IN), RATED_PWR)
    power += rng.normal(0, 20, n)
    return ws, power


class TestPowerCurveBackends(unittest.TestCase):
    def test_gam_monotonic(self):
        ws, power = _sample()
        model = gam(ws, power, n_splines=15)
        grid = np.linspace(CUT_IN, 20.0, 60)
        pred = model.predict(grid)
        self.assertTrue(np.all(np.diff(pred) >= -1e-6), "GAM 应单调递增")

    def test_gam_gridsearch_picks_valid_model(self):
        ws, power = _sample()
        model = gam(ws, power)
        self.assertTrue(callable(model.predict))
        self.assertTrue(np.isfinite(model.predict(np.array([10.0]))[0]))

    def test_fit_power_curve_dispatches(self):
        ws, power = _sample()
        for method in ("gam", "logistic", "iec"):
            f = fit_power_curve(ws, power, method=method)
            out = f(np.array([3.0, 7.0, 12.0, 20.0]))
            self.assertEqual(out.shape, (4,))
            self.assertTrue(np.all(np.isfinite(out)))

    def test_logistic_returns_callable(self):
        ws, power = _sample()
        f = logistic_5_parametric(ws, power, bounds=((1500, 2200), (-10, -1e-3), (1e-3, 30), (1e-3, 1), (1e-3, 10)))
        self.assertTrue(callable(f))

    def test_iec_zero_outside_range(self):
        ws, power = _sample()
        f = IEC(ws, power)
        out = f(np.array([-1.0, 5.0, 50.0]))
        self.assertEqual(out[0], 0.0)
        self.assertEqual(out[2], 0.0)

    def test_unknown_method_raises(self):
        ws, power = _sample()
        with self.assertRaises(ValueError):
            fit_power_curve(ws, power, method="nope")


if __name__ == "__main__":
    unittest.main()
