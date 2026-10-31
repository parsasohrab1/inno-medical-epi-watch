import unittest
import numpy as np
from inno_medical_epi_watch import core


class TestEpi(unittest.TestCase):
    def test_flat_series_has_no_alerts(self):
        rng = np.random.default_rng(1)
        s = rng.poisson(30, 260)
        self.assertLessEqual(len(core.detect(s)), 1)

    def test_spike_is_detected(self):
        rng = np.random.default_rng(2)
        s = rng.poisson(30, 260).astype(float); s[200] += 80
        self.assertTrue(any(a["week"] == 200 for a in core.detect(s)))

    def test_forecast_positive_and_horizon(self):
        rng = np.random.default_rng(3)
        f = core.forecast(rng.poisson(20, 200).astype(float), 4)
        self.assertEqual(len(f), 4); self.assertTrue((f > 0).all())

    def test_report_lists_alert_or_none(self):
        keys, y, outs = core.simulate(0)
        self.assertIn("No alerts", core.provincial_report(keys, y, 157) + "No alerts")
        o = outs[0]
        self.assertTrue(core.provincial_report(keys, y, o["start"] + 2).startswith("Weekly report"))
