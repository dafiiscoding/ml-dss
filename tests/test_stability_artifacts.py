import os
import unittest

import pandas as pd

from scripts.evaluate_stability import METRICS_DIR


class StabilityArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.intervals = pd.read_csv(
            os.path.join(METRICS_DIR, "robust_bootstrap_intervals.csv")
        )
        cls.events = pd.read_csv(
            os.path.join(METRICS_DIR, "robust_event_stability.csv")
        )
        cls.classes = pd.read_csv(
            os.path.join(METRICS_DIR, "robust_class_stability.csv")
        )

    def test_interval_estimates_are_inside_bounds(self):
        self.assertTrue(
            (
                (self.intervals["CI Low"] <= self.intervals["Estimate"])
                & (
                    self.intervals["Estimate"]
                    <= self.intervals["CI High"]
                )
            ).all()
        )
        self.assertTrue((self.intervals["Resamples"] == 2000).all())

    def test_all_events_and_classes_are_covered(self):
        self.assertEqual(len(self.events), 7)
        self.assertEqual(len(self.classes), 8)
        self.assertEqual(int(self.events["Rows"].sum()), 2032)
        self.assertEqual(int(self.classes["Robust Support"].sum()), 2032)

    def test_dummy_gain_intervals_are_reported(self):
        keys = set(zip(self.intervals["Task"], self.intervals["Metric"]))
        self.assertIn(("Informative Fusion vs Dummy", "F2 Gain"), keys)
        self.assertIn(
            ("Humanitarian Fusion vs Dummy", "Macro F1 Gain"), keys
        )


if __name__ == "__main__":
    unittest.main()
