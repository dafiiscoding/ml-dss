import json
import os
import unittest

import pandas as pd

from scripts.evaluate_robustness import METRICS_DIR, evaluate
from src.config import FUSION_CONFIG_PATH


class RobustnessEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FUSION_CONFIG_PATH, encoding="utf-8") as stream:
            cls.config_before = json.load(stream)
        cls.summary, cls.comparison = evaluate()
        with open(FUSION_CONFIG_PATH, encoding="utf-8") as stream:
            cls.config_after = json.load(stream)

    def test_locked_config_is_not_modified(self):
        self.assertEqual(self.config_before, self.config_after)
        self.assertIn("no model fitting", self.summary["method"])

    def test_locked_row_counts(self):
        self.assertEqual(self.summary["canonical_rows"], 2169)
        self.assertEqual(self.summary["robust_rows"], 2032)
        self.assertEqual(self.summary["additional_rows_excluded"], 137)

    def test_comparison_contains_core_metrics(self):
        keys = set(
            zip(self.comparison["Task"], self.comparison["Metric"])
        )
        self.assertIn(("Informative Fusion", "F2"), keys)
        self.assertIn(("Informative Fusion", "MCC"), keys)
        self.assertIn(("Humanitarian Fusion", "Macro F1"), keys)
        self.assertIn(("Manual Review", "Review Rate"), keys)

    def test_artifacts_are_written(self):
        path = os.path.join(
            METRICS_DIR, "robustness_metric_comparison.csv"
        )
        self.assertTrue(os.path.exists(path))
        table = pd.read_csv(path)
        self.assertFalse(table.empty)


if __name__ == "__main__":
    unittest.main()
