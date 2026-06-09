import unittest

import numpy as np

from src.stability_analysis import (
    percentile_interval,
    stratified_bootstrap_indices,
)


class StabilityAnalysisTests(unittest.TestCase):
    def test_stratified_bootstrap_preserves_class_counts(self):
        labels = np.array(["a", "a", "b", "b", "b", "c"])
        indices = stratified_bootstrap_indices(
            labels, np.random.default_rng(42)
        )
        sampled = labels[indices]
        self.assertEqual(len(sampled), len(labels))
        self.assertEqual(
            {label: int((sampled == label).sum()) for label in np.unique(labels)},
            {"a": 2, "b": 3, "c": 1},
        )

    def test_percentile_interval_is_ordered(self):
        interval = percentile_interval(np.arange(100), confidence=0.95)
        self.assertLess(interval["CI Low"], interval["CI High"])
        self.assertGreater(interval["CI Width"], 0)
        self.assertGreater(interval["Bootstrap SE"], 0)


if __name__ == "__main__":
    unittest.main()
