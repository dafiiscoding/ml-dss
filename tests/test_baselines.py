import unittest

from src.baselines import (
    evaluate_humanitarian_baseline,
    evaluate_informative_baselines,
)
from src.data_loader import load_dataset
from src.split_integrity import filter_evaluation_split


class BaselineEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, _, test, _ = load_dataset(use_sample_if_missing=False)
        cls.test = filter_evaluation_split("test", test)

    def test_always_informative_exposes_f2_majority_effect(self):
        metrics = evaluate_informative_baselines(
            self.train["label"], self.test["label"]
        )
        row = metrics.loc["Always informative"]
        self.assertEqual(len(self.test), 2169)
        self.assertAlmostEqual(float(row["Recall"]), 1.0, places=4)
        self.assertAlmostEqual(float(row["F2"]), 0.8939, places=4)
        self.assertAlmostEqual(
            float(row["Balanced Accuracy"]), 0.5, places=4
        )
        self.assertAlmostEqual(float(row["MCC"]), 0.0, places=4)

    def test_humanitarian_majority_baseline(self):
        metrics = evaluate_humanitarian_baseline(
            self.train["label_top"], self.test["label_top"]
        )
        row = metrics.loc["Train majority"]
        self.assertEqual(row["Predicted Class"], "not_humanitarian")
        self.assertAlmostEqual(float(row["Macro F1"]), 0.0686, places=4)
        self.assertAlmostEqual(
            float(row["Balanced Accuracy"]), 0.125, places=4
        )


if __name__ == "__main__":
    unittest.main()
