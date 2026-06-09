import unittest

import pandas as pd

from scripts.build_robust_evaluation_mask import MASK_PATH
from src.data_loader import load_dataset
from src.split_integrity import robust_evaluation_mask


class RobustEvaluationMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask = pd.read_csv(MASK_PATH)

    def test_robust_mask_is_a_subset_of_canonical_mask(self):
        canonical = self.mask["canonical_evaluation_eligible"].astype(bool)
        robust = self.mask["robust_evaluation_eligible"].astype(bool)
        self.assertTrue((~robust | canonical).all())

    def test_locked_robust_counts(self):
        counts = (
            self.mask.groupby("split")["robust_evaluation_eligible"]
            .sum()
            .astype(int)
            .to_dict()
        )
        self.assertEqual(
            counts, {"train": 13608, "val": 2078, "test": 2032}
        )

    def test_additional_exclusion_count(self):
        canonical = self.mask["canonical_evaluation_eligible"].astype(bool)
        robust = self.mask["robust_evaluation_eligible"].astype(bool)
        self.assertEqual(int((canonical & ~robust).sum()), 248)

    def test_runtime_helper_matches_locked_test_count(self):
        _, val_df, test_df, _ = load_dataset(use_sample_if_missing=False)
        self.assertEqual(int(robust_evaluation_mask("val", val_df).sum()), 2078)
        self.assertEqual(
            int(robust_evaluation_mask("test", test_df).sum()), 2032
        )


if __name__ == "__main__":
    unittest.main()
