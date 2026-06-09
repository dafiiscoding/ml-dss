import json
import os
import unittest

import numpy as np

from src.config import FUSION_CONFIG_PATH, MODELS_DIR
from src.data_loader import _derive_informative, load_dataset
from src.decision_rules import get_dss_decision, get_priority_level
from src.fusion import MultimodalFusionPredictor, align_probabilities
from src.risk_scoring import calculate_risk_score
from src.split_integrity import evaluation_mask


class DataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.val, cls.test, _ = load_dataset(
            use_sample_if_missing=False
        )

    def test_fixed_splits_and_schema(self):
        self.assertEqual(
            (len(self.train), len(self.val), len(self.test)),
            (13608, 2237, 2237),
        )
        required = {
            "label",
            "label_top",
            "label_text_cat",
            "label_image_cat",
            "multimodal_agree",
            "informative_agree",
        }
        self.assertTrue(required.issubset(self.train.columns))

    def test_official_informative_target_is_not_derived_from_common_category(self):
        mismatches = sum(
            (
                _derive_informative(df["label_top"]) != df["label"]
            ).sum()
            for df in (self.train, self.val, self.test)
        )
        self.assertEqual(mismatches, 2649)
        for df in (self.train, self.val, self.test):
            self.assertTrue(
                (
                    _derive_informative(df["label_text_cat"])
                    == df["label_text_inf"]
                ).all()
            )

    def test_no_split_overlap(self):
        sets = [
            set(zip(df["tweet_id"], df["image_id"]))
            for df in (self.train, self.val, self.test)
        ]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])

    def test_leakage_safe_evaluation_masks(self):
        self.assertEqual(int(evaluation_mask("val", self.val).sum()), 2189)
        self.assertEqual(int(evaluation_mask("test", self.test).sum()), 2169)

    def test_embedding_shapes(self):
        for split, expected in (("train", 13608), ("val", 2237), ("test", 2237)):
            values = np.load(os.path.join(MODELS_DIR, f"X_{split}_img_emb.npy"))
            self.assertEqual(values.shape, (expected, 512))
            self.assertGreater(float(values.std()), 0)


class FusionAndDecisionTests(unittest.TestCase):
    def test_probability_alignment(self):
        aligned = align_probabilities(
            np.array([0.2, 0.8]),
            np.array(["not_humanitarian", "injured_or_dead_people"]),
        )
        self.assertAlmostEqual(float(aligned.sum()), 1.0)
        self.assertEqual(int(np.argmax(aligned)), 0)

    def test_text_only_fallback(self):
        result = MultimodalFusionPredictor().predict(
            "Urgent rescue needed after bridge collapse.", None
        )
        self.assertFalse(result["image_present"])
        self.assertEqual(result["fused_category"], result["text_category"])
        self.assertEqual(result["conflict_score"], 0.0)

    def test_conflict_override(self):
        with open(FUSION_CONFIG_PATH, encoding="utf-8") as stream:
            threshold = json.load(stream)["manual_review"]["conflict_threshold"]
        result = {
            "fused_informative_prob": 0.8,
            "fused_category": "injured_or_dead_people",
            "fused_category_confidence": 0.8,
            "conflict_score": threshold + 0.01,
            "manual_review_threshold": threshold,
            "text_informative_prob": 0.9,
            "image_informative_prob": 0.1,
        }
        decision = get_dss_decision(result, "injured people need help")
        self.assertTrue(decision["manual_review"])
        self.assertTrue(decision["assigned_team"].startswith("Supervisor"))

    def test_conflict_review_never_downgrades_high_priority(self):
        result = {
            "fused_informative_prob": 0.99,
            "fused_category": "injured_or_dead_people",
            "fused_category_confidence": 0.99,
            "conflict_score": 0.99,
            "manual_review_threshold": 0.54,
        }
        decision = get_dss_decision(
            result, "urgent trapped injured rescue help"
        )
        self.assertEqual(decision["base_priority"], "High")
        self.assertEqual(decision["priority"], "High")
        self.assertIn("Emergency Team", decision["assigned_team"])

    def test_non_humanitarian_confidence_does_not_raise_risk(self):
        low_confidence = calculate_risk_score(
            0.1, "not_humanitarian", "ordinary social post", 0.1
        )
        high_confidence = calculate_risk_score(
            0.1, "not_humanitarian", "ordinary social post", 0.99
        )
        self.assertEqual(low_confidence, high_confidence)

    def test_priority_boundaries(self):
        self.assertEqual(get_priority_level(39), "Low")
        self.assertEqual(get_priority_level(40), "Medium")
        self.assertEqual(get_priority_level(69), "Medium")
        self.assertEqual(get_priority_level(70), "High")


if __name__ == "__main__":
    unittest.main()
