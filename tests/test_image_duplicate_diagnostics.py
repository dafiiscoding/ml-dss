import unittest

from scripts.build_image_duplicate_review_artifacts import _evidence_band


class ImageDuplicateDiagnosticTests(unittest.TestCase):
    def test_evidence_band_thresholds(self):
        self.assertEqual(
            _evidence_band(
                {
                    "clip_cosine_image": 0.995,
                    "pixel_corr_64": 0.99,
                    "pixel_mae_64": 0.01,
                }
            ),
            "strong",
        )
        self.assertEqual(
            _evidence_band(
                {
                    "clip_cosine_image": 0.97,
                    "pixel_corr_64": 0.90,
                    "pixel_mae_64": 0.08,
                }
            ),
            "moderate",
        )
        self.assertEqual(
            _evidence_band(
                {
                    "clip_cosine_image": 0.80,
                    "pixel_corr_64": 0.99,
                    "pixel_mae_64": 0.01,
                }
            ),
            "weak",
        )


if __name__ == "__main__":
    unittest.main()
