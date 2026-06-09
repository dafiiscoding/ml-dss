import unittest

from scripts.review_near_duplicate_texts import (
    CONFIRMED,
    DECISIONS,
    RELATED,
    TEMPLATE,
)


class TextDuplicateReviewTests(unittest.TestCase):
    def test_review_covers_unique_candidate_keys(self):
        self.assertEqual(len(DECISIONS), 16)
        self.assertEqual(len(set(DECISIONS)), len(DECISIONS))

    def test_review_has_expected_status_vocabulary(self):
        statuses = [status for status, _ in DECISIONS.values()]
        self.assertEqual(
            set(statuses), {CONFIRMED, RELATED, TEMPLATE}
        )
        self.assertEqual(statuses.count(CONFIRMED), 5)


if __name__ == "__main__":
    unittest.main()
