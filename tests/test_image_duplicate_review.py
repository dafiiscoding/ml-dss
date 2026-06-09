import unittest

from scripts.review_near_duplicate_images import (
    COLLISION,
    EXCEPTIONS,
    RELATED,
    TEMPLATE,
)


class ImageDuplicateReviewTests(unittest.TestCase):
    def test_exception_keys_are_unique(self):
        self.assertEqual(len(EXCEPTIONS), 17)
        self.assertEqual(len(set(EXCEPTIONS)), len(EXCEPTIONS))

    def test_exception_status_vocabulary(self):
        statuses = [status for status, _ in EXCEPTIONS.values()]
        self.assertEqual(set(statuses), {RELATED, TEMPLATE, COLLISION})
        self.assertEqual(statuses.count(RELATED), 10)
        self.assertEqual(statuses.count(TEMPLATE), 4)
        self.assertEqual(statuses.count(COLLISION), 3)


if __name__ == "__main__":
    unittest.main()
