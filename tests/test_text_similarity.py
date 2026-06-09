import unittest

from src.text_similarity import (
    build_lexical_similarity_space,
    sparse_row_cosine,
    token_jaccard,
)


class TextSimilarityTests(unittest.TestCase):
    def test_related_edit_scores_above_unrelated_text(self):
        texts = [
            "warning letter to harvey and irma survivors",
            "warning letter for harvey irma survivors",
            "sunny afternoon at the beach",
        ]
        space = build_lexical_similarity_space(texts)
        related = sparse_row_cosine(space.combined, 0, 1)
        unrelated = sparse_row_cosine(space.combined, 0, 2)
        self.assertGreater(related, unrelated)
        self.assertGreater(related, 0.5)

    def test_token_jaccard(self):
        self.assertEqual(token_jaccard("a b c", "a b d"), 0.5)
        self.assertEqual(token_jaccard("", ""), 1.0)


if __name__ == "__main__":
    unittest.main()
