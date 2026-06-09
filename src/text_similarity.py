"""Interpretable lexical similarity signals for text duplicate audits."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass
class LexicalSimilaritySpace:
    combined: object
    character: object
    word: object


def build_lexical_similarity_space(
    texts,
    character_weight=0.6,
    word_weight=0.4,
):
    """Build normalized char/word TF-IDF matrices for audit use only."""
    if not np.isclose(character_weight + word_weight, 1.0):
        raise ValueError("Similarity weights must sum to one.")

    character_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        stop_words="english",
        sublinear_tf=True,
        dtype=np.float32,
    )
    character = character_vectorizer.fit_transform(texts)
    word = word_vectorizer.fit_transform(texts)
    combined = normalize(
        hstack(
            [
                character * np.sqrt(character_weight),
                word * np.sqrt(word_weight),
            ],
            format="csr",
        ),
        copy=False,
    )
    return LexicalSimilaritySpace(combined, character, word)


def sparse_row_cosine(matrix, left_index, right_index):
    """Cosine for rows of an already L2-normalized sparse matrix."""
    return float(
        matrix.getrow(left_index)
        .multiply(matrix.getrow(right_index))
        .sum()
    )


def token_jaccard(left, right):
    left_tokens = set(str(left).split())
    right_tokens = set(str(right).split())
    union = left_tokens | right_tokens
    if not union:
        return 1.0
    return len(left_tokens & right_tokens) / len(union)
