"""Tiền xử lý feature: chuẩn hoá số (StandardScaler) + mã hoá phân loại
(OneHotEncoder), gói trong ColumnTransformer.

Quan trọng: preprocessor được **fit trên train** (trong Pipeline) để tránh
leakage. Thuật ngữ xem `docs/GLOSSARY.md`.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pizza_dss.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_features=None, categorical_features=None):
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", _one_hot_encoder(), categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
