"""Helpers for applying the leakage-safe development/test evaluation mask."""
import os

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR

INTEGRITY_PATH = os.path.join(PROCESSED_DATA_DIR, "split_integrity.csv")
ROBUST_INTEGRITY_PATH = os.path.join(
    PROCESSED_DATA_DIR, "robust_evaluation_mask.csv"
)


def _manifest_mask(split, df, path, eligibility_column, missing_command):
    if split == "train":
        return np.ones(len(df), dtype=bool)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Evaluation manifest is missing. Run `{missing_command}`."
        )
    manifest = pd.read_csv(path)
    part = manifest[manifest["split"] == split].sort_values("row_index")
    if len(part) != len(df):
        raise ValueError(
            f"Evaluation manifest/data length mismatch for {split}: "
            f"{len(part)} != {len(df)}"
        )
    for column in ("tweet_id", "image_id"):
        if not part[column].astype(str).reset_index(drop=True).equals(
            df[column].astype(str).reset_index(drop=True)
        ):
            raise ValueError(
                f"Evaluation manifest order mismatch for {split}.{column}"
            )
    if eligibility_column not in part:
        raise ValueError(
            f"Evaluation manifest is missing `{eligibility_column}`."
        )
    return part[eligibility_column].astype(bool).to_numpy()


def evaluation_mask(split, df):
    return _manifest_mask(
        split,
        df,
        INTEGRITY_PATH,
        "evaluation_eligible",
        "python -m scripts.audit_data",
    )


def robust_evaluation_mask(split, df):
    return _manifest_mask(
        split,
        df,
        ROBUST_INTEGRITY_PATH,
        "robust_evaluation_eligible",
        "python -m scripts.build_robust_evaluation_mask",
    )


def filter_evaluation_split(split, df, features=None):
    mask = evaluation_mask(split, df)
    filtered_df = df.loc[mask].reset_index(drop=True)
    if features is None:
        return filtered_df
    return filtered_df, features[mask]


def filter_robust_evaluation_split(split, df, features=None):
    mask = robust_evaluation_mask(split, df)
    filtered_df = df.loc[mask].reset_index(drop=True)
    if features is None:
        return filtered_df
    return filtered_df, features[mask]
