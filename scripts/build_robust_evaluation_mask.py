"""Build a stricter evaluation mask from verified duplicate reviews."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR

PROCESSED_DIR = Path(PROCESSED_DATA_DIR)
METRICS_DIR = Path(REPORTS_DIR) / "metrics"
INTEGRITY_PATH = PROCESSED_DIR / "split_integrity.csv"
IMAGE_REVIEW_PATH = METRICS_DIR / "image_near_duplicate_review.csv"
TEXT_REVIEW_PATH = METRICS_DIR / "text_near_duplicate_review.csv"
MASK_PATH = PROCESSED_DIR / "robust_evaluation_mask.csv"
SUMMARY_PATH = METRICS_DIR / "robust_evaluation_mask_summary.json"


def _confirmed_keys(path):
    review = pd.read_csv(path)
    confirmed = review[review["robust_exclude"]]
    return set(
        zip(
            confirmed["query_split"],
            confirmed["query_row_index"].astype(int),
        )
    )


def build():
    manifest = pd.read_csv(INTEGRITY_PATH)
    image_keys = _confirmed_keys(IMAGE_REVIEW_PATH)
    text_keys = _confirmed_keys(TEXT_REVIEW_PATH)
    union_keys = image_keys | text_keys

    mask = manifest[
        [
            "split",
            "row_index",
            "tweet_id",
            "image_id",
            "evaluation_eligible",
            "exclusion_reason",
        ]
    ].copy()
    mask = mask.rename(
        columns={
            "evaluation_eligible": "canonical_evaluation_eligible",
            "exclusion_reason": "canonical_exclusion_reason",
        }
    )
    mask["robust_evaluation_eligible"] = mask[
        "canonical_evaluation_eligible"
    ].astype(bool)
    mask["robust_exclusion_reason"] = mask[
        "canonical_exclusion_reason"
    ].fillna("")

    known_keys = set(zip(mask["split"], mask["row_index"].astype(int)))
    unknown = sorted(union_keys - known_keys)
    if unknown:
        raise ValueError(f"Reviewed duplicate keys are missing: {unknown}")

    row_lookup = {
        (row.split, int(row.row_index)): index
        for index, row in mask.iterrows()
    }
    for key in sorted(union_keys):
        index = row_lookup[key]
        if not bool(mask.at[index, "canonical_evaluation_eligible"]):
            raise ValueError(
                f"Reviewed near-duplicate was already canonically excluded: {key}"
            )
        reasons = []
        if key in image_keys:
            reasons.append("confirmed_image_near_duplicate")
        if key in text_keys:
            reasons.append("confirmed_text_near_duplicate")
        mask.at[index, "robust_evaluation_eligible"] = False
        mask.at[index, "robust_exclusion_reason"] = ";".join(reasons)

    canonical_counts = (
        mask.groupby("split")["canonical_evaluation_eligible"]
        .sum()
        .astype(int)
        .to_dict()
    )
    robust_counts = (
        mask.groupby("split")["robust_evaluation_eligible"]
        .sum()
        .astype(int)
        .to_dict()
    )
    union_counts = Counter(split for split, _ in union_keys)
    summary = {
        "policy": {
            "canonical_mask": (
                "exact prior-split image SHA or cleaned-text hash exclusions"
            ),
            "robust_mask": (
                "canonical mask plus pairwise-verified image/text near-duplicates"
            ),
            "canonical_mask_changed": False,
        },
        "canonical_eligible_rows": canonical_counts,
        "robust_eligible_rows": robust_counts,
        "additional_verified_exclusions": len(union_keys),
        "additional_verified_exclusions_by_split": {
            key: int(value)
            for key, value in sorted(union_counts.items())
        },
        "confirmed_image_rows": len(image_keys),
        "confirmed_text_rows": len(text_keys),
        "image_text_overlap_rows": len(image_keys & text_keys),
    }
    mask.to_csv(MASK_PATH, index=False)
    with SUMMARY_PATH.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return mask, summary


if __name__ == "__main__":
    build()
