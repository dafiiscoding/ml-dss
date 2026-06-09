"""Record pairwise visual review decisions for image candidates."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR

METRICS_DIR = Path(REPORTS_DIR) / "metrics"
DIAGNOSTIC_PATH = METRICS_DIR / "image_near_duplicate_diagnostics.csv"
REVIEW_PATH = METRICS_DIR / "image_near_duplicate_review.csv"
SUMMARY_PATH = METRICS_DIR / "image_near_duplicate_review_summary.json"

CONFIRMED = "confirmed_near_duplicate"
RELATED = "related_not_duplicate"
TEMPLATE = "template_not_duplicate"
COLLISION = "phash_collision"

# Every candidate not listed here was visually confirmed as the same image or
# graphic after resize, crop, compression, color adjustment, or text overlay.
EXCEPTIONS = {
    ("test", 1684): (
        RELATED,
        "Different tropical outlook advisory with changed storm positions.",
    ),
    ("val", 1918): (
        RELATED,
        "Different hurricane forecast track/advisory frame.",
    ),
    ("val", 2078): (
        RELATED,
        "Different hurricane forecast track/advisory frame.",
    ),
    ("val", 545): (
        RELATED,
        "Different hurricane forecast track and issue time.",
    ),
    ("test", 1758): (
        RELATED,
        "Different hurricane forecast track/advisory frame.",
    ),
    ("test", 1857): (
        RELATED,
        "Different hurricane forecast track/advisory frame.",
    ),
    ("val", 1884): (
        RELATED,
        "Different satellite observation of the same storm.",
    ),
    ("test", 587): (
        RELATED,
        "Different hurricane forecast track/advisory frame.",
    ),
    ("val", 160): (
        RELATED,
        "Different hurricane forecast track/advisory frame.",
    ),
    ("val", 811): (
        RELATED,
        "Same weather-map template but different track data and timestamp.",
    ),
    ("test", 1206): (
        TEMPLATE,
        "Same hourly-news template but different timestamp and headline.",
    ),
    ("val", 948): (
        TEMPLATE,
        "Same hourly-news template but different disaster and headline.",
    ),
    ("val", 1518): (
        TEMPLATE,
        "Same Trendolizer layout but different story, timestamp, and source.",
    ),
    ("val", 107): (
        TEMPLATE,
        "Same Trendolizer layout but different story, timestamp, and source.",
    ),
    ("test", 488): (
        COLLISION,
        "Solid black and solid white images collide under pHash.",
    ),
    ("val", 2103): (
        COLLISION,
        "Different statistical charts with similar global layout.",
    ),
    ("val", 1623): (
        COLLISION,
        "Different aviation maps with similar low-frequency structure.",
    ),
}


def review():
    diagnostics = pd.read_csv(DIAGNOSTIC_PATH)
    candidate_keys = set(
        zip(diagnostics["query_split"], diagnostics["query_row_index"])
    )
    stale = sorted(set(EXCEPTIONS) - candidate_keys)
    if stale:
        raise ValueError(f"Image review exceptions are stale: {stale}")
    if diagnostics[["review_sheet", "review_sheet_position"]].isna().any().any():
        raise ValueError("Review sheet references are incomplete.")

    statuses = []
    reasons = []
    for row in diagnostics.itertuples(index=False):
        key = (row.query_split, int(row.query_row_index))
        status, reason = EXCEPTIONS.get(
            key,
            (
                CONFIRMED,
                "Pairwise visual review confirms the same image/graphic "
                "after resize, crop, compression, color adjustment, or "
                "text overlay.",
            ),
        )
        statuses.append(status)
        reasons.append(reason)

    reviewed = diagnostics.copy()
    reviewed["review_status"] = statuses
    reviewed["review_reason"] = reasons
    reviewed["review_basis"] = (
        "pairwise_visual_contact_sheet_plus_quantitative_diagnostics"
    )
    reviewed["robust_exclude"] = (
        reviewed["review_status"] == CONFIRMED
    )

    status_counts = Counter(reviewed["review_status"])
    exclusions = reviewed[reviewed["robust_exclude"]]
    exclusion_counts = Counter(exclusions["query_split"])
    summary = {
        "criteria": {
            "confirmed_near_duplicate": (
                "same scene or graphic after resize, crop, compression, "
                "color adjustment, or text overlay"
            ),
            "related_not_duplicate": (
                "same storm/event family but a materially different map, "
                "advisory, timestamp, or observation"
            ),
            "template_not_duplicate": (
                "same layout with different story, timestamp, or source"
            ),
            "phash_collision": (
                "visually distinct content sharing a low-frequency hash"
            ),
            "mask_policy": (
                "only confirmed_near_duplicate rows are proposed for the "
                "robust evaluation mask"
            ),
        },
        "reviewed_candidates": len(reviewed),
        "review_status_counts": {
            key: int(value) for key, value in sorted(status_counts.items())
        },
        "confirmed_exclusions": len(exclusions),
        "confirmed_exclusions_by_split": {
            key: int(value)
            for key, value in sorted(exclusion_counts.items())
        },
        "canonical_evaluation_mask_changed": False,
    }
    reviewed.to_csv(REVIEW_PATH, index=False)
    with SUMMARY_PATH.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return reviewed, summary


if __name__ == "__main__":
    review()
