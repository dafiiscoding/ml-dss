"""Record a complete, reproducible manual review of text candidates."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR

METRICS_DIR = Path(REPORTS_DIR) / "metrics"
CANDIDATE_PATH = METRICS_DIR / "text_near_duplicate_candidates.csv"
REVIEW_PATH = METRICS_DIR / "text_near_duplicate_review.csv"
SUMMARY_PATH = METRICS_DIR / "text_near_duplicate_review_summary.json"

CONFIRMED = "confirmed_near_duplicate"
RELATED = "related_not_duplicate"
TEMPLATE = "template_not_duplicate"

# Decisions are keyed by the later-split query row. The reasons deliberately
# distinguish copied wording/content from shared event semantics or templates.
DECISIONS = {
    ("test", 279): (
        TEMPLATE,
        "Same Copernicus activation template, but different activation IDs "
        "and different earthquakes.",
    ),
    ("test", 407): (
        RELATED,
        "Same Mexico earthquake topic, but different death totals and "
        "reporting stage.",
    ),
    ("test", 584): (
        CONFIRMED,
        "Same article headline; differences are source tags and truncated "
        "encoding only.",
    ),
    ("test", 761): (
        RELATED,
        "Same Sri Lanka flood, but distinct death-toll updates from different "
        "publishers.",
    ),
    ("test", 801): (
        TEMPLATE,
        "Generic GoFundMe call with different destination URLs and no shared "
        "campaign identifier.",
    ),
    ("test", 857): (
        RELATED,
        "Same video-game product, but different retail/promotional posts.",
    ),
    ("test", 1505): (
        TEMPLATE,
        "Same trend-alert template, but different named storms.",
    ),
    ("test", 1720): (
        RELATED,
        "Same disaster context, but different reported magnitude and wording.",
    ),
    ("test", 1906): (
        CONFIRMED,
        "Same distinctive article headline; only author/source attribution "
        "differs.",
    ),
    ("val", 183): (
        RELATED,
        "Same race weekend and driver, but pole position and race win are "
        "different facts.",
    ),
    ("val", 283): (
        TEMPLATE,
        "Same news-search template, but different hurricane and date.",
    ),
    ("val", 330): (
        RELATED,
        "Same hurricane context, but different wind-speed update and wording.",
    ),
    ("val", 484): (
        CONFIRMED,
        "Same distinctive article headline; only publisher prefix and text "
        "encoding differ.",
    ),
    ("val", 756): (
        CONFIRMED,
        "Same short promotional text with hashtag order and encoding changes.",
    ),
    ("val", 1302): (
        RELATED,
        "Same Mexico earthquake, but one is a broad report and the other a "
        "specific casualty update.",
    ),
    ("val", 2075): (
        CONFIRMED,
        "Same article proposition and the same two shortened URLs.",
    ),
}


def review():
    candidates = pd.read_csv(CANDIDATE_PATH)
    candidate_keys = set(
        zip(candidates["query_split"], candidates["query_row_index"])
    )
    decision_keys = set(DECISIONS)
    if candidate_keys != decision_keys:
        missing = sorted(candidate_keys - decision_keys)
        stale = sorted(decision_keys - candidate_keys)
        raise ValueError(
            f"Text review is incomplete or stale; missing={missing}, "
            f"stale={stale}"
        )

    statuses = []
    reasons = []
    for row in candidates.itertuples(index=False):
        status, reason = DECISIONS[
            (row.query_split, int(row.query_row_index))
        ]
        statuses.append(status)
        reasons.append(reason)
    reviewed = candidates.copy()
    reviewed["review_status"] = statuses
    reviewed["review_reason"] = reasons
    reviewed["robust_exclude"] = (
        reviewed["review_status"] == CONFIRMED
    )
    reviewed["review_basis"] = "manual_pairwise_content_review"

    status_counts = Counter(reviewed["review_status"])
    exclusions = reviewed[reviewed["robust_exclude"]]
    exclusion_counts = Counter(exclusions["query_split"])
    summary = {
        "criteria": {
            "confirmed_near_duplicate": (
                "same substantive item or copied wording; source prefix, "
                "encoding, hashtag order, or attribution may differ"
            ),
            "related_not_duplicate": (
                "same topic/event but materially different fact, update, "
                "publisher report, or product post"
            ),
            "template_not_duplicate": (
                "shared boilerplate with a different event, identifier, "
                "named entity, or destination"
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
