"""Audit cross-split near-duplicate images without changing evaluation masks."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
from src.data_loader import load_dataset
from src.image_hashing import (
    HammingBKTree,
    hash_to_hex,
    hex_to_hash,
    perceptual_hash,
)

INTEGRITY_PATH = Path(PROCESSED_DATA_DIR) / "split_integrity.csv"
PHASH_CACHE_PATH = Path(PROCESSED_DATA_DIR) / "image_phash.csv"
METRICS_DIR = Path(REPORTS_DIR) / "metrics"
CANDIDATE_PATH = METRICS_DIR / "image_near_duplicate_candidates.csv"
SUMMARY_PATH = METRICS_DIR / "image_near_duplicate_summary.json"
MAX_HAMMING_DISTANCE = 4


def _load_manifest():
    if not INTEGRITY_PATH.exists():
        raise FileNotFoundError(
            "Integrity manifest missing. Run `python -m scripts.audit_data`."
        )
    manifest = pd.read_csv(INTEGRITY_PATH)
    return manifest.sort_values(["split", "row_index"], key=lambda values: (
        values.map({"train": 0, "val": 1, "test": 2})
        if values.name == "split"
        else values
    )).reset_index(drop=True)


def _cache_matches(cache, manifest):
    required = ["split", "row_index", "image", "image_sha256", "phash"]
    if any(column not in cache.columns for column in required):
        return False
    return (
        len(cache) == len(manifest)
        and cache["split"].astype(str).equals(manifest["split"].astype(str))
        and cache["row_index"].astype(int).equals(
            manifest["row_index"].astype(int)
        )
        and cache["image"].astype(str).equals(manifest["image"].astype(str))
        and cache["image_sha256"].astype(str).equals(
            manifest["image_sha256"].astype(str)
        )
    )


def _build_phash_cache(manifest, image_base):
    if PHASH_CACHE_PATH.exists():
        cache = pd.read_csv(PHASH_CACHE_PATH, dtype={"phash": str})
        if _cache_matches(cache, manifest):
            print(f"Reusing pHash cache: {PHASH_CACHE_PATH}")
            return cache

    rows = []
    total = len(manifest)
    for position, row in manifest.iterrows():
        path = Path(image_base) / row["image"]
        rows.append(
            {
                "split": row["split"],
                "row_index": int(row["row_index"]),
                "image": row["image"],
                "image_sha256": row["image_sha256"],
                "phash": hash_to_hex(perceptual_hash(path)),
            }
        )
        if (position + 1) % 2000 == 0 or position + 1 == total:
            print(f"Computed pHash: {position + 1}/{total}", flush=True)
    cache = pd.DataFrame(rows)
    PHASH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(PHASH_CACHE_PATH, index=False)
    return cache


def _attach_labels(manifest, splits):
    lookup_rows = []
    for split, frame in splits.items():
        selected = frame.reset_index(drop=True)[
            ["event_name", "label", "label_top"]
        ].copy()
        selected["split"] = split
        selected["row_index"] = range(len(selected))
        lookup_rows.append(selected)
    labels = pd.concat(lookup_rows, ignore_index=True)
    return manifest.merge(
        labels,
        on=["split", "row_index"],
        how="left",
        validate="one_to_one",
    )


def audit(max_distance=MAX_HAMMING_DISTANCE):
    train_df, val_df, test_df, image_base = load_dataset(
        use_sample_if_missing=False
    )
    manifest = _load_manifest()
    cache = _build_phash_cache(manifest, image_base)
    work = manifest.merge(
        cache[["split", "row_index", "phash"]],
        on=["split", "row_index"],
        how="left",
        validate="one_to_one",
    )
    work = _attach_labels(
        work, {"train": train_df, "val": val_df, "test": test_df}
    )

    tree = HammingBKTree()
    by_key = {
        (row["split"], int(row["row_index"])): row
        for _, row in work.iterrows()
    }
    for _, row in work[work["split"] == "train"].iterrows():
        key = (row["split"], int(row["row_index"]))
        tree.add(hex_to_hash(row["phash"]), key)

    candidates = []
    for split in ("val", "test"):
        part = work[work["split"] == split]
        for _, row in part.iterrows():
            if not bool(row["evaluation_eligible"]):
                continue
            matches = tree.search(
                hex_to_hash(row["phash"]), max_distance
            )
            non_exact = []
            for distance, prior_key in matches:
                prior = by_key[prior_key]
                if prior["image_sha256"] == row["image_sha256"]:
                    continue
                non_exact.append((distance, prior))
            if non_exact:
                distance, prior = min(
                    non_exact,
                    key=lambda item: (
                        item[0],
                        0 if item[1]["split"] == "train" else 1,
                        int(item[1]["row_index"]),
                    ),
                )
                nearest_match_count = sum(
                    match_distance == distance
                    for match_distance, _ in non_exact
                )
                candidates.append(
                    {
                        "query_split": split,
                        "query_row_index": int(row["row_index"]),
                        "query_image": row["image"],
                        "query_sha256": row["image_sha256"],
                        "query_phash": row["phash"],
                        "query_event": row["event_name"],
                        "query_label": row["label"],
                        "query_label_top": row["label_top"],
                        "prior_split": prior["split"],
                        "prior_row_index": int(prior["row_index"]),
                        "prior_image": prior["image"],
                        "prior_sha256": prior["image_sha256"],
                        "prior_phash": prior["phash"],
                        "prior_event": prior["event_name"],
                        "prior_label": prior["label"],
                        "prior_label_top": prior["label_top"],
                        "phash_distance": int(distance),
                        "nearest_match_count": int(nearest_match_count),
                        "same_event": (
                            row["event_name"] == prior["event_name"]
                        ),
                        "same_informative_label": (
                            row["label"] == prior["label"]
                        ),
                        "same_humanitarian_label": (
                            row["label_top"] == prior["label_top"]
                        ),
                        "current_evaluation_eligible": True,
                        "review_status": "candidate_not_verified",
                    }
                )
        for _, row in part.iterrows():
            key = (row["split"], int(row["row_index"]))
            tree.add(hex_to_hash(row["phash"]), key)

    candidate_df = pd.DataFrame(candidates)
    if not candidate_df.empty:
        candidate_df = candidate_df.sort_values(
            ["phash_distance", "query_split", "query_row_index"]
        )

    distance_counts = Counter(
        candidate_df["phash_distance"] if not candidate_df.empty else []
    )
    split_counts = Counter(
        candidate_df["query_split"] if not candidate_df.empty else []
    )
    summary = {
        "method": {
            "hash": "64-bit DCT perceptual hash",
            "max_hamming_distance": int(max_distance),
            "comparison": (
                "validation against train; test against train+validation"
            ),
            "scope": (
                "only rows still eligible under the exact SHA/text mask"
            ),
            "decision": (
                "candidate only; no evaluation row is removed automatically"
            ),
        },
        "rows_hashed": len(work),
        "additional_candidate_rows": len(candidate_df),
        "candidate_rows_by_split": {
            key: int(value) for key, value in sorted(split_counts.items())
        },
        "candidate_rows_by_distance": {
            str(key): int(value)
            for key, value in sorted(distance_counts.items())
        },
        "same_informative_label_candidates": int(
            candidate_df["same_informative_label"].sum()
        ) if not candidate_df.empty else 0,
        "same_humanitarian_label_candidates": int(
            candidate_df["same_humanitarian_label"].sum()
        ) if not candidate_df.empty else 0,
        "same_event_candidates": int(
            candidate_df["same_event"].sum()
        ) if not candidate_df.empty else 0,
        "candidates_with_multiple_nearest_matches": int(
            (candidate_df["nearest_match_count"] > 1).sum()
        ) if not candidate_df.empty else 0,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(CANDIDATE_PATH, index=False)
    with SUMMARY_PATH.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return candidate_df, summary


if __name__ == "__main__":
    audit()
